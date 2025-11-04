"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain
from .guardrails import (
    GuardrailsState,
    create_guardrails_guard,
    create_factuality_guard,
    create_input_guard_node,
    create_output_guard_node,
    create_refinement_node,
    create_error_response_node
)


class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.
    
    Args:
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        List of tools
    """
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))
    
    # Add Arxiv tool
    tools.append(ArxivQueryRun())
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))
    
    return tools


def create_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a simple LangGraph agent.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return END
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    graph.add_edge("action", "agent")
    
    return graph.compile()


def create_guarded_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    valid_topics: Optional[List[str]] = None,
    invalid_topics: Optional[List[str]] = None,
    enable_jailbreak_detection: bool = True,
    enable_pii_protection: bool = True,
    enable_profanity_check: bool = True,
    enable_factuality_check: bool = True,
    max_refinements: int = 3
):
    """Create a LangGraph agent with integrated Guardrails validation.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        valid_topics: List of valid topics to allow
        invalid_topics: List of invalid topics to block
        enable_jailbreak_detection: Whether to enable jailbreak detection
        enable_pii_protection: Whether to enable PII detection
        enable_profanity_check: Whether to enable profanity filtering
        enable_factuality_check: Whether to enable factuality checking
        max_refinements: Maximum number of refinement attempts
        
    Returns:
        Compiled LangGraph agent with guardrails
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    # Create guards
    input_guard = create_guardrails_guard(
        valid_topics=valid_topics or ["student loans", "financial aid", "education financing", "loan repayment"],
        invalid_topics=invalid_topics or ["investment advice", "crypto", "gambling", "politics"],
        enable_jailbreak_detection=enable_jailbreak_detection,
        enable_pii_protection=enable_pii_protection,
        enable_profanity_check=enable_profanity_check
    )
    
    output_guard = None
    if enable_profanity_check or enable_factuality_check:
        # Create output guard for content moderation and factuality
        from guardrails import Guard
        from guardrails.hub import ProfanityFree
        output_guard = Guard()
        if enable_profanity_check:
            output_guard = output_guard.use(
                ProfanityFree(threshold=0.8, validation_method="sentence", on_fail="exception")
            )
    
    # Create guard nodes
    input_guard_node = create_input_guard_node(input_guard, max_refinements=max_refinements)
    output_guard_node = None
    if output_guard:
        output_guard_node = create_output_guard_node(output_guard, max_refinements=max_refinements)
    
    # Create refinement and error nodes
    refinement_node = create_refinement_node(model, max_refinements=max_refinements)
    error_response_node = create_error_response_node()
    
    # Agent nodes
    def call_model(state: GuardrailsState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: GuardrailsState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "output_guard" if output_guard_node else END
    
    # Conditional routing functions
    def route_after_input_guard(state: GuardrailsState):
        """Route after input validation."""
        validation_error = state.get("validation_error")
        if validation_error:
            return "error_response"
        return "agent"
    
    def route_after_output_guard(state: GuardrailsState):
        """Route after output validation."""
        validation_error = state.get("validation_error")
        refinement_count = state.get("refinement_count", 0)
        
        if validation_error and refinement_count < max_refinements:
            return "refine"
        elif validation_error:
            return "error_response"
        return END
    
    # Build graph
    graph = StateGraph(GuardrailsState)
    tool_node = ToolNode(tools)
    
    # Add nodes
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("error_response", error_response_node)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    
    if output_guard_node:
        graph.add_node("output_guard", output_guard_node)
        graph.add_node("refine", refinement_node)
    
    # Set entry point
    graph.set_entry_point("input_guard")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "input_guard",
        route_after_input_guard,
        {
            "error_response": "error_response",
            "agent": "agent"
        }
    )
    
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "action": "action",
            "output_guard": "output_guard" if output_guard_node else END,
            END: END
        }
    )
    
    graph.add_edge("action", "agent")
    
    if output_guard_node:
        graph.add_conditional_edges(
            "output_guard",
            route_after_output_guard,
            {
                "refine": "refine",
                "error_response": "error_response",
                END: END
            }
        )
        
        graph.add_edge("refine", "output_guard")  # Loop back to validate refined output
    
    graph.add_edge("error_response", END)
    
    return graph.compile()
