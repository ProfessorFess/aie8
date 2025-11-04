"""Simple Agent that uses A2A protocol to call the General Purpose Agent server."""
import logging
from typing import Annotated, TypedDict, List, Dict, Any
from uuid import uuid4

import httpx
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAgentState(TypedDict):
    """State schema for the simple agent graph."""
    messages: Annotated[List[BaseMessage], add_messages]


class SimpleAgent:
    """Simple Agent that communicates with the A2A server using LangGraph."""

    def __init__(self, base_url: str = "http://localhost:10000"):
        """Initialize the Simple Agent with A2A server URL."""
        self.base_url = base_url
        self.httpx_client = None
        self.a2a_client = None
        self.agent_card = None

    async def _initialize_client(self):
        """Initialize the A2A client and fetch agent card."""
        if self.a2a_client is None:
            self.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
            resolver = A2ACardResolver(
                httpx_client=self.httpx_client,
                base_url=self.base_url,
            )
            self.agent_card = await resolver.get_agent_card()
            self.a2a_client = A2AClient(
                httpx_client=self.httpx_client,
                agent_card=self.agent_card
            )
            logger.info("A2A client initialized successfully")

    async def _call_a2a_server(self, state: SimpleAgentState) -> Dict[str, Any]:
        """Call the A2A server with the user's message."""
        await self._initialize_client()
        
        # Get the last user message
        messages = state["messages"]
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg
                break
        
        if not user_message:
            return {"messages": [AIMessage(content="No user message found")]}

        # Prepare the message for A2A protocol
        send_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': user_message.content}
                ],
                'message_id': uuid4().hex,
            },
        }
        
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**send_message_payload)
        )

        # Send message to A2A server
        response = await self.a2a_client.send_message(request)
        
        # Extract the response content from the A2A response
        content = ""
        try:
            result = response.root.result
            if result:
                # Try to get content from artifacts first
                if hasattr(result, 'artifacts') and result.artifacts:
                    for artifact in result.artifacts:
                        if hasattr(artifact, 'parts') and artifact.parts:
                            for part in artifact.parts:
                                if hasattr(part, 'root'):
                                    root = part.root
                                    if hasattr(root, 'text'):
                                        content += root.text + "\n"
                
                # Fallback to messages if artifacts don't have content
                if not content and hasattr(result, 'messages') and result.messages:
                    for msg in result.messages:
                        if hasattr(msg, 'parts') and msg.parts:
                            for part in msg.parts:
                                if hasattr(part, 'kind') and part.kind == 'text' and hasattr(part, 'text'):
                                    content += part.text + "\n"
                
                # If still no content, use the model dump as a fallback
                if not content:
                    response_dict = response.model_dump(mode='json', exclude_none=True)
                    content = f"Response from A2A server: {str(response_dict)}"
        except Exception as e:
            logger.error(f"Error extracting response content: {e}")
            content = f"Received response from A2A server, but error extracting content: {str(e)}"
        
        if not content:
            content = "Received response from A2A server, but no content found."

        return {"messages": [AIMessage(content=content.strip())]}

    def build_graph(self):
        """Build the LangGraph graph for the Simple Agent."""
        graph = StateGraph(SimpleAgentState)
        
        # Add the node that calls the A2A server
        graph.add_node("call_a2a_server", self._call_a2a_server)
        
        # Set entry point
        graph.set_entry_point("call_a2a_server")
        
        # After calling A2A server, end
        graph.add_edge("call_a2a_server", END)
        
        return graph.compile()

    async def cleanup(self):
        """Clean up resources."""
        if self.httpx_client:
            await self.httpx_client.aclose()
            self.httpx_client = None
            self.a2a_client = None


async def run_simple_agent(query: str, base_url: str = "http://localhost:10000"):
    """Run the simple agent with a query."""
    agent = SimpleAgent(base_url=base_url)
    graph = agent.build_graph()
    
    try:
        # Run the graph
        initial_state = {"messages": [HumanMessage(content=query)]}
        result = await graph.ainvoke(initial_state)
        
        # Get the final response
        if result.get("messages"):
            final_message = result["messages"][-1]
            if isinstance(final_message, AIMessage):
                return final_message.content
        
        return "No response received"
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    import asyncio
    
    async def main():
        query = "What are the latest developments in artificial intelligence?"
        response = await run_simple_agent(query)
        print(f"\nQuery: {query}")
        print(f"Response: {response}")
    
    asyncio.run(main())

