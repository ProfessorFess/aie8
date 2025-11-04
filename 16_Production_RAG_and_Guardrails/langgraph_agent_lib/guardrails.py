"""Guardrails integration for production-safe LangGraph agents.

This module provides utilities for integrating Guardrails AI validation
into LangGraph agent workflows, including input and output validation.
"""

import logging
from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict, Annotated

from guardrails.hub import (
    RestrictToTopic,
    DetectJailbreak,
    CompetitorCheck,
    LlmRagEvaluator,
    HallucinationPrompt,
    ProfanityFree,
    GuardrailsPII
)
from guardrails import Guard
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

# Set up logging
logger = logging.getLogger(__name__)


class GuardrailsState(TypedDict):
    """State schema for guardrails-enabled agent graphs.
    
    Attributes:
        messages: List of messages in the conversation history.
        validation_results: Optional validation results from guardrails.
        validation_error: Optional error message from failed validation.
        refinement_count: Number of refinement attempts.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    validation_results: Optional[Dict[str, Any]]
    validation_error: Optional[str]
    refinement_count: int


def create_guardrails_guard(
    valid_topics: Optional[List[str]] = None,
    invalid_topics: Optional[List[str]] = None,
    enable_jailbreak_detection: bool = True,
    enable_pii_protection: bool = True,
    enable_profanity_check: bool = True,
    enable_competitor_check: bool = False,
    pii_entities: Optional[List[str]] = None
) -> Guard:
    """Create a Guardrails guard with common production safety checks.
    
    Args:
        valid_topics: List of valid topics to allow. None disables topic restriction.
        invalid_topics: List of invalid topics to block. None disables topic restriction.
        enable_jailbreak_detection: Whether to enable jailbreak detection. Default: True.
        enable_pii_protection: Whether to enable PII detection and redaction. Default: True.
        enable_profanity_check: Whether to enable profanity filtering. Default: True.
        enable_competitor_check: Whether to enable competitor mention detection. Default: False.
        pii_entities: List of PII entity types to detect. Default: Common PII types.
        
    Returns:
        Configured Guard instance.
        
    Raises:
        RuntimeError: If guard configuration fails.
    """
    guard = Guard()
    
    try:
        # Topic restriction
        if valid_topics or invalid_topics:
            guard = guard.use(
                RestrictToTopic(
                    valid_topics=valid_topics or [],
                    invalid_topics=invalid_topics or [],
                    disable_classifier=True,
                    disable_llm=False,
                    on_fail="exception"
                )
            )
            logger.debug("Topic restriction guard configured")
        
        # Jailbreak detection
        if enable_jailbreak_detection:
            guard = guard.use(DetectJailbreak())
            logger.debug("Jailbreak detection guard configured")
        
        # PII protection
        if enable_pii_protection:
            default_entities = ["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"]
            entities = pii_entities or default_entities
            guard = guard.use(
                GuardrailsPII(
                    entities=entities,
                    on_fail="fix"
                )
            )
            logger.debug(f"PII protection guard configured for entities: {entities}")
        
        # Profanity check
        if enable_profanity_check:
            guard = guard.use(
                ProfanityFree(
                    threshold=0.8,
                    validation_method="sentence",
                    on_fail="exception"
                )
            )
            logger.debug("Profanity check guard configured")
        
        # Competitor check (optional)
        if enable_competitor_check:
            guard = guard.use(CompetitorCheck())
            logger.debug("Competitor check guard configured")
        
        logger.info("Guardrails guard configured successfully")
        return guard
        
    except Exception as e:
        logger.error(f"Failed to configure guardrails: {e}", exc_info=True)
        raise RuntimeError(f"Failed to configure guardrails: {e}") from e


def create_factuality_guard(
    eval_model: str = "gpt-4.1-mini",
    on_prompt: bool = True
) -> Guard:
    """Create a factuality guard for RAG responses.
    
    Args:
        eval_model: Model to use for factuality evaluation. Default: "gpt-4.1-mini".
        on_prompt: Whether to validate at prompt stage or response stage. Default: True.
        
    Returns:
        Configured Guard instance for factuality checking.
        
    Raises:
        RuntimeError: If guard configuration fails.
    """
    try:
        guard = Guard().use(
            LlmRagEvaluator(
                eval_llm_prompt_generator=HallucinationPrompt(prompt_name="hallucination_judge_llm"),
                llm_evaluator_fail_response="hallucinated",
                llm_evaluator_pass_response="factual",
                llm_callable=eval_model,
                on_fail="exception",
                on="prompt" if on_prompt else "response"
            )
        )
        logger.info(f"Factuality guard configured with model: {eval_model}")
        return guard
    except Exception as e:
        logger.error(f"Failed to configure factuality guard: {e}", exc_info=True)
        raise RuntimeError(f"Failed to configure factuality guard: {e}") from e


def validate_input(
    guard: Guard,
    user_input: str,
    raise_on_failure: bool = True
) -> Dict[str, Any]:
    """Validate user input using a Guardrails guard.
    
    Args:
        guard: The Guard instance to use for validation.
        user_input: The user input to validate.
        raise_on_failure: Whether to raise an exception on validation failure.
            If False, returns validation result. Default: True.
        
    Returns:
        Dictionary with validation results including:
        - validation_passed: Boolean indicating if validation passed
        - validated_output: The validated (and potentially modified) output
        - error: Error message if validation failed
        
    Raises:
        RuntimeError: If validation fails and raise_on_failure is True.
    """
    try:
        result = guard.validate(user_input)
        
        validation_result = {
            "validation_passed": result.validation_passed,
            "validated_output": result.validated_output if hasattr(result, 'validated_output') else user_input,
            "error": None
        }
        
        if not result.validation_passed and raise_on_failure:
            error_msg = f"Input validation failed: {getattr(result, 'error', 'Unknown error')}"
            logger.warning(f"Input validation failed: {user_input[:100]}...")
            raise RuntimeError(error_msg)
        
        return validation_result
        
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Input validation error: {e}", exc_info=True)
        if raise_on_failure:
            raise RuntimeError(f"Input validation failed: {e}") from e
        return {
            "validation_passed": False,
            "validated_output": user_input,
            "error": str(e)
        }


def validate_output(
    guard: Guard,
    agent_response: str,
    context: Optional[str] = None,
    raise_on_failure: bool = True
) -> Dict[str, Any]:
    """Validate agent output using a Guardrails guard.
    
    Args:
        guard: The Guard instance to use for validation.
        agent_response: The agent's response to validate.
        context: Optional context for factuality checking.
        raise_on_failure: Whether to raise an exception on validation failure.
            If False, returns validation result. Default: True.
        
    Returns:
        Dictionary with validation results.
        
    Raises:
        RuntimeError: If validation fails and raise_on_failure is True.
    """
    try:
        # For factuality guards, include context if provided
        if context:
            result = guard.validate(agent_response, metadata={"context": context})
        else:
            result = guard.validate(agent_response)
        
        validation_result = {
            "validation_passed": result.validation_passed,
            "validated_output": result.validated_output if hasattr(result, 'validated_output') else agent_response,
            "error": None
        }
        
        if not result.validation_passed and raise_on_failure:
            error_msg = f"Output validation failed: {getattr(result, 'error', 'Unknown error')}"
            logger.warning(f"Output validation failed: {agent_response[:100]}...")
            raise RuntimeError(error_msg)
        
        return validation_result
        
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Output validation error: {e}", exc_info=True)
        if raise_on_failure:
            raise RuntimeError(f"Output validation failed: {e}") from e
        return {
            "validation_passed": False,
            "validated_output": agent_response,
            "error": str(e)
        }


def create_input_guard_node(
    input_guard: Guard,
    max_refinements: int = 3
):
    """Create a LangGraph node for input validation with refinement support.
    
    Args:
        input_guard: Guard for validating user inputs.
        max_refinements: Maximum number of refinement attempts. Default: 3.
        
    Returns:
        A function that can be used as a LangGraph node.
    """
    def input_guard_node(state: GuardrailsState) -> Dict[str, Any]:
        """Validate user input with guardrails.
        
        Args:
            state: Current agent state with messages.
            
        Returns:
            Updated state with validation results.
        """
        messages = state.get("messages", [])
        refinement_count = state.get("refinement_count", 0)
        
        if not messages:
            return {"validation_results": None, "validation_error": None}
        
        last_message = messages[-1]
        
        if not isinstance(last_message, HumanMessage):
            return {"validation_results": None, "validation_error": None}
        
        try:
            logger.info(f"Validating user input (attempt {refinement_count + 1}/{max_refinements + 1})")
            result = validate_input(
                input_guard,
                last_message.content,
                raise_on_failure=False  # Don't raise, return error instead
            )
            
            validation_result = {
                "type": "input",
                "passed": result["validation_passed"],
                "validated_output": result.get("validated_output"),
                "error": result.get("error")
            }
            
            if result["validation_passed"]:
                logger.info("Input validation passed")
                return {
                    "validation_results": validation_result,
                    "validation_error": None,
                    "refinement_count": 0
                }
            else:
                error_msg = result.get("error", "Input validation failed")
                logger.warning(f"Input validation failed: {error_msg}")
                
                if refinement_count >= max_refinements:
                    logger.error(f"Max refinements ({max_refinements}) reached")
                    return {
                        "validation_results": validation_result,
                        "validation_error": f"Input validation failed after {max_refinements} attempts: {error_msg}",
                        "refinement_count": refinement_count
                    }
                
                return {
                    "validation_results": validation_result,
                    "validation_error": error_msg,
                    "refinement_count": refinement_count + 1
                }
                
        except Exception as e:
            logger.error(f"Input validation error: {e}", exc_info=True)
            return {
                "validation_results": {
                    "type": "input",
                    "passed": False,
                    "error": str(e)
                },
                "validation_error": str(e),
                "refinement_count": refinement_count
            }
    
    return input_guard_node


def create_output_guard_node(
    output_guard: Guard,
    context_retriever: Optional[Any] = None,
    max_refinements: int = 3
):
    """Create a LangGraph node for output validation with refinement support.
    
    Args:
        output_guard: Guard for validating agent outputs.
        context_retriever: Optional retriever to get context for factuality checking.
        max_refinements: Maximum number of refinement attempts. Default: 3.
        
    Returns:
        A function that can be used as a LangGraph node.
    """
    def output_guard_node(state: GuardrailsState) -> Dict[str, Any]:
        """Validate agent output with guardrails.
        
        Args:
            state: Current agent state with messages.
            
        Returns:
            Updated state with validation results.
        """
        messages = state.get("messages", [])
        refinement_count = state.get("refinement_count", 0)
        
        if not messages:
            return {"validation_results": None, "validation_error": None}
        
        last_message = messages[-1]
        
        if not isinstance(last_message, AIMessage):
            return {"validation_results": None, "validation_error": None}
        
        try:
            logger.info(f"Validating agent output (attempt {refinement_count + 1}/{max_refinements + 1})")
            
            # Get context if available for factuality checking
            context = None
            if context_retriever:
                try:
                    # Try to get context from previous messages
                    for msg in reversed(messages[:-1]):
                        if hasattr(msg, "content") and "context" in str(msg.content).lower():
                            context = str(msg.content)
                            break
                except Exception:
                    pass
            
            result = validate_output(
                output_guard,
                last_message.content,
                context=context,
                raise_on_failure=False  # Don't raise, return error instead
            )
            
            validation_result = {
                "type": "output",
                "passed": result["validation_passed"],
                "validated_output": result.get("validated_output"),
                "error": result.get("error")
            }
            
            if result["validation_passed"]:
                logger.info("Output validation passed")
                return {
                    "validation_results": validation_result,
                    "validation_error": None,
                    "refinement_count": 0
                }
            else:
                error_msg = result.get("error", "Output validation failed")
                logger.warning(f"Output validation failed: {error_msg}")
                
                if refinement_count >= max_refinements:
                    logger.error(f"Max refinements ({max_refinements}) reached")
                    return {
                        "validation_results": validation_result,
                        "validation_error": f"Output validation failed after {max_refinements} attempts: {error_msg}",
                        "refinement_count": refinement_count
                    }
                
                return {
                    "validation_results": validation_result,
                    "validation_error": error_msg,
                    "refinement_count": refinement_count + 1
                }
                
        except Exception as e:
            logger.error(f"Output validation error: {e}", exc_info=True)
            return {
                "validation_results": {
                    "type": "output",
                    "passed": False,
                    "error": str(e)
                },
                "validation_error": str(e),
                "refinement_count": refinement_count
            }
    
    return output_guard_node


def create_guardrails_node(
    input_guard: Optional[Guard] = None,
    output_guard: Optional[Guard] = None,
    strict_mode: bool = True
):
    """Create a LangGraph node that validates inputs and outputs with Guardrails.
    
    Args:
        input_guard: Guard for validating user inputs. If None, input validation is skipped.
        output_guard: Guard for validating agent outputs. If None, output validation is skipped.
        strict_mode: If True, raises exceptions on validation failure.
            If False, logs warnings but continues. Default: True.
        
    Returns:
        A function that can be used as a LangGraph node.
    """
    def guardrails_node(state: GuardrailsState) -> Dict[str, Any]:
        """Validate messages in the agent state.
        
        Args:
            state: Current agent state with messages.
            
        Returns:
            Updated state with validation results.
        """
        messages = state.get("messages", [])
        validation_results = []
        
        if not messages:
            return {"validation_results": []}
        
        # Validate the last message
        last_message = messages[-1]
        
        try:
            if isinstance(last_message, HumanMessage) and input_guard:
                # Validate user input
                logger.debug("Validating user input with guardrails")
                result = validate_input(
                    input_guard,
                    last_message.content,
                    raise_on_failure=strict_mode
                )
                validation_results.append({
                    "type": "input",
                    "passed": result["validation_passed"],
                    "message": last_message.content[:100]
                })
                
                # If validation modified the input, we could update the message here
                if not result["validation_passed"] and strict_mode:
                    logger.error(f"Input validation failed: {result.get('error')}")
            
            elif isinstance(last_message, AIMessage) and output_guard:
                # Validate agent output
                logger.debug("Validating agent output with guardrails")
                result = validate_output(
                    output_guard,
                    last_message.content,
                    raise_on_failure=strict_mode
                )
                validation_results.append({
                    "type": "output",
                    "passed": result["validation_passed"],
                    "message": last_message.content[:100]
                })
                
                if not result["validation_passed"] and strict_mode:
                    logger.error(f"Output validation failed: {result.get('error')}")
                    
        except Exception as e:
            logger.error(f"Guardrails validation error: {e}", exc_info=True)
            if strict_mode:
                raise
            validation_results.append({
                "type": "error",
                "passed": False,
                "error": str(e)
            })
        
        return {"validation_results": validation_results}
    
    return guardrails_node


def create_refinement_node(
    model,
    max_refinements: int = 3
):
    """Create a node that refines failed outputs based on validation errors.
    
    Args:
        model: LLM model to use for refinement.
        max_refinements: Maximum number of refinement attempts. Default: 3.
        
    Returns:
        A function that can be used as a LangGraph node.
    """
    def refinement_node(state: GuardrailsState) -> Dict[str, Any]:
        """Refine agent output based on validation errors.
        
        Args:
            state: Current agent state with messages and validation results.
            
        Returns:
            Updated state with refined message.
        """
        messages = state.get("messages", [])
        validation_error = state.get("validation_error")
        refinement_count = state.get("refinement_count", 0)
        
        if not validation_error or not messages:
            return {}
        
        last_message = messages[-1]
        
        if not isinstance(last_message, AIMessage):
            return {}
        
        if refinement_count >= max_refinements:
            logger.error(f"Max refinements reached, returning error message")
            error_message = AIMessage(
                content=f"I apologize, but I was unable to generate a response that meets the safety requirements. "
                       f"Error: {validation_error}. Please try rephrasing your query."
            )
            return {
                "messages": [error_message],
                "refinement_count": refinement_count
            }
        
        try:
            logger.info(f"Refining output (attempt {refinement_count + 1}/{max_refinements})")
            
            refinement_prompt = f"""Your previous response failed validation with the following error:
{validation_error}

Please refine your response to address this issue. Be sure to:
1. Stay on-topic and relevant
2. Avoid any inappropriate content
3. Ensure factual accuracy
4. Maintain a professional tone

Original response that needs refinement:
{last_message.content}

Provide a refined response:"""
            
            refinement_message = HumanMessage(content=refinement_prompt)
            refined_response = model.invoke([refinement_message])
            
            return {
                "messages": [refined_response],
                "refinement_count": refinement_count + 1
            }
            
        except Exception as e:
            logger.error(f"Refinement error: {e}", exc_info=True)
            error_message = AIMessage(
                content=f"I encountered an error while trying to refine my response. "
                       f"Please try rephrasing your query. Error: {str(e)}"
            )
            return {
                "messages": [error_message],
                "refinement_count": refinement_count + 1
            }
    
    return refinement_node


def create_error_response_node():
    """Create a node that generates error responses for failed validations.
    
    Returns:
        A function that can be used as a LangGraph node.
    """
    def error_response_node(state: GuardrailsState) -> Dict[str, Any]:
        """Generate a user-friendly error response.
        
        Args:
            state: Current agent state with validation error.
            
        Returns:
            Updated state with error message.
        """
        validation_error = state.get("validation_error")
        validation_results = state.get("validation_results")
        
        if not validation_error:
            return {}
        
        error_type = "input" if validation_results and validation_results.get("type") == "input" else "output"
        
        if error_type == "input":
            error_message = AIMessage(
                content="I'm sorry, but I cannot process that request. It appears to be outside my scope "
                       "or may contain inappropriate content. Please try rephrasing your question to focus "
                       "on student loans, financial aid, or education financing."
            )
        else:
            error_message = AIMessage(
                content="I apologize, but I was unable to generate an appropriate response. "
                       "Please try rephrasing your question or ask about a different aspect of the topic."
            )
        
        logger.warning(f"Generating error response for {error_type} validation failure")
        
        return {
            "messages": [error_message],
            "validation_error": None  # Clear error after responding
        }
    
    return error_response_node

