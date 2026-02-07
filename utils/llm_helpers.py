"""
LLM helper utilities for Agentic Chaser.

Centralizes interaction with local Ollama LLM.
All agents should use these helpers instead of directly creating LLM instances.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from config.config import settings


def _raise_ollama_help(e: Exception) -> None:
    """Re-raise with a clearer message when Ollama model is missing (404)."""
    msg = str(e).lower()
    if "404" in msg or "not found" in msg or "model" in msg and "does not exist" in msg:
        model = getattr(settings.ollama, "model", "llama3.1:8b")
        raise RuntimeError(
            f"Ollama model {model!r} is not installed. "
            f"Run: ollama pull {model}  (or set OLLAMA_MODEL in .env to a model from 'ollama list')"
        ) from e
    raise


@lru_cache(maxsize=1)
def get_ollama_llm() -> BaseChatModel:
    """
    Get cached Ollama LLM instance.
    
    Configuration comes from environment:
        OLLAMA_MODEL (default: llama3.1:8b)
        OLLAMA_BASE_URL (default: http://localhost:11434)
    
    Returns:
        BaseChatModel: Configured ChatOllama instance
    """
    ollama_config = settings.ollama
    
    llm = ChatOllama(
        model=ollama_config.model,
        base_url=ollama_config.base_url,
        temperature=ollama_config.temperature,
        timeout=ollama_config.timeout,
    )
    
    return llm


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float] = None,
) -> str:
    """
    Simple chat completion helper.
    
    Args:
        system_prompt: System instruction
        user_prompt: User message
        temperature: Optional temperature override
    
    Returns:
        str: LLM response text
    
    Example:
        response = chat_completion(
            system_prompt="You are a helpful UK financial advisor assistant.",
            user_prompt="Explain LOA to a client in simple terms."
        )
    """
    llm = get_ollama_llm()
    
    # Override temperature if provided
    if temperature is not None:
        llm = ChatOllama(
            model=settings.ollama.model,
            base_url=settings.ollama.base_url,
            temperature=temperature,
            timeout=settings.ollama.timeout,
        )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    try:
        response: AIMessage = llm.invoke(messages)
    except Exception as e:
        _raise_ollama_help(e)
    return response.content


def chat_completion_with_history(
    system_prompt: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
) -> str:
    """
    Chat completion with conversation history.
    
    Args:
        system_prompt: System instruction
        messages: List of {"role": "user"/"assistant", "content": "..."}
        temperature: Optional temperature override
    
    Returns:
        str: LLM response text
    
    Example:
        response = chat_completion_with_history(
            system_prompt="You are a helpful assistant.",
            messages=[
                {"role": "user", "content": "What is an LOA?"},
                {"role": "assistant", "content": "LOA stands for..."},
                {"role": "user", "content": "How long does it take?"}
            ]
        )
    """
    llm = get_ollama_llm()
    
    if temperature is not None:
        llm = ChatOllama(
            model=settings.ollama.model,
            base_url=settings.ollama.base_url,
            temperature=temperature,
            timeout=settings.ollama.timeout,
        )
    
    # Build message list
    msg_list = [SystemMessage(content=system_prompt)]
    
    for msg in messages:
        if msg["role"] == "user":
            msg_list.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            msg_list.append(AIMessage(content=msg["content"]))
    try:
        response: AIMessage = llm.invoke(msg_list)
    except Exception as e:
        _raise_ollama_help(e)
    return response.content


def create_prompt_template(template: str) -> ChatPromptTemplate:
    """
    Create a reusable prompt template.
    
    Args:
        template: Template string with {variables}
    
    Returns:
        ChatPromptTemplate: Reusable template
    
    Example:
        prompt = create_prompt_template(
            "You are assisting {client_name}. They need: {missing_docs}"
        )
        chain = prompt | get_ollama_llm()
        response = chain.invoke({
            "client_name": "Alice",
            "missing_docs": "passport, utility bill"
        })
    """
    return ChatPromptTemplate.from_template(template)
