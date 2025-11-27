"""Chat service module using LangChain Groq"""
import os
from typing import List, Dict, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize the chat model
_chat_model: Optional[ChatGroq] = None


def get_chat_model() -> ChatGroq:
    """Initialize and return the ChatGroq model"""
    global _chat_model
    
    if _chat_model is not None:
        return _chat_model
    
    # Get API key from environment variable
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please set it to use the chat functionality."
        )
    
    _chat_model = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.7,
    )
    
    return _chat_model


def get_system_prompt() -> str:
    """Get the system prompt for the AI assistant"""
    return """You are a helpful AI assistant for a company and founder search platform. 
You help users find information about companies, founders, CEOs, and their LinkedIn profiles.
Be friendly, concise, and helpful. If you don't know something, say so honestly.
Keep responses clear and to the point."""


async def chat_with_ai(
    message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Chat with AI using LangChain Groq
    
    Args:
        message: The user's message
        conversation_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}]
    
    Returns:
        AI's response as a string
    """
    try:
        model = get_chat_model()
        
        # Build messages list
        messages = []
        
        # Add system message
        system_prompt = get_system_prompt()
        messages.append(SystemMessage(content=system_prompt))
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        
        # Add current user message
        messages.append(HumanMessage(content=message))
        
        # Get response from the model
        response = await model.ainvoke(messages)
        
        # Extract the content from the response
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
            
    except Exception as e:
        error_msg = f"Error in chat service: {str(e)}"
        print(error_msg)
        return f"I apologize, but I encountered an error. Please check that GROQ_API_KEY is set correctly. Error: {str(e)}"


def reset_chat_model():
    """Reset the chat model (useful for testing)"""
    global _chat_model
    _chat_model = None

