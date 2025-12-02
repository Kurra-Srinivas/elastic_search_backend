import os
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

def perform_web_search(query: str, max_results: int = 5) -> str:
    """
    Executes a web search using Tavily API (Optimized for LLMs).
    Returns a formatted string of results suitable for RAG context.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY is missing in .env file")
        return ""

    print(f"🌐 [TAVILY] Searching the web for: '{query}'")

    try:
        # Initialize the LangChain Tavily Tool
        # 'include_answer=True' gives a direct short answer calculated by Tavily
        # 'search_depth="advanced"' ensures high quality sources
        tool = TavilySearchResults(
            max_results=max_results,
            search_depth="advanced", 
            include_answer=True,
            include_raw_content=False 
        )
        
        # Execute Search
        # LangChain tools expect a dictionary input for 'query'
        results = tool.invoke({"query": query})
        
        if not results:
            return ""

        # Format the output for the Chat Service
        formatted_context = []
        
        # Check if results is a list of dicts (standard behavior)
        if isinstance(results, list):
            for i, res in enumerate(results):
                content = res.get("content", "").strip()
                url = res.get("url", "No URL")
                
                # Skip empty results
                if len(content) < 10: 
                    continue
                    
                formatted_context.append(f"• [Web Result] {content}\n  Source: {url}")
        else:
            # Fallback if Tavily returns a single string
            return str(results)
            
        return "\n\n".join(formatted_context)

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        # Return empty string so the chat service just says "I don't know" 
        # instead of crashing or showing an error to the user.
        return ""