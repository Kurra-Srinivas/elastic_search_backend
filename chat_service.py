"""Chat service module using LangChain Groq with Agentic RAG Capabilities"""
import os
import json
import logging
from typing import List, Dict, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# --- IMPORT SEARCH TOOLS ---
from elasticsearch_service import (
    es, 
    hybrid_search, 
    check_elasticsearch_connection,
    INDEX_COMPANIES,
    INDEX_PERSONS, 
    INDEX_NOTES
)

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize the chat model
_chat_model: Optional[ChatGroq] = None

def get_chat_model(temperature: float = 0.3) -> ChatGroq:
    """Initialize ChatGroq model"""
    global _chat_model
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=temperature,
    )

async def agent_generate_queries(user_question: str, history: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Step 1: The Agent (With Memory)
    Analyzes user question to decide WHAT to search for.
    """
    model = get_chat_model(temperature=0.1)
    
    # Create context from recent history to handle pronouns like "it", "he", "she"
    history_context = ""
    if history:
        # Get last 2 messages for context
        last_exchange = history[-2:] if len(history) >= 2 else history
        history_context = f"RECENT CONVERSATION:\n{json.dumps(last_exchange)}"

    system_prompt = f"""You are a Search Query Generator. 
    Analyze the user's question and extract search terms.
    Use the RECENT CONVERSATION to resolve pronouns (e.g. if user says "when was it founded", check what company was discussed).
    
    {history_context}
    
    Return ONLY a JSON object with these keys:
    1. "entity_keywords": Keywords to find People, Companies, or Notes (e.g. names).
    2. "document_query": A natural language query to find info in Documents/Files.
    
    Example:
    User: "Where is TechFlow located?"
    JSON: {{"entity_keywords": "TechFlow", "document_query": "TechFlow location"}}
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ]
    
    try:
        response = await model.ainvoke(messages)
        content = response.content.strip()
        # Clean up JSON if LLM wraps it in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content)
    except Exception as e:
        logger.error(f"Agent query generation failed: {e}")
        return {"entity_keywords": user_question, "document_query": user_question}

def search_standard_tables(keywords: str) -> str:
    """
    Step 2: Search People, Companies, Notes (Keyword Search)
    UPDATED: Now explicitly extracts Location, Founded Year, and Education.
    """
    if not keywords or not check_elasticsearch_connection():
        return ""
        
    results_text = []
    print(f"\n--- DEBUG: Searching Database for '{keywords}' ---")

    # Search ALL fields (*) to ensure we don't miss anything
    body = {
        "query": {
            "multi_match": {
                "query": keywords,
                "fields": ["*"], 
                "fuzziness": "AUTO"
            }
        },
        "size": 3
    }
    
    # 1. Search Persons Index
    try:
        res = es.search(index=INDEX_PERSONS, body=body)
        for hit in res["hits"]["hits"]:
            s = hit["_source"]
            meta = s.get('metadata', {})
            
            # --- EXTRACT ALL FIELDS (Handling both root level and metadata level) ---
            name = s.get('name') or s.get('title') or meta.get('name') or "Unknown"
            company = s.get('company') or meta.get('company')
            designation = s.get('designation') or meta.get('designation')
            location = s.get('location') or meta.get('location') # <--- Explicitly added
            education = s.get('education') or meta.get('education') # <--- Explicitly added
            
            # Build the text block for the AI
            info = f"[Person] Name: {name}"
            if designation: info += f", Role: {designation}"
            if company: info += f", Company: {company}"
            if location: info += f", Location: {location}"
            if education: info += f", Education: {education}"
            
            results_text.append(info)
    except Exception as e: 
        print(f"DEBUG: Person search error: {e}")

    # 2. Search Companies Index
    try:
        res = es.search(index=INDEX_COMPANIES, body=body)
        for hit in res["hits"]["hits"]:
            s = hit["_source"]
            meta = s.get('metadata', {})
            
            # --- EXTRACT ALL FIELDS ---
            name = s.get('name') or s.get('title') or meta.get('name') or "Unknown"
            industry = s.get('industry') or meta.get('industry')
            location = s.get('location') or meta.get('location') # <--- Explicitly added
            founded = s.get('founded') or meta.get('founded')    # <--- Explicitly added
            description = s.get('description') or meta.get('description') or ""
            
            # Build the text block for the AI
            info = f"[Company] Name: {name}"
            if industry: info += f", Industry: {industry}"
            if location: info += f", Location: {location}"
            if founded: info += f", Founded: {founded}"
            if description: info += f", Description: {description[:200]}..."
            
            results_text.append(info)
    except Exception: pass

    # 3. Search Notes Index
    try:
        res = es.search(index=INDEX_NOTES, body=body)
        for hit in res["hits"]["hits"]:
            s = hit["_source"]
            content = s.get('content') or s.get('note') or "No content"
            # Try to find who the note belongs to
            card_id = s.get('card_id', 'Unknown')
            results_text.append(f"[Note] Attached to card {card_id}: {content}")
    except Exception: pass
    
    return "\n".join(results_text)

async def chat_with_ai(
    message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Main RAG Loop
    """
    try:
        # 1. GENERATE QUERIES (with history context)
        queries = await agent_generate_queries(message, conversation_history or [])
        kw_query = queries.get("entity_keywords", message)
        doc_query = queries.get("document_query", message)
        
        logger.info(f"Queries -> Entity: '{kw_query}' | Doc: '{doc_query}'")
        
        # 2. SEARCH DATABASES (Persons, Companies, Notes)
        entity_context = search_standard_tables(kw_query)
        
        # 3. SEARCH DOCUMENTS (Vector/Hybrid)
        doc_results = []
        if check_elasticsearch_connection():
            # Uses the hybrid_search function imported from elasticsearch_service
            # Ensure elasticsearch_service.py has hybrid_search aimed at INDEX_DOCUMENTS
            doc_results = hybrid_search(query=doc_query, limit=10)
            
        doc_context_pieces = []
        if doc_results:
            for r in doc_results:
                # Only include if it's actually a document
                if r.get('card_type') == 'document':
                    title = r.get('title', 'Unknown')
                    content = r.get('content', '')[:800] # Truncate chunks
                    doc_context_pieces.append(f"[Document File] Title: {title}\nContent: {content}")
        
        doc_context = "\n\n".join(doc_context_pieces)

        # 4. SYNTHESIZE ANSWER
        final_context = f"""
        === DATABASE RECORDS (People, Companies, Notes) ===
        {entity_context if entity_context else "No matching database records found."}
        
        === UPLOADED FILES & DOCUMENTS ===
        {doc_context if doc_context else "No relevant document sections found."}
        """
        
        # Debug print to see what the AI sees (check your terminal!)
        print(f"DEBUG CONTEXT:\n{final_context[:500]}...") 

        model = get_chat_model(temperature=0.3)
        
        system_prompt = f"""You are a smart Knowledge Assistant for a VC firm. 
        Answer the user's question using ONLY the provided context below.
        
        CONTEXT:
        {final_context}
        
        INSTRUCTIONS:
        1. If looking for a Company: Provide Name, Location, Industry, and Founded Year if available.
        2. If looking for a Person: Provide Name, Company, Designation, Location, and Education if available.
        3. If looking for Documents: Summarize the content from the [Document File] sections.
        4. If the info is in a [Note], mention it explicitly (e.g. "There is a note saying...").
        5. If the answer is not in the context, say "I don't have that information in the database."
        """
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
        
        messages.append(HumanMessage(content=message))
        
        response = await model.ainvoke(messages)
        return response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return f"I encountered an error. Details: {str(e)}"

def reset_chat_model():
    global _chat_model
    _chat_model = None