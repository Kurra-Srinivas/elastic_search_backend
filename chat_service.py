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

# --- IMPORT WEB SEARCH ---
from web_search_service import perform_web_search

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
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        groq_api_key=api_key,
        temperature=temperature,
    )

async def agent_generate_queries(user_question: str, history: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Step 1: The Agent
    Analyzes user question to decide WHAT to search for.
    """
    model = get_chat_model(temperature=0.1)
    
    history_context = ""
    if history:
        last_exchange = history[-4:] if len(history) >= 4 else history
        history_str = json.dumps(last_exchange, indent=2)
        history_context = f"CONVERSATION HISTORY:\n{history_str}"

    system_prompt = f"""You are a Search Query Generator. 
    Analyze the user's question and extract CLEAN search terms.
    
    {history_context}
    
    INSTRUCTIONS:
    1. **entity_keywords**: Extract ONLY the core values (Names, Years, Cities, Roles).
       - CRITICAL: REMOVE generic words like 'company', 'companies', 'firm', 'list', 'show me', 'where is', 'in', 'founded', 'year', 'located', 'based', 'from'.
       - User: "Companies founded in 2020" -> "2020"
       - User: "Companies from Boston" -> "Boston"
       - User: "Who is the CEO of Apple?" -> "Apple CEO"
       - User: "Any meetings on Monday?" -> "Meeting Monday"
       
    2. **web_query**: If the query is general (like "Apple CEO", "Stock Price", "Recipe", "Mahesh Babu Father", "Who is Bhagat Singh") and likely NOT in a private VC database, provide a web search string.
    
    Return ONLY a JSON object:
    {{
        "entity_keywords": "...", 
        "document_query": "...",
        "web_query": "..."
    }}
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ]
    
    try:
        response = await model.ainvoke(messages)
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content)
    except Exception as e:
        logger.error(f"Agent query generation failed: {e}")
        return {
            "entity_keywords": user_question, 
            "document_query": user_question,
            "web_query": user_question
        }

def format_metadata(meta: Dict) -> str:
    """Helper to format metadata dictionary into a readable string for the LLM"""
    if not meta: return ""
    clean_meta = {}
    for k, v in meta.items():
        if k not in ['id', 'card_id', 'text_embedding', 'chunk_index'] and v:
            clean_meta[k] = v
    return ", ".join([f"{k}: {v}" for k, v in clean_meta.items()])

def search_standard_tables(keywords: str) -> Dict[str, str]:
    """
    Step 2: Search People, Companies, Notes
    """
    if not keywords or not check_elasticsearch_connection():
        return {"entities": "", "notes": ""}
        
    entity_results = []
    note_results = []
    
    print(f"\n--- DEBUG: Searching Database for '{keywords}' ---")

    # --- LOGIC A: ENTITY SEARCH (Strict) ---
    body_strict = {
        "query": {
            "multi_match": {
                "query": keywords,
                "fields": ["*"], 
                "fuzziness": "AUTO",
                "minimum_should_match": "2<100%" 
            }
        },
        "size": 10
    }

    # --- LOGIC B: NOTE SEARCH (Loose) ---
    body_loose = {
        "query": {
            "multi_match": {
                "query": keywords,
                "fields": ["content", "title", "metadata.*"], 
                "fuzziness": "AUTO",
                "operator": "or" 
            }
        },
        "size": 15
    }
    
    # 1. Search Persons
    try:
        res = es.search(index=INDEX_PERSONS, body=body_strict)
        for hit in res["hits"]["hits"]:
            s = hit["_source"]
            meta = s.get('metadata', {})
            name = s.get('name') or s.get('title') or "Unknown"
            # Format nicely for the LLM
            info = f"PERSON: {name}\n   - Title: {meta.get('designation', 'N/A')}\n   - Company: {meta.get('company', 'N/A')}\n   - Location: {meta.get('location', 'N/A')}\n   - Education: {meta.get('education', 'N/A')}"
            entity_results.append(info)
    except Exception: pass

    # 2. Search Companies
    try:
        res = es.search(index=INDEX_COMPANIES, body=body_strict)
        for hit in res["hits"]["hits"]:
            s = hit["_source"]
            meta = s.get('metadata', {})
            name = s.get('name') or s.get('title') or "Unknown"
            # Format nicely for the LLM
            info = f"COMPANY: {name}\n   - Industry: {meta.get('industry', 'N/A')}\n   - Location: {meta.get('location', 'N/A')}\n   - Founded: {meta.get('founded', 'N/A')}\n   - Description: {meta.get('description', 'N/A')}"
            entity_results.append(info)
    except Exception: pass

    # 3. Search Notes
    try:
        res = es.search(index=INDEX_NOTES, body=body_loose)
        for hit in res["hits"]["hits"]:
            s = hit["_source"]
            content = s.get('content') or s.get('note') or "No content"
            meta = s.get('metadata', {})
            
            owner_name = meta.get('person_name') or meta.get('company_name')
            if not owner_name:
                owner_name = s.get('title', 'Unknown Entity')
            
            note_results.append(f"NOTE for {owner_name}: {content}")
    except Exception: pass
    
    return {
        "entities": "\n\n".join(entity_results),
        "notes": "\n".join(note_results)
    }

async def chat_with_ai(
    message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Main RAG Loop
    """
    try:
        # 1. GENERATE QUERIES
        queries = await agent_generate_queries(message, conversation_history or [])
        kw_query = queries.get("entity_keywords", message)
        doc_query = queries.get("document_query", message)
        web_query = queries.get("web_query", message)
        
        logger.info(f"Queries -> Entity: '{kw_query}' | Doc: '{doc_query}' | Web: '{web_query}'")
        
        # 2. SEARCH INTERNAL DATABASES (Structured data: People, Companies)
        search_output = search_standard_tables(kw_query)
        entity_context = search_output["entities"]
        notes_context = search_output["notes"]
        
        # 3. SEARCH INTERNAL DOCUMENTS (Semantic search on uploaded files)
        doc_results = []
        doc_context = ""
        if check_elasticsearch_connection():
            doc_results = hybrid_search(query=doc_query, limit=5)
            
        if doc_results:
            doc_pieces = []
            for r in doc_results:
                if r.get('card_type') == 'document':
                    title = r.get('title', 'Unknown')
                    content = r.get('content', '')[:800] 
                    doc_pieces.append(f"[Document File] Title: {title}\nContent: {content}")
            doc_context = "\n\n".join(doc_pieces)

        # --- SMART FALLBACK: Web Search ONLY if ALL internal sources are empty ---
        has_entity_match = bool(entity_context.strip())
        has_notes_match = bool(notes_context.strip())
        has_doc_match = bool(doc_context.strip())
        
        web_context = ""
        
        # CRITICAL FIX: Check ALL three internal sources
        if not has_entity_match and not has_notes_match and not has_doc_match:
             if web_query and len(web_query) > 3:
                 print("⚠️ No internal data found. Fetching Web Results...")
                 web_context = perform_web_search(web_query, max_results=5)
        else:
            print("✅ Using internal data (skipping web search)")
        
        # 4. SYNTHESIZE ANSWER
        final_context = f"""
        === DATABASE RECORDS (Internal - High Confidence) ===
        {entity_context if entity_context else "No direct matches."}
        
        === INTERNAL NOTES (Meetings) ===
        {notes_context if notes_context else "No notes found."}
        
        === DOCUMENTS (Uploaded Files - Context Only) ===
        {doc_context if doc_context else "No relevant files."}
        
        === WEB SEARCH RESULTS (Public Knowledge) ===
        {web_context if web_context else "No web results."}
        """
        
        model = get_chat_model(temperature=0.3)
        
        # --- FINAL FORMATTING PROMPT ---
        system_prompt = f"""You are a professional VC Assistant.
        
        CONTEXT:
        {final_context}
        
        FORMATTING RULES (STRICT):
        1. **OUTPUT STYLE:** Use clean Markdown.
           - Use **Bold** for Company/Person names.
           - Use bullet points (•) for lists.
           - Use nested bullets for details (Location, Founded, etc.).
        
        2. **CONTENT SOURCE PRIORITY (CRITICAL):**
           - **ALWAYS prioritize internal data** from DATABASE RECORDS, NOTES, and DOCUMENTS
           - **LISTS:** If user asks for a list (e.g., "Companies in Boston"), ONLY list items from `=== DATABASE RECORDS ===`. Do NOT include companies found in `=== DOCUMENTS ===` or `=== WEB SEARCH RESULTS ===`.
           - **NOTES:** If user asks about meetings, list all notes from `=== INTERNAL NOTES ===`.
           - **WEB:** ONLY use Web Results if no internal data exists (for definitions, recipes, biographies, public figures).

        3. **SOURCE INDICATION:**
           - **If using WEB SEARCH RESULTS:** Start your response with "Based on web search," or "According to online sources,"
           - **If using DATABASE/NOTES/DOCUMENTS:** Just give the answer directly without mentioning the source.

        4. **NO FLUFF:**
           - Do NOT say "Here is the information...".
           - Just give the answer directly.
        
        5. **EXAMPLE OUTPUT:**
           **TechFlow Solutions**
           • Location: San Francisco
           • Founded: 2020
           
           **DataVault Systems**
           • Location: Boston
           • Founded: 2018
        """
        
        messages = [SystemMessage(content=system_prompt)]
        
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