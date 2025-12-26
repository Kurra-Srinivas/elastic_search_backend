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

    system_prompt = f"""You are a Search Intent Analyzer. 
    Analyze the user's question and extract CLEAN search terms.
    
    {history_context}
    
    INSTRUCTIONS:
    1. **entity_keywords**: Extract core values (Names, Industries, Years, Cities) for the Internal Database.
       - User: "Companies in Boston" -> "Boston"
       - User: "SaaS companies" -> "SaaS"
       - User: "Who is the CEO of TechFlow?" -> "TechFlow" (Search the company, not the CEO name).
       
    2. **web_query**: 
       - CRITICAL RULE: DO NOT generate a web query for "Lists of Companies", "Locations", "Founders", or "CEOs" if the company sounds like a startup/VC portfolio company.
       - ONLY generate a web_query if the user asks for:
         a) Clearly public/general info ("Bitcoin price", "Weather", "Chicken Recipe", "Who is Obama").
         b) A company clearly NOT in a private DB ("Apple", "Tesla", "Microsoft").
         c) A definition ("What is EBITDA?").
       - If unsure, leave `web_query` EMPTY ("").
    
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
            "web_query": ""
        }

def search_standard_tables(keywords: str) -> Dict[str, str]:
    """
    Step 2: Search People, Companies, Notes
    """
    if not keywords or not check_elasticsearch_connection():
        return {"entities": "", "notes": ""}
        
    entity_results = []
    note_results = []
    
    print(f"\n--- DEBUG: Searching Database for '{keywords}' ---")

    # --- LOGIC: WEIGHTED SEARCH ---
    body_weighted = {
        "query": {
            "bool": {
                "should": [
                    { "multi_match": { "query": keywords, "fields": ["*"], "type": "phrase", "boost": 3.0 } },
                    { "multi_match": { "query": keywords, "fields": ["*"], "operator": "and", "fuzziness": "AUTO", "boost": 2.0 } },
                    { "multi_match": { "query": keywords, "fields": ["metadata.industry", "metadata.description", "content"], "fuzziness": "AUTO", "operator": "or", "minimum_should_match": "75%" } }
                ]
            }
        },
        "size": 20 # High limit to capture lists (e.g., "Companies in Boston")
    }

    # --- LOGIC: NOTES ---
    body_notes = {
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
        res = es.search(index=INDEX_PERSONS, body=body_weighted)
        for hit in res["hits"]["hits"]:
            s = hit["_source"]
            meta = s.get('metadata', {})
            name = s.get('name') or s.get('title') or "Unknown"
            info = f"PERSON: {name}\n   - Title: {meta.get('designation', 'N/A')}\n   - Company: {meta.get('company', 'N/A')}\n   - Location: {meta.get('location', 'N/A')}\n   - Education: {meta.get('education', 'N/A')}"
            entity_results.append(info)
    except Exception: pass

    # 2. Search Companies
    try:
        res = es.search(index=INDEX_COMPANIES, body=body_weighted)
        for hit in res["hits"]["hits"]:
            s = hit["_source"]
            meta = s.get('metadata', {})
            name = s.get('name') or s.get('title') or "Unknown"
            founded = meta.get('founded') or meta.get('founded_year') or "N/A"
            info = f"COMPANY: {name}\n   - Industry: {meta.get('industry', 'N/A')}\n   - Location: {meta.get('location', 'N/A')}\n   - Founded: {founded}\n   - Description: {meta.get('description', 'N/A')}"
            entity_results.append(info)
    except Exception: pass

    # 3. Search Notes
    try:
        res = es.search(index=INDEX_NOTES, body=body_notes)
        for hit in res["hits"]["hits"]:
            s = hit["_source"]
            content = s.get('content') or s.get('note') or "No content"
            meta = s.get('metadata', {})
            owner = meta.get('person_name') or meta.get('company_name') or s.get('title', 'Unknown')
            note_results.append(f"NOTE attached to {owner}: {content}")
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
        web_query = queries.get("web_query", "")
        
        logger.info(f"Queries -> Entity: '{kw_query}' | Doc: '{doc_query}' | Web: '{web_query}'")
        
        # 2. RUN INTERNAL SEARCH FIRST
        search_output = search_standard_tables(kw_query)
        entity_context = search_output["entities"]
        notes_context = search_output["notes"]
        
        # 3. RUN DOCUMENT SEARCH
        doc_context = ""
        if check_elasticsearch_connection():
            doc_results = hybrid_search(query=doc_query, limit=5)
            if doc_results:
                doc_pieces = [f"[Document File] Title: {r.get('title')}\nContent: {r.get('content', '')[:800]}" for r in doc_results if r.get('card_type') == 'document']
                doc_context = "\n\n".join(doc_pieces)

        # --- STRICT FALLBACK LOGIC ---
        has_internal_data = bool(entity_context.strip()) or bool(notes_context.strip()) or bool(doc_context.strip())
        
        web_context = ""

        # RULE 1: If we have Internal Data, BLOCK Web Search UNLESS the web query is about something TOTALLY different.
        # Example: Internal="TechFlow", Web="Bitcoin". -> ALLOW.
        # Example: Internal="Boston", Web="Companies in Boston". -> BLOCK.
        
        should_run_web = False
        
        if not has_internal_data:
            # Case A: No internal data. Run web if query exists.
            should_run_web = True
        else:
            # Case B: We have internal data. 
            # Only run web if the web_query is suspiciously different from the internal keywords
            # (Simple heuristic: If web_query is long and distinct, allow it. Otherwise, assume it's redundant.)
            if web_query and len(web_query) > 3:
                # Check for "External Concepts"
                external_triggers = ["bitcoin", "price", "stock", "recipe", "weather", "news", "who is"]
                if any(trigger in web_query.lower() for trigger in external_triggers):
                     should_run_web = True
        
        if should_run_web and web_query:
             print(f"🌍 Fetching Web Data for: {web_query}")
             web_context = perform_web_search(web_query, max_results=4)
        else:
             print("🔒 Web Search BLOCKED (Internal Data took priority).")

        # 4. SYNTHESIZE ANSWER
        final_context = f"""
        === DATABASE RECORDS (Internal - TRUTH) ===
        {entity_context if entity_context else "No direct matches."}
        
        === INTERNAL NOTES ===
        {notes_context if notes_context else "No notes found."}
        
        === DOCUMENTS (Internal Files) ===
        {doc_context if doc_context else "No relevant files."}
        
        === WEB SEARCH RESULTS (Secondary) ===
        {web_context if web_context else "No web results."}
        """
        
        model = get_chat_model(temperature=0.3)
        
        system_prompt = f"""You are a professional VC Assistant.
        
        CONTEXT:
        {final_context}
        
        INSTRUCTIONS:

       1. **ANSWER STYLE & PRECISION:**
           - Answer in **complete, natural sentences**.
           - Example: "TechFlow Solutions is located in San Francisco, CA." (Not just "San Francisco").
           - **CRITICAL:** Answer **ONLY** the specific attribute asked. Do NOT dump unrelated metadata like Industry or Description unless asked for "Details" or "Summary".

        2. **STRICT INTERNAL PRIORITY:** 
           - If `=== DATABASE RECORDS ===` has data, output ONLY that data. 
           - IGNORE Web Results if they contradict or duplicate Internal Data.
           - Example: If DB lists 1 company in Boston, list ONLY that 1 company.
           
        3. **MISSING INFO:**
           - If user asks "Who is CEO of TechFlow?" and the TechFlow record in DB has no CEO:
             - SAY: "I don't have the CEO information for TechFlow."
             - DO NOT guess from Web Results unless the user specifically asked "Search the web".
        
        4. **LISTS:**
           - Format lists using Bullet Points.
           - Include Industry and Location for context.
           
        5. **WEB USAGE:**
           - Only use Web Results if Internal Data is EMPTY or if the user asked for a definition/recipe/stock price.
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