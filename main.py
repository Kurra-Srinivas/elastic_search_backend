from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from chat_service import chat_with_ai
from elasticsearch_service import (
    check_elasticsearch_connection,
    create_all_indices,
    index_company_card,
    index_person_card,
    index_document,
    index_note,
    hybrid_search,
    rebuild_index,
    search_companies_es,
    search_persons_es,
    search_notes_es,
    get_card_by_id_es,
    get_auto_complete_suggestions, # Logic for auto-complete
    es, # Importing es client for direct queries
    INDEX_NOTES,
    INDEX_DOCUMENTS,
    INDEX_PERSONS,
    INDEX_COMPANIES
)

from document_extractor import extract_text_from_file
from upload_status import create_upload_status, update_upload_status, get_upload_status, complete_upload_status
from notes_service import get_note, save_note

app = FastAPI(title="Search UI Backend", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---

class CompanyCardData(BaseModel):
    id: str
    name: str
    industry: Optional[str] = None
    description: Optional[str] = None
    founded: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    linkedin_url: Optional[str] = None
    card_type: str = "company"


class PersonCardData(BaseModel):
    id: str
    name: str
    designation: Optional[str] = None
    company: Optional[str] = None
    linkedin_id: str
    linkedin_url: str
    education: Optional[str] = None
    experience_years: Optional[float] = None
    location: Optional[str] = None
    card_type: str = "person"


class FileUploadResponse(BaseModel):
    success: bool
    file_id: str
    filename: str
    message: str
    status_id: Optional[str] = None


class UploadStatusResponse(BaseModel):
    status: str
    progress: int
    message: str
    chunks_total: int = 0
    chunks_indexed: int = 0


class ChatMessage(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    response: str


class NoteRequest(BaseModel):
    note: str


class NoteResponse(BaseModel):
    note: str


class SearchResultsResponse(BaseModel):
    companies: List[CompanyCardData] = []
    persons: List[PersonCardData] = []
    notes: List[dict] = []
    documents: List[dict] = []


# Create uploads directory if it doesn't exist
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# --- Helper Functions ---

def load_person_cards(count: Optional[int] = None) -> List[PersonCardData]:
    """Load person cards from Elasticsearch"""
    if not check_elasticsearch_connection():
        return []
    
    try:
        # Get all persons from Elasticsearch
        response = es.search(
            index=INDEX_PERSONS,
            body={"size": count or 20, "query": {"match_all": {}}}
        )
        
        data = []
        for hit in response["hits"]["hits"]:
            metadata = hit["_source"].get("metadata", {})
            card_data = PersonCardData(
                id=hit["_source"]["id"],
                name=metadata.get("name", ""),
                designation=metadata.get("designation"),
                company=metadata.get("company"),
                linkedin_id=metadata.get("linkedin_id", ""),
                linkedin_url=metadata.get("linkedin_url", ""),
                education=metadata.get("education"),
                experience_years=metadata.get("experience_years"),
                location=metadata.get("location"),
                card_type="person"
            )
            data.append(card_data)
        
        return data
    except Exception as e:
        logger.error(f"Error loading person cards: {e}")
        return []


def load_company_cards(count: Optional[int] = None) -> List[CompanyCardData]:
    """Load company cards from Elasticsearch"""
    if not check_elasticsearch_connection():
        return []
    
    try:
        # Get all companies from Elasticsearch
        response = es.search(
            index=INDEX_COMPANIES,
            body={"size": count or 20, "query": {"match_all": {}}}
        )
        
        data = []
        for hit in response["hits"]["hits"]:
            metadata = hit["_source"].get("metadata", {})
            card_data = CompanyCardData(
                id=hit["_source"]["id"],
                name=metadata.get("name", ""),
                industry=metadata.get("industry"),
                description=metadata.get("description"),
                founded=metadata.get("founded"),
                location=metadata.get("location"),
                website=metadata.get("website"),
                linkedin_url=metadata.get("linkedin_url"),
                card_type="company"
            )
            data.append(card_data)
        
        return data
    except Exception as e:
        logger.error(f"Error loading company cards: {e}")
        return []


def get_card_by_id(card_id: str) -> Optional[CompanyCardData | PersonCardData]:
    """Get a card (company or person) by its ID from Elasticsearch"""
    card_dict = get_card_by_id_es(card_id)
    if not card_dict:
        return None
    
    if card_dict.get("card_type") == "company":
        return CompanyCardData(**card_dict)
    elif card_dict.get("card_type") == "person":
        return PersonCardData(**card_dict)
    return None

# --- Startup Event ---

@app.on_event("startup")
async def startup_event():
    """Initialize Elasticsearch and load embedding model on startup"""
    # Preload embedding model (loads once, singleton pattern)
    try:
        from embedding_service import get_embedding_model
        get_embedding_model()
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}. Semantic search will be limited.")
    
    # Initialize Elasticsearch
    if check_elasticsearch_connection():
        create_all_indices()
        # Rebuild index on startup to load data from JSON files
        try:
            rebuild_index()
            logger.info("Data indexed from JSON files on startup")
        except Exception as e:
            logger.warning(f"Failed to rebuild index on startup: {e}")
    else:
        logger.warning("Elasticsearch not available. Search functionality will be limited.")


# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "Search UI Backend API", "version": "1.0.0"}

@app.get("/api/cards")
async def get_cards(card_type: Optional[str] = None):
    """Get list of cards - can filter by type (company/person)"""
    if card_type == "company":
        return load_company_cards(20)
    elif card_type == "person":
        return load_person_cards(20)
    else:
        # Return both types mixed
        companies = load_company_cards(10)
        persons = load_person_cards(10)
        return companies + persons

# --- UPDATED SEARCH FUNCTION WITH ENTITY_ID SUPPORT ---

@app.get("/api/cards/search", response_model=SearchResultsResponse)
async def search_cards(query: str = "", entity_id: Optional[str] = None, limit: int = 50):
    """
    Search across all categories.
    If entity_id is provided, fetches ONLY that entity and its related items (Exact Lookup).
    If no entity_id, performs text/fuzzy search using query string.
    """
    results = SearchResultsResponse()
    
    # Check if Elasticsearch is available
    if not check_elasticsearch_connection():
        logger.warning("Elasticsearch not available for search")
        return results

    # --- CASE A: SPECIFIC ENTITY SELECTED (from Dropdown) ---
    if entity_id:
        # 1. Fetch the specific Card
        card = get_card_by_id(entity_id)
        if card:
            if card.card_type == "company":
                results.companies = [card]
            else:
                results.persons = [card]
                
            # 2. Fetch Notes for this Card (Exact Match on card_id)
            try:
                note_response = es.search(
                    index=INDEX_NOTES,
                    body={
                        "query": {"term": {"card_id": entity_id}},
                        "size": 50
                    }
                )
                for hit in note_response["hits"]["hits"]:
                    source = hit["_source"]
                    metadata = source.get("metadata", {})
                    
                    # FIX: Use 'note_text' from metadata if available (Clean), otherwise 'content'
                    clean_note = metadata.get("note_text") or source.get("content", "")
                    
                    note_data = {
                        "id": hit["_id"],
                        "card_id": entity_id,
                        "card_type": card.card_type,
                        "note": clean_note, # <--- Updated
                        "parent_card": {
                            "id": card.id,
                            "name": card.name,
                            "type": card.card_type
                        }
                    }
                    results.notes.append(note_data)
            except Exception as e:
                logger.error(f"Error fetching notes for entity: {e}")

            # 3. Fetch Documents for this Card (Exact Match on card_id)
            try:
                doc_response = es.search(
                    index=INDEX_DOCUMENTS,
                    body={
                        "query": {"term": {"card_id": entity_id}},
                        "size": 50
                    }
                )
                
                seen_doc_ids = set()
                for hit in doc_response["hits"]["hits"]:
                    # Deduping chunks -> show 1 entry per file
                    meta = hit["_source"].get("metadata", {})
                    filename = meta.get("original_filename") or meta.get("filename", "Unknown")
                    
                    if filename not in seen_doc_ids:
                        doc_data = {
                            "id": hit["_id"],
                            "card_id": entity_id,
                            "filename": filename,
                            "stored_filename": meta.get("filename", ""),
                            "chunk_index": 0,
                            "content_preview": hit["_source"].get("content", "")[:200],
                            "score": 1.0, # Exact match
                            "parent_card": {
                                "id": card.id,
                                "name": card.name,
                                "type": card.card_type
                            }
                        }
                        results.documents.append(doc_data)
                        seen_doc_ids.add(filename)
            except Exception as e:
                logger.error(f"Error fetching docs for entity: {e}")

        return results

    # --- CASE B: TEXT SEARCH (Original Logic) ---
    if not query:
        # Return default cards if no query
        return SearchResultsResponse(
            companies=load_company_cards(10),
            persons=load_person_cards(10),
            notes=[],
            documents=[]
        )
    
    # Search companies using Elasticsearch
    try:
        company_results = search_companies_es(query, limit=limit)
        results.companies = []
        for es_result in company_results:
            metadata = es_result.get("metadata", {})
            results.companies.append(
                CompanyCardData(
                    id=es_result["id"],
                    name=metadata.get("name", ""),
                    industry=metadata.get("industry"),
                    description=metadata.get("description"),
                    founded=metadata.get("founded"),
                    location=metadata.get("location"),
                    website=metadata.get("website"),
                    linkedin_url=metadata.get("linkedin_url"),
                    card_type="company"
                )
            )
    except Exception as e:
        logger.error(f"Error searching companies: {e}")
    
    # Search persons using Elasticsearch
    try:
        person_results = search_persons_es(query, limit=limit)
        results.persons = []
        for es_result in person_results:
            metadata = es_result.get("metadata", {})
            results.persons.append(
                PersonCardData(
                    id=es_result["id"],
                    name=metadata.get("name", ""),
                    designation=metadata.get("designation"),
                    company=metadata.get("company"),
                    linkedin_id=metadata.get("linkedin_id", ""),
                    linkedin_url=metadata.get("linkedin_url", ""),
                    education=metadata.get("education"),
                    experience_years=metadata.get("experience_years"),
                    location=metadata.get("location"),
                    card_type="person"
                )
            )
    except Exception as e:
        logger.error(f"Error searching persons: {e}")
    
    # Search notes using Elasticsearch
    try:
        note_results = search_notes_es(query, limit=limit)
        results.notes = []
        for es_result in note_results:
            card_id = es_result["card_id"]
            parent_card = get_card_by_id(card_id)
            
            # FIX: Extract clean note from metadata if available
            metadata = es_result.get("metadata", {})
            clean_note = metadata.get("note_text") or es_result.get("content", "")
            
            note_data = {
                "id": es_result["id"],
                "card_id": card_id,
                "card_type": metadata.get("card_type", "unknown"),
                "note": clean_note, # <--- Updated
                "parent_card": None
            }
            if parent_card:
                if isinstance(parent_card, CompanyCardData):
                    note_data["parent_card"] = {
                        "id": parent_card.id,
                        "name": parent_card.name,
                        "type": "company"
                    }
                elif isinstance(parent_card, PersonCardData):
                    note_data["parent_card"] = {
                        "id": parent_card.id,
                        "name": parent_card.name,
                        "type": "person"
                    }
            results.notes.append(note_data)
    except Exception as e:
        logger.error(f"Error searching notes: {e}")
    
    # Search documents from Elasticsearch (hybrid search)
    try:
        if check_elasticsearch_connection():
            document_results = hybrid_search(query, limit=limit)
            results.documents = []
            seen_card_ids = set()
            
            for doc_result in document_results:
                card_id = doc_result["card_id"]
                if card_id not in seen_card_ids:
                    # Get parent card
                    parent_card = get_card_by_id(card_id)
                    if parent_card:
                        # Get original filename from metadata, fallback to stored filename
                        metadata = doc_result.get("metadata", {})
                        stored_filename = metadata.get("filename", "")
                        original_filename = metadata.get("original_filename", stored_filename)
                        
                        doc_data = {
                            "id": doc_result["id"],
                            "card_id": card_id,
                            "filename": original_filename,
                            "stored_filename": stored_filename,
                            "chunk_index": metadata.get("chunk_index", 0),
                            "content_preview": doc_result["content"][:200] + "..." if len(doc_result["content"]) > 200 else doc_result["content"],
                            "score": doc_result["score"],
                            "highlights": doc_result.get("highlights", {}),
                            "parent_card": {
                                "id": parent_card.id,
                                "name": parent_card.name if isinstance(parent_card, CompanyCardData) else parent_card.name,
                                "type": parent_card.card_type
                            }
                        }
                        results.documents.append(doc_data)
                        seen_card_ids.add(card_id)
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
    
    return results


@app.post("/api/cards/{card_id}/upload", response_model=FileUploadResponse)
async def upload_file(card_id: str, file: UploadFile = File(...)):
    """Upload a file (PDF/DOCX) for a specific card and index its content"""
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.doc'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    filename = f"{card_id}_{file_id}{file_ext}"
    file_path = UPLOADS_DIR / filename
    
    # Create status tracking
    status_id = create_upload_status(file_id, file.filename or filename)
    
    try:
        # Save file
        update_upload_status(status_id, "uploading", 10, "Uploading file...")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text and index in Elasticsearch
        update_upload_status(status_id, "extracting", 30, "Extracting text from document...")
        extracted_text = extract_text_from_file(file_path)
        
        if extracted_text:
            # Check Elasticsearch connection before indexing
            if check_elasticsearch_connection():
                # Index document with progress updates
                update_upload_status(status_id, "chunking", 50, "Chunking document...")
                try:
                    # Store both original filename and stored filename in metadata
                    index_document(
                        card_id=card_id,
                        filename=filename,  # Use stored filename (card_id_uuid.ext) for consistency
                        extracted_text=extracted_text,
                        metadata={
                            "file_id": file_id, 
                            "file_size": len(content),
                            "original_filename": file.filename or filename  # Keep original for reference
                        },
                        status_id=status_id
                    )
                    complete_upload_status(status_id, True, "File indexed successfully")
                    logger.info(f"Indexed document content for {filename}")
                except Exception as e:
                    logger.error(f"Error indexing document: {e}")
                    complete_upload_status(status_id, False, f"Indexing failed: {str(e)}")
            else:
                # File uploaded but Elasticsearch not available - still mark as success for upload
                complete_upload_status(status_id, False, "File uploaded but Elasticsearch not available. Indexing skipped.")
                logger.warning(f"Elasticsearch not available, file uploaded but not indexed: {filename}")
        else:
            complete_upload_status(status_id, False, "Could not extract text from document")
            logger.warning(f"Could not extract text from {filename}")
        
        return FileUploadResponse(
            success=True,
            file_id=file_id,
            filename=filename,
            message=f"File uploaded successfully",
            status_id=status_id
        )
    except Exception as e:
        complete_upload_status(status_id, False, f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/api/upload/status/{status_id}", response_model=UploadStatusResponse)
async def get_upload_status_endpoint(status_id: str):
    """Get upload and indexing status"""
    status = get_upload_status(status_id)
    if not status:
        raise HTTPException(status_code=404, detail="Status not found")
    
    return UploadStatusResponse(
        status=status["status"],
        progress=status["progress"],
        message=status["message"],
        chunks_total=status.get("chunks_total", 0),
        chunks_indexed=status.get("chunks_indexed", 0)
    )


@app.get("/api/cards/{card_id}/files")
async def get_card_files(card_id: str):
    """Get list of files uploaded for a card"""
    files = []
    for file_path in UPLOADS_DIR.glob(f"{card_id}_*"):
        if file_path.is_file():
            stored_filename = file_path.name
            
            # Try to get original filename from Elasticsearch metadata
            original_filename = stored_filename  # Default to stored filename
            if check_elasticsearch_connection():
                try:
                    from elasticsearch_service import es, INDEX_DOCUMENTS
                    # Search for any chunk with this filename to get original_filename
                    search_response = es.search(
                        index=INDEX_DOCUMENTS,
                        body={
                            "query": {
                                "bool": {
                                    "must": [
                                        {"term": {"card_id": card_id}},
                                        {"match": {"metadata.filename": stored_filename}}
                                    ]
                                }
                            },
                            "size": 1
                        }
                    )
                    if search_response["hits"]["hits"]:
                        metadata = search_response["hits"]["hits"][0]["_source"].get("metadata", {})
                        original_filename = metadata.get("original_filename", stored_filename)
                except Exception as e:
                    logger.debug(f"Could not fetch original filename: {e}")
            
            files.append({
                "filename": stored_filename,  # Keep stored filename for deletion
                "original_filename": original_filename,  # Display name
                "size": file_path.stat().st_size,
                "uploaded_at": file_path.stat().st_mtime
            })
    return {"card_id": card_id, "files": files}


@app.delete("/api/cards/{card_id}/files/{filename}")
async def delete_card_file(card_id: str, filename: str):
    """Delete a file and remove it from Elasticsearch index"""
    try:
        # Verify the file belongs to this card
        file_path = UPLOADS_DIR / filename
        if not file_path.exists():
            # File might already be deleted, but we should still try to remove from index
            logger.warning(f"File {filename} not found on disk, but will attempt to remove from index")
        else:
            if not filename.startswith(f"{card_id}_"):
                raise HTTPException(status_code=403, detail="File does not belong to this card")
        
        # Delete from Elasticsearch first (this is the critical part)
        deletion_success = False
        if check_elasticsearch_connection():
            from elasticsearch_service import delete_document_by_filename
            deletion_success = delete_document_by_filename(card_id, filename)
            if not deletion_success:
                logger.warning(f"Failed to delete chunks from Elasticsearch for {filename}")
        else:
            logger.warning("Elasticsearch not available, skipping index deletion")
        
        # Delete the file from disk
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Deleted file {filename} for card {card_id}")
            except Exception as e:
                logger.error(f"Failed to delete file {filename}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")
        
        if not deletion_success:
            logger.warning(f"File {filename} deleted from disk but chunks may still exist in index")
        
        return {
            "success": True, 
            "message": f"File {filename} deleted successfully",
            "index_deleted": deletion_success
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@app.post("/api/search/index/rebuild")
async def rebuild_search_index():
    """Rebuild the entire Elasticsearch index"""
    if not check_elasticsearch_connection():
        raise HTTPException(status_code=503, detail="Elasticsearch not available")
    
    try:
        stats = rebuild_index()
        return {
            "success": True,
            "message": "Index rebuilt successfully",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")


@app.get("/api/cards/{card_id}/note", response_model=NoteResponse)
async def get_card_note(card_id: str):
    """Get note for a card"""
    note = get_note(card_id)
    return NoteResponse(note=note)


@app.post("/api/cards/{card_id}/note", response_model=NoteResponse)
async def save_card_note(card_id: str, note_request: NoteRequest):
    """Save note for a card"""
    success = save_note(card_id, note_request.note)
    
    # Index note in Elasticsearch if available
    if success and check_elasticsearch_connection():
        try:
            # Get card metadata for indexing
            card = get_card_by_id(card_id)
            card_metadata = None
            if card:
                if isinstance(card, CompanyCardData):
                    card_metadata = {
                        "name": card.name,
                        "industry": card.industry,
                        "location": card.location,
                        "description": card.description
                    }
                elif isinstance(card, PersonCardData):
                    card_metadata = {
                        "name": card.name,
                        "company": card.company,
                        "designation": card.designation,
                        "education": card.education,
                        "location": card.location
                    }
            
            card_type = "company" if isinstance(card, CompanyCardData) else "person" if isinstance(card, PersonCardData) else "unknown"
            index_note(card_id, note_request.note, card_type, card_metadata)
        except Exception as e:
            logger.warning(f"Failed to index note in Elasticsearch: {e}")
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save note")
    
    return NoteResponse(note=note_request.note)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatMessage):
    """Chat endpoint using LangChain Groq"""
    try:
        response = await chat_with_ai(
            message=chat_request.message,
            conversation_history=chat_request.conversation_history
        )
        return ChatResponse(response=response)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")


@app.get("/api/suggest")
async def auto_suggest_endpoint(query: str):
    """
    Fast endpoint for search bar auto-completion.
    Returns list of names/titles matching the prefix.
    """
    if not query or len(query) < 2:
        return []  # Don't search for 1 letter
        
    try:
        return get_auto_complete_suggestions(query)
    except Exception as e:
        logger.error(f"Auto-suggest failed: {e}")
        return []