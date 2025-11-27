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

from profile_loader import (
    load_profiles,
    get_recent_education,
    get_company_info
)
from company_loader import load_companies
from chat_service import chat_with_ai
from elasticsearch_service import (
    check_elasticsearch_connection,
    create_all_indices,
    index_company_card,
    index_person_card,
    index_document,
    index_note,
    hybrid_search,
    rebuild_index
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

# Data models
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


# Create uploads directory if it doesn't exist
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

def load_person_cards(count: Optional[int] = None) -> List[PersonCardData]:
    """Load person cards from enriched profiles"""
    profiles = load_profiles()
    
    data = []
    for profile in profiles:
        profile_data = profile.get("profile_data", {})
        
        # Get person name
        name = profile_data.get("name", "Unknown")
        
        # Get company and designation
        company, designation = get_company_info(profile)
        
        # Get LinkedIn info
        linkedin_id = profile.get("linkedin_username", "")
        linkedin_url = profile.get("linkedin_url", f"https://linkedin.com/in/{linkedin_id}")
        
        # Get education
        education = get_recent_education(profile)
        
        # Get experience years
        experience_years = profile.get("total_experience_years")
        
        # Get location
        location = profile_data.get("location")
        
        card_data = PersonCardData(
            id=f"person_{linkedin_id}",
            name=name,
            designation=designation,
            company=company,
            linkedin_id=linkedin_id,
            linkedin_url=linkedin_url,
            education=education,
            experience_years=experience_years,
            location=location,
            card_type="person"
        )
        data.append(card_data)
        
        # Limit if count specified
        if count and len(data) >= count:
            break
    
    return data


def load_company_cards(count: Optional[int] = None) -> List[CompanyCardData]:
    """Load company cards from companies data"""
    companies = load_companies()
    
    data = []
    for company in companies:
        card_data = CompanyCardData(
            id=company.get("id", ""),
            name=company.get("name", ""),
            industry=company.get("industry"),
            description=company.get("description"),
            founded=company.get("founded"),
            location=company.get("location"),
            website=company.get("website"),
            linkedin_url=company.get("linkedin_url"),
            card_type="company"
        )
        data.append(card_data)
        
        # Limit if count specified
        if count and len(data) >= count:
            break
    
    return data


def get_card_by_id(card_id: str) -> Optional[CompanyCardData | PersonCardData]:
    """Get a card (company or person) by its ID"""
    # Try to find in company cards
    companies = load_company_cards()
    for company in companies:
        if company.id == card_id:
            return company
    
    # Try to find in person cards
    persons = load_person_cards()
    for person in persons:
        if person.id == card_id:
            return person
    
    return None

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
        # Optionally rebuild index on startup (comment out if not needed)
        # rebuild_index()
    else:
        logger.warning("Elasticsearch not available. Search functionality will be limited.")


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

@app.get("/api/cards/search")
async def search_cards(query: str = "", card_type: Optional[str] = None, limit: int = 50):
    """Search cards using Elasticsearch hybrid search (keyword + semantic)"""
    if not query:
        # Return default cards if no query
        if card_type == "company":
            return load_company_cards(20)
        elif card_type == "person":
            return load_person_cards(20)
        else:
            companies = load_company_cards(10)
            persons = load_person_cards(10)
            return companies + persons
    
    # Use Elasticsearch if available, but fallback to basic search if no results
    if check_elasticsearch_connection():
        try:
            search_results = hybrid_search(query, limit=limit)
            
            # Convert search results back to card format
            cards = []
            seen_card_ids = set()  # Avoid duplicates
            
            for result in search_results:
                if result["card_type"] == "company":
                    card_id = result["card_id"]
                    if card_id not in seen_card_ids:
                        cards.append(CompanyCardData(
                            id=card_id,
                            name=result["title"],
                            industry=result["metadata"].get("industry"),
                            description=result["metadata"].get("description"),
                            founded=result["metadata"].get("founded"),
                            location=result["metadata"].get("location"),
                            website=result["metadata"].get("website"),
                            linkedin_url=result["metadata"].get("linkedin_url"),
                            card_type="company"
                        ))
                        seen_card_ids.add(card_id)
                elif result["card_type"] == "person":
                    card_id = result["card_id"]
                    if card_id not in seen_card_ids:
                        cards.append(PersonCardData(
                            id=card_id,
                            name=result["title"],
                            designation=result["metadata"].get("designation"),
                            company=result["metadata"].get("company"),
                            linkedin_id=result["metadata"].get("linkedin_id", ""),
                            linkedin_url=result["metadata"].get("linkedin_url", ""),
                            education=result["metadata"].get("education"),
                            experience_years=result["metadata"].get("experience_years"),
                            location=result["metadata"].get("location"),
                            card_type="person"
                        ))
                        seen_card_ids.add(card_id)
                elif result["card_type"] == "document":
                    # Document chunk found - fetch the parent card
                    card_id = result["card_id"]
                    if card_id not in seen_card_ids:
                        parent_card = get_card_by_id(card_id)
                        if parent_card:
                            cards.append(parent_card)
                            seen_card_ids.add(card_id)
                            logger.debug(f"Found document match, returning parent card: {card_id}")
                elif result["card_type"] == "note":
                    # Note found - fetch the parent card
                    card_id = result["card_id"]
                    if card_id not in seen_card_ids:
                        parent_card = get_card_by_id(card_id)
                        if parent_card:
                            cards.append(parent_card)
                            seen_card_ids.add(card_id)
                            logger.debug(f"Found note match, returning parent card: {card_id}")
            
            # Filter by card_type if specified
            if card_type:
                cards = [c for c in cards if c.card_type == card_type]
            
            # If Elasticsearch returned results, use them
            if cards:
                return cards
            # If no results from Elasticsearch, fall through to basic search
            logger.info(f"Elasticsearch returned no results for '{query}', falling back to basic search")
        except Exception as e:
            logger.warning(f"Elasticsearch search failed: {e}, falling back to basic search")
            # Fall through to basic search
    
    # Fallback to basic keyword search (always works)
    all_companies = load_company_cards()
    all_persons = load_person_cards()
    
    query_lower = query.lower()
    filtered = []
    
    if not card_type or card_type == "company":
        for company in all_companies:
            if (query_lower in company.name.lower() or
                (company.industry and query_lower in company.industry.lower()) or
                (company.description and query_lower in company.description.lower()) or
                (company.location and query_lower in company.location.lower())):
                filtered.append(company)
    
    if not card_type or card_type == "person":
        for person in all_persons:
            if (query_lower in person.name.lower() or
                (person.company and query_lower in person.company.lower()) or
                (person.designation and query_lower in person.designation.lower()) or
                (person.education and query_lower in person.education.lower()) or
                (person.location and query_lower in person.location.lower())):
                filtered.append(person)
    
    return filtered[:limit]


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
                    index_document(
                        card_id=card_id,
                        filename=file.filename or filename,
                        extracted_text=extracted_text,
                        metadata={"file_id": file_id, "file_size": len(content)},
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
            files.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "uploaded_at": file_path.stat().st_mtime
            })
    return {"card_id": card_id, "files": files}


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
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save note")
    
    # Index the note in Elasticsearch if available
    if check_elasticsearch_connection():
        # Get the card to extract metadata
        card = get_card_by_id(card_id)
        if card:
            # Determine card type and build metadata
            if isinstance(card, CompanyCardData):
                card_type = "company"
                card_metadata = {
                    "name": card.name,
                    "industry": card.industry,
                    "location": card.location,
                    "description": card.description
                }
            else:  # PersonCardData
                card_type = "person"
                card_metadata = {
                    "name": card.name,
                    "company": card.company,
                    "designation": card.designation,
                    "education": card.education,
                    "location": card.location
                }
            index_note(card_id, note_request.note, card_type, card_metadata)
        else:
            # Fallback if card not found
            card_type = "person" if card_id.startswith("person_") else "company"
            index_note(card_id, note_request.note, card_type)
    
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

