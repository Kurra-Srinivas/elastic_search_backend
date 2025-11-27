"""Elasticsearch service for indexing and searching cards"""
from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Optional
import os
import logging
from datetime import datetime

from embedding_service import generate_embedding

logger = logging.getLogger(__name__)

# Elasticsearch connection
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
ES_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
ES_USER = os.getenv("ELASTICSEARCH_USER", None)
ES_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", None)

# Initialize Elasticsearch client
es_url = f"http://{ES_HOST}:{ES_PORT}"
es_config = {
    "hosts": [es_url],
    "request_timeout": 5,
    "max_retries": 2
}

if ES_USER and ES_PASSWORD:
    es_config["basic_auth"] = (ES_USER, ES_PASSWORD)

es = Elasticsearch(**es_config)

# Separate indices for each data type
INDEX_COMPANIES = "companies"
INDEX_PERSONS = "persons"
INDEX_DOCUMENTS = "documents"
INDEX_NOTES = "notes"

# All indices list
ALL_INDICES = [INDEX_COMPANIES, INDEX_PERSONS, INDEX_DOCUMENTS, INDEX_NOTES]


def check_elasticsearch_connection() -> bool:
    """Check if Elasticsearch is available with timeout"""
    try:
        # Try info() first as it's more reliable than ping()
        info = es.info(request_timeout=5)
        if info and info.get("cluster_name"):
            logger.debug(f"Elasticsearch connection successful: {info.get('cluster_name')}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Elasticsearch connection check failed: {e}")
        # Try ping as fallback
        try:
            result = es.ping(request_timeout=5)
            if result:
                logger.debug("Elasticsearch connection successful (via ping)")
                return True
        except Exception as e2:
            logger.debug(f"Elasticsearch ping also failed: {e2}")
        return False


def _get_index_mapping() -> Dict[str, Any]:
    """Get the common index mapping for all indices"""
    return {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "card_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "text_embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
                "metadata": {
                    "type": "object",
                    "enabled": True
                },
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"}
            }
        },
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
    }


def create_index(index_name: str) -> bool:
    """Create a specific index with dense vector support"""
    if es.indices.exists(index=index_name):
        logger.info(f"Index {index_name} already exists")
        return True
    
    mapping = _get_index_mapping()
    
    try:
        es.indices.create(index=index_name, body=mapping)
        logger.info(f"Created index {index_name} with dense vector support for embeddings")
        return True
    except Exception as e:
        logger.error(f"Failed to create index {index_name}: {e}")
        return False


def create_all_indices() -> bool:
    """Create all indices"""
    success = True
    for index_name in ALL_INDICES:
        if not create_index(index_name):
            success = False
    return success


def index_company_card(company_data: Dict[str, Any]) -> bool:
    """Index a company card"""
    try:
        # Ensure index exists
        if not es.indices.exists(index=INDEX_COMPANIES):
            create_index(INDEX_COMPANIES)
        # Build searchable content
        content_parts = []
        if company_data.get("description"):
            content_parts.append(company_data["description"])
        if company_data.get("industry"):
            content_parts.append(f"Industry: {company_data['industry']}")
        if company_data.get("location"):
            content_parts.append(f"Location: {company_data['location']}")
        if company_data.get("founded"):
            content_parts.append(f"Founded: {company_data['founded']}")
        
        content = " ".join(content_parts)
        
        # Generate embedding for semantic search
        text_for_embedding = f"{company_data['name']} {content}"
        embedding = generate_embedding(text_for_embedding)
        
        doc = {
            "id": company_data["id"],
            "card_id": company_data["id"],
            "title": company_data["name"],
            "content": content,
            "text_embedding": embedding,
            "metadata": {
                "industry": company_data.get("industry"),
                "description": company_data.get("description"),
                "founded": company_data.get("founded"),
                "location": company_data.get("location"),
                "website": company_data.get("website"),
                "linkedin_url": company_data.get("linkedin_url")
            },
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        es.index(index=INDEX_COMPANIES, id=company_data["id"], document=doc)
        logger.info(f"Indexed company card: {company_data['id']}")
        return True
    except Exception as e:
        logger.error(f"Failed to index company card {company_data.get('id')}: {e}")
        return False


def index_person_card(person_data: Dict[str, Any]) -> bool:
    """Index a person card"""
    try:
        # Ensure index exists
        if not es.indices.exists(index=INDEX_PERSONS):
            create_index(INDEX_PERSONS)
        # Build searchable content
        content_parts = []
        if person_data.get("designation"):
            content_parts.append(person_data["designation"])
        if person_data.get("company"):
            content_parts.append(f"Company: {person_data['company']}")
        if person_data.get("education"):
            content_parts.append(f"Education: {person_data['education']}")
        if person_data.get("location"):
            content_parts.append(f"Location: {person_data['location']}")
        if person_data.get("experience_years"):
            content_parts.append(f"Experience: {person_data['experience_years']} years")
        
        content = " ".join(content_parts)
        
        # Generate embedding for semantic search
        text_for_embedding = f"{person_data['name']} {content}"
        embedding = generate_embedding(text_for_embedding)
        
        doc = {
            "id": person_data["id"],
            "card_id": person_data["id"],
            "title": person_data["name"],
            "content": content,
            "text_embedding": embedding,
            "metadata": {
                "name": person_data["name"],
                "designation": person_data.get("designation"),
                "company": person_data.get("company"),
                "linkedin_id": person_data.get("linkedin_id"),
                "linkedin_url": person_data.get("linkedin_url"),
                "education": person_data.get("education"),
                "experience_years": person_data.get("experience_years"),
                "location": person_data.get("location")
            },
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        es.index(index=INDEX_PERSONS, id=person_data["id"], document=doc)
        logger.info(f"Indexed person card: {person_data['id']}")
        return True
    except Exception as e:
        logger.error(f"Failed to index person card {person_data.get('id')}: {e}")
        return False


def index_document(card_id: str, filename: str, extracted_text: str, metadata: Optional[Dict[str, Any]] = None, status_id: Optional[str] = None) -> bool:
    """Index a document (PDF/DOCX) with chunking"""
    try:
        # Ensure index exists
        if not es.indices.exists(index=INDEX_DOCUMENTS):
            create_index(INDEX_DOCUMENTS)
        from text_chunker import chunk_text_by_sentences
        
        # Chunk the document text
        chunks = chunk_text_by_sentences(extracted_text, max_chunk_size=500, overlap_sentences=1)
        
        if not chunks:
            logger.warning(f"No chunks created for document {filename}")
            return False
        
        # Update status with total chunks
        if status_id:
            from upload_status import update_upload_status
            update_upload_status(status_id, "indexing", 60, f"Indexing {len(chunks)} chunks...", chunks_total=len(chunks))
        
        # Index each chunk as a separate document
        indexed_count = 0
        for chunk_idx, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
                
            # Create unique ID for each chunk
            chunk_id = f"doc_{card_id}_{filename}_chunk_{chunk_idx}"
            
            # Generate embedding for this chunk
            embedding = generate_embedding(chunk_text)
            
            doc = {
                "id": chunk_id,
                "card_id": card_id,
                "title": f"{filename} (chunk {chunk_idx + 1})",
                "content": chunk_text,
                "text_embedding": embedding,
                "metadata": {
                    "filename": filename,
                    "card_id": card_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    **(metadata or {})
                },
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            es.index(index=INDEX_DOCUMENTS, id=chunk_id, document=doc)
            indexed_count += 1
            
            # Update progress
            if status_id:
                progress = 60 + int((chunk_idx + 1) / len(chunks) * 35)  # 60-95%
                update_upload_status(status_id, "indexing", progress, 
                                    f"Indexing chunk {chunk_idx + 1}/{len(chunks)}...",
                                    chunks_indexed=chunk_idx + 1)
        
        logger.info(f"Indexed document {filename} as {indexed_count} chunks")
        return True
    except Exception as e:
        logger.error(f"Failed to index document {filename}: {e}")
        return False


def delete_card_from_index(card_id: str, card_type: str) -> bool:
    """Delete a card and all its associated documents and notes from indices"""
    try:
        # Determine which index to delete from
        if card_type == "company":
            index_name = INDEX_COMPANIES
        elif card_type == "person":
            index_name = INDEX_PERSONS
        else:
            logger.warning(f"Unknown card type: {card_type}")
            return False
        
        # Delete the card itself
        es.delete(index=index_name, id=card_id, ignore=[404])
        
        # Delete all documents associated with this card
        query = {
            "query": {
                "term": {"card_id": card_id}
            }
        }
        es.delete_by_query(index=INDEX_DOCUMENTS, body=query)
        
        # Delete notes associated with this card
        es.delete_by_query(index=INDEX_NOTES, body=query)
        
        logger.info(f"Deleted card, documents, and notes: {card_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete card {card_id}: {e}")
        return False


def hybrid_search(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining keyword (BM25) and semantic (vector) search
    Uses sentence-transformers embeddings for semantic search
    Falls back to keyword-only if embeddings fail
    """
    try:
        # Keyword search (BM25) - always use this
        keyword_query = {
            "multi_match": {
                "query": query,
                "fields": ["title^3", "content^2"],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }
        
        # Try to generate embedding for vector search
        query_embedding = None
        try:
            query_embedding = generate_embedding(query)
        except Exception as e:
            logger.debug(f"Failed to generate embedding: {e}, using keyword search only")
        
        # Build search body - use hybrid if embedding available, otherwise keyword-only
        if query_embedding and len(query_embedding) > 0:
            # Hybrid search with both keyword and vector
            search_body = {
                "knn": {
                    "field": "text_embedding",
                    "query_vector": query_embedding,
                    "k": limit,
                    "num_candidates": limit * 2,
                    "boost": 0.5  # Weight for vector search
                },
                "query": {
                    "bool": {
                        "should": [
                            keyword_query
                        ],
                        "boost": 1.0  # Weight for keyword search
                    }
                },
                "size": limit,
                "_source": ["id", "card_type", "card_id", "title", "content", "metadata", "_score"],
                "highlight": {
                    "fields": {
                        "title": {},
                        "content": {}
                    }
                }
            }
        else:
            # Keyword-only search (fallback)
            search_body = {
                "query": keyword_query,
                "size": limit,
                "_source": ["id", "card_type", "card_id", "title", "content", "metadata", "_score"],
                "highlight": {
                    "fields": {
                        "title": {},
                        "content": {}
                    }
                }
            }
        
        # Search across all indices
        all_results = []
        
        for index_name in ALL_INDICES:
            try:
                response = es.search(index=index_name, body=search_body)
                
                for hit in response["hits"]["hits"]:
                    # Determine card_type from index name
                    if index_name == INDEX_COMPANIES:
                        card_type = "company"
                    elif index_name == INDEX_PERSONS:
                        card_type = "person"
                    elif index_name == INDEX_DOCUMENTS:
                        card_type = "document"
                    elif index_name == INDEX_NOTES:
                        card_type = "note"
                    else:
                        card_type = "unknown"
                    
                    result = {
                        "id": hit["_source"]["id"],
                        "card_type": card_type,
                        "card_id": hit["_source"]["card_id"],
                        "title": hit["_source"]["title"],
                        "content": hit["_source"].get("content", ""),
                        "metadata": hit["_source"].get("metadata", {}),
                        "score": hit["_score"],
                        "highlights": hit.get("highlight", {}),
                        "index": index_name
                    }
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Search failed for index {index_name}: {e}")
                continue
        
        # Group results by card_id to handle multiple document chunks/notes per card
        # For each card, keep only the highest scoring match
        card_results = {}
        for result in all_results:
            card_id = result["card_id"]
            # For direct card matches (company/person), use them directly
            if result["card_type"] in ["company", "person"]:
                if card_id not in card_results or result["score"] > card_results[card_id]["score"]:
                    card_results[card_id] = result
            else:
                # For documents/notes, aggregate by taking the max score
                if card_id not in card_results:
                    card_results[card_id] = result
                else:
                    # Keep the highest scoring chunk/note for this card
                    if result["score"] > card_results[card_id]["score"]:
                        card_results[card_id] = result
        
        # Convert back to list and sort by score
        aggregated_results = list(card_results.values())
        aggregated_results.sort(key=lambda x: x["score"], reverse=True)
        
        return aggregated_results[:limit]
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def index_note(card_id: str, note: str, card_type: str, card_metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Index a note for a card with full card metadata"""
    try:
        # Ensure index exists
        if not es.indices.exists(index=INDEX_NOTES):
            create_index(INDEX_NOTES)
        
        if not note or not note.strip():
            # Delete note if empty
            es.delete(index=INDEX_NOTES, id=f"note_{card_id}", ignore=[404])
            return True
        
        # Build searchable content including card info
        content_parts = [note]
        if card_metadata:
            if card_type == "company" and card_metadata.get("name"):
                content_parts.append(f"Company: {card_metadata['name']}")
                if card_metadata.get("industry"):
                    content_parts.append(f"Industry: {card_metadata['industry']}")
            elif card_type == "person" and card_metadata.get("name"):
                content_parts.append(f"Person: {card_metadata['name']}")
                if card_metadata.get("company"):
                    content_parts.append(f"Company: {card_metadata['company']}")
                if card_metadata.get("designation"):
                    content_parts.append(f"Designation: {card_metadata['designation']}")
        
        searchable_content = " ".join(content_parts)
        
        # Generate embedding for semantic search (includes note + card context)
        embedding = generate_embedding(searchable_content)
        
        # Build metadata with full card info
        metadata = {
            "card_id": card_id,
            "card_type": card_type
        }
        
        # Add card-specific metadata
        if card_metadata:
            if card_type == "company":
                metadata.update({
                    "company_name": card_metadata.get("name"),
                    "industry": card_metadata.get("industry"),
                    "location": card_metadata.get("location"),
                    "description": card_metadata.get("description")
                })
            elif card_type == "person":
                metadata.update({
                    "person_name": card_metadata.get("name"),
                    "company": card_metadata.get("company"),
                    "designation": card_metadata.get("designation"),
                    "education": card_metadata.get("education"),
                    "location": card_metadata.get("location")
                })
        
        doc = {
            "id": f"note_{card_id}",
            "card_id": card_id,
            "title": f"Note for {card_metadata.get('name', card_id) if card_metadata else card_id}",
            "content": searchable_content,  # Includes note + card context for better search
            "text_embedding": embedding,
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        es.index(index=INDEX_NOTES, id=f"note_{card_id}", document=doc)
        logger.info(f"Indexed note for card: {card_id} with metadata")
        return True
    except Exception as e:
        logger.error(f"Failed to index note for card {card_id}: {e}")
        return False


def rebuild_index() -> Dict[str, Any]:
    """Rebuild all indices from source data"""
    from company_loader import load_companies
    from profile_loader import load_profiles, get_recent_education, get_company_info
    from notes_service import _load_notes
    
    stats = {
        "companies_indexed": 0,
        "persons_indexed": 0,
        "notes_indexed": 0,
        "errors": []
    }
    
    # Delete existing indices
    for index_name in ALL_INDICES:
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
    
    # Create all indices
    if not create_all_indices():
        stats["errors"].append("Failed to create indices")
        return stats
    
    # Index companies
    try:
        companies = load_companies()
        for company in companies:
            company_data = {
                "id": company.get("id", ""),
                "name": company.get("name", ""),
                "industry": company.get("industry"),
                "description": company.get("description"),
                "founded": company.get("founded"),
                "location": company.get("location"),
                "website": company.get("website"),
                "linkedin_url": company.get("linkedin_url")
            }
            if index_company_card(company_data):
                stats["companies_indexed"] += 1
    except Exception as e:
        stats["errors"].append(f"Error indexing companies: {e}")
    
    # Index persons
    try:
        profiles = load_profiles()
        for profile in profiles:
            profile_data = profile.get("profile_data", {})
            name = profile_data.get("name", "Unknown")
            company, designation = get_company_info(profile)
            linkedin_id = profile.get("linkedin_username", "")
            linkedin_url = profile.get("linkedin_url", f"https://linkedin.com/in/{linkedin_id}")
            education = get_recent_education(profile)
            experience_years = profile.get("total_experience_years")
            location = profile_data.get("location")
            
            person_data = {
                "id": f"person_{linkedin_id}",
                "name": name,
                "designation": designation,
                "company": company,
                "linkedin_id": linkedin_id,
                "linkedin_url": linkedin_url,
                "education": education,
                "experience_years": experience_years,
                "location": location
            }
            if index_person_card(person_data):
                stats["persons_indexed"] += 1
    except Exception as e:
        stats["errors"].append(f"Error indexing persons: {e}")
    
    # Index notes
    try:
        from main import get_card_by_id, CompanyCardData, PersonCardData
        notes = _load_notes()
        for card_id, note_text in notes.items():
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
                if index_note(card_id, note_text, card_type, card_metadata):
                    stats["notes_indexed"] += 1
            else:
                # Fallback if card not found
                card_type = "person" if card_id.startswith("person_") else "company"
                if index_note(card_id, note_text, card_type):
                    stats["notes_indexed"] += 1
    except Exception as e:
        stats["errors"].append(f"Error indexing notes: {e}")
    
    return stats

