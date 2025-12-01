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

# All Elasticsearch indices
INDEX_COMPANIES = "companies"
INDEX_PERSONS = "persons"
INDEX_NOTES = "notes"
INDEX_DOCUMENTS = "documents"

# All indices list
ALL_INDICES = [INDEX_COMPANIES, INDEX_PERSONS, INDEX_NOTES, INDEX_DOCUMENTS]


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
    """Create all Elasticsearch indices"""
    success = True
    success = create_index(INDEX_COMPANIES) and success
    success = create_index(INDEX_PERSONS) and success
    success = create_index(INDEX_NOTES) and success
    success = create_index(INDEX_DOCUMENTS) and success
    return success


def index_company_card(company_data: Dict[str, Any]) -> bool:
    """Index a company card in Elasticsearch for keyword and fuzzy search"""
    try:
        # Ensure index exists
        if not es.indices.exists(index=INDEX_COMPANIES):
            create_index(INDEX_COMPANIES)
        
        company_id = company_data.get("id")
        if not company_id:
            logger.warning("Cannot index company without ID")
            return False
        
        # Build searchable content from all fields
        searchable_fields = []
        if company_data.get("name"):
            searchable_fields.append(company_data["name"])
        if company_data.get("industry"):
            searchable_fields.append(company_data["industry"])
        if company_data.get("description"):
            searchable_fields.append(company_data["description"])
        if company_data.get("location"):
            searchable_fields.append(company_data["location"])
        
        searchable_content = " ".join(searchable_fields)
        
        doc = {
            "id": company_id,
            "card_id": company_id,
            "card_type": "company",
            "title": company_data.get("name", ""),
            "content": searchable_content,
            "metadata": {
                "name": company_data.get("name"),
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
        
        es.index(index=INDEX_COMPANIES, id=company_id, document=doc)
        logger.debug(f"Indexed company: {company_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to index company {company_data.get('id')}: {e}")
        return False


def index_person_card(person_data: Dict[str, Any]) -> bool:
    """Index a person card in Elasticsearch for keyword and fuzzy search"""
    try:
        # Ensure index exists
        if not es.indices.exists(index=INDEX_PERSONS):
            create_index(INDEX_PERSONS)
        
        person_id = person_data.get("id")
        if not person_id:
            logger.warning("Cannot index person without ID")
            return False
        
        # Build searchable content from all fields
        searchable_fields = []
        if person_data.get("name"):
            searchable_fields.append(person_data["name"])
        if person_data.get("designation"):
            searchable_fields.append(person_data["designation"])
        if person_data.get("company"):
            searchable_fields.append(person_data["company"])
        if person_data.get("education"):
            searchable_fields.append(person_data["education"])
        if person_data.get("location"):
            searchable_fields.append(person_data["location"])
        
        searchable_content = " ".join(searchable_fields)
        
        doc = {
            "id": person_id,
            "card_id": person_id,
            "card_type": "person",
            "title": person_data.get("name", ""),
            "content": searchable_content,
            "metadata": {
                "name": person_data.get("name"),
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
        
        es.index(index=INDEX_PERSONS, id=person_id, document=doc)
        logger.debug(f"Indexed person: {person_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to index person {person_data.get('id')}: {e}")
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
    """Delete a card and all associated documents from Elasticsearch"""
    try:
        # Delete from appropriate index based on card_type
        index_map = {
            "company": INDEX_COMPANIES,
            "person": INDEX_PERSONS,
            "note": INDEX_NOTES
        }
        
        # Delete the card itself
        if card_type in index_map:
            index_name = index_map[card_type]
            if es.indices.exists(index=index_name):
                es.delete(index=index_name, id=card_id, ignore=[404])
                logger.debug(f"Deleted {card_type} {card_id} from index {index_name}")
        
        # Delete associated documents
        if es.indices.exists(index=INDEX_DOCUMENTS):
            query = {
                "query": {
                    "term": {"card_id": card_id}
                }
            }
            es.delete_by_query(index=INDEX_DOCUMENTS, body=query)
            logger.debug(f"Deleted documents for card: {card_id}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to delete card {card_id} from index: {e}")
        return False


def search_companies_es(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Search companies using Elasticsearch with keyword and fuzzy matching
    Direct matches rank higher than fuzzy matches
    """
    try:
        if not query or not query.strip():
            return []
        
        # Build query with direct matches boosted higher than fuzzy
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        # Direct keyword matches (boosted higher)
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "content^2"],
                                "type": "phrase_prefix",  # Exact phrase matches rank highest
                                "boost": 3.0
                            }
                        },
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "content^2"],
                                "type": "best_fields",  # Direct keyword matches
                                "boost": 2.0
                            }
                        },
                        # Fuzzy matches (lower boost)
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^2", "content"],
                                "type": "best_fields",
                                "fuzziness": "AUTO",  # Automatic fuzziness based on term length
                                "boost": 1.0
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": limit,
            "_source": ["id", "card_id", "card_type", "title", "content", "metadata", "_score"],
            "highlight": {
                "fields": {
                    "title": {},
                    "content": {}
                }
            }
        }
        
        if not es.indices.exists(index=INDEX_COMPANIES):
            return []
        
        response = es.search(index=INDEX_COMPANIES, body=search_body)
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_source"]["id"],
                "card_id": hit["_source"]["card_id"],
                "card_type": "company",
                "title": hit["_source"]["title"],
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("metadata", {}),
                "score": hit["_score"],
                "highlights": hit.get("highlight", {})
            })
        
        return results
    except Exception as e:
        logger.error(f"Elasticsearch company search failed: {e}")
        return []


def search_persons_es(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Search persons using Elasticsearch with keyword and fuzzy matching
    Direct matches rank higher than fuzzy matches
    """
    try:
        if not query or not query.strip():
            return []
        
        # Build query with direct matches boosted higher than fuzzy
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        # Direct keyword matches (boosted higher)
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "content^2"],
                                "type": "phrase_prefix",
                                "boost": 3.0
                            }
                        },
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "content^2"],
                                "type": "best_fields",
                                "boost": 2.0
                            }
                        },
                        # Fuzzy matches (lower boost)
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^2", "content"],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                                "boost": 1.0
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": limit,
            "_source": ["id", "card_id", "card_type", "title", "content", "metadata", "_score"],
            "highlight": {
                "fields": {
                    "title": {},
                    "content": {}
                }
            }
        }
        
        if not es.indices.exists(index=INDEX_PERSONS):
            return []
        
        response = es.search(index=INDEX_PERSONS, body=search_body)
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_source"]["id"],
                "card_id": hit["_source"]["card_id"],
                "card_type": "person",
                "title": hit["_source"]["title"],
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("metadata", {}),
                "score": hit["_score"],
                "highlights": hit.get("highlight", {})
            })
        
        return results
    except Exception as e:
        logger.error(f"Elasticsearch person search failed: {e}")
        return []


def search_notes_es(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Search notes using Elasticsearch with keyword and fuzzy matching
    Direct matches rank higher than fuzzy matches
    """
    try:
        if not query or not query.strip():
            return []
        
        # Build query with direct matches boosted higher than fuzzy
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        # Direct keyword matches (boosted higher)
                        {
                            "match": {
                                "content": {
                                    "query": query,
                                    "boost": 3.0
                                }
                            }
                        },
                        {
                            "match_phrase_prefix": {
                                "content": {
                                    "query": query,
                                    "boost": 2.5
                                }
                            }
                        },
                        # Fuzzy matches (lower boost)
                        {
                            "match": {
                                "content": {
                                    "query": query,
                                    "fuzziness": "AUTO",
                                    "boost": 1.0
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": limit,
            "_source": ["id", "card_id", "card_type", "title", "content", "metadata", "_score"],
            "highlight": {
                "fields": {
                    "content": {}
                }
            }
        }
        
        if not es.indices.exists(index=INDEX_NOTES):
            return []
        
        response = es.search(index=INDEX_NOTES, body=search_body)
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_source"]["id"],
                "card_id": hit["_source"]["card_id"],
                "card_type": "note",
                "title": hit["_source"]["title"],
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("metadata", {}),
                "score": hit["_score"],
                "highlights": hit.get("highlight", {})
            })
        
        return results
    except Exception as e:
        logger.error(f"Elasticsearch note search failed: {e}")
        return []


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
        
        # Search only documents index (companies, persons, notes use separate search functions)
        all_results = []
        
        for index_name in [INDEX_DOCUMENTS]:
            try:
                response = es.search(index=index_name, body=search_body)
                
                for hit in response["hits"]["hits"]:
                    # Only documents are indexed now
                    result = {
                        "id": hit["_source"]["id"],
                        "card_type": "document",
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
        
        # Group results by card_id to handle multiple document chunks per card
        # For each card, keep only the highest scoring match
        card_results = {}
        for result in all_results:
            card_id = result["card_id"]
            # For documents, aggregate by taking the max score
            if card_id not in card_results:
                card_results[card_id] = result
            else:
                # Keep the highest scoring chunk for this card
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
            "metadata": {
                **metadata,
                "note_text": note  # Store original note text separately for easy retrieval
            },
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        es.index(index=INDEX_NOTES, id=f"note_{card_id}", document=doc)
        logger.info(f"Indexed note for card: {card_id} with metadata")
        return True
    except Exception as e:
        logger.error(f"Failed to index note for card {card_id}: {e}")
        return False


def get_card_by_id_es(card_id: str) -> Optional[Dict[str, Any]]:
    """Get a card (company or person) by its ID from Elasticsearch - returns dict with card data"""
    try:
        # Try company index first
        if es.indices.exists(index=INDEX_COMPANIES):
            try:
                response = es.get(index=INDEX_COMPANIES, id=card_id)
                metadata = response["_source"].get("metadata", {})
                return {
                    "id": response["_source"]["id"],
                    "name": metadata.get("name", ""),
                    "industry": metadata.get("industry"),
                    "description": metadata.get("description"),
                    "founded": metadata.get("founded"),
                    "location": metadata.get("location"),
                    "website": metadata.get("website"),
                    "linkedin_url": metadata.get("linkedin_url"),
                    "card_type": "company"
                }
            except Exception:
                pass
        
        # Try person index
        if es.indices.exists(index=INDEX_PERSONS):
            try:
                response = es.get(index=INDEX_PERSONS, id=card_id)
                metadata = response["_source"].get("metadata", {})
                return {
                    "id": response["_source"]["id"],
                    "name": metadata.get("name", ""),
                    "designation": metadata.get("designation"),
                    "company": metadata.get("company"),
                    "linkedin_id": metadata.get("linkedin_id", ""),
                    "linkedin_url": metadata.get("linkedin_url", ""),
                    "education": metadata.get("education"),
                    "experience_years": metadata.get("experience_years"),
                    "location": metadata.get("location"),
                    "card_type": "person"
                }
            except Exception:
                pass
        
        return None
    except Exception as e:
        logger.error(f"Failed to get card {card_id} from Elasticsearch: {e}")
        return None


def rebuild_index() -> Dict[str, Any]:
    """Rebuild all indices from JSON files"""
    stats = {
        "companies_indexed": 0,
        "persons_indexed": 0,
        "notes_indexed": 0,
        "documents_indexed": 0,
        "errors": []
    }
    
    try:
        # Recreate all indices
        for index_name in ALL_INDICES:
            if es.indices.exists(index=index_name):
                es.indices.delete(index=index_name)
            if not create_index(index_name):
                stats["errors"].append(f"Failed to create index {index_name}")
        
        # Index companies from JSON file
        try:
            from company_loader import load_companies
            companies = load_companies()
            for company in companies:
                if index_company_card(company):
                    stats["companies_indexed"] += 1
        except Exception as e:
            stats["errors"].append(f"Failed to index companies: {e}")
        
        # Index persons from JSON file
        try:
            from profile_loader import load_profiles, get_recent_education, get_company_info
            profiles = load_profiles()
            for profile in profiles:
                profile_data = profile.get("profile_data", {})
                name = profile_data.get("name", "Unknown")
                if not name or name == "Unknown":
                    continue
                
                linkedin_id = profile.get("linkedin_username", "")
                if not linkedin_id:
                    continue
                
                person_id = f"person_{linkedin_id}"
                company, designation = get_company_info(profile)
                linkedin_url = profile.get("linkedin_url", f"https://linkedin.com/in/{linkedin_id}")
                education = get_recent_education(profile)
                experience_years = profile.get("total_experience_years")
                location = profile_data.get("location")
                
                person_data = {
                    "id": person_id,
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
            stats["errors"].append(f"Failed to index persons: {e}")
        
        # Note: Notes are indexed when saved, not during rebuild
        # Documents are indexed when uploaded, not during rebuild
        
        logger.info(f"Index rebuild complete: {stats['companies_indexed']} companies, "
                   f"{stats['persons_indexed']} persons indexed")
        
    except Exception as e:
        stats["errors"].append(f"Rebuild failed: {e}")
        logger.error(f"Index rebuild failed: {e}")
    
    return stats




# For auto Suggestions...

# ... (Keep previous imports and code) ...

def get_auto_complete_suggestions(query_text: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Fast prefix search for auto-complete dropdowns.
    Updated: Now includes NOTES and uses better field targeting.
    """
    if not query_text:
        return []

    suggestions = []
    
    search_body = {
        "query": {
            "multi_match": {
                "query": query_text,
                "type": "bool_prefix", 
                "fields": [
                    "title",                        
                    "metadata.name",                
                    "metadata.designation",         
                    "metadata.location",            
                    "metadata.industry",            
                    "metadata.education",           
                    "metadata.company",
                    "content" # Added content for Notes/Docs prefix search
                ]
            }
        },
        "size": limit,
        "_source": ["title", "card_type", "metadata", "id", "content"] 
    }

    # ADDED: INDEX_NOTES to the list
    target_indices = [INDEX_PERSONS, INDEX_COMPANIES, INDEX_DOCUMENTS, INDEX_NOTES]
    
    for index in target_indices:
        try:
            if es.indices.exists(index=index):
                res = es.search(index=index, body=search_body)
                
                for hit in res["hits"]["hits"]:
                    src = hit["_source"]
                    meta = src.get("metadata", {})
                    
                    # 1. Determine Display Text
                    text = src.get("title") or meta.get("name")
                    
                    # 2. Determine Type & Context Hint
                    type_ = "unknown"
                    sub_text = ""
                    
                    if index == INDEX_COMPANIES: 
                        type_ = "Company"
                        if meta.get("location"): sub_text = meta.get("location")
                        elif meta.get("industry"): sub_text = meta.get("industry")
                        
                    elif index == INDEX_PERSONS: 
                        type_ = "Person"
                        if meta.get("designation"): sub_text = meta.get("designation")
                        elif meta.get("location"): sub_text = meta.get("location")
                        
                    elif index == INDEX_DOCUMENTS: 
                        type_ = "Document"
                        # For docs, maybe show the first few words of content?
                        # sub_text = "File" 
                        
                    elif index == INDEX_NOTES:
                        type_ = "Note"
                        # For notes, show who it's attached to
                        if meta.get("person_name"): sub_text = f"on {meta['person_name']}"
                        elif meta.get("company_name"): sub_text = f"on {meta['company_name']}"

                    if text:
                        # Format: "Name (Extra Info)"
                        display_text = f"{text} ({sub_text})" if sub_text else text
                        
                        suggestions.append({
                            "text": display_text,
                            "type": type_,
                            "id": src.get("id")
                        })
        except Exception as e:
            logger.warning(f"Suggestion error on {index}: {e}")

    # 3. Deduplicate
    seen = set()
    unique_suggestions = []
    for s in suggestions:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique_suggestions.append(s)
            
    return unique_suggestions[:limit]