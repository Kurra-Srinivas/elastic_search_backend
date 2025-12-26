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


# --- MAPPINGS ---

def _get_company_mapping() -> Dict[str, Any]:
    """Specific Schema for Companies (Includes mapped_person_ids)"""
    return {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "card_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword", "ignore_above": 256}
                    }
                },
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "metadata": {
                    "properties": {
                        # Store list of person IDs here
                        "mapped_person_ids": {"type": "keyword"},
                        "name": {"type": "text"},
                        "industry": {"type": "keyword"},
                        "location": {"type": "text"},
                        "founded": {"type": "keyword"},
                        "linkedin_url": {"type": "keyword"},
                        "website": {"type": "keyword"}
                    }
                },
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"}
            }
        }
    }

def _get_generic_mapping() -> Dict[str, Any]:
    """Generic mapping for Persons, Notes, Documents"""
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
        }
    }


def create_index(index_name: str) -> bool:
    """Create a specific index with appropriate mapping"""
    if es.indices.exists(index=index_name):
        logger.info(f"Index {index_name} already exists")
        return True
    
    # Use specific mapping for Companies, generic for others
    if index_name == INDEX_COMPANIES:
        mapping = _get_company_mapping()
    else:
        mapping = _get_generic_mapping()
    
    try:
        es.indices.create(index=index_name, body=mapping)
        logger.info(f"Created index {index_name} with specific mapping")
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
    """Index a company card with mapped_person_ids"""
    try:
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
        
        # Prepare Metadata
        metadata = {
            "name": company_data.get("name"),
            "industry": company_data.get("industry"),
            "description": company_data.get("description"),
            "founded": company_data.get("founded"),
            "location": company_data.get("location"),
            "website": company_data.get("website"),
            "linkedin_url": company_data.get("linkedin_url"),
            "mapped_person_ids": company_data.get("mapped_person_ids", [])
        }

        doc = {
            "id": company_id,
            "card_id": company_id,
            "card_type": "company",
            "title": company_data.get("name", ""),
            "content": searchable_content,
            "metadata": metadata,
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
        if not es.indices.exists(index=INDEX_DOCUMENTS):
            create_index(INDEX_DOCUMENTS)
        from text_chunker import chunk_text_by_sentences
        
        chunks = chunk_text_by_sentences(extracted_text, max_chunk_size=500, overlap_sentences=1)
        
        if not chunks:
            logger.warning(f"No chunks created for document {filename}")
            return False
        
        if status_id:
            from upload_status import update_upload_status
            update_upload_status(status_id, "indexing", 60, f"Indexing {len(chunks)} chunks...", chunks_total=len(chunks))
        
        indexed_count = 0
        for chunk_idx, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
                
            chunk_id = f"doc_{card_id}_{filename}_chunk_{chunk_idx}"
            embedding = generate_embedding(chunk_text)
            original_filename = (metadata or {}).get("original_filename", filename)
            
            doc = {
                "id": chunk_id,
                "card_id": card_id,
                "title": f"{original_filename} (chunk {chunk_idx + 1})",
                "content": chunk_text,
                "text_embedding": embedding,
                "metadata": {
                    "filename": filename,
                    "card_id": card_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "original_filename": original_filename,
                    **(metadata or {})
                },
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            es.index(index=INDEX_DOCUMENTS, id=chunk_id, document=doc)
            indexed_count += 1
            
            if status_id:
                progress = 60 + int((chunk_idx + 1) / len(chunks) * 35)
                update_upload_status(status_id, "indexing", progress, 
                                    f"Indexing chunk {chunk_idx + 1}/{len(chunks)}...",
                                    chunks_indexed=chunk_idx + 1)
        
        logger.info(f"Indexed document {filename} as {indexed_count} chunks")
        return True
    except Exception as e:
        logger.error(f"Failed to index document {filename}: {e}")
        return False


def delete_document_by_filename(card_id: str, filename: str) -> bool:
    """Delete all document chunks for a specific filename from Elasticsearch"""
    try:
        if not es.indices.exists(index=INDEX_DOCUMENTS):
            return False
        
        search_all = es.search(
            index=INDEX_DOCUMENTS,
            body={"query": {"term": {"card_id": card_id}}, "size": 10000}
        )
        
        chunks_to_delete = []
        for hit in search_all["hits"]["hits"]:
            metadata = hit["_source"].get("metadata", {})
            stored_filename = metadata.get("filename", "")
            chunk_id = hit["_id"]
            
            if stored_filename == filename or stored_filename.endswith(filename) or filename in stored_filename:
                chunks_to_delete.append(chunk_id)
        
        for chunk_id in chunks_to_delete:
            try:
                es.delete(index=INDEX_DOCUMENTS, id=chunk_id)
            except Exception: pass
        
        es.indices.refresh(index=INDEX_DOCUMENTS)
        return True
    except Exception as e:
        logger.error(f"Failed to delete document chunks for {filename}: {e}")
        return False


def delete_card_from_index(card_id: str, card_type: str) -> bool:
    """Delete a card and all associated documents from Elasticsearch"""
    try:
        index_map = {
            "company": INDEX_COMPANIES,
            "person": INDEX_PERSONS,
            "note": INDEX_NOTES
        }
        
        if card_type in index_map:
            index_name = index_map[card_type]
            if es.indices.exists(index=index_name):
                es.delete(index=index_name, id=card_id, ignore=[404])
        
        if es.indices.exists(index=INDEX_DOCUMENTS):
            query = {"query": {"term": {"card_id": card_id}}}
            es.delete_by_query(index=INDEX_DOCUMENTS, body=query)
        
        return True
    except Exception as e:
        logger.error(f"Failed to delete card {card_id} from index: {e}")
        return False


def search_companies_es(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search companies using Elasticsearch"""
    try:
        if not query or not query.strip(): return []
        
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {"multi_match": {"query": query, "fields": ["title^3", "content^2"], "type": "phrase_prefix", "boost": 3.0}},
                        {"multi_match": {"query": query, "fields": ["title^3", "content^2"], "type": "best_fields", "boost": 2.0}},
                        {"multi_match": {"query": query, "fields": ["title^2", "content"], "type": "best_fields", "fuzziness": "AUTO", "boost": 1.0}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": limit,
            "_source": ["id", "card_id", "card_type", "title", "content", "metadata", "_score"]
        }
        
        if not es.indices.exists(index=INDEX_COMPANIES): return []
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
                "score": hit["_score"]
            })
        return results
    except Exception as e:
        logger.error(f"Elasticsearch company search failed: {e}")
        return []


def search_persons_es(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search persons using Elasticsearch"""
    try:
        if not query or not query.strip(): return []
        
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {"multi_match": {"query": query, "fields": ["title^3", "content^2"], "type": "phrase_prefix", "boost": 3.0}},
                        {"multi_match": {"query": query, "fields": ["title^3", "content^2"], "type": "best_fields", "boost": 2.0}},
                        {"multi_match": {"query": query, "fields": ["title^2", "content"], "type": "best_fields", "fuzziness": "AUTO", "boost": 1.0}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": limit,
            "_source": ["id", "card_id", "card_type", "title", "content", "metadata", "_score"]
        }
        
        if not es.indices.exists(index=INDEX_PERSONS): return []
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
                "score": hit["_score"]
            })
        return results
    except Exception as e:
        logger.error(f"Elasticsearch person search failed: {e}")
        return []


def search_notes_es(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search notes using Elasticsearch"""
    try:
        if not query or not query.strip(): return []
        
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"content": {"query": query, "boost": 3.0}}},
                        {"match_phrase_prefix": {"content": {"query": query, "boost": 2.5}}},
                        {"match": {"content": {"query": query, "fuzziness": "AUTO", "boost": 1.0}}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": limit,
            "_source": ["id", "card_id", "card_type", "title", "content", "metadata", "_score"]
        }
        
        if not es.indices.exists(index=INDEX_NOTES): return []
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
                "score": hit["_score"]
            })
        return results
    except Exception as e:
        logger.error(f"Elasticsearch note search failed: {e}")
        return []


def hybrid_search(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Hybrid search combining keyword (BM25) and semantic (vector) search"""
    try:
        keyword_query = {"multi_match": {"query": query, "fields": ["title^3", "content^2"], "type": "best_fields", "fuzziness": "AUTO"}}
        
        query_embedding = None
        try:
            query_embedding = generate_embedding(query)
        except Exception: pass
        
        if query_embedding and len(query_embedding) > 0:
            search_body = {
                "knn": {
                    "field": "text_embedding", "query_vector": query_embedding, "k": limit, "num_candidates": limit * 2, "boost": 0.5
                },
                "query": {
                    "bool": {"should": [keyword_query], "boost": 1.0}
                },
                "size": limit,
                "_source": ["id", "card_type", "card_id", "title", "content", "metadata", "_score"],
                "highlight": {"fields": {"title": {}, "content": {}}}
            }
        else:
            search_body = {
                "query": keyword_query,
                "size": limit,
                "_source": ["id", "card_type", "card_id", "title", "content", "metadata", "_score"],
                "highlight": {"fields": {"title": {}, "content": {}}}
            }
        
        all_results = []
        for index_name in [INDEX_DOCUMENTS]:
            try:
                response = es.search(index=index_name, body=search_body)
                for hit in response["hits"]["hits"]:
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
            except Exception: continue
        
        card_results = {}
        for result in all_results:
            card_id = result["card_id"]
            if card_id not in card_results:
                card_results[card_id] = result
            else:
                if result["score"] > card_results[card_id]["score"]:
                    card_results[card_id] = result
        
        aggregated_results = list(card_results.values())
        aggregated_results.sort(key=lambda x: x["score"], reverse=True)
        return aggregated_results[:limit]
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def index_note(card_id: str, note: str, card_type: str, card_metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Index a note"""
    try:
        if not es.indices.exists(index=INDEX_NOTES):
            create_index(INDEX_NOTES)
        
        if not note or not note.strip():
            es.delete(index=INDEX_NOTES, id=f"note_{card_id}", ignore=[404])
            return True
        
        content_parts = [note]
        if card_metadata:
            if card_type == "company" and card_metadata.get("name"):
                content_parts.append(f"Company: {card_metadata['name']}")
                if card_metadata.get("industry"): content_parts.append(f"Industry: {card_metadata['industry']}")
            elif card_type == "person" and card_metadata.get("name"):
                content_parts.append(f"Person: {card_metadata['name']}")
                if card_metadata.get("company"): content_parts.append(f"Company: {card_metadata['company']}")
                if card_metadata.get("designation"): content_parts.append(f"Designation: {card_metadata['designation']}")
        
        searchable_content = " ".join(content_parts)
        embedding = generate_embedding(searchable_content)
        
        metadata = {"card_id": card_id, "card_type": card_type}
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
            "content": searchable_content,
            "text_embedding": embedding,
            "metadata": {**metadata, "note_text": note},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        es.index(index=INDEX_NOTES, id=f"note_{card_id}", document=doc)
        return True
    except Exception as e:
        logger.error(f"Failed to index note: {e}")
        return False


def get_card_by_id_es(card_id: str) -> Optional[Dict[str, Any]]:
    """Get a card by ID"""
    try:
        if es.indices.exists(index=INDEX_COMPANIES):
            try:
                res = es.get(index=INDEX_COMPANIES, id=card_id)
                return {**res["_source"].get("metadata", {}), "id": res["_source"]["id"], "card_type": "company"}
            except: pass
        if es.indices.exists(index=INDEX_PERSONS):
            try:
                res = es.get(index=INDEX_PERSONS, id=card_id)
                return {**res["_source"].get("metadata", {}), "id": res["_source"]["id"], "card_type": "person"}
            except: pass
    except: pass
    return None


def get_auto_complete_suggestions(query_text: str, limit: int = 5) -> List[Dict[str, str]]:
    """Fast prefix search for auto-complete"""
    if not query_text: return []
    suggestions = []
    search_body = {
        "query": {
            "multi_match": {
                "query": query_text,
                "type": "bool_prefix",
                "fields": ["title", "metadata.name", "metadata.designation", "metadata.location", "metadata.industry", "metadata.company"]
            }
        },
        "size": limit,
        "_source": ["title", "card_type", "metadata", "id"]
    }
    
    for index in [INDEX_PERSONS, INDEX_COMPANIES, INDEX_DOCUMENTS, INDEX_NOTES]:
        try:
            if es.indices.exists(index=index):
                res = es.search(index=index, body=search_body)
                for hit in res["hits"]["hits"]:
                    src = hit["_source"]
                    meta = src.get("metadata", {})
                    text = src.get("title") or meta.get("name")
                    type_ = "unknown"
                    if index == INDEX_COMPANIES: type_ = "Company"
                    elif index == INDEX_PERSONS: type_ = "Person"
                    elif index == INDEX_DOCUMENTS: 
                        text = meta.get("original_filename") or src.get("title", "").split(" (chunk")[0]
                        type_ = "Document"
                    elif index == INDEX_NOTES: type_ = "Note"
                    
                    if text:
                        suggestions.append({"text": text, "type": type_, "id": src.get("id")})
        except: pass
        
    seen = set()
    unique = []
    for s in suggestions:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique.append(s)
    return unique[:limit]


# --- REBUILD LOGIC ---

def rebuild_index() -> Dict[str, Any]:
    """
    Rebuild indices with Reverse Mapping Strategy.
    1. Reset Indices
    2. Load Persons First -> Map them to Companies (In Memory)
    3. Load Companies Second -> Inject Person IDs from Map
    """
    stats = {
        "companies_indexed": 0,
        "persons_indexed": 0,
        "errors": []
    }
    
    try:
        # 1. Reset Indices (Delete & Recreate)
        for index_name in ALL_INDICES:
            if es.indices.exists(index=index_name):
                es.indices.delete(index=index_name)
            create_index(index_name) # Uses new mapping for companies
        
        # 2. LOAD PERSONS FIRST (To build the map)
        company_roster = {} # Map: "Company Name" -> ["person_id_1", "person_id_2"]
        
        try:
            from profile_loader import load_profiles, get_recent_education, get_company_info
            profiles = load_profiles()
            
            for profile in profiles:
                # Extract Data
                profile_data = profile.get("profile_data", {})
                name = profile_data.get("name", "Unknown")
                linkedin_id = profile.get("linkedin_username", "")
                
                if not name or name == "Unknown" or not linkedin_id:
                    continue
                
                # Generate ID (Must match what we store)
                person_id = f"person_{linkedin_id}"
                
                # Get Company Info
                company_name, designation = get_company_info(profile)
                
                # --- BUILD THE MAP (Key Step) ---
                if company_name:
                    # Clean company name to match (strip spaces, lowercase optionally if fuzzy needed)
                    # For now, strict string match
                    clean_c_name = company_name.strip()
                    if clean_c_name not in company_roster:
                        company_roster[clean_c_name] = []
                    company_roster[clean_c_name].append(person_id)
                
                # Prepare Person Data
                person_data = {
                    "id": person_id,
                    "name": name,
                    "designation": designation,
                    "company": company_name,
                    "linkedin_id": linkedin_id,
                    "linkedin_url": profile.get("linkedin_url", ""),
                    "education": get_recent_education(profile),
                    "experience_years": profile.get("total_experience_years"),
                    "location": profile_data.get("location")
                }
                
                # Index Person
                if index_person_card(person_data):
                    stats["persons_indexed"] += 1
                    
        except Exception as e:
            stats["errors"].append(f"Failed to index persons: {e}")
            logger.error(f"Person load error: {e}")

        # 3. LOAD COMPANIES SECOND (Inject the map)
        try:
            from company_loader import load_companies
            companies = load_companies()
            
            for company in companies:
                c_name = company.get("name", "").strip()
                
                # --- INJECT MAPPED IDS ---
                # Check if we found people for this company in step 2
                associated_people = company_roster.get(c_name, [])
                company["mapped_person_ids"] = associated_people
                
                if associated_people:
                    logger.info(f"Mapped {len(associated_people)} people to {c_name} (IDs: {associated_people})")

                # Index Company
                if index_company_card(company):
                    stats["companies_indexed"] += 1
                    
        except Exception as e:
            stats["errors"].append(f"Failed to index companies: {e}")
            logger.error(f"Company load error: {e}")
            
        logger.info(f"Rebuild Complete: {stats}")
        
    except Exception as e:
        stats["errors"].append(f"Rebuild failed: {e}")
        logger.error(f"Index rebuild failed: {e}")
    
    return stats