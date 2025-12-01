# Search UI Backend - System Architecture-(changes made by Srinivas)

## Overview

This project implements a Unified Hybrid Search & Agentic RAG System designed for Venture Capital use cases. It enables users to retrieve information across structured entities (People, Companies) and unstructured data (Documents, Notes) through a natural language interface.

## Core Architecture

### 1. Data Layer: The 4-Index Strategy

We utilize Elasticsearch (Dockerized) with four distinct indices to handle different data modalities:

| Index Name | Type        | Search Method   | Content                               |
|------------|-------------|------------------|----------------------------------------|
| persons    | Structured  | Keyword (BM25)   | Founders, Employees, LinkedIns         |
| companies  | Structured  | Keyword (BM25)   | Startups, VC Firms, Locations          |
| notes      | Structured  | Keyword (BM25)   | Private meeting notes, diligence comments |
| documents  | Unstructured| Vector (Dense)   | PDF chunks, Pitch Decks, Reports       |

**Note:** In the current architecture, only documents are embedded for vector search. The other indices rely on high-performance text matching.

---

### 2. The Intelligence Layer: Agentic RAG

Unlike traditional RAG which simply embeds the user query, we use an Agentic Approach to "think" before searching. This solves the vocabulary mismatch problem between structured database fields and natural language questions.

## The Pipeline Flow (chat_service.py)

### Phase 1: Intent Analysis (Query Generation)

The LLLM (llama-3.3-70b) analyzes the user's question first. It splits the intent into two optimized streams:

- **entity_keywords:** Short keywords for finding specific names/companies in the DB (e.g., "Tencent").  
- **document_query:** A natural language string for finding concepts in PDFs (e.g., "Tencent employee roles").

**Example:**  
> User: "Who is the guy at Tencent?"  
>  
> Agent Output: {"entity_keywords": "Tencent", "document_query": "Tencent employee"}

---

### Phase 2: Parallel Retrieval

The system executes two search strategies simultaneously:

- **Stream A (Standard Tables):** Executes a Multi-Match Keyword search across persons, companies, and notes indices. It scans all fields (*) including nested metadata like location and founded.
- **Stream B (Vector Search):** Executes a Hybrid (k-NN + BM25) search on the documents index using all-MiniLM-L6-v2 embeddings.

---

### Phase 3: Context Synthesis

Results from all 4 indices are aggregated into a structured context window. Metadata fields (Location, Founded Year, Education) are explicitly extracted to enrich the context before sending it to the AI.

---

### Phase 4: Response Generation

The LLM generates a final answer, citing specific sources ("According to the Database..." or "According to the file...").

---

## Feature: Smart Auto-Suggest (/api/suggest)

To improve UX latency, we implemented a dedicated autocomplete endpoint separate from the main search.

- **Endpoint:** `GET /api/suggest?query=...`
- **Mechanism:** Uses Elasticsearch bool_prefix query type.
- **Performance:** Optimized for speed (<50ms) by searching only specific fields: name, title, metadata.location, metadata.designation.
- **Context Aware:** Returns suggestions with context (e.g., "TechFlow Solutions (San Francisco)" or "Weixuan Fu (Full Stack Engineer)").
- **Index Flagging:** updated SearchBar with Index flags (Company/Person/Note) for auto-suggestions.

---

## Key Components

- **main.py:** FastAPI entry point. Handles POST /upload, POST /chat, and GET /api/suggest.  
- **chat_service.py:** Contains the Agentic logic, prompt engineering, and context assembly.  
- **elasticsearch_service.py:** Manages low-level index operations, hybrid_search, and auto-complete logic.  
- **embedding_service.py:** Uses sentence-transformers to vectorize document chunks upon upload.  

---

## Setup & Deployment

### Prerequisites

- Docker Desktop (running Elasticsearch 8.11)  
- Python 3.10+  
- Groq API Key  

### Quick Start

- **Start Database:** `docker start es_project`  
- **Run Backend:** `uvicorn main:app --reload`  
- **Ingest Data:** `POST /api/search/index/rebuild` (loads JSON data).  
- **Ingest Files:** Upload PDFs via the Frontend UI (loads Vector data).  
