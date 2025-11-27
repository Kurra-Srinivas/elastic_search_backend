# Elasticsearch Setup Guide

## Prerequisites

1. **Install Elasticsearch locally**
   - Download from: https://www.elastic.co/downloads/elasticsearch
   - Or use Docker (recommended for local dev):
     ```bash
     docker run -d -p 9200:9200 -p 9300:9300 \
       -e "discovery.type=single-node" \
       -e "xpack.security.enabled=false" \
       -e "xpack.security.enrollment.enabled=false" \
       docker.elastic.co/elasticsearch/elasticsearch:8.11.0
     ```
   - **Note**: Security is disabled for local development. For production, enable security and use proper authentication.

2. **Verify Elasticsearch is running**
   ```bash
   curl http://localhost:9200
   ```

## ELSER Model Setup (Optional)

ELSER (Elastic Learned Sparse Encoder) is used for semantic search. **Note**: ELSER requires a license that supports Machine Learning features (not available in Basic license).

The system will work with **keyword search only** if ELSER is not available. If you have an appropriate license:

1. **Deploy the ELSER model** (PowerShell):
   ```powershell
   # For ELSER v2 (Elasticsearch 8.11+)
   Invoke-RestMethod -Method PUT -Uri "http://localhost:9200/_ml/trained_models/.elser_model_2" -ContentType "application/json" -Body '{"input": {"field_names": ["text_field"]}}'
   
   # Or for ELSER v1
   Invoke-RestMethod -Method PUT -Uri "http://localhost:9200/_ml/trained_models/.elser_model_1" -ContentType "application/json" -Body '{"input": {"field_names": ["text_field"]}}'
   ```

2. **Start the model**:
   ```powershell
   Invoke-RestMethod -Method POST -Uri "http://localhost:9200/_ml/trained_models/.elser_model_2/_start"
   ```

3. **Verify model is available**:
   ```powershell
   Invoke-RestMethod -Method GET -Uri "http://localhost:9200/_ml/trained_models/.elser_model_2"
   ```

**Note**: Without ELSER, the system uses keyword (BM25) search which works well for most use cases.

## Environment Variables (Optional)

Create a `.env` file in the backend directory:

```env
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_USER=
ELASTICSEARCH_PASSWORD=
```

## Initial Index Setup

1. **Start the backend server**:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

2. **Rebuild the index** (first time or after data changes):
   ```bash
   curl -X POST "http://localhost:8000/api/search/index/rebuild"
   ```

   Or use the Swagger UI at `http://localhost:8000/docs`

## Search Features

- **Keyword Search**: Traditional BM25 full-text search
- **Semantic Search**: ELSER-powered semantic understanding
- **Hybrid Search**: Combines both for best results

## Troubleshooting

1. **Elasticsearch connection failed**:
   - Ensure Elasticsearch is running on port 9200
   - Check firewall settings

2. **ELSER model not found**:
   - Run the model load command (see above)
   - Check Elasticsearch logs for errors

3. **Index creation failed**:
   - Check Elasticsearch logs
   - Ensure you have proper permissions

## Testing

Test the search endpoint:
```bash
curl "http://localhost:8000/api/cards/search?query=technology&limit=10"
```

