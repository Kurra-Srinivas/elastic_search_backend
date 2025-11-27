"""Upload and indexing status tracking"""
from typing import Dict, Optional
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

# In-memory status store (in production, use Redis or database)
_status_store: Dict[str, Dict] = {}


def create_upload_status(file_id: str, filename: str) -> str:
    """Create a new upload status entry"""
    status_id = str(uuid.uuid4())
    _status_store[status_id] = {
        "file_id": file_id,
        "filename": filename,
        "status": "uploading",  # uploading -> extracting -> chunking -> indexing -> completed
        "progress": 0,
        "message": "Uploading file...",
        "chunks_total": 0,
        "chunks_indexed": 0,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    return status_id


def update_upload_status(status_id: str, status: str, progress: int, message: str, 
                        chunks_total: Optional[int] = None, chunks_indexed: Optional[int] = None):
    """Update upload status"""
    if status_id in _status_store:
        _status_store[status_id]["status"] = status
        _status_store[status_id]["progress"] = progress
        _status_store[status_id]["message"] = message
        _status_store[status_id]["updated_at"] = datetime.now().isoformat()
        if chunks_total is not None:
            _status_store[status_id]["chunks_total"] = chunks_total
        if chunks_indexed is not None:
            _status_store[status_id]["chunks_indexed"] = chunks_indexed


def get_upload_status(status_id: str) -> Optional[Dict]:
    """Get upload status"""
    return _status_store.get(status_id)


def complete_upload_status(status_id: str, success: bool = True, message: str = "Completed"):
    """Mark upload as completed"""
    if status_id in _status_store:
        _status_store[status_id]["status"] = "completed" if success else "failed"
        _status_store[status_id]["progress"] = 100 if success else 0
        _status_store[status_id]["message"] = message
        _status_store[status_id]["updated_at"] = datetime.now().isoformat()


def cleanup_old_statuses(max_age_hours: int = 24):
    """Clean up old status entries (call periodically)"""
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    to_remove = [
        status_id for status_id, status in _status_store.items()
        if datetime.fromisoformat(status["created_at"]) < cutoff
    ]
    for status_id in to_remove:
        del _status_store[status_id]

