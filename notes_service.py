"""Notes service for storing and retrieving notes per card"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Path to notes storage file
BASE_DIR = Path(__file__).parent
NOTES_FILE = BASE_DIR / "notes.json"

# In-memory cache
_notes_cache: Optional[Dict[str, str]] = None


def _load_notes() -> Dict[str, str]:
    """Load notes from file"""
    global _notes_cache
    
    if _notes_cache is not None:
        return _notes_cache
    
    if NOTES_FILE.exists():
        try:
            with open(NOTES_FILE, 'r', encoding='utf-8') as f:
                _notes_cache = json.load(f)
                return _notes_cache
        except Exception as e:
            logger.error(f"Failed to load notes: {e}")
            _notes_cache = {}
            return _notes_cache
    else:
        _notes_cache = {}
        return _notes_cache


def _save_notes(notes: Dict[str, str]) -> bool:
    """Save notes to file"""
    global _notes_cache
    
    try:
        with open(NOTES_FILE, 'w', encoding='utf-8') as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)
        _notes_cache = notes
        return True
    except Exception as e:
        logger.error(f"Failed to save notes: {e}")
        return False


def get_note(card_id: str) -> str:
    """Get note for a card"""
    notes = _load_notes()
    return notes.get(card_id, "")


def save_note(card_id: str, note: str) -> bool:
    """Save note for a card"""
    notes = _load_notes()
    notes[card_id] = note
    return _save_notes(notes)


def delete_note(card_id: str) -> bool:
    """Delete note for a card"""
    notes = _load_notes()
    if card_id in notes:
        del notes[card_id]
        return _save_notes(notes)
    return True

