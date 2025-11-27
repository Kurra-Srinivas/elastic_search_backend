"""Company loader module for loading company data from JSON file"""
import json
import os
from typing import List, Dict, Any, Optional

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPANIES_FILE = os.path.join(BASE_DIR, "companies_data.json")

# Cache for loaded companies
_companies_cache: Optional[List[Dict[str, Any]]] = None


def load_companies() -> List[Dict[str, Any]]:
    """Load companies from JSON file with caching"""
    global _companies_cache
    
    if _companies_cache is not None:
        return _companies_cache
    
    try:
        with open(COMPANIES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            _companies_cache = data.get("companies", [])
        return _companies_cache
    except FileNotFoundError:
        raise FileNotFoundError(f"Companies file not found: {COMPANIES_FILE}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in companies file: {e}")


def clear_cache():
    """Clear the companies cache (useful for testing or reloading)"""
    global _companies_cache
    _companies_cache = None

