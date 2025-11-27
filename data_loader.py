"""Data loader module for loading synthetic data from JSON file"""
import json
import os
from typing import List, Dict, Any

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.json")

# Cache for loaded data
_data_cache: Dict[str, List[str]] = None


def load_data() -> Dict[str, List[str]]:
    """Load data from JSON file with caching"""
    global _data_cache
    
    if _data_cache is not None:
        return _data_cache
    
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            _data_cache = json.load(f)
        return _data_cache
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in data file: {e}")


def get_companies() -> List[str]:
    """Get list of company names"""
    data = load_data()
    return data.get("companies", [])


def get_first_names() -> List[str]:
    """Get list of first names"""
    data = load_data()
    return data.get("first_names", [])


def get_last_names() -> List[str]:
    """Get list of last names"""
    data = load_data()
    return data.get("last_names", [])


def get_designations() -> List[str]:
    """Get list of designations"""
    data = load_data()
    return data.get("designations", [])


def clear_cache():
    """Clear the data cache (useful for testing or reloading)"""
    global _data_cache
    _data_cache = None

