"""Profile loader module for loading enriched profiles from JSON file"""
import json
import os
from typing import List, Dict, Any, Optional, Tuple

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_FILE = os.path.join(BASE_DIR, "enriched_profiles.json")

# Cache for loaded profiles
_profiles_cache: Optional[List[Dict[str, Any]]] = None


def load_profiles() -> List[Dict[str, Any]]:
    """Load profiles from JSON file with caching"""
    global _profiles_cache
    
    if _profiles_cache is not None:
        return _profiles_cache
    
    try:
        with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            _profiles_cache = data.get("results", [])
        return _profiles_cache
    except FileNotFoundError:
        raise FileNotFoundError(f"Profiles file not found: {PROFILES_FILE}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in profiles file: {e}")


def get_recent_education(profile: Dict[str, Any]) -> Optional[str]:
    """Get the most recent education from profile"""
    profile_data = profile.get("profile_data", {})
    education = profile_data.get("education_background", [])
    
    if not education:
        return None
    
    # Get the most recent education (last in list or by end_date)
    recent_edu = education[-1] if education else None
    if not recent_edu:
        return None
    
    degree = recent_edu.get("degree_name", "")
    field = recent_edu.get("field_of_study", "")
    school = recent_edu.get("institute_name", "")
    
    parts = []
    if degree:
        parts.append(degree)
    if field:
        parts.append(field)
    if school:
        parts.append(f"@ {school}")
    
    return ", ".join(parts) if parts else None


def get_company_info(profile: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Get company name and designation from profile"""
    profile_data = profile.get("profile_data", {})
    
    # Try current employer first
    current_employers = profile_data.get("current_employers", [])
    if current_employers:
        employer = current_employers[0]
        company = employer.get("employer_name")
        designation = employer.get("employee_title") or profile_data.get("title")
        return company, designation
    
    # Fall back to most recent past employer
    past_employers = profile_data.get("past_employers", [])
    if past_employers:
        # Get the most recent one (first in list if sorted by date)
        employer = past_employers[0]
        company = employer.get("employer_name")
        designation = employer.get("employee_title") or profile_data.get("title")
        return company, designation
    
    # Fall back to title only
    designation = profile_data.get("title")
    return None, designation


def clear_cache():
    """Clear the profiles cache (useful for testing or reloading)"""
    global _profiles_cache
    _profiles_cache = None

