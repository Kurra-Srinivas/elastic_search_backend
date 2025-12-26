import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_FILE = os.path.join(BASE_DIR, "data", "enriched_profiles.json")
COMPANIES_FILE = os.path.join(BASE_DIR, "data", "companies_data.json")

def generate_companies_file():
    try:
        with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
            profiles = profile_data.get("results", [])
    except FileNotFoundError:
        print(f"Error: Could not find {PROFILES_FILE}")
        return

    extracted_companies = {}
    company_counter = 1

    print(f"Processing {len(profiles)} profiles (Latest Employer Only)...")

    for profile in profiles:
        profile_data = profile.get("profile_data", {})
        current_employers = profile_data.get("current_employers", [])
        past_employers = profile_data.get("past_employers", [])
        
        # 1. CHECK: Try current employer first, fallback to past employer
        latest_emp = None
        is_past = False
        
        if current_employers:
            latest_emp = current_employers[0]
        elif past_employers:
            # Get the most recent past employer (first in the list)
            latest_emp = past_employers[0]
            is_past = True
        
        # 2. VALIDATE: Skip if no employer found at all
        if not latest_emp:
            person_name = profile_data.get("name", "Unknown")
            print(f"    Skipped: {person_name} (No employers found)")
            continue

        name = latest_emp.get("employer_name", "").strip()
        
        # 3. VALIDATE: Skip only if truly empty
        if not name:
            continue
            
        # 4. STORE: Add to our list if we haven't seen it yet
        if name not in extracted_companies:
            # Extract website safely
            websites = latest_emp.get("employer_company_website_domain", [])
            website_url = f"https://{websites[0]}" if websites else ""

            # Extract LinkedIn URL properly
            linkedin_id = latest_emp.get("employer_linkedin_id", "")
            linkedin_url = ""
            
            if linkedin_id:
                # Case 1: Already a full URL (starts with http)
                if linkedin_id.startswith("http"):
                    linkedin_url = linkedin_id
                # Case 2: Partial URL with linkedin.com/company/
                elif "linkedin.com/company/" in linkedin_id:
                    # Make sure it has https://
                    if not linkedin_id.startswith("https://"):
                        linkedin_url = f"https://{linkedin_id}"
                    else:
                        linkedin_url = linkedin_id
                # Case 3: Just the numeric ID or company slug
                else:
                    linkedin_url = f"https://linkedin.com/company/{linkedin_id}"
            
            # Extract location - handle multiple location formats
            location = latest_emp.get("employee_location", "")
            if location:
                # Clean up location - take first part if comma-separated
                location_parts = location.split(",")
                if len(location_parts) >= 2:
                    # Take city and country or last meaningful part
                    location = location_parts[0].strip()

            # Extract description
            description = latest_emp.get("employer_linkedin_description", "")
            if description and len(description) > 200:
                description = description[:200] + "..."

            new_company = {
                "id": f"comp_{company_counter:03d}",
                "name": name,
                "industry": "",  # Placeholder
                "description": description,
                "founded": "",
                "location": location,
                "website": website_url,
                "linkedin_url": linkedin_url
            }
            
            extracted_companies[name] = new_company
            company_counter += 1
            
            # Show if it's from past employment
            status = "(past employer)" if is_past else ""
            print(f"  → Added: {name} {status}")

    # 5. SAVE: Overwrite the file
    output_data = {"companies": list(extracted_companies.values())}

    os.makedirs(os.path.dirname(COMPANIES_FILE), exist_ok=True)
    
    with open(COMPANIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n Success! Generated {len(extracted_companies)} unique companies.")
    print(f" File saved to: {COMPANIES_FILE}")

if __name__ == "__main__":
    generate_companies_file()