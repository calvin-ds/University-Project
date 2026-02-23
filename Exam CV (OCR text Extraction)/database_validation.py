import pandas as pd
from fuzzywuzzy import process
import os

# Load the student database
def load_student_database(csv_path):
    """Loads the student database from an uploaded CSV file."""
    if not os.path.exists(csv_path):
        print(f"Error: Student database file '{csv_path}' not found.")
        return None

    student_db = pd.read_csv(csv_path)

    # Convert all text columns to uppercase for case-insensitive comparison
    student_db["Family Name"] = student_db["Family Name"].astype(str).str.upper()
    student_db["First Name"] = student_db["First Name"].astype(str).str.upper()
    student_db["Student Number"] = student_db["Student Number"].astype(str)

    return student_db

# Function to match extracted text with database entries
def match_student_info(extracted_family_name, extracted_first_name, extracted_student_number, student_db):
    """
    Matches extracted names and student number with the database using fuzzy matching.

    Returns:
        - Matched Family Name
        - Matched First Name
        - Matched Student Number
    """
    if student_db is None or student_db.empty:
        return extracted_family_name, extracted_first_name, extracted_student_number  # No database, return extracted values

    # Fuzzy match Family Name
    family_match = process.extractOne(extracted_family_name, student_db["Family Name"])
    matched_family_name = family_match[0] if family_match else extracted_family_name

    # Fuzzy match First Name
    first_name_match = process.extractOne(extracted_first_name, student_db["First Name"])
    matched_first_name = first_name_match[0] if first_name_match else extracted_first_name

    # Match Student Number (exact match)
    student_number_match = student_db[student_db["Student Number"] == extracted_student_number]

    # If the student number exists in the database, retrieve the correct details
    if not student_number_match.empty:
        matched_family_name = student_number_match.iloc[0]["Family Name"]
        matched_first_name = student_number_match.iloc[0]["First Name"]
        matched_student_number = student_number_match.iloc[0]["Student Number"]
    else:
        matched_student_number = extracted_student_number  # Keep extracted number if no match found

    return matched_family_name, matched_first_name, matched_student_number
