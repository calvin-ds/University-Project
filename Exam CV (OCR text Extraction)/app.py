import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import time
from PIL import Image
from io import BytesIO
from fuzzywuzzy import fuzz
from image_processing import process_and_save_image
from text_extraction import extract_text, extract_student_number
from database_validation import load_student_database, match_student_info
from score_extraction import extract_scores

# Set up directories
MANUAL_INSPECTION_DIR = "manual_inspection"

# Ensure manual inspection directory exists
if not os.path.exists(MANUAL_INSPECTION_DIR):
    os.makedirs(MANUAL_INSPECTION_DIR)

# Set page config
st.set_page_config(page_title="Exam Processing", layout="wide", initial_sidebar_state="collapsed")

# Sidebar for Database Uploads (Collapsible)
with st.sidebar:
    st.header("📂 Database Uploads")

    # Upload Student Database
    database_file = st.file_uploader("Upload Student Database (CSV)", type=["csv"])
    
    # Upload Blank Results Database
    results_db_file = st.file_uploader("Upload Blank Results Database (CSV)", type=["csv"])

    # Load databases if uploaded
    student_db, results_db = None, None
    if database_file is not None:
        student_db = pd.read_csv(database_file)
        st.success("✅ Student database uploaded successfully!")

    if results_db_file is not None:
        results_db = pd.read_csv(results_db_file)
        results_db_path = results_db_file.name  # Store path for updates
        st.success("✅ Results database uploaded successfully!")

# Main App Title
st.title("📄 ExamCV - Text extraction and Automation")

# Upload Exam Sheet Image
st.subheader("📄 Upload Exam Sheet Image")
uploaded_file = st.file_uploader("📤 Choose an exam sheet image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    rotation_angle = st.selectbox("🔄 Rotate Image (Clockwise)", [0, 90, 180, 270])

    def rotate_image(image, angle):
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) if angle == 90 else \
               cv2.rotate(image, cv2.ROTATE_180) if angle == 180 else \
               cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) if angle == 270 else image

    rotated_image = rotate_image(image, rotation_angle)
    processed_image = process_and_save_image(rotated_image)
    processed_pil_image = Image.fromarray(processed_image)

    # Layout for Image & Extraction Data (Left & Right Side)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 Processed Image")
        st.image(processed_pil_image, caption="Processed & Resized Image", use_column_width=True)

    processed_image_path = "processed_exam_sheet.jpg"
    cv2.imwrite(processed_image_path, processed_image)

    # Extract text from image
    extracted_data = extract_text(processed_image_path)
    extracted_family_name = extracted_data.get("Family Name", "Not Detected")
    extracted_first_name = extracted_data.get("First Name", "Not Detected")
    extracted_student_number = extract_student_number(processed_image_path)

    matched_family_name, matched_first_name, matched_student_number = match_student_info(
        extracted_family_name, extracted_first_name, extracted_student_number, student_db
    )

    # Calculate Accuracy
    def calculate_accuracy(extracted, matched):
        return fuzz.ratio(extracted, matched) if extracted and matched else 0

    family_name_accuracy = calculate_accuracy(extracted_family_name, matched_family_name)
    first_name_accuracy = calculate_accuracy(extracted_first_name, matched_first_name)
    student_number_accuracy = 100 if extracted_student_number == matched_student_number else fuzz.ratio(extracted_student_number, matched_student_number)

    # Display Extracted Data in Right Column
    with col2:
        st.subheader("📌 Extracted & Matched Data")
        
        comparison_data = pd.DataFrame({
            "Field": ["Family Name", "First Name", "Student Number"],
            "Extracted (OCR)": [extracted_family_name, extracted_first_name, extracted_student_number],
            "Matched (Database)": [matched_family_name, matched_first_name, matched_student_number],
            "Accuracy (%)": [family_name_accuracy, first_name_accuracy, student_number_accuracy]
        })
        st.dataframe(comparison_data)

    # Accuracy Check for Further Extraction
    if all(acc >= 50 for acc in [family_name_accuracy, first_name_accuracy, student_number_accuracy]):
        st.success("✅ More than 50 percent accuracy is attained. Proceeding with automated score extraction!")

        # Extract Scores
        scores = extract_scores(processed_image_path)
        extracted_scores = {q: int(scores[q]) for q in scores if q != "Total"}
        extracted_total = int(scores["Total"])
        calculated_total = sum(extracted_scores.values())

        # Verify if scores match the extracted total
        if extracted_total == calculated_total:
            st.success("✅ Scores match the total! Proceeding with validation.")
            valid_scores = True
        else:
            st.warning("⚠️ Scores do not add up to the total. Manual correction required.")
            valid_scores = False

            # Allow user correction
            for q in extracted_scores:
                extracted_scores[q] = st.number_input(f"Enter corrected score for {q}:", min_value=0, max_value=10, value=extracted_scores[q])
            calculated_total = sum(extracted_scores.values())

            if extracted_total == calculated_total:
                st.success("✅ Correction accepted! Scores now match the total.")
                valid_scores = True
            else:
                st.error("❌ Scores still do not match. Flagging for manual review.")
                valid_scores = False

        # Append to Results Database Only If Valid
        if valid_scores and results_db is not None:
            new_data = {
                "Family Name": matched_family_name,
                "First Name": matched_first_name,
                "Student Number": matched_student_number,
                **extracted_scores,
                "Total Score": extracted_total
            }

            results_db = pd.concat([results_db, pd.DataFrame([new_data])], ignore_index=True)

            # Save updated database
            results_db.to_csv(results_db_path, index=False)
            st.success("✅ Exam results successfully added to the database!")

            # Provide option to download updated database
            st.download_button("💾 Download Updated Results Database", data=open(results_db_path, "rb"), file_name=results_db_path, mime="text/csv")

        elif not valid_scores:
            st.warning("⚠️ Scores are not valid. Moving the image to the manual inspection folder.")
            timestamp = int(time.time())
            flagged_image_filename = f"{matched_student_number}_{timestamp}.jpg"
            flagged_image_path = os.path.join(MANUAL_INSPECTION_DIR, flagged_image_filename)
            cv2.imwrite(flagged_image_path, processed_image)
            st.error(f"🚨 Flagged for manual inspection: {flagged_image_filename}")
    else:
        st.warning("⚠️ Low accuracy detected, moving image for manual inspection.")
        timestamp = int(time.time())
        flagged_image_filename = f"{extracted_student_number}_{timestamp}.jpg"
        flagged_image_path = os.path.join(MANUAL_INSPECTION_DIR, flagged_image_filename)
        cv2.imwrite(flagged_image_path, processed_image)
        st.error(f"🚨 Flagged for manual inspection: {flagged_image_filename}")
