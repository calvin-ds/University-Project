import cv2
import time
from PIL import Image
import google.generativeai as genai
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load TrOCR model for text extraction (Names)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Configure Gemini API
genai.configure(api_key="AIzaSyCdRBRihIfq3gZRbRoz3xSd8xI_dAWW5pg")  # Replace with your actual API key

# Define coordinates for student details (update as per your exam sheet format)
student_info_coordinates = {
    "Family Name": (215, 64, 448, 103),
    "First Name": (216, 100, 446, 133),
    "Student Number": (216, 132, 364, 155)
}

def extract_text(image_path):
    """Extracts Family Name and First Name using OCR (TrOCR)."""
    image = cv2.imread(image_path)
    extracted_values = {}

    for field, (x1, y1, x2, y2) in student_info_coordinates.items():
        if field == "Student Number":
            continue  # Skip student number, as it's handled separately

        cropped_image = image[y1:y2, x1:x2]
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
        outputs = model.generate(pixel_values)
        extracted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip().upper()
        extracted_values[field] = extracted_text

    return extracted_values

def extract_student_number(image_path):
    """Extracts Student Number using Gemini API."""
    x1, y1, x2, y2 = student_info_coordinates["Student Number"]
    image = cv2.imread(image_path)
    student_number_roi = image[y1:y2, x1:x2]

    temp_image_path = "temp_student_number.jpg"
    cv2.imwrite(temp_image_path, student_number_roi)

    pil_image = Image.open(temp_image_path)
    model = genai.GenerativeModel("gemini-1.5-flash")

    time.sleep(3)  # Prevent rate limiting
    response = model.generate_content([pil_image, "Extract only the student number from this image. Return digits only."])

    return response.text.strip()
