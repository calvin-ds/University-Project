import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load OCR model for text extraction (TrOCR)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Define coordinates for extracting exam scores
score_coordinates = {
    "Q1": (673, 474, 734, 499),
    "Q2": (673, 503, 734, 528),
    "Q3": (673, 533, 734, 558),
    "Q4": (673, 563, 734, 588),
    "Q5": (673, 591, 734, 616),
    "Q6": (673, 620, 734, 647),
    "Q7": (673, 650, 734, 676),
    "Q8": (673, 680, 734, 706),
    "Q9": (673, 708, 734, 735),
    "Q10": (673, 739, 734, 765),
    "Q11": (673, 767, 734, 794),
    "Total": (659, 806, 742, 838),
}

def extract_scores(image_path):
    """
    Extracts scores for each question and the total using OCR.

    Parameters:
        image_path (str): Path to the processed exam sheet image.

    Returns:
        dict: Extracted scores for each question and total.
    """
    image = cv2.imread(image_path)
    extracted_scores = {}

    for field, (x1, y1, x2, y2) in score_coordinates.items():
        cropped_image = image[y1:y2, x1:x2]
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
        outputs = model.generate(pixel_values)
        extracted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip().upper()

        # Ensure extracted text is numeric (handling OCR errors)
        extracted_text = ''.join(filter(str.isdigit, extracted_text)) if extracted_text else "0"
        extracted_scores[field] = extracted_text

    return extracted_scores
