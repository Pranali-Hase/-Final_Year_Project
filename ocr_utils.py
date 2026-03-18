import easyocr
import os

reader = easyocr.Reader(['en'], gpu=False)

def extract_text(image_path):
    if not os.path.exists(image_path):
        return ""

    results = reader.readtext(image_path)
    text = " ".join([res[1] for res in results])
    return text
