import cv2
from google import genai
from pydantic import BaseModel, Field
from typing import List

# Sources/Docs use :
#https://aistudio.google.com/docs/structured-output?hl=fr
#https://docs.opencv.org/4.13.0/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
#https://aistudio.google.com/docs/image-understanding

class schema(BaseModel):
    coord: List[List[int]] = Field(description="Coordinates in format [[ymin, xmin, ymax, xmax], ...], normalize from 0 to 1000")

def check(image_path, target):
    client = genai.Client()
    image = cv2.imread(image_path)

    uploaded_file = client.files.upload(file=image_path)

    prompt = f"Detect and locate every occurrence of '{target}'. If none exist, return an empty list."

    interaction = client.interactions.create(
        model="gemini-flash-lite-latest",
        input=[
            {"type": "text", "text": prompt},
            {"type": "image", "uri": uploaded_file.uri, "mime_type": uploaded_file.mime_type},
            ],
        response_format={ 
            "type": "text",
            "mime_type": "application/json",
            "schema": schema.model_json_schema()},
        )

    detections = schema.model_validate_json(interaction.output_text).coord
    print(detections)

    if not detections:
        return False

    height, width, _ = image.shape
    for object in detections:
        ymin, xmin, ymax, xmax = object
        pt1 = (xmin * width // 1000, ymin * height // 1000)
        pt2 = (xmax * width // 1000, ymax * height // 1000)

        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)

    cv2.imwrite("result4.jpg", image)
    return True


if __name__ == "__main__":
    img = input("Image Path : ")
    target = input("Object to Highlight : ")

    is_present = check(img, target)

    if is_present:
        print(f"Yes : '{target}' is on the picture (result save as result4.jpg).")
    else:
        print(f"No : '{target}' has not been detected.")