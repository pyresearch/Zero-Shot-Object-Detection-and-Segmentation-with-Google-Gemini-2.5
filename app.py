import os
import google.generativeai as genai
from google.generativeai import types
from PIL import Image
import supervision as sv

# Set your API key (use environment variable for security)
os.environ["GEMINI_API_KEY"] = "AIzaSyAye3z3gJJyyTMqRavDre-jBQ3jHRzJi00"  # Replace if setting via env

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

safety_settings = {
    'DANGEROUS': 'BLOCK_ONLY_HIGH',
}

MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.5

# Create the model instance
model = genai.GenerativeModel(MODEL_NAME)

# First part: Object detection for yellow taxi
# Update path to your local file
IMAGE_PATH = "demotest.jpg"  # Adjust as needed
PROMPT = (
    "Detect People present: Mark Zuckerberg, Elon Musk, Jensen Huang, Sundar Pichai, Tim Cook, Satya Nadella, Sam Altman."
    "Output a JSON list of bounding boxes where each entry contains the 2D bounding box in the key \"box_2d\", "
    "and the text label in the key \"label\". Use descriptive labels."
)

image = Image.open(IMAGE_PATH)
width, height = image.size
target_height = int(1024 * height / width)
resized_image = image.resize((1024, target_height), Image.Resampling.LANCZOS)

response = model.generate_content(
    [resized_image, PROMPT],
    generation_config=types.GenerationConfig(
        temperature=TEMPERATURE,
    ),
    safety_settings=safety_settings,
)

print(response.text)

resolution_wh = image.size

detections = sv.Detections.from_vlm(
    vlm=sv.VLM.GOOGLE_GEMINI_2_5,
    result=response.text,
    resolution_wh=resolution_wh,
)

thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

box_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    smart_position=True,
    text_color=sv.Color.BLACK,
    text_scale=text_scale,
    text_position=sv.Position.CENTER,
)

annotated = image
for annotator in (box_annotator, label_annotator):
    annotated = annotator.annotate(scene=annotated, detections=detections)

sv.plot_image(annotated)

# Second part: Segmentation for shoes and jacket
# Update path to your local file
IMAGE_PATH = "demotest.jpg"  # Adjust as needed
PROMPT = (
    "Give the segmentation masks for People present: Mark Zuckerberg, Elon Musk, Jensen Huang, Sundar Pichai, Tim Cook, Satya Nadella, Sam Altman."
    "Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key \"box_2d\", "
    "the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive labels."
)

image = Image.open(IMAGE_PATH)
width, height = image.size
target_height = int(1024 * height / width)
resized_image = image.resize((1024, target_height), Image.Resampling.LANCZOS)

response = model.generate_content(
    [resized_image, PROMPT],
    generation_config=types.GenerationConfig(
        temperature=TEMPERATURE,
    ),
    safety_settings=safety_settings,
)

print(response.text)

resolution_wh = image.size

detections = sv.Detections.from_vlm(
    vlm=sv.VLM.GOOGLE_GEMINI_2_5,
    result=response.text,
    resolution_wh=resolution_wh,
)

thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

box_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    smart_position=True,
    text_color=sv.Color.BLACK,
    text_scale=text_scale,
    text_position=sv.Position.CENTER,
)
masks_annotator = sv.MaskAnnotator()

annotated = image
for annotator in (box_annotator, label_annotator, masks_annotator):
    annotated = annotator.annotate(scene=annotated, detections=detections)

sv.plot_image(annotated)