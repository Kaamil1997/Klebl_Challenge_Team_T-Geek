import os
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Base")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Base",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

output_folder = "Output_data"
os.makedirs(output_folder, exist_ok=True)

def load_model(model_path):
    return YOLO(model_path)

def process_image(input_path, model):
    original_image = cv2.imread(input_path)
    if original_image is None:
        raise ValueError(f"Failed to load image: {input_path}")

    original_height, original_width = original_image.shape[:2]
    resized_image = cv2.resize(original_image, (1080, 720))
    results = model(resized_image, conf=0.2, iou=0.8)[0]

    if hasattr(results, 'boxes'):
        detections = results.boxes
        for box in detections:
            confidence = box.conf[0]
            if confidence > 0.60:
                x1_resized, y1_resized, x2_resized, y2_resized = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = results.names[class_id]

                scale_x = original_width / 1080
                scale_y = original_height / 720

                x1_original = int(x1_resized * scale_x)
                y1_original = int(y1_resized * scale_y)
                x2_original = int(x2_resized * scale_x)
                y2_original = int(y2_resized * scale_y)

                cropped_object = original_image[y1_original:y2_original, x1_original:x2_original]
                output_cropped_path = os.path.join(output_folder, f"{label}.jpg")
                cv2.imwrite(output_cropped_path, cropped_object)
    else:
        print("No bounding boxes found in results.")

def extract_data_from_image(image_path, prompt):
    # Load image using PIL
    image = Image.open(image_path)
    
    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        },
    ]
    
    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)
    
    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]
    
    return generated_text

def main(image_path):
    model = load_model("best_2.pt")
    process_image(image_path, model)

    json_image_path = "/content/Output_data/plankopf.jpg"
    json_prompt = """extract all these the values of the given keys in a json format:
                    1. Planschlüssel
                    2. Stat.Pos
                    3. Auftr. Nr.
                    4. Index
                    5. Fertigteil Position
                    6. Stück
                    7. Volumen (m3)
                    8. Gewicht (to)"""
    json_output = extract_data_from_image(json_image_path, json_prompt)
    print("JSON Output:", json_output)

    table_image_path_1 = "/content/Output_data/liste_stahl.jpg"
    table_prompt_1 = """extract all these the values of the given image and print it in table format"""
    table_output_1 = extract_data_from_image(table_image_path_1, table_prompt_1)
    print("Table Output 1:", table_output_1)

    table_image_path_2 = "/content/Output_data/liste_einbauteile.jpg"
    table_prompt_2 = """extract all these the values of the given image and print it in table format"""
    table_output_2 = extract_data_from_image(table_image_path_2, table_prompt_2)
    print("Table Output 2:", table_output_2)

    return json_output, table_output_1, table_output_2

if __name__ == "__main__":
    image_path = "/content/FT_XX_10-221_-_F.jpg"
    json_output, table_output_1, table_output_2 = main(image_path)