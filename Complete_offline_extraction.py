import os
import cv2
import torch
import json
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

def main_table(image_path):
    model = load_model("best_2.pt")
    process_image(image_path, model)

    final_data = {}

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
    final_data['JSON Data'] = json_output

    table_image_path_1 = "/content/Output_data/liste_stahl.jpg"
    table_prompt_1 = """extract all these the values of the given image and print it in table format"""
    table_output_1 = extract_data_from_image(table_image_path_1, table_prompt_1)
    final_data['Table Data 1'] = table_output_1

    table_image_path_2 = "/content/Output_data/liste_einbauteile.jpg"
    table_prompt_2 = """extract all these the values of the given image and print it in table format"""
    table_output_2 = extract_data_from_image(table_image_path_2, table_prompt_2)
    final_data['Table Data 2'] = table_output_2

    # Save final data to JSON file
    with open(os.path.join(output_folder, 'final_data_table.json'), 'w') as fp:
        json.dump(final_data, fp, indent=4)

    return final_data



image_path = "/content/FT_XX_10-221_-_F.jpg"
final_output = main_table(image_path)




from transformers import AutoProcessor, AutoModelForVisionAndLanguage
from PIL import Image
import torch
from io import BytesIO
import json

model_name = "smolvlm/smol-vlm-7b"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVisionAndLanguage.from_pretrained(model_name)

def extract_information(image_path, prompt):
    image = Image.open(image_path)
    
    base_width = 2640
    w_percent = base_width / float(image.size[0])
    h_size = int(float(image.size[1]) * float(w_percent))
    image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
        
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
        padding=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.5
        )
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    try:
        response_json = json.loads(response)
        return response_json
    except json.JSONDecodeError:
        return response

image_path = "/content/FT_XX_10-221_-_F.jpg"

prompt1 = "Extract the Bauteil, Länge Gesamt, height, Höhe Gesamt, Ausklinkung links, Länge, Höhe, Ausklinkung rechts, Länge and Höhe of Vorderansicht 2D diagram"
result1 = extract_information(image_path, prompt1)
print("Vorderansicht Information:")
print(result1)

prompt2 = "Extract the Bauteil and Breite of Seitenansicht 2D diagram"
result2 = extract_information(image_path, prompt2)
print("\nSeitenansicht Information:")
print(result2)

# Generate a final JSON output combining both results
final_json = {
    "Vorderansicht": result1,
    "Seitenansicht": result2
}
print("\nCombined JSON Output:")
json.dumps(final_json_2d_diagram, indent=4)








import json
from transformers import AutoProcessor, AutoModelForVisionAndLanguage
import torch

# Initialize the processor and model
processor = AutoProcessor.from_pretrained("smolvlm/smol-vlm-7b")
model = AutoModelForVisionAndLanguage.from_pretrained("smolvlm/smol-vlm-7b")

def generate_no_sql_json(json_data_1, json_data_2):
    
    combined_json = {**json_data_1, **json_data_2}

    
    json_as_text = json.dumps(combined_json)
    prompt = f"Convert this combined JSON data to a NoSQL object-oriented format: {json_as_text}"

    inputs = processor(text=prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Send input to the model and get output
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.5)
    
    # Decode the generated tokens to text
    nosql_json = processor.decode(outputs[0], skip_special_tokens=True)

    return nosql_json




# Generate the NoSQL JSON
final_no_sql_json = generate_no_sql_json(final_data_table, final_json_2d_diagram)
print("Final NoSQL Object-Oriented JSON:")
print(final_no_sql_json)
