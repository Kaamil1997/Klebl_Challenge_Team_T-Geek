import os
import cv2
import base64
import requests
import json  # Import JSON module to handle JSON operations
from google.colab import userdata
from ultralytics import YOLO
from langchain_openai import ChatOpenAI
from google import genai
from google.genai import types
import google.generativeai as genai
from PIL import Image
import io
import requests
from io import BytesIO
import json
from transformers import AutoProcessor, AutoModelForVisionAndLanguage
import torch


llm = ChatOpenAI(model="gpt-4o-mini")

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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_data_from_image(image_path, prompt):
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
    payload = {
        "model": "gpt- -mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 10000
    }
    response = requests.post("https://gramener.com/llmproxy/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def main_table(image_path):
    model = load_model("best_2.pt")
    process_image(image_path, model)

    results_dict = {}

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
    results_dict['JSON Output'] = json_output

    table_image_path_1 = "Output_data/liste_stahl.jpg"
    table_prompt_1 = """extract all these the values of the given image and print it in table format"""
    table_output_1 = extract_data_from_image(table_image_path_1, table_prompt_1)
    results_dict['Table Output 1'] = table_output_1

    table_image_path_2 = "Output_data/liste_einbauteile.jpg"
    table_prompt_2 = """extract all these the values of the given image and print it in table format"""
    table_output_2 = extract_data_from_image(table_image_path_2, table_prompt_2)
    results_dict['Table Output 2'] = table_output_2

    # Converting the results dictionary to a JSON string
    final_json_output = json.dumps(results_dict, indent=4)
    print("Final JSON Output:", final_json_output)
    return final_json_output


image_path = "FT_XX_10-221_-_F.jpg"
final_results = main_table(image_path)



client = genai.Client(api_key=GOOGLE_API_KEY)

model_name = "gemini-2.0-flash-exp"

bounding_box_system_instructions = """
    Return a JSON array in a key value pair. Never return masks or code fencing. Limit to 25 objects.
    """

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

image = "FT_XX_10-221_-_F.jpg" 

img = Image.open(BytesIO(open(image, "rb").read()))
im = Image.open(image).resize((2640, int(2640 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS)

# Initialize a dictionary to collect responses
all_responses = {}

# First prompt
prompt_1 = "Extract the Bauteil, Länge Gesamt, height, Höhe Gesamt, Ausklinkung links, Länge, Höhe, Ausklinkung rechts, Länge and Höhe of Vorderansicht 2D diagram"  
response_1 = client.models.generate_content(
    model=model_name,
    contents=[prompt_1, im],
    config=types.GenerateContentConfig(
        system_instruction=bounding_box_system_instructions,
        temperature=0.5,
        safety_settings=safety_settings,
    )
)
all_responses['Vorderansicht'] = json.loads(response_1.text)  # Assuming the API returns JSON string

# Second prompt
prompt_2 = "Extract the Bauteil and Breite of Seitenansicht 2D diagram"  
response_2 = client.models.generate_content(
    model=model_name,
    contents=[prompt_2, im],
    config=types.GenerateContentConfig(
        system_instruction=bounding_box_system_instructions,
        temperature=0.5,
        safety_settings=safety_settings,
    )
)
all_responses['Seitenansicht'] = json.loads(response_2.text) 


# Save the final merged JSON to a file
output_file_path = 'merged_output_table.json'
with open(output_file_path, 'w') as file:
    json.dump(all_responses, file, indent=4)








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
final_no_sql_json = generate_no_sql_json(results_dict, all_responses)
print("Final NoSQL Object-Oriented JSON:")
print(final_no_sql_json)