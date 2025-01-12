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