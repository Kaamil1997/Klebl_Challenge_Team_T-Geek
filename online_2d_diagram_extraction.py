from google import genai
from google.genai import types
import google.generativeai as genai
from PIL import Image
import io
import os
import requests
from io import BytesIO



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

image = "/content/FT_XX_10-221_-_F.jpg" 

im = Image.open(image)


prompt = "Extract the Bauteil, Länge Gesamt, height, Höhe Gesamt, Ausklinkung links, Länge, Höhe, Ausklinkung rechts, Länge and Höhe of Vorderansicht 2D diagram"  


img = Image.open(BytesIO(open(image, "rb").read()))
im = Image.open(image).resize((2640, int(2640 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS)


response = client.models.generate_content(
    model=model_name,
    contents=[prompt, im],
    config = types.GenerateContentConfig(
        system_instruction=bounding_box_system_instructions,
        temperature=0.5,
        safety_settings=safety_settings,
    )
)

print(response.text)


prompt = "Extract the Bauteil and Breite of Seitenansicht 2D diagram"  


response = client.models.generate_content(
    model=model_name,
    contents=[prompt, im],
    config = types.GenerateContentConfig(
        system_instruction=bounding_box_system_instructions,
        temperature=0.5,
        safety_settings=safety_settings,
    )
)


print(response.text)