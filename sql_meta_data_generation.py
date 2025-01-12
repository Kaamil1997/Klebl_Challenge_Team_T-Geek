import os
import requests 

response = requests.post(
    "https://llmfoundry.straive.com/gemini/v1beta/openai/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['LLMFOUNDRY_TOKEN']}:my-test-project"},
    json={
        "model": "gemini-2.0-flash-exp",
        "messages": [
            {"role": "system", "content": """You are a helpful assistant, who generates a no SQL data object oriented data model output json."""},
            {"role": "user", "content": f"Here is the data, generate a data model out of this all data: {json_output},{table_output_1},{table_output_2}"}
        ]
    }
)
print(response.json()["choices"][0]["message"]["content"])