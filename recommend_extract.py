from dotenv import load_dotenv
from google import genai
import json

load_dotenv()

client = genai.Client()

myfile = client.files.upload(
    file="path_to_the_file"
)

prompt = """
You are an AI assistant who will recommend entity and table names from the document that was uploaded

Example:
"Hey, from the document you uploaded here are some entities that can be extarcted:
- InvoiceNumber
-CustomerName
-InvoiceDate

If you want to extract any entities in particular please let me know."

Please follow the above example.
"""

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[prompt, myfile],
)

print(response.text)
