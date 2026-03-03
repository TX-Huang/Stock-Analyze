import os
from google import genai
import sys

key = os.environ.get("GEMINI_API_KEY", "")
if not key:
    print("No key")
    sys.exit()

client = genai.Client(api_key=key)
try:
    res = client.models.generate_content(model="gemini-2.0-flash", contents="What is the ticker for 台積電? return just the ticker")
    print("2.0-flash:", res.text)
except Exception as e:
    print("2.0-flash error:", e)
