from google import genai

# Is gemini-3-pro-preview a valid model in standard genai client?
try:
    print(genai.models.list_models())
except Exception as e:
    print(e)
