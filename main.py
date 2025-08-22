import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Define the data model for the request body using Pydantic
class TranslationRequest(BaseModel):
    text: str

# Initialize the FastAPI app
app = FastAPI()

origins = [
    "*", # Allow all origins
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repo_name = "Amboara001/t5-betsim-mlg-translator"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    model = T5ForConditionalGeneration.from_pretrained(repo_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(repo_name)
    print("Model and tokenizer loaded successfully from Hugging Face Hub.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Could not load the translation model.")

TASK_PREFIX = "translate Betsimisaraka to official Malagasy: "

@torch.inference_mode()
def translate_sentence(text: str, max_new_tokens=128, num_beams=5):
    prompt = TASK_PREFIX + text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def translate_paragraph(paragraph: str):
    # Split on punctuation and line breaks, keeping delimiters
    # This regex splits on punctuation or line breaks, keeping them as separate tokens
    parts = re.split(r'(\n|[.!?;])', paragraph)
    translated = []
    buffer = ""
    for part in parts:
        if part in ['\n', '.', '!', '?', ';']:
            if buffer.strip():
                translated.append(translate_sentence(buffer.strip()))
                buffer = ""
            translated.append(part)
        else:
            buffer += part
    if buffer.strip():
        translated.append(translate_sentence(buffer.strip()))
    # Recombine, removing unnecessary spaces before punctuation/line breaks
    result = ""
    for t in translated:
        if t in ['\n', '.', '!', '?', ';']:
            result = result.rstrip() + t
        else:
            result += " " + t
    return result.strip()

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    text_to_translate = request.text
    if not text_to_translate:
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        translated_text = translate_paragraph(text_to_translate)
        return {"translated_text": translated_text}
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# To run: uvicorn main:app --reload