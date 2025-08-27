import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class TranslationRequest(BaseModel):
    text: str

app = FastAPI()

origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model repo names
repo_bmm_mg = "Amboara001/bmm-to-mg-t5-base-v3"
repo_mg_bmm = "Amboara001/malagasy-to-betsim-t5-base-translator"
#hf_token = "nlpmodel"  # Replace with your actual Hugging Face token if needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load both models and tokenizers
try:
    model_bmm_mg = T5ForConditionalGeneration.from_pretrained(repo_bmm_mg).to(device)
    tokenizer_bmm_mg = T5Tokenizer.from_pretrained(repo_bmm_mg)
    model_mg_bmm = T5ForConditionalGeneration.from_pretrained(repo_mg_bmm).to(device)
    tokenizer_mg_bmm = T5Tokenizer.from_pretrained(repo_mg_bmm)
    print("Both models and tokenizers loaded successfully from Hugging Face Hub.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Could not load the translation models.")

TASK_PREFIX_BMM_MG = "translate Betsimisaraka to official Malagasy: "
TASK_PREFIX_MG_BMM = "translate official Malagasy to Betsimisaraka: "

@torch.inference_mode()
def translate_sentence(text: str, model, tokenizer, task_prefix, max_new_tokens=128, num_beams=5):
    prompt = task_prefix + text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def translate_paragraph(paragraph: str, model, tokenizer, task_prefix):
    parts = re.split(r'(\n|[.!?;])', paragraph)
    translated = []
    buffer = ""
    for part in parts:
        if part in ['\n', '.', '!', '?', ';']:
            if buffer.strip():
                translated.append(translate_sentence(buffer.strip(), model, tokenizer, task_prefix))
                buffer = ""
            translated.append(part)
        else:
            buffer += part
    if buffer.strip():
        translated.append(translate_sentence(buffer.strip(), model, tokenizer, task_prefix))
    result = ""
    for t in translated:
        if t in ['\n', '.', '!', '?', ';']:
            result = result.rstrip() + t
        else:
            result += " " + t
    return result.strip()

@app.post("/translate-bmm-to-mg")
async def translate_bmm_to_mg(request: TranslationRequest):
    text_to_translate = request.text
    if not text_to_translate:
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        translated_text = translate_paragraph(
            text_to_translate, model_bmm_mg, tokenizer_bmm_mg, TASK_PREFIX_BMM_MG
        )
        return {"translated_text": translated_text}
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.post("/translate-mg-to-bmm")
async def translate_mg_to_bmm(request: TranslationRequest):
    text_to_translate = request.text
    if not text_to_translate:
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        translated_text = translate_paragraph(
            text_to_translate, model_mg_bmm, tokenizer_mg_bmm, TASK_PREFIX_MG_BMM
        )
        return {"translated_text": translated_text}
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# To run: uvicorn main:app --reload