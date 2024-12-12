# pip install transformers datasets faiss-cpu PyPDF2 beautifulsoup4 requests torch huggingface_hub google

# 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

import os

hf_token = "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"

from huggingface_hub import login

login(token=hf_token)

# echo $hf_token | huggingface-cli login

# huggingface-cli whoami

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)


drive.mount("/content/drive")

import os
import PyPDF2


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text


pdf_dir = "/content/drive/MyDrive/LLM_Projesi/"
pdf_texts = []

for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        text = extract_text_from_pdf(pdf_path)
        pdf_texts.append(text)

        import requests
from bs4 import BeautifulSoup


def extract_text_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve {url}")
        return ""
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join([para.get_text() for para in paragraphs])
    return text


urls = [
    "https://ogrenciisleri.duzce.edu.tr/sayfa/b149/mevzuat",
    "https://www.bozok.edu.tr/mevzuat",
]

web_texts = []

for url in urls:
    text = extract_text_from_url(url)
    web_texts.append(text)

    import re


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9.,;:!?İĞÜŞÇÖığüşçö ]+", "", text)
    return text.strip()


def split_into_chunks(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    return chunks


cleaned_texts = [clean_text(text) for text in pdf_texts + web_texts]
chunks = []

for text in cleaned_texts:
    chunks.extend(split_into_chunks(text))

with open("dataset.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n")

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model.resize_token_embeddings(len(tokenizer))

from datasets import load_dataset

dataset = load_dataset("text", data_files={"train": "dataset.txt"}, split="train")


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

trainer.train()

import torch


def generate_response(question, max_length=150):
    input_ids = tokenizer.encode(question, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response


question = "Yaz okulu nasıl işliyor?"
response = generate_response(question)
print(response)

model.save_pretrained("/content/drive/MyDrive/LLM_Projesi_LLaMA_Model/")
tokenizer.save_pretrained("/content/drive/MyDrive/LLM_Projesi_LLaMA_Model/")

from transformers import LlamaTokenizer, LlamaForCausalLM

model_path = "/content/drive/MyDrive/LLM_Projesi_LLaMA_Model/"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype="float16", load_in_8bit=True
)


def generate_response(question, max_length=150):
    input_ids = tokenizer.encode(question, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response


question = "Dönem dondurma nasıl yapılır?"
response = generate_response(question)
print(response)
