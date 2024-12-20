# !pip install ragatouille transformers datasets faiss-cpu PyPDF2 beautifulsoup4 requests torch huggingface_hub

# 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

import os

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


pdf_dir = "./pdf/"
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
    "https://haber.duzce.edu.tr/duyurular",
    "https://www.duzce.edu.tr/sayfa/2bd9/akademik-takvim",
    "https://ebs.duzce.edu.tr/tr-TR/GenelBilgi/Index/12",
    "https://ebs.duzce.edu.tr/tr-TR/GenelBilgi/Index/27",
    "https://ebs.duzce.edu.tr/tr-TR/GenelBilgi/Index/36",
    "https://ebs.duzce.edu.tr/tr-TR/Bolum/Index/14?bot=14",
    "https://bm.mf.duzce.edu.tr/personel/akademik",
    "https://www.bozok.edu.tr/mevzuat",
    "https://bozok.edu.tr/",
    "https://bozok.edu.tr/ogrenci",
    "https://bozok.edu.tr/sayfa/genel-kayit-ve-kabul-kosullari/9253",
    "https://bozok.edu.tr/sayfa/kredi-hareketliligi-ve-onceki-ogrenmenin-taninmasi/9254",
    "https://bozok.edu.tr/sayfa/yurtdisi-ogreniminin-taninmasi/9255",
    "https://bozok.edu.tr/sayfa/akts-kredilerinin-belirlenmesi/9256",
    "https://bozok.edu.tr/sayfa/akademik-danismanlik/9257",
    "https://bozok.edu.tr/diplomaeki",
    "https://bozok.edu.tr/duyurular/ogrenci",
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

file_path = "/content/drive/MyDrive/sorucevap.txt"


def parse_questions_and_answers(file_path):
    questions = []
    answers = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("S:"):
                questions.append(line[3:].strip())
            elif line.startswith("C:"):
                answers.append(line[3:].strip())

    return questions, answers


questions, answers = parse_questions_and_answers(file_path)

print("Sorular:", questions[:3])
print("Cevaplar:", answers[:3])

import transformers
import torch

model_id = "ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {
        "role": "system",
        "content": "Sen öğrenci işleri departmanı yerine soruları cevaplayan bir yapay zeka asistanısın. Cevapların kısa, doğru ve profesyonel olmalıdır.",
    },
    {"role": "user", "content": "Soru: Mezuniyet koşulları nelerdir?"},
    {
        "role": "assistant",
        "content": "Cevap: Mezuniyet için 240 AKTS tamamlamanız ve zorunlu derslerden başarılı olmanız gerekir.",
    },
    {
        "role": "system",
        "content": "Soruları yanıtlarken adım adım düşün ve cevabı net bir şekilde ver.",
    },
    {"role": "user", "content": "Soru: Kayıt yenileme işlemi nasıl yapılır?"},
    {
        "role": "assistant",
        "content": "Cevap: OBS sistemine giriş yaparak ders kayıt ekranından gerekli işlemleri yapabilirsiniz.",
    },
    {
        "role": "system",
        "content": "Eğer kesin bir bilgiye sahip değilsen, 'Bu konuda tam bilgiye sahip değilim ancak tahminim...' diyerek kullanıcıyı yönlendir.",
    },
    {"role": "user", "content": "Soru: Sınav tarihleri ne zaman?"},
    {
        "role": "assistant",
        "content": "Cevap: Bu konuda tam bilgiye sahip değilim, ancak sınav takvimine OBS sisteminden erişebilirsiniz.",
    },
    {
        "role": "system",
        "content": "Kullanıcıların anlaması için gerekirse cevaplarına kısa açıklamalar ekle.",
    },
    {"role": "user", "content": "Soru: Öğrenci belgesi nasıl alınır?"},
    {
        "role": "assistant",
        "content": "Cevap: Öğrenci belgenizi e-Devlet veya OBS sistemi üzerinden alabilirsiniz. Eğer fiziksel bir belge isterseniz öğrenci işleri ofisine başvurabilirsiniz.",
    },
]

for idx, text in enumerate(cleaned_texts[:10]):
    question = f"Soru {idx+1}: Bu metindeki önemli bilgi nedir?"
    answer = f"Cevap {idx+1}: {text[:200]}"
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

for output in outputs:
    print(output["generated_text"])

from datasets import load_metric
import string
import re

metric_em = load_metric("exact_match")
metric_f1 = load_metric("f1")


def normalize_answer(s):
    """Normalize metinleri karşılaştırma için hazırlar."""

    def remove_punctuation(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    return remove_punctuation(lower(remove_articles(s))).strip()


def compute_metrics(predictions, references):
    em_scores = []
    f1_scores = []

    for pred, ref in zip(predictions, references):
        pred_norm = normalize_answer(pred)
        ref_norm = normalize_answer(ref)

        em_scores.append(int(pred_norm == ref_norm))

        pred_tokens = set(pred_norm.split())
        ref_tokens = set(ref_norm.split())
        common = pred_tokens & ref_tokens

        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall)
            else 0
        )
        f1_scores.append(f1)

    return {
        "exact_match": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
    }


real_answers = answers
model_predictions = [
    "Öğrenci kimlik kartı gereklidir.",
    "Hayır, cep telefonu hesap makinesi olarak kullanılamaz.",
    "Kopya teşebbüsü nedeniyle disiplin işlemi uygulanır.",
]

metrics = compute_metrics(model_predictions, real_answers)
print("Exact Match (EM):", metrics["exact_match"])
print("F1 Score:", metrics["f1"])
