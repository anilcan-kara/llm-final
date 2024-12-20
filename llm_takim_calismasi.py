# !pip install transformers evaluate faiss-cpu PyPDF2 beautifulsoup4 requests torch huggingface_hub

import os


def get_cleaned_texts_from_folder(folder_path):
    cleaned_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()
                cleaned_texts.append(text)
    return cleaned_texts


folder_path = "./veriseti/"
cleaned_texts = get_cleaned_texts_from_folder(folder_path)

print("Toplam metin sayısı:", len(cleaned_texts))

file_path = "./sorucevap.txt"


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

hf_token = "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"
from huggingface_hub import login

login(token=hf_token)

import transformers
import torch

model_id = "ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

print(pipeline.model.config)

messages = [
    {
        "role": "system",
        "content": "Sen öğrenci işleri departmanı yerine soruları cevaplayan bir yapay zeka asistanısın. Cevapların kısa, doğru ve profesyonel olmalıdır.",
    },
    {
        "role": "system",
        "content": "Soruları kısa ve net bir şekilde, yalnızca istenen bilgiyi içerecek şekilde cevapla. Gereksiz detay verme.",
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

print("Template message sayisi:", len(messages))

generated_questions = []

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

for idx, text in enumerate(cleaned_texts):
    print(f"Soru uretiliyor {idx + 1} / {len(cleaned_texts)}")
    question = f"Bu metin hakkında öğrenciler tarafından öğrenci işleri departmanına yöneltilebilecek bir adet soru sor: ```{text[:1000]}```"
    messages_copy = messages.copy()
    messages_copy.append({"role": "user", "content": question})

    outputs = pipeline(
        messages_copy,
        max_new_tokens=16,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )

    generated_questions.append(outputs[-1]["generated_text"][1]["content"])

print("Üretilen soru sayisi:", len(generated_questions))

with open("./generated_questions.txt", "w", encoding="utf-8") as f:
    for question in generated_questions:
        f.write(question + "\n")

generated_answers = []

for idx, generated_question in enumerate(generated_questions):
    print(f"Cevap uretiliyor {idx + 1} / {len(generated_questions)}")
    messages_copy = messages.copy()
    messages_copy.append({"role": "user", "content": generated_question})

    outputs = pipeline(
        messages_copy,
        max_new_tokens=16,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )

    generated_answers.append(outputs[-1]["generated_text"][1]["content"])

print("Üretilen cevap sayisi:", len(generated_answers))

from evaluate import load
import string
import re

metric_em = load("exact_match")
metric_f1 = load("f1")


def normalize_answer(s):
    return s


def compute_metrics_single(prediction, reference):
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)

    em = int(pred_norm == ref_norm)

    pred_tokens = set(pred_norm.split())
    ref_tokens = set(ref_norm.split())
    common = pred_tokens & ref_tokens

    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    return em, f1


def evaluate_model(questions, real_answers):
    em_scores = []
    f1_scores = []
    i = 1

    for question, real_answer in zip(questions, real_answers):
        print("Metrik hesaplanıyor:", i, "/", len(questions))
        i += 1
        print("Soru:", question)
        print("Gerçek Cevap:", real_answer)

        messages_copy = messages.copy()

        for idx, generated_question in enumerate(generated_questions):
            generated_answer = generated_answers[idx]
            messages_copy.append({"role": "user", "content": generated_question})
            messages_copy.append({"role": "assistant", "content": generated_answer})

        messages_copy.append({"role": "user", "content": question})

        model_output = pipeline(
            messages_copy,
            max_new_tokens=16,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )

        model_answer = model_output[0]["generated_text"][1]["content"]

        em, f1 = compute_metrics_single(model_answer, real_answer)
        em_scores.append(em)
        f1_scores.append(f1)

    return {
        "exact_match": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
    }


metrics = evaluate_model(questions[:2], answers[:2])

print("Exact Match (EM):", metrics["exact_match"])
print("F1 Score:", metrics["f1"])
