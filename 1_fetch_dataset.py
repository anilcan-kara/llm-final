# !pip install transformers evaluate faiss-cpu PyPDF2 beautifulsoup4 requests torch huggingface_hub

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
pdf_list = os.listdir(pdf_dir)

print("PDF dosya sayisi:", len(pdf_list))

for filename in pdf_list:
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        text = extract_text_from_pdf(pdf_path)
        pdf_texts.append(text)

print("PDF text sayisi:", len(pdf_texts))

import requests
from bs4 import BeautifulSoup


def extract_text_from_url(url):
    response = requests.get(url, timeout=5)
    if response.status_code != 200:
        print(f"Failed to retrieve {url}")
        return ""
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    spans = soup.find_all("span")
    titles = soup.find_all("title")
    h1s = soup.find_all("h1")
    h2s = soup.find_all("h2")
    h3s = soup.find_all("h3")
    h4s = soup.find_all("h4")
    h5s = soup.find_all("h5")
    h6s = soup.find_all("h6")
    bs = soup.find_all("b")
    is_ = soup.find_all("i")
    us = soup.find_all("u")
    text = "\n".join(
        [
            item.get_text()
            for item in paragraphs
            + spans
            + titles
            + h1s
            + h2s
            + h3s
            + h4s
            + h5s
            + h6s
            + bs
            + is_
            + us
        ]
    )
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

print("Web sayfa sayisi:", len(urls))

for url in urls:
    text = extract_text_from_url(url)
    web_texts.append(text)


print("Web text sayisi:", len(web_texts))

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

print("Toplam text sayisi:", len(cleaned_texts))

for text in cleaned_texts:
    chunks.extend(split_into_chunks(text))


for i, text in enumerate(cleaned_texts):
    with open(f"./veriseti/{i}.txt", "w", encoding="utf-8") as f:
        f.write(text)


with open("./dataset.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n")
