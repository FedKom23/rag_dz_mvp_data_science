import os
import pickle
import re
import numpy as np
import faiss
import pypdf
from razdel import sentenize
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

PDF_PATH = "data/data_text.pdf"
MAX_CHUNK_CHARS = 600   
OVERLAP_CHARS = 120     


def read_pdf(path: str) -> str:
    reader = pypdf.PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def _split_paragraph(para: str, max_chars: int, overlap: int) -> list[str]:
    """Разбивает длинный абзац сначала по предложениям (razdel),
    затем при необходимости — посимвольно с перекрытием."""
    sents = [s.text.strip() for s in sentenize(para) if s.text.strip()]
    if not sents:
        return []

    chunks = []
    current = ""
    for sent in sents:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            # если одно предложение длиннее лимита — бьём посимвольно
            if len(sent) > max_chars:
                start = 0
                while start < len(sent):
                    chunks.append(sent[start : start + max_chars])
                    start += max_chars - overlap
            else:
                current = sent

    if current:
        chunks.append(current)
    return chunks


def split_into_chunks(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> list[str]:
    """
    Гибридное разбиение:
    1. Делим по пунктам документа (строки вида «12. Текст» или «а) текст»).
    2. Длинные пункты дробим по предложениям через razdel.
    3. При необходимости — посимвольно с перекрытием.
    """
    # Разбиваем по пронумерованным пунктам («12. », «а) », «1) »)
    raw_parts = re.split(r'\n{2,}|(?=\b\d{1,3}\.\s)', text)

    chunks = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            chunks.append(part)
        else:
            chunks.extend(_split_paragraph(part, max_chars, overlap))

    # Убираем дубликаты и пустые
    seen = set()
    result = []
    for c in chunks:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            result.append(c)
    return result


def build_index(pdf_path: str = PDF_PATH):
    print(f"Чтение PDF: {pdf_path}")
    text = read_pdf(pdf_path)
    print(f"Извлечено символов: {len(text)}")

    chunks = split_into_chunks(text)
    print(f"Чанков: {len(chunks)}")

    print(f"Загрузка модели {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE_DIR)

    passages = ["passage: " + c for c in chunks]
    print("Кодирование чанков...")
    embeddings = model.encode(passages, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"FAISS индекс построен: {index.ntotal} векторов, размерность {dimension}")

    faiss.write_index(index, "data/faiss_index.bin")
    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("\nСохранено:")
    print("  data/faiss_index.bin — FAISS индекс")
    print("  data/chunks.pkl      — текстовые чанки")

    return index, chunks, model


def load_index():
    index = faiss.read_index("data/faiss_index.bin")
    with open("data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE_DIR)
    return index, chunks, model


def semantic_search(query: str, index, chunks: list[str], model, top_k: int = 5) -> list[tuple[str, float]]:
    query_emb = model.encode(["query: " + query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(query_emb, top_k)
    return [(chunks[i], float(scores[0][j])) for j, i in enumerate(ids[0])]


if __name__ == "__main__":
    index, chunks, model = build_index()

