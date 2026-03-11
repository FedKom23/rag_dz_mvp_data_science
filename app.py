import os
import requests
import streamlit as st
from engine import load_index, semantic_search

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"


@st.cache_resource
def load_resources():
    return load_index()


def call_ollama(prompt: str) -> str | None:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception:
        return None


def make_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    return (
        "Ты — точный ассистент по Национальной стратегии развития ИИ до 2030 года.\n"
        "Правила:\n"
        "1. Отвечай СТРОГО на заданный вопрос — первое предложение должно быть прямым ответом.\n"
        "2. Используй ТОЛЬКО информацию из контекста ниже. Не добавляй ничего от себя.\n"
        "3. Ответ — 1-2 предложения, без вводных слов и повторения вопроса.\n"
        "4. Если ответа в контексте нет — ответь: «В контексте нет информации по данному вопросу.»\n"
        "5. Отвечай на русском языке.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}\n\n"
        "Ответ:"
    )


st.set_page_config(
    page_title="Справочник по стратегии ИИ",
    page_icon="🤖",
    layout="centered",
)
st.title("🤖 Справочник по стратегии развития ИИ до 2030")
st.caption("Задайте вопрос — система найдёт ответ в документе Национальной стратегии.")

question = st.text_area(
    "Вопрос",
    placeholder="Например: Какова цель стратегии развития ИИ до 2030 года?",
    height=120,
)

if st.button("Получить ответ", use_container_width=True):
    question = question.strip()
    if not question:
        st.warning("Введите вопрос.")
        st.stop()

    with st.spinner("Ищем релевантные фрагменты..."):
        index, chunks, model = load_resources()
        results = semantic_search(question, index, chunks, model, top_k=5)

    context_chunks = [chunk for chunk, _ in results]

    with st.spinner("Генерируем ответ..."):
        prompt = make_prompt(question, context_chunks)
        answer = call_ollama(prompt)

    st.subheader("Ответ")
    if answer:
        st.markdown(
            f"""
            <div style="
                background-color: #1e1e2e;
                border: 1px solid #444;
                border-radius: 8px;
                padding: 16px 20px;
                min-height: 120px;
                font-size: 16px;
                font-weight: 700;
                color: #ffffff;
                white-space: pre-wrap;
                line-height: 1.6;
            ">{answer}</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("LLM недоступен. Показываем релевантные фрагменты из документа:")
        for i, (chunk, score) in enumerate(results, 1):
            with st.expander(f"Фрагмент {i} (релевантность: {score:.3f})"):
                st.write(chunk)
