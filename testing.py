"""
RAGAS-оценка RAG-системы (RAGAS 0.4.x + Ollama).

Метрики:
  - faithfulness        — ответ подкреплён контекстом (нет галлюцинаций)
  - answer_relevancy    — ответ релевантен вопросу
  - context_precision   — найденные чанки релевантны вопросу
  - context_recall      — найденные чанки покрывают нужную информацию

Улучшения:
  - Кастомные русскоязычные промпты для answer_relevancy и context_recall
  - Чанкинг через razdel (предложения) вместо символьного — лучшее покрытие текста.

Запуск:
    python3.10 testing.py

Требует:
  - Ollama с моделью llama3.1:8b (`ollama serve` в фоне)
  - Готового FAISS индекса (сначала `python3.10 engine.py`)
"""

import warnings
warnings.filterwarnings("ignore")

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.metrics._answer_relevance import (
    ResponseRelevancePrompt,
    ResponseRelevanceInput,
    ResponseRelevanceOutput,
)
from ragas.metrics._context_recall import (
    ContextRecallClassificationPrompt,
    QCA,
    ContextRecallClassifications,
    ContextRecallClassification,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings

from engine import load_index, semantic_search

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Кастомный промпт для Answer Relevancy с русскими примерами ────────────────
class RussianResponseRelevancePrompt(ResponseRelevancePrompt):
    """
    Добавляем русскоязычные примеры, чтобы модель не добавляла английских
    объяснений перед JSON и не ломала парсер.
    """
    examples = ResponseRelevancePrompt.examples + [
        (
            ResponseRelevanceInput(
                response="Стратегия развития искусственного интеллекта в России утверждена на период до 2030 года.",
            ),
            ResponseRelevanceOutput(
                question="На какой период утверждена стратегия развития ИИ в России?",
                noncommittal=0,
            ),
        ),
        (
            ResponseRelevanceInput(
                response="Правовую основу стратегии составляют Конституция Российской Федерации и федеральные законы.",
            ),
            ResponseRelevanceOutput(
                question="Какие документы составляют правовую основу стратегии?",
                noncommittal=0,
            ),
        ),
        (
            ResponseRelevanceInput(
                response="Я не знаю ответа на этот вопрос, так как информации в предоставленном контексте недостаточно.",
            ),
            ResponseRelevanceOutput(
                question="О чём спрашивал пользователь?",
                noncommittal=1,
            ),
        ),
    ]


# ── Кастомный промпт для Context Recall с русскими примерами ─────────────────
class RussianContextRecallPrompt(ContextRecallClassificationPrompt):
    """
    Добавляем русскоязычный пример, чтобы модель выдавала JSON
    без английских пояснений.
    """
    examples = ContextRecallClassificationPrompt.examples + [
        (
            QCA(
                question="Какова цель стратегии развития ИИ?",
                context=(
                    "Настоящая Стратегия определяет цели, основные задачи и меры "
                    "по развитию искусственного интеллекта в Российской Федерации "
                    "на период до 2030 года. Стратегия направлена на обеспечение "
                    "ускоренного развития искусственного интеллекта и достижение "
                    "технологической независимости."
                ),
                answer=(
                    "Цель стратегии — ускоренное развитие искусственного интеллекта "
                    "в Российской Федерации до 2030 года и достижение технологической "
                    "независимости страны в сфере ИИ."
                ),
            ),
            ContextRecallClassifications(
                classifications=[
                    ContextRecallClassification(
                        statement="Цель стратегии — ускоренное развитие искусственного интеллекта в Российской Федерации до 2030 года.",
                        reason="В контексте прямо сказано об ускоренном развитии ИИ на период до 2030 года.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement="Стратегия направлена на достижение технологической независимости страны в сфере ИИ.",
                        reason="В контексте явно упоминается технологическая независимость.",
                        attributed=1,
                    ),
                ]
            ),
        ),
    ]


# ── Тестовый датасет ──────────────────────────────────────────────────────────
TEST_DATASET = [
    {
        "question": "Какова главная цель Национальной стратегии развития ИИ до 2030 года?",
        "ground_truth": (
            "Главная цель стратегии — обеспечение ускоренного развития искусственного "
            "интеллекта в Российской Федерации, исследований и разработок в этой области, "
            "достижение технологической независимости страны в сфере ИИ."
        ),
    },
    {
        "question": "Какие законы составляют правовую основу стратегии?",
        "ground_truth": (
            "Правовую основу стратегии составляют Конституция Российской Федерации и "
            "федеральные законы, в частности Федеральный закон от 27 июля 2006 г. № 149-ФЗ "
            "«Об информации, информационных технологиях и о защите информации»."
        ),
    },
    {
        "question": "Что понимается под термином «искусственный интеллект» в стратегии?",
        "ground_truth": (
            "Под искусственным интеллектом понимается комплекс технологических решений, "
            "позволяющий имитировать когнитивные функции человека, в том числе самообучение "
            "и поиск решений без заранее заданного алгоритма."
        ),
    },
    {
        "question": "Какие задачи ставятся в области подготовки кадров для сферы ИИ?",
        "ground_truth": (
            "Стратегия предусматривает подготовку специалистов в области ИИ через систему "
            "образования, повышение квалификации работников, привлечение иностранных "
            "специалистов, формирование компетенций в области ИИ у широкой аудитории."
        ),
    },
    {
        "question": "Какие приоритетные отрасли для внедрения ИИ выделяет стратегия?",
        "ground_truth": (
            "Стратегия выделяет приоритетные направления: здравоохранение, транспорт, "
            "сельское хозяйство, государственное управление, финансовый сектор и "
            "промышленность."
        ),
    },
]


def make_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    return (
        "Ты помощник, отвечающий на вопросы по Национальной стратегии развития "
        "искусственного интеллекта на период до 2030 года.\n"
        "Используй только предоставленный контекст. Отвечай на русском языке. "
        "Если ответа в контексте нет — так и скажи.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}\n\n"
        "Ответ:"
    )


def run_rag_pipeline(question: str, index, chunks, embed_model, ollama_llm, top_k: int = 5):
    results = semantic_search(question, index, chunks, embed_model, top_k=top_k)
    context_chunks = [chunk for chunk, _ in results]
    response = ollama_llm.invoke(make_prompt(question, context_chunks))
    return response.content.strip(), context_chunks


def build_evaluation_dataset(index, chunks, embed_model, ollama_llm) -> EvaluationDataset:
    samples = []
    for i, item in enumerate(TEST_DATASET, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i}/{len(TEST_DATASET)}] {q[:65]}...")

        answer, ctx = run_rag_pipeline(q, index, chunks, embed_model, ollama_llm, top_k=5)

        samples.append(SingleTurnSample(
            user_input=q,
            response=answer,
            retrieved_contexts=ctx,
            reference=gt,
        ))

    return EvaluationDataset(samples=samples)


def setup_ragas(ollama_llm):
    """Настраивает RAGAS метрики с Ollama и русскоязычными промптами."""
    ragas_llm = LangchainLLMWrapper(ollama_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    )

    faithfulness.llm = ragas_llm

    # Кастомный промпт с русскими примерами
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings
    answer_relevancy.question_generation = RussianResponseRelevancePrompt()

    context_precision.llm = ragas_llm

    # Кастомный промпт с русским примером
    context_recall.llm = ragas_llm
    context_recall.context_recall_prompt = RussianContextRecallPrompt()

    return [faithfulness, answer_relevancy, context_precision, context_recall]


def main():
    print("=" * 60)
    print("RAGAS-оценка RAG-системы")
    print(f"LLM: {OLLAMA_MODEL} (Ollama)")
    print("Промпты: с русскоязычными примерами")
    print("Чанкинг: по предложениям (razdel)")
    print("=" * 60)

    print("\nЗагрузка FAISS индекса и модели...")
    index, chunks, embed_model = load_index()
    print(f"  Чанков в индексе: {len(chunks)}")

    ollama_llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    print("\nНастройка RAGAS метрик (Ollama + кастомные промпты)...")
    metrics = setup_ragas(ollama_llm)

    print("\nЗапуск RAG-пайплайна для тестового датасета...")
    dataset = build_evaluation_dataset(index, chunks, embed_model, ollama_llm)

    print("\nВычисление RAGAS метрик...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,
        show_progress=True,
        run_config=RunConfig(timeout=600, max_retries=3, max_wait=60),
    )

    scores = result.to_pandas()

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ RAGAS")
    print("=" * 60)
    metric_names = {
        "faithfulness": "Faithfulness       (достоверность)",
        "answer_relevancy": "Answer Relevancy   (релевантность)",
        "context_precision": "Context Precision  (точность)",
        "context_recall": "Context Recall     (полнота)",
    }
    for col, label in metric_names.items():
        if col in scores.columns:
            mean_val = scores[col].mean()
            print(f"  {label}: {mean_val:.4f}")

    print("=" * 60)

    scores.to_csv("data/ragas_results.csv", index=False)
    print("\nДетальные результаты: data/ragas_results.csv")


if __name__ == "__main__":
    main()
