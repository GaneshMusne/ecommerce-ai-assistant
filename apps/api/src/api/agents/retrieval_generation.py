import os
from functools import lru_cache
from threading import Lock
import instructor
from langsmith import traceable, get_current_run_tree

from google import genai
from google.genai import types
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from api.agents.models import RAGGenerationResponse

COLLECTION_NAME = "Amazon-items-minilm-l6-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDING_LOCK = Lock()

@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={
        "ls_provider": "huggingface",
        "ls_model_name": EMBEDDING_MODEL_NAME,
    },
)
def get_embedding(text: str) -> list[float]:
    # lru_cache can initialize twice on concurrent cache misses. Serialize
    # loading and inference through the shared local embedding model.
    with _EMBEDDING_LOCK:
        return (
            get_embedding_model()
            .encode(text, normalize_embeddings=True)
            .tolist()
        )


@traceable(
    name="retrieve_data",
    run_type="retriever"
)
def retrieve_data(query, qdrant_client, k=5):

    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        similarity_scores.append(result.score)
        retrieved_context_ratings.append(result.payload["average_rating"])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings
    }


@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context):

    formatted_context = ""

    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context


@traceable(
    name="build_prompt",
    run_type="prompt"
)
def build_prompt(preprocessed_context, question):

    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructions:
- Answer the question based on the provided context only.
- Never use word context and refer to it as the available products.

Context:
{preprocessed_context}

Question:
{question}
"""

    return prompt


@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_provider": "google", "ls_model_name": GEMINI_MODEL}
)
def generate_answer(prompt):
    # with genai.Client(
    #     api_key=os.getenv("GOOGLE_API_KEY"),
    #     http_options=types.HttpOptions(
    #         retry_options=types.HttpRetryOptions(
    #             attempts=6,
    #             initial_delay=2.0,
    #             max_delay=30.0,
    #             exp_base=2.0,
    #             jitter=1.0,
    #         )
    #     ),
    # ) as client:
    #     response = client.models.generate_content(
    #         model=GEMINI_MODEL,
    #         contents=prompt,
    #     )

    client = instructor.from_provider(
        f"google/{GEMINI_MODEL}",
        mode=instructor.Mode.TOOLS,
    )

    response, raw_response = client.create_with_completion(
        messages=[{"role": "user", "content": prompt}],
        response_model=RAGGenerationResponse,
    )

    if not response:
        raise RuntimeError("Gemini returned no text. Check the response for blocked content.")
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metdata"] = {
            "input_tokens": raw_response.usage_metadata.prompt_token_count,
            "total_tokens": raw_response.usage_metadata.total_token_count
        }

    return response


@traceable(
    name="rag_pipeline"
)
def rag_pipeline(question, qdrant_client, top_k=5):
    retrieved_context = retrieve_data(question, qdrant_client, top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    res = generate_answer(prompt)
    final_result = {
        "answer": res.answer,
        "question": question,
        "references": res.references,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"],
        "similarity_scores": retrieved_context["similarity_scores"]
    }

    return final_result

def rag_pipeline_wrapper(question, top_k=5):

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    result = rag_pipeline(question, qdrant_client, top_k)

    used_context = []
    dummy_vector = np.zeros(384)

    for item in result.get("references", []):
        payload = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=dummy_vector,
            limit=1,
            with_payload=True,
            with_vectors=False,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.id)
                    )
                ]
            )
        ).points[0].payload
        image_url = payload.get("image")
        price = payload.get("price")
        if image_url:
            used_context.append(
                {
                    "image_url": image_url,
                    "price": price,
                    "description": item.description
                }
            )

    return {
        "answer": result["answer"],
        "used_context": used_context
    }
