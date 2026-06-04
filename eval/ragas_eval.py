from langchain_groq import ChatGroq

##from langchain_google_genai import (
##   ChatGoogleGenerativeAI,
##   GoogleGenerativeAIEmbeddings
##}

from langchain_huggingface import HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(
    PROJECT_ROOT
)

import json
import pandas as pd

from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)

from src.retriever import MultiHopRetriever
from src.pipeline import PRMPipeline

import os

def build_groq_llm():

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    return LangchainLLMWrapper(llm)

def build_embeddings():

    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return LangchainEmbeddingsWrapper(emb)

# def build_gemini_llm():

#     print("BUILDING GEMINI")

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         google_api_key=os.getenv("GOOGLE_API_KEY")
#     )

#     return LangchainLLMWrapper(llm)


# def build_gemini_embeddings():

#     emb = GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001"
#     )

#     return LangchainEmbeddingsWrapper(
#         emb
#     )

# print("USING GEMINI LLM CONFIG")

def run_experiment(
    threshold,
    num_questions=3
):

    retriever = MultiHopRetriever()

    pipeline = PRMPipeline()

    dataset = retriever.load_hotpot()

    validation = dataset["validation"]

    rows = []

    total = min(
        num_questions,
        len(validation)
    )

    for i in range(total):

        sample = validation[i]

        question = sample["question"]

        ground_truth = sample["answer"]

        paragraphs, titles = (
            retriever.extract_paragraphs(
                sample
            )
        )

        try:

            result = pipeline.run(
                question=question,
                paragraphs=paragraphs,
                titles=titles,
                threshold=threshold
            )

            rows.append(
                        {
                            "question":
                                question,

                            "answer":
                                result["answer"],

                            "contexts":
                                result["contexts"],

                            "ground_truth":
                                ground_truth,

                            "hop1_kept":
                                len(
                                    result["hop1"]
                                ),

                            "hop2_kept":
                                len(
                                    result["hop2"]
                                )
                        }
                    )

            print(
                f"[{i+1}/{total}] done"
            )

        except Exception as e:

            print(
                f"Failed example {i}: {e}"
            )

    return pd.DataFrame(
        rows
    )


def compute_ragas(df):

    ragas_dataset = Dataset.from_dict(
        {
            "question": df["question"].tolist(),
            "answer": df["answer"].tolist(),
            "contexts": df["contexts"].tolist(),
            "ground_truth": df["ground_truth"].tolist()
        }
    )

    local_llm = build_groq_llm()
    local_embeddings = build_embeddings()

    result = evaluate(
        ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness
        ],
        llm=local_llm,
        embeddings=local_embeddings
    )

    return result.to_pandas().mean(numeric_only=True).to_dict()


if __name__ == "__main__":

    os.makedirs(
        "results",
        exist_ok=True
    )

    print(
        "\nRunning t=0.4"
    )

    df04 = run_experiment(
        threshold=0.4,
        num_questions=500
    )

    df04.to_csv(
        "results/results_04.csv",
        index=False
    )

    score04 = compute_ragas(
        df04
    )

    print(
        "\nRunning t=0.6"
    )

    df06 = run_experiment(
        threshold=0.6,
        num_questions=500
    )

    df06.to_csv(
        "results/results_06.csv",
        index=False
    )

    score06 = compute_ragas(
        df06
    )

    summary = pd.DataFrame(
            [
                {
                    "threshold":0.4,
                    **score04
                },
                {
                    "threshold":0.6,
                    **score06
                }
            ]
        )

    summary.to_csv(
        "results/summary.csv",
        index=False
    )

    with open(
        "results/results.json",
        "w"
    ) as f:

        json.dump(
            summary.to_dict(
                orient="records"
            ),
            f,
            indent=4
        )

    print(
        "\nEvaluation Complete"
    )
