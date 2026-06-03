import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class ProcessRewardModel:

    def __init__(
        self,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    ):

        self.embedder = SentenceTransformer(
            embedding_model
        )

        self.alpha = 0.7
        self.beta = 0.3

    def encode(
        self,
        text
    ):

        return self.embedder.encode(
            [text],
            convert_to_numpy=True
        )

    def similarity(
        self,
        text_a,
        text_b
    ):

        emb_a = self.encode(
            text_a
        )

        emb_b = self.encode(
            text_b
        )

        score = cosine_similarity(
            emb_a,
            emb_b
        )[0][0]

        return float(score)

    def relevance_score(
        self,
        question,
        paragraph
    ):

        return self.similarity(
            question,
            paragraph
        )

    def reasoning_score(
        self,
        reasoning_state,
        paragraph
    ):

        if reasoning_state is None:

            return 0.5

        if len(reasoning_state.strip()) == 0:

            return 0.5

        return self.similarity(
            reasoning_state,
            paragraph
        )

    def score_step(
        self,
        question,
        paragraph,
        reasoning_state=""
    ):

        relevance = self.relevance_score(
            question,
            paragraph
        )

        reasoning = self.reasoning_score(
            reasoning_state,
            paragraph
        )

        reward = (
            self.alpha * relevance
            +
            self.beta * reasoning
        )

        return {
            "reward":
                float(reward),

            "relevance":
                float(relevance),

            "reasoning":
                float(reasoning)
        }
    
    def train(
        self,
        gold_passages,
        distractor_passages
    ):

        gold_scores = []

        distractor_scores = []

        for g in gold_passages:

            gold_scores.append(
                self.relevance_score(
                    g,
                    g
                )
            )

        for d in distractor_passages:

            distractor_scores.append(
                self.relevance_score(
                    gold_passages[0],
                    d
                )
            )

        return {
            "gold_mean":
                float(
                    np.mean(
                        gold_scores
                    )
                ),

            "distractor_mean":
                float(
                    np.mean(
                        distractor_scores
                    )
                )
        }

    def score_retrievals(
        self,
        question,
        retrieved_passages,
        reasoning_state=""
    ):

        scored = []

        for item in retrieved_passages:

            paragraph = item[
                "paragraph"
            ]

            scores = self.score_step(
                question,
                paragraph,
                reasoning_state
            )

            scored.append(
                {
                    "paragraph":
                        paragraph,

                    "reward":
                        scores[
                            "reward"
                        ],

                    "relevance":
                        scores[
                            "relevance"
                        ],

                    "reasoning":
                        scores[
                            "reasoning"
                        ]
                }
            )

        scored = sorted(
            scored,
            key=lambda x:
            x["reward"],
            reverse=True
        )

        return scored

    def prune(
        self,
        question,
        retrieved_passages,
        threshold=0.4,
        reasoning_state=""
    ):

        scored = self.score_retrievals(
            question,
            retrieved_passages,
            reasoning_state
        )

        kept = []

        for item in scored:

            if item["reward"] >= threshold:

                kept.append(
                    item
                )

        return kept

    def update_reasoning_state(
        self,
        current_state,
        selected_passages
    ):

        if len(selected_passages) == 0:

            return current_state

        text = []

        if current_state:

            text.append(
                current_state
            )

        for p in selected_passages:

            text.append(
                p["paragraph"][:300]
            )

        return "\n".join(
            text
        )

    def evaluate_gold_support(
        self,
        supporting_titles,
        retrieved_passages
    ):

        hits = 0

        for passage in retrieved_passages:

            text = passage[
                "paragraph"
            ].lower()

            for title in supporting_titles:

                if (
                    title.lower()
                    in text
                ):

                    hits += 1

                    break

        return hits


if __name__ == "__main__":

    prm = ProcessRewardModel()

    question = (
        "What is the capital of France?"
    )

    paragraph = (
        "Paris is the capital city of France."
    )

    result = prm.score_step(
        question,
        paragraph
    )

    print(
        "\nReward:",
        result["reward"]
    )

    print(
        "Relevance:",
        result["relevance"]
    )

    print(
        "Reasoning:",
        result["reasoning"]
    )