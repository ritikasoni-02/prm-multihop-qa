from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re


class MultiHopRetriever:

    def __init__(
        self,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    ):

        self.embedder = SentenceTransformer(
            embedding_model
        )

    def load_hotpot(self):

        return load_dataset(
            "hotpot_qa",
            "distractor"
        )

    def extract_paragraphs(
        self,
        sample
    ):

        paragraphs = []
        titles = []

        for title, sentences in zip(
            sample["context"]["title"],
            sample["context"]["sentences"]
        ):

            paragraph = " ".join(sentences)

            paragraphs.append(
                paragraph
            )

            titles.append(
                title
            )

        return paragraphs, titles

    def build_index(
        self,
        paragraphs
    ):

        embeddings = self.embedder.encode(
            paragraphs,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        embeddings = embeddings.astype(
            np.float32
        )

        dim = embeddings.shape[1]

        index = faiss.IndexFlatL2(
            dim
        )

        index.add(
            embeddings
        )

        return index

    def retrieve(
        self,
        query,
        index,
        paragraphs,
        titles,
        top_k=5
    ):

        q_emb = self.embedder.encode(
            [query],
            convert_to_numpy=True
        )

        q_emb = q_emb.astype(
            np.float32
        )

        distances, indices = index.search(
            q_emb,
            top_k
        )

        results = []

        for idx, dist in zip(
            indices[0],
            distances[0]
        ):

            results.append(
                {
                    "title":
                        titles[idx],

                    "paragraph":
                        paragraphs[idx],

                    "distance":
                        float(dist)
                }
            )

        return results

    def decompose_question(
            self,
            question
        ):
        question = question.strip()

        sub_questions = [question]

        entities = self.extract_bridge_entities(
            question
        )

        for entity in entities:

            sub_questions.append(
                f"Who is {entity}?"
            )

        return sub_questions

    def extract_bridge_entities(
        self,
        text
    ):

        pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"

        entities = re.findall(
            pattern,
            text
        )

        entities = list(
            dict.fromkeys(
                entities
            )
        )

        return entities

    def build_second_hop_query(
        self,
        question,
        first_hop_text
    ):

        entities = self.extract_bridge_entities(
            first_hop_text
        )

        if len(entities) == 0:

            return question

        bridge = entities[0]

        return (
            f"{question} information about {bridge}"
        )

    def first_hop(
        self,
        question,
        paragraphs,
        titles,
        top_k=5
    ):

        index = self.build_index(
            paragraphs
        )

        return self.retrieve(
            question,
            index,
            paragraphs,
            titles,
            top_k
        )

    def second_hop(
        self,
        question,
        first_hop_results,
        paragraphs,
        titles,
        top_k=5
    ):

        if len(first_hop_results) == 0:

            return []

        bridge_text = (
            first_hop_results[0]
            ["paragraph"]
        )

        second_query = (
            self.build_second_hop_query(
                question,
                bridge_text
            )
        )

        index = self.build_index(
            paragraphs
        )

        return self.retrieve(
            second_query,
            index,
            paragraphs,
            titles,
            top_k
        )

    def multi_hop_retrieve(
        self,
        question,
        paragraphs,
        titles,
        top_k=5
    ):

        history = {}

        hop1 = self.first_hop(
            question,
            paragraphs,
            titles,
            top_k
        )

        history["hop1"] = hop1

        hop2 = self.second_hop(
            question,
            hop1,
            paragraphs,
            titles,
            top_k
        )

        history["hop2"] = hop2

        return history


if __name__ == "__main__":

    retriever = MultiHopRetriever()

    dataset = retriever.load_hotpot()

    sample = dataset[
        "validation"
    ][0]

    paragraphs, titles = (
        retriever.extract_paragraphs(
            sample
        )
    )

    history = (
        retriever.multi_hop_retrieve(
            sample["question"],
            paragraphs
        )
    )

    print(
        "\nQUESTION:"
    )

    print(
        sample["question"]
    )

    print(
        "\nFIRST HOP:"
    )

    for r in history["hop1"]:

        print(
            r["paragraph"][:200]
        )

    print(
        "\nSECOND HOP:"
    )

    for r in history["hop2"]:

        print(
            r["paragraph"][:200]
        )