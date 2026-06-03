from transformers import pipeline

from src.retriever import MultiHopRetriever
from src.prm import ProcessRewardModel


class PRMPipeline:

    def __init__(self):

        self.retriever = MultiHopRetriever()

        self.prm = ProcessRewardModel()

        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base"
        )

    def generate_answer(
        self,
        question,
        contexts
    ):

        context_text = "\n".join(
            contexts
        )

        prompt = f"""
Context:
{context_text}

Question:
{question}

Answer:
"""

        output = self.generator(
            prompt,
            max_new_tokens=64,
            do_sample=False
        )

        return output[0]["generated_text"]

    def run(
        self,
        question,
        paragraphs,
        titles,
        threshold=0.4
    ):

        reasoning_state = ""

        sub_questions = (
            self.retriever.decompose_question(
                question
            )
        )

        retrieval_history = (
            self.retriever.multi_hop_retrieve(
                question,
                paragraphs,
                titles,
                top_k=5
            )
        )

        hop1_raw = retrieval_history["hop1"]

        hop1_kept = self.prm.prune(
            question=question,
            retrieved_passages=hop1_raw,
            threshold=threshold,
            reasoning_state=reasoning_state
        )

        reasoning_state = (
            self.prm.update_reasoning_state(
                reasoning_state,
                hop1_kept
            )
        )

        hop2_raw = retrieval_history["hop2"]

        hop2_kept = self.prm.prune(
            question=question,
            retrieved_passages=hop2_raw,
            threshold=threshold,
            reasoning_state=reasoning_state
        )

        reasoning_state = (
            self.prm.update_reasoning_state(
                reasoning_state,
                hop2_kept
            )
        )

        contexts = []

        for item in hop1_kept:
            contexts.append(
                item["paragraph"]
            )

        for item in hop2_kept:
            contexts.append(
                item["paragraph"]
            )

        unique_contexts = []
        seen = set()

        for ctx in contexts:

            if ctx not in seen:

                unique_contexts.append(
                ctx[:400]
                )

                seen.add(
                    ctx
                )

        answer = self.generate_answer(
                    question,
                    unique_contexts
                )

        return {

            "question":
                question,
            
            "sub_questions":
                sub_questions,

            "answer":
                answer,

            "contexts":
                unique_contexts,

            "hop1":
                hop1_kept,

            "hop2":
                hop2_kept,

            "reasoning_state":
                reasoning_state,

            "threshold":
                threshold
        }


if __name__ == "__main__":

    retriever = MultiHopRetriever()

    dataset = retriever.load_hotpot()

    sample = dataset["validation"][0]

    question = sample["question"]

    paragraphs, titles = (
        retriever.extract_paragraphs(
            sample
        )
    )

    pipeline_obj = PRMPipeline()

    result = pipeline_obj.run(
        question=question,
        paragraphs=paragraphs,
        threshold=0.4
    )

    print("\nQUESTION:")
    print(question)

    print("\nGROUND TRUTH:")
    print(sample["answer"])

    print("\nMODEL ANSWER:")
    print(result["answer"])

    print("\nHOP1 KEPT:")
    print(len(result["hop1"]))

    print("\nHOP2 KEPT:")
    print(len(result["hop2"]))