from src.pipeline import PRMPipeline

paragraphs = [
    "Paris is the capital of France.",
    "France is located in Europe.",
    "Berlin is the capital of Germany.",
    "Germany is located in Europe."
]

pipeline = PRMPipeline()

titles = [
    "France",
    "France",
    "Germany",
    "Germany"
]

result = pipeline.run(
    question="What is the capital of France?",
    paragraphs=paragraphs,
    titles=titles,
    threshold=0.4
)

print(result["answer"])