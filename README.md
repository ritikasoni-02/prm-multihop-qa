# PRM-Guided Multi-Hop Question Answering

## Overview

This project implements a multi-hop question answering pipeline using a Process Reward Model (PRM) to filter retrieved evidence before answer generation.

The system is evaluated on the HotpotQA dataset and compares retrieval quality under different PRM confidence thresholds.

---

## Project Objectives

- Build a multi-hop retrieval pipeline.
- Retrieve supporting paragraphs from HotpotQA.
- Use a PRM to score retrieved evidence.
- Filter low-quality evidence.
- Generate answers using an LLM.
- Evaluate performance using RAGAS metrics.

---

## Project Structure

```
prm-multihop-qa/
│
├── src/
│   ├── retriever.py
│   ├── prm.py
│   └── pipeline.py
│
├── eval/
│   └── ragas_eval.py
│
├── results/
│   ├── results_04.csv
│   ├── results_06.csv
│   ├── summary.csv
│   └── results.json
│
├── requirements.txt
└── README.md
```

---

## Methodology

### Retrieval

Questions are taken from the HotpotQA validation set.

For each question:

1. Supporting paragraphs are extracted.
2. First-hop retrieval is performed.
3. Second-hop retrieval is performed.

### PRM Filtering

Retrieved evidence is scored by a Process Reward Model.

Two thresholds are evaluated:

- t = 0.4
- t = 0.6

Only evidence above the threshold is retained.

### Answer Generation

The filtered evidence is provided to an LLM for answer generation.

### Evaluation

The following RAGAS metrics are used:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall
- Answer Correctness

---