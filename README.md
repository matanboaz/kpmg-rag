# KPMG RAG System — Home Assignment

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain to answer historical and industrial questions from document collections. It was developed for the KPMG home assignment.

## 📂 Features

- Loads and chunks `.docx` documents (e.g., food, steel, textile, auto).
- Vectorizes text and stores it in Pinecone.
- Performs semantic search and filters results by topic.
- Applies smart prompt engineering for edge cases:
  - Ambiguity
  - Contradictions
  - Misspellings
  - Temporal and topical vagueness
- Supports debug mode with similarity scores for diagnostics.

## 🧰 Installation

```bash
pip install -r requirements.txt
```

You’ll also need:
- An OpenAI API key (set as `OPENAI_API_KEY`)
- A Pinecone account and environment variables:
  - `PINECONE_API_KEY`
  - `PINECONE_ENV`

## 🚀 How to Use

1. Place your `.docx` documents in the working directory.
2. Run the notebook `KPMG_Solution.ipynb` step by step.
3. Use the test block at the end to evaluate predefined edge cases.
4. Use `debug_retrieval()` to explore how retrieval worked.

## 🧪 Testing

```python
debug_retrieval("What is Fordism?")
```

Returns the top-matching chunks and similarity scores to diagnose underperformance.

## 📁 Structure

```
.
├── KPMG_Solution.ipynb       # Main notebook
├── requirements.txt          # Install dependencies
├── README.md                 # You're reading it
└── your_doc_files.docx       # Input documents
```

## 📫 Contact

For questions or feedback, please reach out to the developer.
