@"
# 📚 Intelligent Document Analysis & Q&A System

> An AI-powered learning and document comprehension system combining **Conventional Machine Learning** (Classification & Clustering) with **Generative AI** (LLMs, RAG, Embeddings) to help readers grasp complex documents faster and practice active recall.

---

## 🌟 Key Features

- **📄 Smart Document Parsing & Semantic Chunking**: Supports PDF and TXT document ingestion with page-level tracking and metadata extraction.
- **📊 Conventional ML Layer**:
  - **Topic Clustering**: Uses **TF-IDF + K-Means** to automatically group document chunks into conceptual topics and mind-map themes.
  - **Content Classification**: Categorizes text chunks into functional reader content (*Definitions*, *Core Explanations*, *Examples*, *Summaries*).
- **🗄️ Vector Store & Embeddings**: Uses **ChromaDB** with **HuggingFace / Gemini Embeddings** for high-dimensional semantic indexing.
- **💬 Context-Aware RAG Q&A**: Powered by **LangChain** and **Google Gemini / OpenAI** to answer reader questions with precise source and page citations.
- **🎯 Interactive Practice Quiz Arena**: Automatically generates practice quizzes (MCQs and Short Answer questions) with difficulty levels, dynamic scoring, and constructive AI feedback.

---

## 🏗️ Architecture

```text
  +-----------------------+
  |    Input Document     | (PDF / TXT)
  +-----------------------+
              |
              v
  +-----------------------+
  | Document Processing   | (PyPDF & Recursive Character Splitter)
  +-----------------------+
         /         \
        /           \
       v             v
+-----------------------+   +-----------------------+
|  Conventional ML      |   |     Gen AI Layer      |
|  - TF-IDF + K-Means   |   | - ChromaDB Vector DB  |
|  - Content Classifier |   | - LangChain RAG Chain |
+-----------------------+   +-----------------------+
        \             /
         \           /
          v         v
  +-----------------------+
  | Practice & Quiz Arena | (MCQ & Short Answer Generator + AI Grader)
  +-----------------------+
              |
              v
  +-----------------------+
  | Streamlit Web App     | (app.py Dashboard)
  +-----------------------+
