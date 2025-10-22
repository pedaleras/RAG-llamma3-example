# 🤖 Local RAG Agent using Llama 3 + LangChain

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** pipeline running **entirely locally**,
combining **Llama 3** (via Ollama) with **LangChain** and **Hugging Face embeddings**.

The agent is capable of reading and reasoning over web content — in this example, an article
by [Lilian Weng](https://lilianweng.github.io/posts/2023-06-23-agent/) about AI agents — and then answering questions
about it.

---

## 🧩 Credits

This project is inspired by a [YouTube tutorial from Código Fonte TV](https://www.youtube.com/watch?v=CuPKOGdA46Q),
where they created a RAG agent using **ChatGPT and the OpenAI API** and based on research from Lilian Weng’s
post [LLM-powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/).

I followed the same concept but did something **different**: this version runs **entirely locally** using **Llama 3 via
Ollama** and **free Hugging Face embeddings**, so no API key or paid plan is needed.

---

## 🧩 Tech Stack

- **Python 3.10+**
- **LangChain** (orchestration)
- **Ollama** (to run the Llama 3 model locally)
- **Chroma** (vector database for document retrieval)
- **Hugging Face embeddings** (`sentence-transformers/all-MiniLM-L6-v2`)
- **BeautifulSoup4** (HTML parsing for web content)

---

## ⚙️ Setup

1. **Install Ollama**  
   👉 [https://ollama.com/download](https://ollama.com/download)

   Then pull the model:
   ```bash
   ollama pull llama3
   ```

2. **Clone this repository**
    ```bash
    git clone https://github.com/<your-username>/<repo-name>.git
    cd <repo-name>/src
    ```

3. **Create and activate a virtual environment**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate # Windows
    source .venv/bin/activate # Linux/Mac
    ```

4. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

**Minimal requirements.txt for this project:**

```text
beautifulsoup4==4.14.2
langchain==1.0.1
langchain-chroma==1.0.0
langchain-classic==1.0.0
langchain-community==0.4
langchain-core==1.0.0
langchain-huggingface==1.0.0
langchain-ollama==1.0.0
langchain-text-splitters==1.0.0
chromadb==1.2.1
huggingface-hub==0.35.3
```
    
5. Run the app

    ```bash
    python main.py
    ```

---

## 💬 Example

```cmd
Question (type 'exit' to quit): What is task decomposition?
> Task decomposition is the process of breaking down complex goals into smaller, more manageable steps...
```