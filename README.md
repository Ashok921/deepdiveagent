# 🔬 DeepDive: Enterprise Cyclical Multi-Agent Research Workbench

An enterprise-grade, autonomous multi-agent research system built with **LangGraph**, **ChromaDB**, **Hugging Face Embeddings**, and **Gemini Models**.

The pipeline uses a **cyclical graph workflow** (Planner → Researcher → Verifier → Synthesizer) designed to maximize inference efficiency, eliminate token bloat, and maintain high data integrity through iterative cross-examination and local vector indexing.

---

## 🏛️ System Architecture & Workflow

```
                                    +-----------------------+
                                    |         START         |
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    |      1. PLANNER       |
                                    | Decomposes goal into  |
                                    | interrogative sub-Qs  |
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    |     2. RESEARCHER     |
                                    | Web scraping (Tavily) |
                                    | Local Chroma Indexing |
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    |      3. VERIFIER      |
                                    | Semantic Top-k Search |
                                    | Evaluates sufficiency |
                                    +-----------+-----------+
                                                |
                         +----------------------+----------------------+
                         |                                             |
             [is_sufficient == False]                       [is_sufficient == True]
           [loops < MAX_RESEARCH_LOOPS]                    [or MAX_LOOPS reached]
                         |                                             |
                         v                                             v
            (Route gaps to re-search)                       +-----------------------+
                         |                                  |    4. SYNTHESIZER     |
                         +--------------------------------->| Queries top-8 chunks  |
                                                            | Compiles final report |
                                                            +-----------+-----------+
                                                                        |
                                                                        v
                                                            +-----------------------+
                                                            |          END          |
                                                            +-----------------------+
```

---

## 🔑 Key Features

- **Local Embeddings Strategy:** Uses Hugging Face's `all-MiniLM-L6-v2` locally to embed chunks, completely eliminating embedding API quota dependencies and cost overhead.
- **Top-k Vector Retrieval:** Both the Verifier and Synthesizer nodes execute similarity searches directly against a local ChromaDB instance instead of dumping raw evidence text into prompts.
- **Deduplication Engine:** Prevents vector storage inflation using content-prefix deduplication across multiple search loops.
- **Strict Guardrails:** Enforces deterministic iteration caps (`MAX_LOOPS = 3`) to prevent infinite reasoning loops while capturing missing context gaps.
- **Streamlit Interactive UI:** Features real-time state logging, native status progress spinners, vector store metrics, and markdown export support.

---

## 🛠️ Tech Stack

- **Orchestration:** LangGraph (Stateful cyclical graph runtime)
- **Vector Store:** ChromaDB (Local persistent storage)
- **Embeddings:** `langchain-huggingface` (`all-MiniLM-L6-v2`)
- **Text Processing:** `RecursiveCharacterTextSplitter`
- **Search & Scraping:** Tavily Search API & FireCrawl Loader
- **LLM Engine:** Gemini 2.5 / 3.0 Flash (`langchain-google-genai`)
- **Frontend UI:** Streamlit

---

## 🚀 Quick Start & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Clone & Setup Environment
```bash
git clone https://github.com/your-org/deepdive-research-workbench.git
cd deepdive-research-workbench

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install streamlit langgraph langchain-core langchain-community langchain-huggingface chromadb tavily-python firecrawl-py langchain-google-genai
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY="your-google-gemini-api-key"
TAVILY_API_KEY="your-tavily-api-key"
FIRECRAWL_API_KEY="your-firecrawl-api-key"
```

### 5. Launch the Application
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```text
deepdive-research-workbench/
├── app.py                  # Streamlit UI dashboard & progress tracker
├── state.py                # TypedDict state schema definition
├── graph.py                # LangGraph workflow compilation & conditional routing
├── nodes/                  # Pure function node logic
│   ├── planner.py          # Deconstructs prompt into sub-questions
│   ├── researcher.py       # Searches, scrapes & indexes chunks into ChromaDB
│   ├── verifier.py         # Performs similarity search & checks gap sufficiency
│   └── synthesizer.py      # Retrieves top context & writes final markdown report
├── chroma_db/              # Persistent vector store output directory
├── requirements.txt        # Python dependencies
└── README.md               # Architecture documentation
```