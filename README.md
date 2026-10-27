DeepDive: Cyclical Multi-Agent Research Workbench

An enterprise-grade, cyclical multi-agent research system built with LangGraph, ChromaDB, Hugging Face Embeddings, and a Streamlit interactive frontend. Designed to maximize inference efficiency, minimize API token consumption, and maintain data integrity through structured outputs and local vector retrieval.
🏗️ System Architecture & Workflow
The pipeline operates as a controlled cyclical graph with four specialized nodes:
Planner (Planner Node): Deconstructs the original user query into targeted, independent interrogative sub-questions using structured outputs.
Researcher (Researcher Node): Executes web searches via Tavily, scrapes clean markdown content using FireCrawl, chunks text using RecursiveCharacterTextSplitter, and indexes embeddings into a persistent local ChromaDB store.
Verifier (Verifier Node): Cross-examines retrieved evidence chunks from ChromaDB against sub-questions. Determines sufficiency and loops back to the researcher with identified gaps if necessary (bounded by a maximum loop count).
Synthesizer (Synthesizer Node): Pulls top-$k$ relevant chunks across research loops using semantic vector search and compiles a clean, comprehensive executive markdown report.
🛠️ Tech Stack
Orchestration: LangGraph (Cyclical state graph)
Vector Store: ChromaDB (Persistent local storage)
Embeddings: Hugging Face (all-MiniLM-L6-v2 via langchain-huggingface)
Text Splitting: RecursiveCharacterTextSplitter
Search & Scraping: Tavily API & FireCrawl Loader
Frontend: Streamlit (Interactive UI with telemetry and status containers)
