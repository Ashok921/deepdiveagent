import os
import json
import asyncio
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.deepdive_graph import ResearchState, build_compiled_graph

load_dotenv()

app = FastAPI(title="DeepDive LangGraph State Visualizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_RESEARCH_STATE = {
    "original_query": "What are the latest enterprise adoption trends and architectural patterns for multi-agent LLM systems in 2026?",
    "sub_questions": [
        "What are the dominant architectural orchestration topologies (hierarchical vs. peer-to-peer) used in production in 2026?",
        "How do modern multi-agent systems handle cross-agent consensus and automated fact verification?",
        "What are the leading latency, token-cost, and failure-mode benchmarks reported by enterprise adopters?",
        "Which frameworks (LangGraph, AutoGen, CrewAI, Swarm) lead in production deployments and why?"
    ],
    "is_sufficient": True,
    "research_loops": 1,
    "evidence_bank": [
        {
            "sub_question": "What are the dominant architectural orchestration topologies (hierarchical vs. peer-to-peer) used in production in 2026?",
            "source_url": "https://arxiv.org/abs/2601.08942",
            "title": "State of Multi-Agent AI Architectures in Enterprise Systems (2026 Survey)",
            "content": "In 2026, enterprise multi-agent deployments converged toward hybrid hierarchical graphs. A primary supervisor agent routes tasks to specialized workers (retrieval, code execution, auditing), with localized subgraphs utilizing peer-to-peer consensus for verification."
        },
        {
            "sub_question": "How do modern multi-agent systems handle cross-agent consensus and automated fact verification?",
            "source_url": "https://langchain.com/blog/multi-agent-verification-loops",
            "title": "Closing the Hallucination Gap: Verification Loops in Stateful LangGraph",
            "content": "By implementing deterministic verification nodes that evaluate evidence sufficiency before synthesis, systems reduced hallucination rates by 42%. If verification fails, targeted sub-queries trigger targeted re-search loops with bounded iterations."
        },
        {
            "sub_question": "What are the leading latency, token-cost, and failure-mode benchmarks reported by enterprise adopters?",
            "source_url": "https://venturebeat.com/ai/enterprise-agent-economics-2026/",
            "title": "Enterprise Agent Economics: Token Budgets, Latency & Cascading Failures",
            "content": "Average agent execution times dropped from 24s in 2024 to 4.2s in 2026 due to speculative parallel execution and localized small-model routing. Cascading failure states are mitigated via state checkpoints and circuit breakers."
        },
        {
            "sub_question": "Which frameworks (LangGraph, AutoGen, CrewAI, Swarm) lead in production deployments and why?",
            "source_url": "https://github.com/langchain-ai/langgraph",
            "title": "LangGraph: Cyclical Multi-Agent Graph Orchestration in Production",
            "content": "LangGraph established dominance in production enterprise systems (>58% adoption) primarily due to its first-class support for cyclical graphs, human-in-the-loop state inspection, time-travel debugging, and durable persistence."
        }
    ],
    "audit_logs": [
        "[Planner] Generated 4 targeted sub-questions for: 'What are the latest enterprise adoption trends and architectural patterns for multi-agent LLM systems in 2026?'",
        "[Researcher] Searching web for: 'What are the dominant architectural orchestration topologies used in production in 2026?'",
        "[Researcher] Retrieved source: State of Multi-Agent AI Architectures in Enterprise Systems (2026 Survey) (https://arxiv.org/abs/2601.08942)",
        "[Researcher] Searching web for: 'How do modern multi-agent systems handle cross-agent consensus and automated fact verification?'",
        "[Researcher] Retrieved source: Closing the Hallucination Gap: Verification Loops in Stateful LangGraph (https://langchain.com/blog/multi-agent-verification-loops)",
        "[Researcher] Searching web for: 'What are the leading latency, token-cost, and failure-mode benchmarks reported by enterprise adopters?'",
        "[Researcher] Retrieved source: Enterprise Agent Economics: Token Budgets, Latency & Cascading Failures (https://venturebeat.com/ai/enterprise-agent-economics-2026/)",
        "[Researcher] Searching web for: 'Which frameworks lead in production deployments and why?'",
        "[Researcher] Retrieved source: LangGraph: Cyclical Multi-Agent Graph Orchestration in Production (https://github.com/langchain-ai/langgraph)",
        "[Researcher] Successfully gathered 4 new evidence chunks into evidence bank.",
        "[Verifier] Cross-examining evidence bank against query and sub-questions...",
        "[Verifier] Evidence sufficient: True. Feedback: Evidence bank contains comprehensive benchmarks, architectural taxonomy, and production adoption data.",
        "[Verifier] Verification Passed! Evidence bank is comprehensive and verified.",
        "[Synthesizer] Compiling evidence into comprehensive research report...",
        "[Synthesizer] Final research report compiled successfully."
    ],
    "final_report": """# Executive Briefing: Enterprise Multi-Agent LLM Systems in 2026

## 1. Executive Summary
Enterprise adoption of multi-agent Large Language Model (LLM) architectures has transitioned from experimental single-turn prompts to **resilient, cyclical state graphs**. Key drivers include the necessity for deterministic self-correction, rigorous compliance audit logs, and specialized domain decomposition.

---

## 2. Dominant Architectural Topologies
* **Hybrid Hierarchical Orchestration:** Production systems predominantly deploy a **Supervisor-Worker** graph pattern. High-level planning agents decompose goals into distinct sub-questions, dispatching them to parallel domain researchers.
* **Bounded Verification Cycles:** Replacing linear chains with cyclical graphs containing verification barriers prevents ungrounded hallucinations from polluting the final synthesis.

---

## 3. Production Framework Landscape
1. **LangGraph (58% Production Market Share):** Preferred for enterprise mission-critical systems due to:
   * First-class cyclical graph support with conditional routing.
   * State persistence, rollback, and human-in-the-loop checkpoints.
   * Native streaming of granular state updates.
2. **Specialized Swarms:** Used for lightweight localized tasks with ephemeral memory requirements.

---

## 4. Key Performance Indicators & Economics
* **End-to-End Latency:** Speculative parallel sub-question execution has reduced research latency by ~70%.
* **Hallucination Mitigation:** Grounded vector retrieval coupled with verifier cross-examination nodes cuts citation errors by >40%.

---

## 5. Verified Source References
1. [State of Multi-Agent AI Architectures in Enterprise Systems (2026 Survey)](https://arxiv.org/abs/2601.08942)
2. [Closing the Hallucination Gap: Verification Loops in Stateful LangGraph](https://langchain.com/blog/multi-agent-verification-loops)
3. [Enterprise Agent Economics: Token Budgets, Latency & Cascading Failures](https://venturebeat.com/ai/enterprise-agent-economics-2026/)
4. [LangGraph: Cyclical Multi-Agent Graph Orchestration in Production](https://github.com/langchain-ai/langgraph)
"""
}


@app.get("/api/status")
async def get_system_status():
    return {
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "google_configured": bool(os.getenv("GOOGLE_API_KEY")),
        "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
        "firecrawl_configured": bool(os.getenv("FIRECRAWL_API_KEY")),
        "default_provider": "groq" if os.getenv("GROQ_API_KEY") else ("gemini" if os.getenv("GOOGLE_API_KEY") else "simulation")
    }


@app.get("/api/sample-state")
async def get_sample_state():
    return JSONResponse(content=SAMPLE_RESEARCH_STATE)


async def simulate_stream(query: str) -> AsyncGenerator[str, None]:
    """Generates realistic step-by-step LangGraph execution simulation."""
    state = {
        "original_query": query,
        "sub_questions": [],
        "is_sufficient": False,
        "evidence_bank": [],
        "audit_logs": [],
        "research_loops": 0,
        "final_report": ""
    }

    # STEP 1: Planner
    yield f"event: node_start\ndata: {json.dumps({'node': 'planner'})}\n\n"
    await asyncio.sleep(0.8)
    
    sub_qs = [
        f"What are the foundational architectures and core mechanics of {query}?",
        f"How does {query} compare with existing alternative paradigms and benchmarks?",
        f"What are the verified production trade-offs, failure modes, and latency characteristics?",
        f"What are leading real-world implementations and case studies for {query}?"
    ]
    planner_log = f"[Planner] Generated {len(sub_qs)} targeted sub-questions for: '{query}'"
    state["sub_questions"] = sub_qs
    state["audit_logs"].append(planner_log)
    state["research_loops"] = 0
    
    yield f"event: log\ndata: {json.dumps({'log': planner_log})}\n\n"
    yield f"event: state_update\ndata: {json.dumps({'node': 'planner', 'state': state})}\n\n"
    await asyncio.sleep(1.0)

    # STEP 2: Researcher
    yield f"event: node_start\ndata: {json.dumps({'node': 'researcher'})}\n\n"
    evidence_items = []
    
    for i, q in enumerate(sub_qs):
        search_log = f"[Researcher] Searching web for: '{q}'"
        state["audit_logs"].append(search_log)
        yield f"event: log\ndata: {json.dumps({'log': search_log})}\n\n"
        await asyncio.sleep(0.6)
        
        item = {
            "sub_question": q,
            "source_url": f"https://arxiv.org/abs/2602.{1000 + i}",
            "title": f"Technical Research Benchmark: {q[:40]}...",
            "content": f"Empirical findings regarding '{q}'. Multi-agent graph state coordination improves contextual accuracy and eliminates single-agent failure points."
        }
        evidence_items.append(item)
        state["evidence_bank"].append(item)
        
        found_log = f"[Researcher] Retrieved source: {item['title']} ({item['source_url']})"
        state["audit_logs"].append(found_log)
        yield f"event: log\ndata: {json.dumps({'log': found_log})}\n\n"
        yield f"event: state_update\ndata: {json.dumps({'node': 'researcher', 'state': state})}\n\n"
        await asyncio.sleep(0.4)

    finish_res_log = f"[Researcher] Gathered {len(evidence_items)} verified evidence chunks into evidence bank."
    state["audit_logs"].append(finish_res_log)
    yield f"event: log\ndata: {json.dumps({'log': finish_res_log})}\n\n"
    yield f"event: state_update\ndata: {json.dumps({'node': 'researcher', 'state': state})}\n\n"
    await asyncio.sleep(0.8)

    # STEP 3: Verifier
    yield f"event: node_start\ndata: {json.dumps({'node': 'verifier'})}\n\n"
    ver_start_log = "[Verifier] Cross-examining evidence bank against query and sub-questions..."
    state["audit_logs"].append(ver_start_log)
    yield f"event: log\ndata: {json.dumps({'log': ver_start_log})}\n\n"
    await asyncio.sleep(1.0)

    state["is_sufficient"] = True
    state["research_loops"] = 1
    ver_pass_log = "[Verifier] Evidence sufficient: True. Feedback: All sub-questions comprehensively addressed with high citation rigor."
    state["audit_logs"].append(ver_pass_log)
    state["audit_logs"].append("[Verifier] Verification Passed! Proceeding to synthesis.")
    yield f"event: log\ndata: {json.dumps({'log': ver_pass_log})}\n\n"
    yield f"event: state_update\ndata: {json.dumps({'node': 'verifier', 'state': state})}\n\n"
    await asyncio.sleep(0.8)

    # STEP 4: Synthesizer
    yield f"event: node_start\ndata: {json.dumps({'node': 'synthesizer'})}\n\n"
    synth_start_log = "[Synthesizer] Compiling evidence into comprehensive research report..."
    state["audit_logs"].append(synth_start_log)
    yield f"event: log\ndata: {json.dumps({'log': synth_start_log})}\n\n"
    await asyncio.sleep(1.2)

    report_content = f"""# Executive Research Report: {query}

## 1. Executive Summary
This report analyzes **{query}** utilizing multi-agent state graph evaluation. Evidence retrieved across multiple sources demonstrates high technical viability, architectural maturity, and verifiable performance guarantees.

---

## 2. In-Depth Technical Analysis & Architecture
- **State Partitioning:** Breaking complex queries into distinct sub-questions allows targeted, parallel investigation without hallucination spillover.
- **Verification Gates:** Enforcing deterministic sufficiency thresholds prior to synthesis guarantees citation fidelity.
- **Iterative Feedback:** The cyclical LangGraph architecture autonomously handles knowledge gaps before delivering the final artifact.

---

## 3. Comparative Benchmarks, Strengths & Failure Modes
- **Latency Optimization:** Speculative parallel retrieval reduces total pipeline execution duration by up to 70% compared to serialized research chains.
- **Hallucination Reduction:** Cross-examination of retrieved chunks against sub-questions reduces ungrounded factual assertions by 42%.
- **Cascading Failure Mitigation:** Bounded retry loops and fallback generation prevent infinite graph traversal and pipeline deadlocks.

---

## 4. Key Takeaways & Practical Recommendations
- **Adopt State Graph Topology:** Standardize on stateful graph architectures that support conditional edge routing and checkpointing.
- **Enforce Semantic Deduplication:** Filter web boilerplate and redundant content chunks to maximize synthesization quality.
- **Implement Verification Barriers:** Always validate evidence bank density before committing to final document generation.

---

## 5. Verified Source References & Citations
""" + "\n".join([f"- [{e['title']}]({e['source_url']}) — *Addresses: {e['sub_question']}*" for e in state["evidence_bank"]])

    state["final_report"] = report_content
    synth_done_log = "[Synthesizer] Final research report compiled successfully."
    state["audit_logs"].append(synth_done_log)
    yield f"event: log\ndata: {json.dumps({'log': synth_done_log})}\n\n"
    yield f"event: state_update\ndata: {json.dumps({'node': 'synthesizer', 'state': state})}\n\n"
    
    # Complete
    yield f"event: complete\ndata: {json.dumps({'final_state': state})}\n\n"


async def live_graph_stream(query: str, provider: str, model_name: Optional[str]) -> AsyncGenerator[str, None]:
    """Runs compiled LangGraph and yields granular SSE events."""
    try:
        app_graph = build_compiled_graph(provider=provider, model_name=model_name)
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': f'Failed to compile graph: {str(e)}'})}\n\n"
        return

    current_state = {
        "original_query": query,
        "sub_questions": [],
        "is_sufficient": False,
        "evidence_bank": [],
        "audit_logs": [],
        "research_loops": 0,
        "final_report": ""
    }

    try:
        inputs = ResearchState(original_query=query)
        # Stream updates from the LangGraph execution
        for chunk in app_graph.stream(inputs, stream_mode="updates"):
            for node_name, updates in chunk.items():
                yield f"event: node_start\ndata: {json.dumps({'node': node_name})}\n\n"
                
                # Merge state updates
                if "sub_questions" in updates:
                    current_state["sub_questions"] = updates["sub_questions"]
                if "is_sufficient" in updates:
                    current_state["is_sufficient"] = updates["is_sufficient"]
                if "research_loops" in updates:
                    current_state["research_loops"] = updates["research_loops"]
                if "evidence_bank" in updates:
                    current_state["evidence_bank"].extend(updates["evidence_bank"])
                if "audit_logs" in updates:
                    for l in updates["audit_logs"]:
                        current_state["audit_logs"].append(l)
                        yield f"event: log\ndata: {json.dumps({'log': l})}\n\n"
                if "final_report" in updates:
                    current_state["final_report"] = updates["final_report"]

                yield f"event: state_update\ndata: {json.dumps({'node': node_name, 'state': current_state})}\n\n"
                await asyncio.sleep(0.3)

        yield f"event: complete\ndata: {json.dumps({'final_state': current_state})}\n\n"

    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


@app.get("/api/stream")
async def stream_research(
    query: str = Query(..., description="Research query"),
    provider: str = Query("groq", description="LLM provider: groq, gemini, or simulation"),
    model: Optional[str] = Query(None, description="Specific model name")
):
    if provider == "simulation" or not (os.getenv("GROQ_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return StreamingResponse(simulate_stream(query), media_type="text/event-stream")
    
    return StreamingResponse(live_graph_stream(query, provider, model), media_type="text/event-stream")


# Mount static frontend directory
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
