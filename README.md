# AI Coding Sessions

> Real transcripts from AI-assisted development sessions — showing how I break down problems, use AI iteratively, and refine outputs into production-ready patterns.

---

## Sessions

### 1. ForgeIoT — Industrial IoT Predictive Maintenance Platform
**File:** `forgeiot-session.md`

A predictive maintenance platform for factory equipment running on AWS. Two real problems from development:

- **Debugging a silent Glue ETL pipeline failure** — traced a conditional trigger misconfiguration, moved to an EventBridge + `StartJobRun` pattern, added concurrency controls
- **Designing a Bedrock AgentCore multi-agent layer** — supervisor/specialist architecture, structured output schema for safety-critical claim traceability, ECS Fargate credential fix

**Stack:** AWS Glue · EventBridge · Bedrock AgentCore · ECS Fargate · Python · Streamlit

---

### 2. SAP Support Bot — RAG Pipeline
**File:** `sap-support-bot-session.md`

An internal support bot that answers SAP configuration queries by retrieving from SAP Notes, runbooks, and resolved ticket history — built specifically to stop the LLM hallucinating T-codes and config paths.

- **Retrieval design** — hybrid BM25 + dense embeddings, heuristic query routing, section-aware SAP Note chunking
- **Hallucination control** — citation-gated generation, post-generation T-code validation, module-specific confidence thresholds
- **API layer** — why I dropped streaming in favour of async validated responses, compliance audit logging

**Stack:** FastAPI · Python · OpenSearch · LLM (mocked) · AWS Bedrock (production target)

---

## What these show

- How I describe problems to AI (system context, not just code)
- Where I accepted suggestions and where I pushed back
- How I think about safety and reliability in AI-assisted systems
- Domain constraints I applied myself that the model couldn't know

---

## Tools used

- **Claude** — primary AI pair programming tool
- **Python** — both projects
- **FastAPI** — SAP support bot API layer
- **AWS** — infrastructure for ForgeIoT
