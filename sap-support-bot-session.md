# AI Coding Session — SAP Support Bot (RAG Pipeline)
**Tool:** Claude (claude.ai)
**Date:** October 2025
**Focus:** Building a RAG pipeline for internal SAP query resolution — grounding LLM responses in SAP documentation and ticket history to eliminate hallucinated T-codes and config paths
**Format:** Condensed transcript with key exchanges, decisions, and refinements

---

## Session Overview

Tech M's internal SAP support team handles ~200 tickets/week — mostly repetitive configuration questions (MM, FI, SD modules). The goal: a bot that answers SAP queries by retrieving from a curated knowledge base (SAP Notes, internal runbooks, resolved ticket history) rather than generating from parametric memory, which hallucinates T-codes confidently and dangerously.

---

## Session 1 — Retrieval Architecture: Sparse vs. Dense

**Me:**
I'm building a RAG bot for SAP support queries. The knowledge base has three source types: SAP Notes PDFs (~4,000 docs), internal runbooks (markdown, ~300 docs), and resolved ticket history (structured JSON, ~15,000 tickets). What retrieval architecture would you recommend, and should I use dense or sparse retrieval?

**Claude:**
For SAP support specifically, I'd argue strongly for a hybrid retrieval setup — sparse (BM25) as primary, dense (embeddings) as secondary, with a reranker on top. Here's why the domain changes the usual recommendation:

SAP documentation is full of highly specific, low-frequency tokens: T-codes (`MIGO`, `FB60`, `VF01`), transaction paths (`IMG > MM > Purchasing > ...`), error message IDs (`M7 123`, `F5 702`), OSS note numbers. BM25 handles exact-match retrieval of these identifiers extremely well. Dense retrieval will semantically cluster "goods receipt" and "GR posting" correctly, but may dilute precision when someone asks about a specific error code.

Recommended stack:
```
Query
  │
  ├─► BM25 (Elasticsearch / OpenSearch) ──────────────┐
  │   top-20 candidates                                │
  │                                                    ▼
  └─► Dense Embeddings (text-embedding-3-small) ──► Reranker (cross-encoder)
      top-20 candidates                               top-5 final context
```

For the reranker, a cross-encoder fine-tuned on MS-MARCO works well out of the box. If you have labeled SAP ticket pairs (query + correct resolution), fine-tuning the reranker on those will give significant precision gains.

Source-type routing matters too: a query containing a T-code should weight SAP Notes higher; a "how do I..." process question should weight runbooks higher; an error message ID should hit ticket history first.

**Me:**
The source routing idea is interesting. How do I implement that without it becoming a complex classifier I have to maintain?

**Claude:**
Keep it heuristic-based — regex on the query before retrieval, not a trained classifier:

```python
import re

SAP_TCODE_RE = re.compile(r'\b[A-Z]{2,4}\d{0,3}\b')         # MIGO, FB60, VA01
SAP_MSG_RE   = re.compile(r'\b[A-Z]{1,2}\s?\d{3,4}\b')      # M7 123, F5 702  
SAP_NOTE_RE  = re.compile(r'\b(note|oss|sap note)\s*\d+\b', re.IGNORECASE)

def route_query(query: str) -> dict:
    weights = {"sap_notes": 0.33, "runbooks": 0.33, "tickets": 0.34}
    
    if SAP_TCODE_RE.search(query):
        weights = {"sap_notes": 0.55, "runbooks": 0.30, "tickets": 0.15}
    
    if SAP_MSG_RE.search(query):
        weights = {"sap_notes": 0.25, "runbooks": 0.15, "tickets": 0.60}
    
    if SAP_NOTE_RE.search(query):
        weights = {"sap_notes": 0.80, "runbooks": 0.10, "tickets": 0.10}
    
    return weights
```

This is easy to audit, easy to extend, and doesn't require retraining when SAP releases new T-codes. The tradeoff is it won't handle ambiguous queries well — but for SAP support, most queries are specific enough that one of these patterns will fire.

**Me:**
Good. What about chunking strategy for the SAP Notes PDFs? They have a very specific structure — header, symptom, cause, solution, keywords section.

**Claude:**
Don't chunk SAP Notes by token count — chunk by section. The "Solution" and "Symptom" sections have very different retrieval utility; mixing them in the same chunk confuses the reranker.

```python
SAP_NOTE_SECTIONS = ["Symptom", "Cause", "Solution", "Other Terms", "Keywords"]

def chunk_sap_note(text: str, note_id: str) -> list[dict]:
    chunks = []
    current_section = "Header"
    current_text = []
    
    for line in text.split('\n'):
        matched = next((s for s in SAP_NOTE_SECTIONS if line.strip().startswith(s)), None)
        if matched:
            if current_text:
                chunks.append({
                    "note_id": note_id,
                    "section": current_section,
                    "text": '\n'.join(current_text).strip(),
                    "retrieval_weight": 1.5 if current_section == "Solution" else 1.0
                })
            current_section = matched
            current_text = []
        else:
            current_text.append(line)
    
    return chunks
```

The `retrieval_weight` field lets your reranker stage boost Solution-section chunks at scoring time. Keywords section chunks are also valuable for BM25 — they're SAP's own synonym list for the note.

---

**Outcome:** Adopted hybrid retrieval with heuristic routing. Section-aware chunking implemented for SAP Notes. This became the core retrieval architecture for the bot.

**What I changed:** The regex patterns were a good starting point but missed multi-word T-codes and some module-specific patterns (e.g. `SPRO` navigation paths). Extended the patterns with a small lookup table of known high-frequency SAP terms from our ticket history.

---

## Session 2 — Hallucination Control in the Generation Layer

**Me:**
The retrieval works well but the LLM generation layer is still hallucinating — it confidently fills gaps in retrieved context with plausible-but-wrong SAP config steps. I need structural controls, not just prompt instructions.

**Claude:**
Prompt instructions alone won't hold against a model that has strong SAP priors from training data — it will "helpfully" complete partial config paths from memory. You need a combination of:

**1. Citation-gated generation**

Force every procedural step to cite a chunk ID. If no chunk supports it, the model must not include it:

```python
SYSTEM_PROMPT = """
You are an SAP support assistant. Answer ONLY using the provided context chunks.

Rules:
- Every config step, T-code, or path you mention MUST be followed by [chunk_id]
- If the context does not contain enough information to answer fully, say:
  "I can partially answer this. The following is supported by documentation: [answer].
   The following is NOT in the retrieved context and I cannot confirm it: [gap description]"
- Never complete a navigation path (IMG > ...) unless the full path appears in a chunk
- T-codes: only use T-codes that appear verbatim in retrieved chunks

Context:
{chunks}
"""
```

**2. Post-generation validation**

Extract all T-codes and config paths from the generated response and check them against retrieved chunks:

```python
def validate_response(response: str, chunks: list[dict]) -> dict:
    chunk_text = ' '.join(c['text'] for c in chunks)
    
    # Extract T-codes from response
    mentioned_tcodes = SAP_TCODE_RE.findall(response)
    ungrounded = [t for t in mentioned_tcodes if t not in chunk_text]
    
    return {
        "response": response,
        "ungrounded_tcodes": ungrounded,
        "safe_to_display": len(ungrounded) == 0
    }
```

**3. Confidence tiering**

Don't return a single response — return a tiered answer:

```python
def tier_response(validation_result: dict) -> dict:
    if validation_result["safe_to_display"]:
        return {"tier": "CONFIRMED", "response": validation_result["response"]}
    
    elif len(validation_result["ungrounded_tcodes"]) <= 1:
        return {
            "tier": "REVIEW_RECOMMENDED", 
            "response": validation_result["response"],
            "warning": f"Could not verify: {validation_result['ungrounded_tcodes']}"
        }
    
    else:
        return {
            "tier": "ESCALATE",
            "response": "Retrieved context is insufficient for a safe answer.",
            "partial": validation_result["response"],
            "recommend": "Please raise with SAP Basis team"
        }
```

**Me:**
The `ESCALATE` tier is important — we have strict SLAs and I'd rather the bot says "I don't know" than gives a wrong config path that causes a production issue. Can we make the threshold configurable per module?

**Claude:**
Yes — parameterize the tier thresholds by module, and detect the module from the query routing step:

```python
MODULE_THRESHOLDS = {
    "FI":  {"max_ungrounded": 0, "default_tier": "ESCALATE"},   # Finance: zero tolerance
    "MM":  {"max_ungrounded": 1, "default_tier": "REVIEW_RECOMMENDED"},
    "SD":  {"max_ungrounded": 1, "default_tier": "REVIEW_RECOMMENDED"},
    "BASIS": {"max_ungrounded": 0, "default_tier": "ESCALATE"},  # Basis: zero tolerance
    "DEFAULT": {"max_ungrounded": 1, "default_tier": "REVIEW_RECOMMENDED"}
}

def tier_response(validation_result: dict, module: str = "DEFAULT") -> dict:
    thresholds = MODULE_THRESHOLDS.get(module, MODULE_THRESHOLDS["DEFAULT"])
    ungrounded_count = len(validation_result["ungrounded_tcodes"])
    
    if ungrounded_count == 0:
        return {"tier": "CONFIRMED", "response": validation_result["response"]}
    elif ungrounded_count <= thresholds["max_ungrounded"]:
        return {"tier": "REVIEW_RECOMMENDED", ...}
    else:
        return {"tier": thresholds["default_tier"], ...}
```

FI and BASIS modules touching financial postings and system configuration should have zero tolerance for ungrounded T-codes — a wrong `F-02` path in FI could corrupt ledger entries.

**Me:**
Exactly. This also helps with audit — compliance needs to know whether a response was confirmed or escalated. Let me add response logging. What's the minimum I should log for a compliance audit trail?

**Claude:**
For a compliance trail in a support context:

```python
@dataclass
class SupportResponseLog:
    timestamp: str
    query_hash: str          # SHA256 of query — no PII in logs
    module: str
    retrieved_chunk_ids: list[str]
    response_tier: str       # CONFIRMED | REVIEW_RECOMMENDED | ESCALATE
    ungrounded_tcodes: list[str]
    model_id: str            # e.g. "claude-3-sonnet-20240229"
    responded_by: str        # "bot" or "human" (if escalated and human-resolved)
    resolution_confirmed: bool  # Did engineer mark it as correct?
```

Don't log the raw query or raw response in the audit table — log the hash and keep the full text in a separate encrypted store with a shorter retention window. Audit trail needs the decision metadata, not the content.

---

**Outcome:** Citation-gated generation + post-generation T-code validation implemented. Module-specific confidence thresholds deployed (FI and BASIS set to zero tolerance). Compliance logging schema defined and integrated with the response handler.

**What I changed:**
- The T-code regex from Session 1 needed updating here too — `validate_response` was generating false positives on SAP module names (MM, FI) that aren't T-codes. Added a known-T-codes allowlist from SAP documentation as a secondary check
- Added IMG path validation alongside T-code validation — the LLM was hallucinating navigation paths even when T-codes were correct
- The compliance log schema: added `query_category` field (how-to / error / config) for analytics — Claude didn't include it but it turned out critical for tracking which query types had the highest escalation rates

---

## Session 3 — Streaming Responses for the FastAPI Layer

**Me:**
The generation step takes 4–8 seconds for complex queries. I want to stream the response token-by-token through FastAPI to the frontend. How do I handle streaming while also running post-generation validation — validation needs the full response.

**Claude:**
You can't validate while streaming in the same response — you need to choose between latency and safety. For SAP support, I'd recommend safety wins: buffer the full response, validate, then stream the validated response. Users waiting 6 seconds for a CONFIRMED answer is better than instantly receiving a hallucinated config path.

But you can give the user immediate feedback:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

async def safe_generate_stream(query: str, chunks: list):
    # Phase 1: Immediate feedback
    yield "data: {\"status\": \"retrieving\", \"message\": \"Searching SAP documentation...\"}\n\n"
    await asyncio.sleep(0)  # Yield control
    
    # Phase 2: Generate (buffered)
    yield "data: {\"status\": \"generating\", \"message\": \"Generating response...\"}\n\n"
    full_response = await generate_response(query, chunks)  # Full generation
    
    # Phase 3: Validate
    validation = validate_response(full_response, chunks)
    tiered = tier_response(validation, detect_module(query))
    
    # Phase 4: Stream validated response
    yield f"data: {{\"status\": \"validated\", \"tier\": \"{tiered['tier']}\"}}\n\n"
    
    for word in tiered["response"].split():
        yield f"data: {{\"token\": \"{word} \"}}\n\n"
        await asyncio.sleep(0.02)  # Simulate streaming feel
    
    yield "data: [DONE]\n\n"

@app.get("/support/query")
async def support_query(q: str):
    chunks = await retrieve(q)
    return StreamingResponse(
        safe_generate_stream(q, chunks),
        media_type="text/event-stream"
    )
```

The status events (retrieving → generating → validated) give the user a progress signal. The `tier` field arrives before the text, so the frontend can render a CONFIRMED badge or a warning banner before the first token appears.

**Me:**
That's a clean pattern. One issue — `asyncio.sleep(0.02)` for fake streaming feels wrong. If I'm buffering the full response anyway, is the streaming feel actually valuable here?

**Claude:**
You're right to question it — fake streaming is a UX antipattern if the underlying operation is already synchronous. For a support bot where accuracy is the primary value, I'd drop the token-by-token stream entirely and return the full validated response with a well-designed loading state.

The real streaming benefit kicks in only if your generation is genuinely incremental — e.g. if you're streaming from the LLM directly and doing lightweight inline validation. For your architecture (buffer → validate → respond), just use a standard async response with a frontend skeleton loader. Engineers care more about "is this answer safe to follow" than "watching text appear."

Revised approach:

```python
@app.post("/support/query")
async def support_query(request: QueryRequest) -> SupportResponse:
    chunks = await retrieve(request.query)
    raw = await generate_response(request.query, chunks)
    validation = validate_response(raw, chunks)
    tiered = tier_response(validation, detect_module(request.query))
    
    await log_response(request, tiered, chunks)
    
    return SupportResponse(
        tier=tiered["tier"],
        response=tiered["response"],
        warning=tiered.get("warning"),
        chunk_ids=[c["chunk_id"] for c in chunks]
    )
```

Cleaner, testable, and doesn't create false expectations about response latency.

---

**Outcome:** Dropped streaming in favour of async validated response. Cleaner architecture, easier to test, and aligned with the safety-first design principle.

**What I took from this exchange:** Claude's willingness to reverse its own earlier suggestion when I pushed back was the right call. The initial streaming implementation was technically correct but architecturally wrong for the use case. This is the kind of iterative refinement that makes AI-assisted development valuable — not accepting the first plausible answer.

---

## Overall Session Reflections

**Problems AI assistance solved quickly:**
- Hybrid retrieval architecture decision — validated my instinct toward BM25 primacy for exact-match SAP tokens and gave me the right mental model for why dense retrieval underperforms on domain-specific identifiers
- Citation-gating prompt pattern — a structural approach I hadn't seen documented for SAP specifically
- Module-threshold parameterization — turned a generic safety control into a domain-calibrated one with a single prompt exchange

**Places where I corrected or extended the output:**
- T-code regex had false positives on SAP module abbreviations — required a lookup-table fix not surfaced in the session
- IMG navigation path hallucination is a distinct failure mode from T-code hallucination — added separate validation
- Dropped the fake-streaming pattern after pushing back; Claude agreed immediately (good sign the reasoning was sound, not just confident)

**Core workflow observation:**
The bot produces better architecture when I front-load domain constraints: "SAP Notes have this specific structure," "FI module means zero tolerance." Generic prompts produce generic architecture. Specificity in → specificity out.
