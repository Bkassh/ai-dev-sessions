# ForgeIoT – Debugging & Architecture Notes
*Informal session log – April 2025*

---

## The Glue pipeline mess

Okay so this took way longer than it should have.

Background: ForgeIoT pulls sensor telemetry from three machines on the factory floor — a turning center, a milling station, a drill press. The Glue ETL job is supposed to pick up the raw data after the crawler finishes and push it through to Athena. Simple enough in theory.

The job kept completing as SUCCEEDED. No errors. No output. S3 target prefix completely empty.

I spent probably an hour and a half looking at the wrong thing — kept digging into the job script thinking there was a schema mismatch or a silent filter dropping all rows. Ran it manually a couple of times, same result. Eventually I just described the whole setup to Claude and asked what it thought.

It came back pretty quickly with something I hadn't considered: the conditional trigger was evaluating `CrawlState: SUCCEEDED` at registration time, not at runtime. By the time I activated the trigger, the crawler had already finished and reset to `READY` — so the condition never fired correctly. The job was running on essentially nothing.

Honestly a bit embarrassing in hindsight. But also the kind of thing that's easy to miss because the job *does* run, it just has no data to process.

Claude suggested switching to an EventBridge rule that fires on the crawler's CloudWatch event and calls `StartJobRun` directly. Sketched out the rule pattern and the IAM role I'd need:

```json
{
  "source": ["aws.glue"],
  "detail-type": ["Glue Crawler State Change"],
  "detail": {
    "crawlerName": ["forgeiot-raw-crawler"],
    "state": ["Succeeded"]
  }
}
```

That part I took as-is — it's a standard pattern and the JSON was correct.

The thing that was actually useful was the DLQ suggestion. I hadn't thought about what happens if the Glue job is already running when the EventBridge rule fires — it fails silently. Adding an SQS dead-letter queue on the target means I can catch those and replay them. Small thing but it would've bitten me during the backfill.

I also asked about concurrency since the crawler sometimes runs twice in quick succession during backfill periods. Claude said to set `MaxConcurrentRuns: 1` and enable job bookmarks. I took the concurrency limit but skipped the bookmarks — ForgeIoT uses time-partitioned S3 prefixes and bookmarks interact weirdly with custom partition schemes. Handled idempotency at the partition key level instead.

**Result:** Pipeline runs reliably now. Historical data flows through to Athena correctly. Took about half a day total, probably 2 hours of which was me looking in the wrong place.

---

## Multi-agent design for the Bedrock layer

This one was more of a design session than a debugging session.

The idea for ForgeIoT is that an engineer can ask something like "is Machine 101A likely to fail this week?" and get a meaningful answer. That means I need multiple things to happen: anomaly detection on recent sensor data, a look at maintenance history, and something that puts it together into a coherent response. A single agent trying to do all of that gets messy fast.

I wanted to use Bedrock AgentCore and set up a supervisor + specialist structure. The supervisor receives the natural language query, figures out what to ask, and calls specialist agents as tools. Each specialist handles one thing — anomaly detection, history lookup, report generation.

I talked through the architecture with Claude and the basic structure made sense:

```
User Query
    │
    ▼
Supervisor Agent
    ├── invoke_anomaly_agent(machine_id, time_window)
    ├── invoke_history_agent(machine_id, lookback_days)
    └── invoke_report_agent(anomaly_result, history_result)
```

One thing Claude flagged that I hadn't fully thought through: the supervisor's system prompt needs to include the machine ID mapping. The underlying data uses system IDs like `VMC_TurningCenter_01` but engineers talk about "Machine 101A." If the supervisor doesn't know that mapping, it'll either fail silently or reason about the wrong machine. Obvious once you hear it.

Claude drafted a system prompt. The structure was good — it had the machine registry table, reasoning rules for which agents to invoke in what order, explicit instructions not to fabricate sensor readings. I rewrote most of the wording though. The draft was a bit too polite and generic for an industrial tool that maintenance engineers use to make real decisions. Made it more direct.

The thing I'm most happy with from this session is the output schema. I was originally relying on prompt instructions alone to stop the model from making things up — "never fabricate readings," that kind of thing. Claude pointed out that for a safety-critical output, instructions alone aren't reliable enough. Suggested a structured JSON schema where every claim has to cite a source, and a separate `unsupported_claims` field where the model explicitly declares what it can't back up:

```python
RESPONSE_SCHEMA = {
    "summary": str,
    "claims": [
        {
            "statement": str,
            "source": str,        # which agent provided this
            "data_ref": str,      # specific data point or "null"
            "confidence": float
        }
    ],
    "recommended_action": str,
    "unsupported_claims": []
}
```

The `unsupported_claims` field is the forcing function. The model has to actively declare uncertainty rather than paper over it. That felt like the right design for something where a wrong answer could mean a machine goes down unexpectedly — or worse, an engineer greenlights something they shouldn't.

I added a confidence threshold on top: anything below 0.4 triggers a human-review flag. Claude gave me the range validation but the threshold number itself is a judgment call based on what "low confidence" actually means for a maintenance alert. That part I decided.

Also fixed an ECS credentials issue during this session. I'd been injecting AWS credentials as environment variables into the Fargate containers — worked fine initially but caused refresh problems on long-running sessions. Claude pointed out I should just use an ECS task role and let the container metadata endpoint handle rotation automatically. No `profile_name`, no explicit keys:

```python
client = boto3.client("bedrock-agent-runtime", region_name="eu-central-1")
```

That was embarrassingly simple. I'd been overcomplicating it.

---

## What I actually took away

The Glue trigger issue taught me to describe the *system* to Claude, not just the code. When I explained the full setup — crawler, trigger, job, S3 prefix — it diagnosed the problem straight away. When I was reading the job script on my own I was looking at the wrong layer entirely.

The multi-agent session confirmed something I already half-believed: prompt instructions are soft guarantees. If the output matters — and in a maintenance context it really does — you need structural validation. The schema approach is more work upfront but it's the right call.

The pattern I keep coming back to: use Claude for architecture, boilerplate, and catching things you haven't considered. Own the domain-specific constraints yourself — it doesn't know your partition scheme, your tolerance for false positives, or what a wrong answer costs in your context. That part doesn't get outsourced.
