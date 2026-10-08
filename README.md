# Support Desk Crew

A multi-agent customer support desk, built in seven phases for learning.
See [PLAN.md](PLAN.md) for the full design and the phase-by-phase path.

## Quickstart

This project is managed with [uv](https://docs.astral.sh/uv/). No manual venv, no `pip`.

```powershell
uv sync                      # creates .venv, installs deps, pins the lockfile
copy .env.example .env       # only needed from Phase 1 onward

uv run support-desk-verify
uv run pytest
```

`verify` rebuilds `.local/support_desk.db` from `data/seed.sql` and checks that the golden
ticket set, the policy corpus and the database all agree with each other. Delete `.local/`
at any time; everything regenerates.

### Working with uv

| Task | Command |
|---|---|
| Install / refresh the environment | `uv sync` |
| Run anything in the environment | `uv run <cmd>` |
| Add a runtime dependency | `uv add <pkg>` |
| Add a dev-only dependency | `uv add --dev <pkg>` |
| Add to a phase extra | `uv add --optional agents <pkg>` |
| Lint / format | `uv run ruff check .` / `uv run ruff format .` |
| Upgrade the lockfile | `uv lock --upgrade` |

Heavy dependencies are gated behind extras so each phase stays fast to install:

| Phase | Command |
|---|---|
| 0 | `uv sync` |
| 1–3 | `uv sync --extra agents` |
| 4–5 | `uv sync --extra agents --extra rag` |
| 6–7 | `uv sync --all-extras` |

`uv.lock` is committed on purpose — it is what makes your eval numbers reproducible across
machines and across phases.

### Behind a corporate proxy

If `uv sync` fails with `invalid peer certificate: UnknownIssuer`, your network is
inspecting TLS. Tell uv to trust the OS certificate store:

```powershell
setx UV_SYSTEM_CERTS 1     # once, then reopen the terminal
```

Or pass `--system-certs` on each command.

### If live model calls are blocked

Some corporate proxies run a CASB/DLP policy that blocks POST requests to AI chat-completion
endpoints (OpenRouter, OpenAI, Anthropic) and model downloads (Hugging Face) specifically —
even when plain `GET` to the same domains succeeds. You'll see an `OpenAIPermissionDeniedError`
wrapping an HTML "justification required" page, not a connection error.

This is a policy decision, not a bug — don't try to route around it. Options, in order:
1. Ask IT to allow-list the specific endpoint you need.
2. Ask IT (or someone with admin rights) to install [Ollama](https://ollama.com) for a fully
   local model — no outbound AI traffic at all.
3. Ask around for an internal/approved LLM gateway other tools in your org already use.

Until one of those lands, agent code can still be written and unit-tested offline: every
agent accepts an injectable `llm` client (see `tests/test_triage.py`), so retry/fallback
logic is fully verified without a live endpoint. Only the golden-set accuracy scorer
(`support-desk-score-triage`) needs real access.

## Status

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Foundations — seed data, policy corpus, golden set, budget meter | **done** |
| 1 | One agent, no tools — triage with structured output | **code-complete, live scoring blocked** ([details](#if-live-model-calls-are-blocked)) |
| 2 | Tools — parameterised account lookups | **done** — tool-calling loop + deterministic fallback, fully offline |
| 3 | Multiple agents and a graph | **code-complete, action accuracy blocked** ([details](#if-live-model-calls-are-blocked)) |
| 4 | RAG and parallelism | **done** — 100% topical retrieval recall, branches verified concurrent |
| 5 | Guardrails, critic, budgets | **done** — zero hard-rule violations, adversarial ticket refused |
| 6 | Human in the loop | **done** — survives a real process kill, no high-risk action escapes |
| 7 | Evaluation, observability, refactor | todo |

## Scorecard

Update at the end of every phase.

| Metric | Target | Current |
|---|---|---|
| Routing accuracy | ≥ 90% | — |
| Action accuracy | ≥ 85% | — |
| Groundedness (citations valid) | ≥ 95% | — |
| Hard-rule violations | 0 | — |
| High-risk actions escaping approval | 0 | — |
| Mean cost / ticket | ≤ $0.05 | — |
| p95 latency | < 20s | — |

## Ground rules

- `config.AS_OF` is a fixed date. Never replace it with `date.today()`.
- The model never writes SQL. Phase 2 exposes parameterised functions only.
- Guardrails are Python, not prompts.
- No phase starts until the previous phase's acceptance criteria pass.
- Dependencies change only via `uv add` / `uv lock`, never by hand-editing the lockfile.
