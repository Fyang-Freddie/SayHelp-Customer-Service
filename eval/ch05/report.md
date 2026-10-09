# Chapter 5 acceptance evidence (current status: 2026-10-09)

Current status: **all8 scoped-review findings ADDRESSED; current code67bef1a verified by688 passed in764.44s, exit0,0 failures/errors/skips/warnings, owned Milvus cleanup confirmed**. Only explicit live destination/payload authorization and real-provider acceptance remain pending. The reviewed feature branch was pushed and its remote SHA verified at cb4054361fb735d2e1fb833a24fa2dc8cce8f4b4; this delivery record is a later documentation-only commit. Overall Task9 is not marked complete. Original protected fingerprints remain4/4 unchanged (controller reverified).

The following chronological records preserve the results and pending statements as they stood on their stated dates. Earlier deferred UI/review/full-suite statements are superseded by the current status above and the final verification section.

Historical status at the initial2026-10-08 checkpoint: **incomplete — live provider acceptance awaits direct user authorization; controller review/delivery pending**. Simulation, local regression and browser fixtures are separate evidence. The live counter is **0 / 10 complete chat requests**, counting failures; none was sent. The planned small set is seven workflow cases plus one bare multi-step (8 turns).

## Demo scope

`python scripts/demo_ch05.py --mode bare|workflow --case logistics|multi-step|policy|complaint|chitchat|weak-evidence|missing-info` works without ignored helpers. All seven questions were labelled in `acceptance_cases.json` before execution. Default simulation uses a scripted model, fixed evidence, existing seeded random business tools and temporary official SQLite checkpoints. `--live` uses configured ModelService and existing read-only retrieval. Order/logistics remain simulated even with a live model. There are no CLI history or ticket writes; actual independent confirmation is covered by MySQL/API and browser tests.

The bare loop has no outer classifier/knowledge gate. Workflow routes deterministically and gates before Agent output. Multi-step asks for logistics only when the preceding order result says shipped/completed. Seed5 makes the original order mock return shipped; a subsequent model decision consumes that returned result. Scripted success is not measured model capability.

Automatic review rejected the first live command before execution: retrieved corpus content/prompts would be sent to an unspecified provider without sufficiently explicit destination/payload authorization. Safe metadata inspection found **api.deepseek.com / deepseek-v4-flash**. Controller asked the user to authorize authored questions, prompts, simulated tool results and read-only snippets from the existing ch04 document/FAQ corpus, with no history/tickets and at most10 turns. No answer yet; generic continuation did not override the pending question and no indirect retry occurred.

## Commands and results

Python below is the worktree `.venv/Scripts/python.exe`. The ignored test helpers insert repository root and load only TEST_DATABASE_URL/MILVUS_URI/BGE_CACHE_DIR. Their use records execution; final demos do not depend on them.

| Actual command | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest tests/test_demo_ch05.py -q --tb=short` before implementation |14 failed:missing script; earlier sandbox14 setup errors were not counted as RED|
| Same after implementation |14 passed in24.28s;0 live calls|
| `.venv/Scripts/python.exe .superpowers/sdd/2026-10-07-sayhelp-ch05-workflow-agent/run_tests.py tests/test_rag_api.py tests/test_rag_latency.py -q --tb=short` before migration |10 failed,4 passed in25.26s:obsolete injection/main symbols|
| Same after migration |1 failed,13 passed in29.60s:valid product filter exposed real serialization issue|
| Single citation test before fix |1 failed in4.01s:session/error instead of token/citations|
| `.venv/Scripts/python.exe .superpowers/sdd/2026-10-07-sayhelp-ch05-workflow-agent/run_tests.py tests/test_rag_api.py tests/test_rag_latency.py tests/test_rag_chat.py -q --tb=short` after fix |28 passed in56.15s,0 skips; independent old ChatService retained|
| `.venv/Scripts/python.exe .superpowers/sdd/2026-10-07-sayhelp-ch05-workflow-agent/run_full_task9.py tests/test_knowledge_index.py::test_real_milvus_mysql_recovery -q --tb=short` |1 passed in5.24s;owned database created/removed|
| `.venv/Scripts/python.exe .superpowers/sdd/2026-10-07-sayhelp-ch05-workflow-agent/run_full_task9.py -q --tb=short` |**16 failed,656 passed,0 skipped in658.50s**;all failures obsolete extraction test injection;owned Milvus database removed|
| `.venv/Scripts/python.exe .superpowers/sdd/2026-10-07-sayhelp-ch05-workflow-agent/run_tests.py tests/test_extract_api.py tests/test_chat_api.py tests/test_rag_api.py -q --tb=short` after test-only extraction migration |18 passed,36 setup errors in91.61s:local MySQL connection refused after full-suite completion|
| `.venv/Scripts/python.exe -m pip check` |exit0:No broken requirements found|
| bundled Node `scripts/validate_ch05_actions.cjs` |exit0:36 passed|
| bundled Node `scripts/validate_ch04_citations.cjs` |exit0:25 passed,render_ms10;ch04 generated artifact restored|
| `.venv/Scripts/python.exe .superpowers/sdd/2026-10-07-sayhelp-ch05-workflow-agent/run_simulation_task9.py` |exit0:14 simulation results saved,0 live requests|

Exact browser environment: `PLAYWRIGHT_MODULE=C:/Users/YF202/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright`; executable `C:/Users/YF202/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe`. These are actual headless Edge tests against authored localhost HTTP/SSE fixtures, not a real-provider end-to-end run.

Full-suite settings: TEST_BGE_M3=1, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, real cached BGE models and torch4 threads. Existing MySQL fixture creates/cleans random databases. Approved TEST_MILVUS_DATABASE extension accepts only sayhelp_test_<32hex>; runner verifies UUID absence, creates/owns it, injects MilvusClient(db_name=...), retains empty-knowledge guard, and drops only its verified owned database. Production default collection was not deleted/rebuilt. First isolated runner failed before connecting because MILVUS_URI was absent, then used the existing default local endpoint successfully. No skipped real-model test was counted as passed.

Full-suite failures were all16 in `tests/test_extract_api.py`:three basic validation cases,valid structured result,four malformed-result cases,JSON-mode secret-safe failure,and seven real ChatOpenAI-pipeline JSON validation parameters. Each failed before route execution with `TypeError: create_app() got an unexpected keyword argument 'knowledge_search'`. The test-only unused keyword and import were removed; extraction assertions remain. No runtime compatibility branch was added. The immediate correction was followed by affected API checks, not a duplicate expensive full suite. A final clean full run remains required after whole-branch review/fix convergence; the original full command is not called green.

An additional extraction-only check (`.venv/Scripts/python.exe -m pytest tests/test_extract_api.py -q --tb=short`) returned **16 passed in1.51s**, before controller instruction to avoid repeating for counts arrived. The combined36 errors all occurred creating temporary MySQL databases with WinError10061/connection refused; no production restart was attempted. Thus all16 obsolete-injection failures have covering GREEN evidence, but a current full-green integrated run is not claimed.

Captured log warning (no pytest warning summary): `langchain-openai injected a custom httpx transport to apply http_socket_options, which disables httpx's proxy auto-detection (system proxy configuration detected). Set LANGCHAIN_OPENAI_TCP_KEEPALIVE=0 or pass http_socket_options=() to restore default proxy behavior, or supply openai_proxy / your own http_client / http_async_client to take full control.` This occurred in the mocked upstream pipeline test; it was not a real provider request.

## Simulation observations

Counts are real runtime observations with scripted model inputs. Classification is separate from the six-call Agent budget. Tokens here are estimated, never paid usage. Node count includes parent agent and child nodes; bare has no graph.

|Case|Bare Agent/tools|Workflow Agent/tools/nodes|Workflow estimated tokens|
|---|---:|---:|---:|
|logistics|3/1|3/1/9|2706|
|multi-step|4/2|4/2/11|3868|
|policy|3/1|2/0/9|1904|
|complaint|3/0|0/0/5|657|
|chitchat|2/0|0/0/5|643|
|weak-evidence|2/0|0/0/7|666|
|missing-info|2/0|2/0/7|1675|

Each workflow case performs one scripted classification. Agent finals are two scripted chunks; fixed routes are not described as provider streams. Detailed answers, events, nodes, suggestions and usage are in acceptance_results.json. Live capability, usage and stream behavior remain **unmeasured**.

## Failure contracts and review focus

- Task1 selection text before tool calls remains hidden; final ModelService stream is unbound. Budgets reserve final input/output, reject invalid/reused tool IDs, preserve call pairing, bound actual retries and stop repeated failures. Oversized tool results become bounded paired errors rather than silently clipped evidence.
- Task6 tests cover MySQL final success then SQLite failure without completion/replay; actual child SQLite completed checkpoint before MySQL final failure; incomplete/cancelled history omission; bounded complete history reconciliation; deleted conversation during final stream; early first final token before stream end. Parent terminal SQLite necessarily follows MySQL in this architecture; no fictitious root-terminal-first scenario is claimed.
- Task7 real MySQL/API tests cover same-action retries, payload conflicts, cross-owner/deleted/handoff rejection, active reservation through cancelled writes and unknown-result503 with no automatic retry. Browser36 covers both button orders, ignored suggestions, cancelled confirmation, stable retry identity and stale SSE.
- Task5 resets evidence/suggestions each turn and gates before Agent events; original offline RAG/ChatService contracts remain separately tested. Runtime/SQLite/MySQL failure fixtures use synthetic models and are not live-provider demonstrations.

## Earlier evaluations preserved, not rerun

Unchanged intent42 and knowledge calibration were not repeated. Intent core35/35,boundary7/7,mandatory20/20. Initial raw-query threshold5.026695251464844 accepted known4/6 with unknown0/6 false accepts; first held-out acceptance **failed**. Original knowledge_results.json and knowledge_results.initial.json retain that failure.

After aggregate failure, controller approved calibration-only midpoint for separated labels, leaving ch04 policy untouched. Old held-out set became development feedback. Frozen separable-label-gap-midpoint-v1 uses midpoint2.217834621667862 between highest-negative -0.5910260081291199 and lowest-positive5.026695251464844. Fresh prelabelled confirmatory12 accepted known6/6,unknown0/6 false accepts,0 missing-gold/service errors,accepted=true. Original initial artifacts and first fresh confirmation remain unchanged. Small-set results do not establish general sufficiency.

## Versions and remaining limits

Python3.14.5,FastAPI0.142.2,SQLAlchemy2.0.54,langchain-core1.6.6,langchain-openai1.6.7,LangGraph1.2.2,SQLite saver3.1.1,pymilvus2.6.17,pytest9.1.1. Worktree environment reads original packages without modifying them and adds only constrained dependencies. Lockfile/pipcheck document this environment.

Task9 runtime repair: Pydantic filter serialization included None defaults, then graph reconstruction rejected them as explicit nulls. Valid-filter HTTP RED reproduced the issue. Context7 confirmed model_dump(exclude_none=True) before the one-line fix; targeted28 passed,invalid null/empty/unknown still rejected before writes. Context7 also verified Milvus database isolation and graph streaming before use.

Outstanding: live authorization/acceptance; independent controller Task9 and whole-branch reviews; finish/push/remote SHA verification. No deployment or production service restart performed.

Deferred UI Minor remains for final triage: open history A from B, delete A before its delayed GET returns; response may redisplay deleted A snapshot, although currentAction blocks writes. Historical process Minor: Task6 Session.refresh-specific docs check happened after first edit; subsequent official API/race verification does not retroactively satisfy prelookup timing. Single-worker SQLite/locks, no distributed transaction, local weak-question JSONL with no automatic retention remain documented limits.

Four protected original dirty-file SHA256 fingerprints matched after Task9 edits; contents were not printed/staged. Credentials,runtime files,venv and ignored SDD helpers are excluded. Report creation was interrupted once by account-limit approval failure (not unsafe-action ruling); resumed using existing full-suite output without repeating it.

Dependency availability after interruption: read-only Docker status failed because `//./pipe/dockerDesktopLinuxEngine` does not exist. No listener on3307 and no Docker-named processes were observed. Existing executables were found at `E:/docker/Docker Desktop.exe` and `E:/docker/resources/Docker Desktop.exe`; no startup/restart performed. Controller owns dependency recovery and final green integration evidence.


## Post-recovery verification and checkpoint review

Controller recovered the installed DockerDesktop hidden and ran `docker compose start --wait --wait-timeout60`; existing `contact_agent-mysql-1` became Healthy, without container recreation/configuration changes. Worker ran exactly the36 previously environment-failed tests:

`.venv/Scripts/python.exe .superpowers/sdd/2026-10-07-sayhelp-ch05-workflow-agent/run_tests.py tests/test_chat_api.py tests/test_rag_api.py -k 'not test_app_requires_database_url_or_injected_storage and not test_model_service_forwards_nonempty_text_chunks_with_configured_model' -q --tb=short`

Result: **36 passed,2 deselected,0 failed/errors/skipped in86.41s**. The two previously passing non-DB chat tests were deliberately excluded and the16 extraction tests were not rerun. This supersedes the connection-unavailable condition for these cases while preserving the original18passed/36setup-errors record. No production deployment or data rebuild occurred.

Controller independent checkpoint review of f3bc87e found no Critical/Important code defects. Overall gate remains incomplete until live authorization/acceptance and final clean full-suite evidence after review convergence. Two deferred Minors: demo `provider_requests` counts logical method operations, while SDK max_retries=1 can add physical HTTP attempts; custom-httpx/proxy warning is in the mocked pipeline test configuration. Future real evidence must distinguish complete chat turns, logical model operations and physical retries. No extra Minor fix loop started. Live remains0/10.

## Final review repairs and full-run findings (2026-10-09)
The combined review repair code was frozen at675f43c. Browser validation passed38 action checks and25 citation checks (render8ms), and pip check reported no broken requirements. Request context now uses the configured input allowance, tool citations share stable per-turn numbers, chapter5 SDK retries are disabled separately from legacy calls, and safe correlated audit events cover exceptions/cancellation. Prompt source-consistency wording was corrected; actual model quality remains pending explicit live authorization, not established by scripted cases.

The final guarded full-suite command `.venv/Scripts/python.exe .superpowers/sdd/2026-10-07-sayhelp-ch05-workflow-agent/run_full_task9.py -q --tb=short` returned **684 passed,2 failed in728.05s**, no skips or warnings summary. TEST_BGE_M3=1 and cached offline models were used. The owned temporary Milvus database was confirmed removed; production databases were not recreated. This failed result remains part of the record and is not relabelled green.

The FAQ API failure used an incomplete synthetic gate payload with only citation n and no chunk_id/evidence id. Its realistic fixture now preserves the full payload assertions and adds matching citation SSE/history/message-ID checks. The weak-log API failure was a production regression: the new gate_decision write preceded the existing exception guard. The write now fails closed with empty evidence/citations, logging_error=true and low_confidence_recorded=false; weak/strong gate-log failures were RED2failed/24deselected/0.83s before the fix.

Exact failing tests plus tests/test_workflow_knowledge.py, tests/test_workflow_service.py, tests/test_workflow_graph.py, tests/test_ch05_final_repairs.py returned **68 passed in43.81s**, no failures/errors/skips/warnings. No full-suite repeat has yet run; controller coordinates final scoped review and the justified final green run following this production repair. Live0/10 remains pending and Task9 is not declared fully accepted.

Controller also independently ran workflow/multi-step and bare/logistics CLI examples on current code: workflow4model/2tool/2stream chunks/11nodes with sequential order then logistics, bare3model/1tool/2chunks, both zero ticket writes and explicitly simulation/zero external requests. These validate demonstration mechanics, not real-provider quality. Earlier historical provider_requests fields remain untouched; current CLI calls this logical_model_requests and explicitly does not claim HTTP telemetry.

## Final current-code full verification (2026-10-09)
Following the actual production logging-failure repair, the controller authorized one final full run on frozen code67bef1aedf7376db6547c028ef512cc516ef54d9. Command: `.venv/Scripts/python.exe .superpowers/sdd/2026-10-07-sayhelp-ch05-workflow-agent/run_full_task9.py -q --tb=short`. Result: **688 passed in764.44s (12:44), exit0; 0 failures,0 errors,0 skips,0 warnings**. The runner printed `Owned disposable Milvus database removed`. The same guarded owned temporary MySQL/Milvus isolation, TEST_BGE_M3=1 and cached offline models were used. No concurrent full run or code changes occurred. Earlier684passed/2failed remains preserved above; this run is separately labelled current-code evidence.

The single scoped final repair review concluded **all8 findings ADDRESSED**, with no new Critical or Important findings (SDD final-fix-review.md). Browser actions38/citations25 and pip check evidence remain as previously recorded; they were not rerun after this documentation step. A late attempt to read suite progress was not executed because automatic approval review could not complete under the account usage limit; this was not an unsafe-action ruling. After the user resumed, the original process was reaped and confirmed exit0 without restarting tests.

This evidence closes the code-repair verification step. **Live calls remain0/10 and explicit destination/payload authorization is pending. Task9 overall acceptance and real-provider/prompt quality are not marked complete.** No provider calls, new implementation, push, merge or worktree cleanup were performed for this record update.
