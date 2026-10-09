# 第 5 章代码评审归档

当前结论：最终修复代码 `67bef1aedf7376db6547c028ef512cc516ef54d9` 的八项问题全部关闭，无新增 Critical/Important。后续独立证据为 688 项全量测试通过和 9/9 真实模型验收通过；只在已批准的小集内作能力结论。以下评审原文保留当时的未完成状态，现已由上述证据关闭。

## 整分支评审原文

### Strengths

- The fixed outer graph and bounded Agent subgraph follow the intended seven-way routing. Knowledge gating precedes Agent execution; complaint/chitchat paths avoid it.
- Selection drafts stay internal, while final generation uses an unbound stream.
- Stable MySQL message/action identities, ticket request digests, cancellation settlement, and conversation reservations address difficult persistence and duplicate-write cases.
- Tests exercise real MySQL constraints, official SQLite restart recovery, both relevant cross-database failure orderings, and browser behavior. Evaluation records candidly preserve failed calibration and distinguish simulation from live evidence.

### Critical

None found.

### Important

1. **Knowledge-tool citations never reach the persisted/UI citation map.**  
   **Locations:** `app/agent_runtime.py:221`, `app/workflow_service.py:144–152`, `app/workflow_service.py:72,125–126`.  
   `knowledge_tool_answer()` embeds citations in the tool result, but `execute_calls()` updates neither `citations` nor evidence numbering. Business routes therefore finish with `citations=[]`; on knowledge routes, a later retrieval can introduce another `[1]` while the public citation map still points to the initial retrieval. This breaks the required reuse of evidence/citations and can display the wrong source.  
   **Focused offline proof:** a synthetic gated `query_faq` returned source `new`, the final answer was `Answer [1]`, and graph state contained `citations=[]`.  
   **Fix:** maintain one per-turn citation registry for all admitted knowledge results, with stable numbering supplied to the model and persisted/emitted by the service. Exclude rejected tool results. Cover business-route retrieval and initial-plus-subsequent retrieval with distinct sources.

2. **The per-request context budget is lost after initial history preparation.**  
   **Locations:** `app/agent_runtime.py:24–29,94–99,156–157,212–216,232–234`; `app/workflow_service.py:172–175`.  
   Runtime admission checks only the cumulative turn allowance. `AgentLimits` receives no context limit, so admitted tool results, added evidence, and tool schemas can push later requests beyond the existing context budget. Spec §5 explicitly retains the context budget for each call.  
   **Focused offline proof:** a synthetic 18,000-character read result produced subsequent selection and final inputs of **4,542 tokens each**, exceeding the default **3,584-token input allowance**, while cumulative usage remained **9,589/12,000** and completion was reported.  
   **Fix:** propagate and enforce the configured context allowance for selection, tool-result admission, and final generation, including ephemeral evidence and schemas. Preserve whole evidence and paired messages when rejecting oversized results. Also add classification preflight against the cumulative allowance: `workflow_graph.py:61–63` currently invokes classification unconditionally, even when a valid small configured turn budget cannot cover it.

3. **SDK retries bypass the Agent call and token accounting.**  
   **Locations:** `app/model_service.py:23,47–57`; `scripts/demo_ch05.py:156`.  
   Chapter 5 selection/final generation reuse `_model` with `max_retries=1`. The runtime counts one method invocation and records only returned usage; an additional transport attempt is invisible. A timeout after provider processing can consume tokens without returned usage. Consequently the six-call/cumulative-budget guarantee does not cover actual attempts, and `provider_requests` mislabels logical operations.  
   **Focused offline proof:** real ChatOpenAI with local HTTP MockTransport returning 503 then success made **2 HTTP attempts for 1 counted selection**, returning only the successful response’s usage. No external request occurred.  
   **Fix:** use a Chapter 5 client with SDK retries disabled, preserving legacy policy separately; alternatively implement explicit attempts that each consume the budgets and record uncertain usage. Rename the demo metric unless it measures actual HTTP attempts.

4. **Required production audit logging is substantially incomplete.**  
   **Locations:** `app/workflow_log.py:7–8,21–32`; `app/workflow_graph.py:99–102`; `app/workflow_service.py:131–140`; `app/workflow_knowledge.py:83–84`.  
   The logger accepts only four event types and strips conversation/turn IDs, route, node names, tool/call IDs, duration, evidence IDs, and token usage. Runtime custom events are not persisted. Model/transport/cancellation failures bypass `log_turn` entirely. Weak-question records also lack conversation/turn identity. This fails spec §§4/8 and prevents reconstruction of a particular production turn or verification of its budget/gate behavior.  
   **Fix:** add safe correlated node/tool/gate/terminal events and measured/estimated usage, including failure/cancellation paths. Carry turn identity into weak-question records. Keep reasoning, credentials, and raw sensitive tool outputs excluded. Test interleaved conversations and an exceptional termination.

### Minor

- **Deleted history can reappear after a delayed GET.** `app/web/index.html:635–637,691–694`: opening A from B leaves `conversationId=B` until GET completion. Deleting A does not invalidate that pending load; its response can later restore deleted A and localStorage. Existing action guards still block writes. Check `deletedConversations` before applying the response and add the delayed GET/delete browser case.
- **Duplicate action kinds are accepted.** `app/agent_runtime.py:120–146`: two `create_ticket` suggestions pass validation and become separate action IDs, permitting two tickets for duplicate suggestions; two handoffs also produce duplicate controls. Enforce the specified subset semantics by rejecting duplicate kinds.
- **The workflow prompt describes grounded product lookup as simulated.** `app/prompts.py` `_WORKFLOW_SERVICE_TEMPLATE` says `query_product` returns random mock data, while `app/tools.py` replaces it with real document retrieval under `knowledge_answer`. Align the prompt with the production adapter so genuine product evidence is not falsely labelled simulated.

### Deferred finding triage

| Deferred item | Disposition |
|---|---|
| Task 4 stale ORM status transition | **Resolved.** `workflow_repository.py:147–148` refreshes after the conversation lock; the targeted race test covers it. |
| Task 6 late Context7 lookup | **Historical process deviation acknowledged.** Subsequent verification supports the implementation; it cannot change lookup chronology. No new code defect assigned. |
| Task 8 delayed history/delete race | **Retained Minor**, above. |
| Task 9 logical `provider_requests` | **Confirmed**, included in Important retry/accounting finding. |
| Task 9 custom-httpx/proxy warning | **Test-output hygiene only.** The recorded occurrence used mocked transport. Configure that test’s client explicitly to suppress the irrelevant warning; no evidence here establishes a production proxy failure. |

### Coverage and evidence

All five named Review Focus areas have meaningful implementation/tests: hidden selection content, actual dual-store failure orderings, idempotent ticket confirmation/payload conflict, stale SSE/action protection, and complete-history/reset behavior. The delayed history GET remains the UI exception above.

I accepted the existing test evidence rather than rerunning it. The later documented recovery is **36 passed, 2 deselected**, resolving the earlier MySQL setup errors. The original full run remains **656 passed / 16 obsolete-fixture failures**, with covering repair evidence; it is not a green full-suite command.

Live acceptance remains **0/10**, awaiting explicit destination/payload authorization after automatic-review rejection. Simulation does not establish provider tool-choice, streaming, or multistep capability. No live calls were made during this review.

### Declined to judge

- Multi-instance coordination, distributed transactions, and high availability: explicitly excluded; single-worker limitations are documented.
- A root terminal SQLite checkpoint preceding the MySQL final write: impossible in this orchestration; the actual child-checkpoint-first failure is tested instead.
- Retroactive compliance with the Task 6 pre-edit documentation lookup: historical fact, not repairable by a code change.
- Real-provider quality/streaming behavior and generalization beyond the small labelled sets: absent authorized live evidence.
- Production proxy behavior: the observed warning came from a mocked pipeline; it does not demonstrate an actual network failure.
- Durable persistence of the local handoff animation/greeting: the approved requirement is a local visual simulation.

### Assessment

**Ready to merge? No.**

The core architecture is sound, but citation propagation, context limits, hidden retries, and correlated logging need one combined repair wave. Final acceptance must also retain the explicit live-evidence and verification gates; the branch should not yet be described as requirements-complete.


## 同一次最终修复定向复审原文

- **Knowledge-tool citations reach the persisted/UI map** — **ADDRESSED**. `app/agent_runtime.py:179` creates a candidate source registry, rewrites complete matches/citations, and reuses source numbers. `:262` and `:273` promote only admitted results; `:275` returns the registry to graph state. `tests/test_ch05_final_repairs.py:26` covers initial/subsequent/repeated sources, and `:196` covers business/knowledge routes through SSE and isolated real MySQL history. The corrected real-shaped FAQ fixture at `tests/test_chat_api.py:398` preserves the original complete-payload/tool-allowlist checks and strengthens emitted/persisted citation and message-ID assertions.
- **Per-request context allowance and cumulative classification preflight** — **ADDRESSED**. `app/workflow_service.py:191` propagates the configured input allowance; `app/agent_runtime.py:94` includes schemas, `:96` checks selection input, `:176` checks final and cumulative allowances for whole-result admission, and `:287` checks final input. `app/workflow_graph.py:105` preflights classification input plus its output reserve. Tests at `tests/test_ch05_final_repairs.py:51`, `:67`, and `:85` cover schema allowance, rejection without citation promotion, paired results, and zero-call classification refusal.
- **Chapter 5 SDK retries and misleading request metric** — **ADDRESSED**. `app/model_service.py:36` creates the separate no-retry client used at `:54` and `:61`, preserving the legacy client. `scripts/demo_ch05.py:157` names logical requests and disclaims HTTP telemetry. `tests/test_ch05_final_repairs.py:97` checks one physical attempt for both phases with real ChatOpenAI and local MockTransport.
- **Correlated audit events, exceptional usage, cancellation, and weak-question IDs** — **ADDRESSED**. `app/workflow_log.py:18` scopes identity; `:38` admits required events and `:44` restricts usage fields. `app/workflow_graph.py:58` retains attempted counts/usage and `:69` records node duration and exceptional termination. `app/agent_runtime.py:109` and `:303` estimate failed-attempt usage. `app/workflow_knowledge.py:81` guards gate-decision logging; failure now returns empty evidence/citations, logging_error=true and low_confidence_recorded=false. Weak records inherit identity. `app/workflow_service.py:138` records completion after graph exhaustion and `:143` covers persistence errors/cancellation. Tests at `tests/test_ch05_final_repairs.py:125`, `:158`, and `:168` cover interleaving, sensitive-output exclusion, errors, weak IDs and external cancellation; `tests/test_workflow_knowledge.py:312` covers weak/strong gate-log failure.
- **Delayed GET restores deleted history** — **ADDRESSED**. `app/web/index.html:691` checks requested/returned IDs against deleted conversations before mutating the view or localStorage. `scripts/validate_ch05_actions.cjs:123` and `:128` implement the delayed-GET/delete browser case; reported action checks are 38 passed.
- **Duplicate action kinds** — **ADDRESSED**. `app/agent_runtime.py:150` rejects repeated kinds; `tests/test_ch05_final_repairs.py:119` covers duplicate tickets and handoffs.
- **Grounded product lookup incorrectly described as simulated** — **ADDRESSED** for source/prompt consistency. `app/prompts.py:40` identifies gated product/FAQ evidence and returned numbering; `app/bare_agent.py:33` explicitly labels its unconnected product demo as simulated. Real-model wording/citation-quality validation remains **DEFERRED**, live acceptance **0/10**; scripted behavior does not establish model quality.
- **Mocked-client proxy warning hygiene** — **ADDRESSED**. Explicit socket options are limited to mocks at `tests/test_extract_api.py:220`, `tests/test_intent.py:114`, `:134`, and `tests/test_ch05_final_repairs.py:106`; no production proxy change or global warning suppression.

- **New breakage in the final combined fix diff:** None remaining at Critical, Important, or Minor severity. The intermediate gate-decision logging escape at 675f43c was a production regression; the covering 67bef1a delta handles it without admitting evidence or claiming recording succeeded. The FAQ failure was an incomplete synthetic envelope and its replacement retains/strengthens behavioral assertions rather than relaxing the production contract.
- **Out-of-scope observations:** None newly identified.
- **Check — scoped static inspection:** Reviewed supplied ae3dd1a..675f43c and covering 675f43c..67bef1a packages in the same unfinished scoped review, with no restart or Git recomputation. Recovered only an output-truncated bare/model hunk. The one focused unchanged-interface check followed admitted knowledge evidence through `app/evidence.py:10`, `app/workflow_service.py:159`, and `app/workflow_graph.py:34`: source IDs match the registry contract, initial numbering is contiguous, and whole ephemeral evidence reaches runtime input accounting.
- **Check — reported verification:** Initial final-repair/graph/service run was **40 passed in 36.85s**. Full run at 675f43c was **684 passed, 2 failed in 728.05s** and remains recorded as failed. The covering report names both exact failed tests plus knowledge/service/graph/final-repair tests and reports **68 passed in 43.81s**, no warnings/skips/errors, after new gate-log regressions first failed. Browser actions **38**, citations **25**, and dependency check are reported clean. Claims are consistent with inspected tests/diffs; no tests, probes or live calls were rerun by this reviewer.
- **Fix round:** **All findings addressed, no new Critical/Important breakage** at **67bef1aedf7376db6547c028ef512cc516ef54d9**. This is one scoped repair verdict, not a new whole-branch review or completion/merge/push approval. A green full-suite run of the final code remains pending in the evidence reviewed; live validation remains deferred pending explicit destination/payload authorization, and Task 9 remains incomplete.
