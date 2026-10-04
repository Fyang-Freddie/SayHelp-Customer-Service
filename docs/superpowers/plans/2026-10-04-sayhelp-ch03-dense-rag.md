# SayHelp Chapter 3 Dense Knowledge Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the literal `query_faq` lookup with BGE-M3 dense retrieval while building a resumable MySQL/Milvus knowledge base from Markdown, FAQ, and mined conversations.

**Architecture:** MySQL `knowledge_chunks` owns text and status; Milvus `knowledge` owns only aligned INT64 IDs and 1024-dimensional vectors. Offline producers write pending rows, one indexer upserts them into Milvus and marks them done, and online retrieval maps Milvus hits back to MySQL rows without changing the tool contract.

**Tech Stack:** Python 3.14, FastAPI, SQLAlchemy 2.0, MySQL 8.4, local `SentenceTransformer("BAAI/bge-m3")` with Sentence Transformers 6.1, Milvus standalone 2.6.24 with PyMilvus 2.6, LangChain's existing chat model, APScheduler 3.11.

**Spec:** `docs/superpowers/specs/2026-10-04-sayhelp-ch03-dense-rag-design.md`

## Global Constraints

- Base this branch on `codex/ch02-function-calling`; preserve `query_faq(keyword: str)` and `{'keyword', 'matches', 'message'}` with match keys `question`, `answer`, `category`.
- Copy both Chapter 3 MySQL tables exactly from the spec appendix. Keep the Chapter 2 tables and other four tools intact.
- Embed only labeled `category`, `questions`, and `answer` text. Metadata and conversation source references never enter vectors.
- Use only BGE-M3 dense retrieval. No keyword recall, hybrid search, reranker, or silent model/database substitution.
- Milvus primary key equals MySQL `knowledge_chunks.id`; its collection name is `knowledge`, vector dimension 1024, metric COSINE, `auto_id=False`.
- Pin new dependencies to `sentence-transformers>=6.1,<7`, `pymilvus>=2.6.17,<2.7`, and `APScheduler>=3.11.3,<4`; verify the resolved versions on Python 3.14 before using their APIs.
- Never include tokens, secrets, `.env`, model weights, virtual environments, or test database credentials in Git or command output.
- Before each library/API implementation step, inspect Context7 official docs for the installed version. The design-stage checks are recorded in the spec.
- After each task, immediately append user wording, output/review, corrections, and failures/rework to `dev-notes/ch03.md`, then commit that entry with the task.

## File Structure

- `db/ch03_schema.sql`, `app/knowledge_db.py`, `app/init_knowledge_db.py`: exact DDL, mappings, and additive initialization.
- `app/knowledge_chunking.py`, `app/knowledge_ingest.py`: structure-aware Markdown chunking and repeatable MySQL ingestion.
- `app/embedding.py`, `app/vector_store.py`, `app/knowledge_index.py`, `app/build_knowledge.py`: local model adapter, Milvus adapter, pending-row indexer, and one-shot CLI.
- `app/qa_mining.py`, `app/qa_scheduler.py`, `app/qa_prompts.py`: batched extraction, global deduplication, and a single dedicated daily scheduler process.
- `app/knowledge_search.py`, `app/tools.py`, `app/main.py`, `app/prompts.py`, `app/config.py`: online retrieval and existing HTTP/tool composition.
- `compose.milvus.yaml`, `requirements.txt`, `README.md`, `tests/fixtures/ch03_cases.json`: reproducible local services, dependencies, demonstration, and labeled acceptance.

## Review Focus

1. Chinese/English punctuation and decimal points: a chunk must end on a sentence boundary and overlap complete sentences. Task 2 tests this.
2. A single table row or sentence beyond the model-safe limit: reject with location instead of truncating. Task 2 tests this.
3. Crash after Milvus upsert but before MySQL commit: rerun must keep one vector ID and mark the row done. Task 4 tests this.
4. Milvus returns a deleted, pending, or missing MySQL ID: online results must omit it without breaking rank or output shape. Task 6 tests this.
5. Similar questions with conflicting answers: global dedup must keep both for review instead of discarding one silently. Task 5 tests this.

---

### Task 1: Add exact MySQL knowledge schema and additive initializer

**Files:** Create `db/ch03_schema.sql`, `app/knowledge_db.py`, `app/init_knowledge_db.py`, `tests/test_knowledge_db.py`; modify `dev-notes/ch03.md`.

**Interfaces:** `KnowledgeChunk` and `QaExtractionStaging` map the exact DDL; `initialize_knowledge_database(database_url: str) -> None` creates both tables only when absent and rejects a partial Chapter 3 schema. Chapter 2 `initialize_database` remains valid; the new initializer calls it before Chapter 3 setup. Seed FAQ copying is left to Task 3.

- [ ] **Step 1: Write failing schema tests** named `test_ch03_schema_exactly_matches_supplied_ddl` (assert every column/type/nullability/default, two self-FKs, indexes, enums, charset, and `SET NAMES utf8mb4`) and `test_ch03_initializer_is_additive_and_repeatable` (assert empty setup and repeat succeed, partial schema raises). Use a uniquely named disposable MySQL test schema.
- [ ] **Step 2: Run RED**: `python -m pytest tests/test_knowledge_db.py -q`; expect a missing `app.knowledge_db`/initializer or table failure.
- [ ] **Step 3: Implement exact DDL and mappings**, preserving the user's table statements in the spec appendix. Serialize initialization with a MySQL advisory lock and do not drop or rewrite Chapter 2 data.
- [ ] **Step 4: Run GREEN**: the focused test with `TEST_DATABASE_URL` must pass, followed by `python -m pytest -q` and `git diff --check`.
- [ ] **Step 5: Append the four-field task note and commit** only Task 1 files plus `dev-notes/ch03.md`.

### Task 2: Structure-aware Markdown chunking

**Files:** Create `app/knowledge_chunking.py`, `tests/test_knowledge_chunking.py`, `tests/fixtures/ch03_policy.md`, `tests/fixtures/ch03_manual.md`; modify `dev-notes/ch03.md`.

**Interfaces:** `chunk_markdown(text: str, *, content_type: str, max_chars: int = 800, overlap_chars: int = 120) -> list[KnowledgeDraft]`; `KnowledgeDraft` contains `category`, `questions`, `answer`, `section_path`, `content_type`, `is_key_clause`, and source-order information. Reject oversized indivisible sentences/rows with source location.

- [ ] **Step 1: Write failing tests** named `test_heading_path_and_policy_fields`, `test_sentence_overlap_preserves_decimal_and_full_stops`, `test_table_groups_repeat_header`, and `test_indivisible_input_rejected_with_location`. Assert nested paths, parent category/current-title question, complete-sentence overlap without duplicate-only blocks, repeated table header/separator, oversize row/sentence errors, and `> [!IMPORTANT]` as the only key-clause marker.
- [ ] **Step 2: Run RED**: `python -m pytest tests/test_knowledge_chunking.py -q`; expect missing module or assertions.
- [ ] **Step 3: Implement parser and splitter** with Markdown table detection before ordinary prose splitting. Never hard-cut a sentence or row; preserve deterministic source order.
- [ ] **Step 4: Run GREEN** focused and full tests, then run the labeled fixtures and inspect every produced chunk boundary and table header.
- [ ] **Step 5: Append the four-field task note and commit** Task 2 files.

### Task 3: Repeatable FAQ and Markdown ingestion into pending MySQL rows

**Files:** Create `app/knowledge_ingest.py`, `tests/test_knowledge_ingest.py`; modify `dev-notes/ch03.md`. Task 4 owns CLI composition.

**Interfaces:** `ingest_faq(session_factory) -> list[int]` copies existing `Faq` rows with real `questions`; `ingest_markdown(session_factory, path, content_type) -> list[int]` consumes Task 2 drafts. Both insert `pending` rows and return stable IDs. Write adjacent pointers after flushing IDs within the same transaction. Serialize ingestion with a MySQL advisory lock and normalize complete authoritative fields for unchanged-input repeatability.

- [ ] **Step 1: Write failing MySQL tests** named `test_faq_seed_ingests_once_with_real_question`, `test_markdown_ingestion_links_chunks_atomically`, and `test_failed_ingestion_rolls_back_links`. Assert the `运费` row fields, equal IDs/counts after two runs, reciprocal adjacency, and no rows after injected failure. Task 4 tests the vector text itself.
- [ ] **Step 2: Run RED**: `python -m pytest tests/test_knowledge_ingest.py -q`; expect missing interface or wrong rows.
- [ ] **Step 3: Implement short transactional ingestion**, no vector calls, no automatic replacement of changed documents, and no deletion of Chapter 2 FAQ.
- [ ] **Step 4: Run GREEN** focused and full tests plus `git diff --check`.
- [ ] **Step 5: Append the four-field task note and commit** Task 3 files.

### Task 4: BGE-M3 embedding, Milvus schema, and resumable indexing

**Files:** Create `app/embedding.py`, `app/vector_store.py`, `app/knowledge_index.py`, `app/build_knowledge.py`, `compose.milvus.yaml`, `tests/test_knowledge_index.py`; modify `requirements.txt`, `app/config.py`, `README.md`, `dev-notes/ch03.md`.

**Interfaces:** `BgeM3Embedder.encode(texts: list[str]) -> list[list[float]]` returns 1024 finite floats per item; `knowledge_text(row) -> str` joins only the three vectorized fields; `MilvusKnowledgeStore.ensure_collection()`, `.upsert(id: int, vector: list[float]) -> int`, `.search(vector: list[float], limit: int) -> list[SearchHit]`; `index_pending(session_factory, embedder, store, limit: int) -> int` returns completed count. `MILVUS_URI` defaults to `http://127.0.0.1:19530`; the model ID stays fixed at `BAAI/bge-m3`, with optional `BGE_CACHE_DIR`. `python -m app.build_knowledge` initializes schema, ingests FAQ/Markdown inputs, and resumes pending rows.

- [ ] **Step 1: Write failing tests** named `test_vector_text_has_only_three_labeled_fields`, `test_collection_is_1024d_cosine_with_mysql_primary_key`, `test_failed_vector_write_stays_pending`, and `test_rerun_repairs_crash_after_upsert`. Assert field exclusion, finite 1024-vector validation, pending-before-I/O order, one Milvus ID after a crash/rerun, `vector_id == str(id)`, `done`, and skip of already-done rows.
- [ ] **Step 2: Run RED**: `python -m pytest tests/test_knowledge_index.py -q`; expect missing adapter/indexer.
- [ ] **Step 3: Implement local model lazy loading**, pin compatible Sentence Transformers/PyMilvus versions, vendor a pinned official Milvus standalone Compose layout with persistent volumes and loopback ports, and add safe config for Milvus URI/model cache. Use PyMilvus `MilvusClient.upsert` by the MySQL ID; validate returned ID before status update. Do not retain a MySQL transaction during embedding or Milvus I/O.
- [ ] **Step 4: Run GREEN** focused fake-adapter tests, a real Milvus + MySQL integration test, complete regression suite, and `docker compose -f compose.milvus.yaml config --quiet`.
- [ ] **Step 5: Append the four-field task note and commit** Task 4 files.

### Task 5: Batch conversation QA extraction, global dedup, and daily schedule

**Files:** Create `app/qa_mining.py`, `app/qa_prompts.py`, `app/qa_scheduler.py`, `tests/test_qa_mining.py`, `tests/fixtures/ch03_qa_cases.json`; modify `requirements.txt`, `README.md`, `dev-notes/ch03.md`.

**Interfaces:** `mine_conversations(session_factory, llm, *, batch_size: int = 20) -> MiningSummary` reads completed conversation turns, stages validated QA with `batch_no`/`source_ref`, globally deduplicates staged rows, and inserts kept `pending` knowledge with `category='历史客服对话'` in one transaction with status updates. `run_scheduled_mining(...)` uses APScheduler 3.11 `BlockingScheduler`, daily 02:00 `Asia/Shanghai`, one instance, coalescing, and calls the same one-shot mining+indexing path. Add a CLI `--once` mode for deterministic execution.

- [ ] **Step 1: Write failing code tests** named `test_mining_batches_complete_conversations_only`, `test_staging_precedes_global_dedup`, `test_conflicting_answers_survive_dedup`, `test_mining_rerun_is_idempotent`, and `test_scheduler_runs_one_daily_job`. Assert source order, no tool payloads/incomplete turns, `extracted` before final statuses, no false merge of conflicting answers, atomic knowledge/status updates, stable row count on rerun, and one scheduled invocation.
- [ ] **Step 2: Run RED**: `python -m pytest tests/test_qa_mining.py -q`; expect missing module/behavior.
- [ ] **Step 3: Implement extraction and scheduler** using the configured chat model, bounded prompts, validated JSON, and existing DB sessions. Use normalized exact matches then BGE-M3 near-duplicate candidates; only discard a candidate when both question and answer are equivalent, and use labeled fixtures to choose thresholds. Avoid marking a batch complete before staging writes commit.
- [ ] **Step 4: Run GREEN** code tests and full suite. For the pure extraction prompt and dedup judgment, run the labeled `ch03_qa_cases.json` evaluation once against the configured model and record observed precision/false merges; adjust and rerun only for concrete failures.
- [ ] **Step 5: Append the four-field task note and commit** Task 5 files.

### Task 6: Replace `query_faq` internals without changing its contract

**Files:** Create `app/knowledge_search.py`, `tests/test_knowledge_search.py`; modify `app/tools.py`, `app/chat_service.py`, `app/main.py`, `app/prompts.py`, `app/config.py`, `tests/test_tools.py`, `tests/test_chat_service.py`, `tests/test_chat_api.py`, `tests/fixtures/ch03_cases.json`, `dev-notes/ch03.md`.

**Interfaces:** `KnowledgeSearch.search(keyword: str, *, limit: int = 5) -> list[KnowledgeMatch]` embeds the query, searches Milvus, and loads done MySQL rows in rank order. `build_tools(repository, conversation_id, knowledge_search: KnowledgeSearch)` and `ChatService(..., knowledge_search: KnowledgeSearch)` require an injected search adapter; tests inject a fake. The production composition root creates the real adapter. Initial `KNOWLEDGE_MIN_SCORE=0.55` is tunable only against the labeled set. The old `Repository.find_faq` can remain for legacy database tests but is never called by the online tool. `query_faq` preserves all result keys and match keys; unavailable dependencies produce empty matches plus a safe temporary-unavailable message.

- [ ] **Step 1: Write failing tests** named `test_query_faq_contract_and_semantic_postage_hit`, `test_search_preserves_rank_and_skips_stale_rows`, `test_low_score_or_backend_error_is_safe_miss`, and `test_other_tools_and_sse_unchanged`. Assert the exact tool field sets, `邮费` to `运费` result, Top-K/rank, omission of deleted/pending/missing IDs, safe empty matches on failure, and the prior four tools/SSE order.
- [ ] **Step 2: Run RED**: focused retrieval/tool/API tests; expect old SQL behavior or missing interface.
- [ ] **Step 3: Implement retrieval composition** with one lazy BGE-M3 instance per API process. Query Milvus only, hydrate from MySQL, filter by a labeled-evaluation score threshold, update tool description and service prompt for paraphrases and grounded answers, and keep the chapter 2 HTTP contract.
- [ ] **Step 4: Run GREEN** focused and full tests; run labeled ch03 retrieval cases, including the exact Chapter 2 `邮费` miss now expected to hit, plus unrelated-query guardrails.
- [ ] **Step 5: Append the four-field task note and commit** Task 6 files.

### Task 7: End-to-end acceptance and delivery

**Files:** Modify `README.md`, `tests/fixtures/ch03_cases.json`, `dev-notes/ch03.md`; repair only defects found by acceptance.

**Interfaces:** Document runnable install/start/ingest/retry/scheduler/chat/test commands and observed results. No new product interface is planned.

- [ ] **Step 1: Start MySQL and Milvus** from committed Compose definitions, initialize schema, run the one-shot build twice, and inspect MySQL pending/done counts plus Milvus IDs.
- [ ] **Step 2: Deliberately interrupt indexing** after a MySQL pending commit and after a Milvus upsert in separate disposable runs; rerun and verify no missing or duplicate vector IDs.
- [ ] **Step 3: Run the labeled live chat** with “邮费是多少”; confirm `query_faq`, the shipping-fee source row, final grounded answer, unchanged SSE/tool output shape, and correct MySQL/Milvus evidence.
- [ ] **Step 4: Run the full deterministic suite** with disposable MySQL and real Milvus, `docker compose config`, `git diff --check`, and a credential/staged-file audit; record exact counts and limitations.
- [ ] **Step 5: Update README and append the four-field acceptance note**, then commit verified acceptance documentation.

## Final review and finish

- [ ] Request a fresh code review of the whole `codex/ch03-rag` diff against the approved spec. Fix actionable findings with focused tests; append a four-field review conclusion immediately.
- [ ] Run required verification again, append a four-field finish entry, commit it, and push the reviewed branch to the configured GitHub remote. Provide demonstration commands, exact test results, and the absolute `dev-notes/ch03.md` path.
