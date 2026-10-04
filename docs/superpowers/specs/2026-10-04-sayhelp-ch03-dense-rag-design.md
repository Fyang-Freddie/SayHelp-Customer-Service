# Chapter 3: BGE-M3 dense knowledge retrieval

## Outcome and constraints

Upgrade the existing Chapter 2 `query_faq(keyword: str)` implementation from literal SQL `LIKE` to dense semantic retrieval. Keep the tool name, input schema, and result keys `keyword`, `matches`, and `message`; each match keeps `question`, `answer`, and `category`. The acceptance question “邮费是多少” must retrieve the shipping-fee answer and lead to a grounded response. A deliberately interrupted indexing run must recover every MySQL `pending` chunk on rerun.

Use BGE-M3 through local `SentenceTransformer("BAAI/bge-m3").encode()`. Hugging Face supplies the public model files; inference runs locally. No Hugging Face token is required or committed. Milvus stores 1024-dimensional dense vectors; MySQL is the authoritative source of text and metadata. This chapter has one dense retrieval path only: no keyword recall, hybrid search, or reranking.

The Chapter 2 implementation branch `codex/ch02-function-calling` is the starting point because `main` currently contains only its design and plan. Preserve the other four tools, SSE events, after-sales extraction, and message persistence.

## Chosen architecture and alternatives

The API process holds one lazily initialized BGE-M3 model instance and one Milvus client; the offline command loads its own instance and uses the same local model cache. Run the demo API with one worker to avoid multiple copies of the model. A separate embedding service could centralize model memory but adds another deployment and failure boundary. Calling the hosted Hugging Face `sentence_similarity` task returns pairwise scores, not vectors, so it cannot populate the fixed Milvus collection.

Split responsibilities into a Markdown/FAQ chunk producer, a conversation QA extraction job, a MySQL knowledge repository, an idempotent Milvus indexer, an embedding adapter, and a retrieval adapter. Keep the online tool a thin mapping from retrieved knowledge rows into its existing result shape. A dedicated scheduler process invokes the same resumable offline command on a configurable daily schedule; a one-shot CLI command remains available for setup, recovery, and demonstration. The scheduler does not run in every web worker.

## Knowledge records and exact schema

Apply the user's `knowledge_chunks` and `qa_extraction_staging` MySQL DDL exactly, including `SET NAMES utf8mb4`, column types, indexes, foreign keys, enum values, and comments. Add SQLAlchemy mappings consistent with that DDL. Do not alter the Chapter 2 `faq` table or discard it. Copy its seeded rows into `knowledge_chunks` for Chapter 3 retrieval; repeated ingestion must not create duplicate knowledge rows.

For each chunk, `category`, `questions`, and `answer` are joined in a stable labeled format and **only those fields** are embedded. Real FAQ and extracted QA use actual customer question wording in `questions`. Policy/manual prose uses the current section title as `questions` and the parent title path as `category`. `section_path`, `content_type`, `is_key_clause`, `prev_chunk_id`, and `next_chunk_id` stay in MySQL for traceability and adjacency; they do not enter the embedded text or Milvus scalar fields. Milvus collection `knowledge` stores a non-auto-generated INT64 primary key `id` equal to `knowledge_chunks.id` and one FLOAT_VECTOR field of dimension 1024, indexed for cosine similarity. `vector_id` is the string representation of that same ID.

## Offline document processing

Parse Markdown heading levels into a hierarchy before splitting body text. Preserve heading context even when a section becomes several chunks. Keep sentences whole: recursively split oversized paragraphs at paragraph, then sentence boundaries (`。！？.!?`), choosing the nearest valid sentence end within the size target. Use 800 characters per chunk and 120 characters of complete-sentence overlap as initial defaults. If one sentence cannot fit within the BGE-M3 input limit, reject it with a source-location error; never emit half a sentence. Avoid empty or duplicate-only chunks. For a large Markdown table, repeat its header and separator in each row-group chunk; never split one table row across chunks. Record source order through `prev_chunk_id`/`next_chunk_id` after MySQL IDs are assigned. Reject malformed or oversized rows with a source-location error rather than silently truncating.

Chunk limits and overlap are configurable and verified on labeled policy, FAQ, manual, and table examples. Ingestion is repeatable for unchanged input: normalize source content and compare the complete authoritative fields within a serialized ingestion run before inserting. Existing rows retain their primary keys so the Milvus key remains stable. Replacing or deleting previously indexed source documents is outside this chapter's ingestion command; it must not silently overwrite a done vector or claim stale content was removed. Set `is_key_clause` only for Markdown `> [!IMPORTANT]` callouts; otherwise use the DDL default `0`.

## Conversation mining

Read existing MySQL `messages` grouped by conversation and preserve message order. Include user and final assistant content as evidence; exclude tool-call JSON, tool result payloads, incomplete turns, and credentials. Process a bounded number of conversations per LLM request to prevent cross-conversation leakage. Use the configured chat model to extract only supported question-answer pairs; require a valid structured result, reject invented or unsupported answers, and retain `batch_no` and `source_ref` in `qa_extraction_staging`.

After all selected batches are staged, run a global deduplication pass across the staged rows and existing knowledge, not a per-batch insert. Normalize question and answer text for exact duplicates, then use BGE-M3 similarity to identify candidate near-duplicates; keep distinct or conflicting answers unless evidence confirms they are equivalent. Mark every staged row `kept` or `discarded`. Insert kept QA into `knowledge_chunks` with `content_type='faq'`, actual `questions`, and `category='历史客服对话'` because the supplied staging schema has no topic category. Commit final knowledge insertion and staging status changes in one MySQL transaction. Reruns skip terminal staging rows and recover unfinished extraction and vectorization work. Keep the staging table for audit until explicitly cleared.

## Idempotent dual write and recovery

Commit a chunk in MySQL with `vectorize_status='pending'` before calling the embedding model or Milvus. For each pending primary key, compute its vector from the current authoritative fields, then `upsert` Milvus by the matching INT64 ID. Only after the Milvus call succeeds, set `vector_id` and `vectorize_status='done'` in MySQL. If the process stops after Milvus upsert but before the MySQL update, rerunning upserts the same ID and completes the status transition. An already-done row is skipped unless its authoritative vectorized fields changed and were explicitly reset to pending. Never mark a row done after a failed or malformed embedding or failed Milvus write. Keep transactions short; do not hold a MySQL transaction open through model inference or network calls.

## Online retrieval and failure behavior

Embed the incoming `keyword` as the query, search Milvus `knowledge` by cosine similarity for a bounded Top-K, and fetch the matching `done` rows by ID from MySQL. Preserve Milvus rank order, skip missing/stale IDs, and return `question=row.questions`, `answer=row.answer`, and `category=row.category` in the existing `matches` list. Use a labeled evaluation to set a minimum score that avoids unrelated policy answers; an empty or low-confidence result retains the existing no-match shape and safe message. If embedding or Milvus is unavailable, return the same keys with an explicit temporary-unavailable message and no invented answer. Update the tool description and service prompt to allow semantic paraphrases while requiring the final response to stay grounded in retrieved answers.

## Verification and delivery

Use TDD for parser/chunking, ORM and idempotent indexing, recovery, retrieval contract, and scheduler behavior. For LLM extraction prompts and document-only rules, use labeled examples or an evaluation set instead of tests that merely mirror text. Cover heading paths, sentence-aligned overlap, table headers, FAQ questions, global deduplication, interrupted upsert recovery, stale IDs, and low-confidence misses. Run MySQL/Milvus integration checks and the full Chapter 2 regression suite. Demonstrate “邮费是多少” through the unchanged `query_faq` and the chat flow; inspect both its `knowledge_chunks` row and Milvus ID. Publish exact setup, one-shot build, scheduled job, recovery, chat demo, and test commands in `README.md`.

Append `dev-notes/ch03.md` immediately after brainstorm approval, plan approval, every implementation task, code review, and finish. Each entry records the user's key wording, the resulting artifact or review decision, rejected/corrected ideas, and failures/rework. Commit and push completed, verified changes to the configured GitHub remote; never commit `.env`, tokens, model cache, virtual environments, or credentials.

## Documentation checked during design

- BGE-M3 model and local Sentence Transformers example: https://huggingface.co/BAAI/bge-m3
- SentenceTransformer constructor and `encode` API (Context7): https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/model.py
- MilvusClient collection, upsert, and search API (Context7): https://github.com/milvus-io/pymilvus/blob/master/_autodocs/api-reference/milvus-client.md
- SQLAlchemy 2.0 transaction and locking API (Context7): https://docs.sqlalchemy.org/en/20/orm/session_api.html
- Hugging Face sentence similarity result contract (Context7): https://huggingface.co/docs/huggingface_hub/package_reference/inference_client

## Authoritative Chapter 3 MySQL DDL supplied by the user

Copy this SQL into the executable schema file without changing definitions. The surrounding implementation may add setup logic but must not alter these two tables.

```sql
SET NAMES utf8mb4;

CREATE TABLE knowledge_chunks (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'chunk 主键,与 Milvus 集合主键对齐',
  category         VARCHAR(255)    NOT NULL                COMMENT '分类 / 上级标题路径,进向量化文本',
  questions        TEXT            NOT NULL                COMMENT '问法或本节标题,多个问法换行分隔,进向量化文本',
  answer           TEXT            NOT NULL                COMMENT '正文答案,进向量化文本',
  section_path     VARCHAR(512)    NULL                    COMMENT '章节路径,元数据,溯源用,不进向量',
  content_type     VARCHAR(32)     NULL                    COMMENT '内容类型:faq / policy / manual 等,元数据',
  is_key_clause    TINYINT(1)      NOT NULL DEFAULT 0      COMMENT '是否关键条款,0 否 1 是,元数据',
  prev_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '前一块指针,元数据',
  next_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '后一块指针,元数据',
  vector_id        VARCHAR(64)     NULL                    COMMENT 'Milvus 集合 knowledge 里的主键,写入后回填',
  vectorize_status ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT '待向量化 / 已向量化,双写幂等靠它',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category),
  KEY idx_vectorize_status (vectorize_status),
  CONSTRAINT fk_chunks_prev FOREIGN KEY (prev_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL,
  CONSTRAINT fk_chunks_next FOREIGN KEY (next_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库 chunk 原文权威源';

CREATE TABLE qa_extraction_staging (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '暂存行主键',
  batch_no         VARCHAR(64)     NOT NULL                COMMENT '抽取批次号,一批几十个会话跑一次,分批防串味、按批追溯',
  source_ref       VARCHAR(255)    NULL                    COMMENT '来源会话 / 导出文件标识,溯源用,不入最终知识库',
  question         TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的用户问法',
  answer           TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的客服答案',
  status           ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted' COMMENT '已抽出待去重 / 去重保留 / 去重丢弃',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '抽取写入时间',
  PRIMARY KEY (id),
  KEY idx_batch_no (batch_no),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史对话抽 QA 的离线中转暂存表:分批抽取、整体去重,保留项入 knowledge_chunks,建库完成可清空';
```
