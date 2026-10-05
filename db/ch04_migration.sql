-- Explicit additive migration; run through app.init_ch04_db for compatibility checks.
SET NAMES utf8mb4;
ALTER TABLE knowledge_chunks
  ADD COLUMN source_file VARCHAR(255) NULL COMMENT '仓库相对原文路径',
  ADD COLUMN source_start_line INT UNSIGNED NULL COMMENT '原文开始行',
  ADD COLUMN source_end_line INT UNSIGNED NULL COMMENT '原文结束行',
  ADD COLUMN product_category VARCHAR(64) NULL COMMENT '标准商品品类',
  ADD COLUMN source_digest VARCHAR(64) NULL COMMENT '原文 SHA256 版本';
ALTER TABLE conversations
  ADD COLUMN is_pinned TINYINT(1) NOT NULL DEFAULT 0 COMMENT '历史置顶状态',
  ADD COLUMN pinned_at DATETIME NULL COMMENT '置顶时间',
  ADD COLUMN deleted_at DATETIME NULL COMMENT '历史逻辑删除时间',
  ADD KEY idx_history_order (deleted_at,is_pinned,pinned_at,id);
ALTER TABLE messages ADD COLUMN citations JSON NULL COMMENT '本轮实际喂入的完整证据快照';
CREATE TABLE knowledge_index_states (
  collection_name VARCHAR(128) NOT NULL COMMENT '索引集合',
  chunk_id BIGINT UNSIGNED NOT NULL COMMENT '权威 chunk',
  payload_digest VARCHAR(64) NOT NULL COMMENT '文本与过滤元数据指纹',
  status ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT '新集合写入状态',
  error TEXT NULL COMMENT '安全错误原因',
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '状态更新时间',
  PRIMARY KEY (collection_name,chunk_id),
  CONSTRAINT fk_index_chunk FOREIGN KEY (chunk_id) REFERENCES knowledge_chunks(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ch04 独立集合构建进度';
