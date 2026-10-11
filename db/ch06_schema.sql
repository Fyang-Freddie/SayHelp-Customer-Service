CREATE TABLE workflow_interactions (
	interrupt_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL, 
	conversation_id BIGINT UNSIGNED NOT NULL, 
	turn_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL, 
	message_id BIGINT UNSIGNED NOT NULL, 
	kind ENUM('order_selection','customer_input','clarification') NOT NULL, 
	payload JSON NOT NULL, 
	checkpoint_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL, 
	status ENUM('waiting','processing','consumed','cancelled') NOT NULL DEFAULT 'waiting', 
	created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, 
	PRIMARY KEY (interrupt_id), 
	CONSTRAINT fk_interaction_conversation FOREIGN KEY(conversation_id) REFERENCES conversations (id), 
	CONSTRAINT fk_interaction_turn FOREIGN KEY(turn_id) REFERENCES workflow_turns (turn_id), 
	CONSTRAINT fk_interaction_message FOREIGN KEY(message_id) REFERENCES messages (id)
)ENGINE=InnoDB CHARSET=utf8mb4 COMMENT='ch06 等待用户交互';

CREATE TABLE workflow_resume_requests (
	request_id VARCHAR(128) COLLATE utf8mb4_bin NOT NULL, 
	conversation_id BIGINT UNSIGNED NOT NULL, 
	turn_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL, 
	interrupt_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL, 
	payload_digest VARCHAR(64) NOT NULL, 
	status ENUM('processing','complete','cancelled') NOT NULL DEFAULT 'processing', 
	result JSON, 
	PRIMARY KEY (request_id), 
	CONSTRAINT fk_resume_conversation FOREIGN KEY(conversation_id) REFERENCES conversations (id), 
	CONSTRAINT fk_resume_turn FOREIGN KEY(turn_id) REFERENCES workflow_turns (turn_id), 
	CONSTRAINT fk_resume_interaction FOREIGN KEY(interrupt_id) REFERENCES workflow_interactions (interrupt_id)
)ENGINE=InnoDB CHARSET=utf8mb4 COMMENT='ch06 恢复请求幂等记录';

CREATE TABLE refund_actions (
	action_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL, 
	conversation_id BIGINT UNSIGNED NOT NULL, 
	turn_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL, 
	item JSON NOT NULL, 
	eligibility JSON NOT NULL, 
	fingerprints JSON NOT NULL, 
	status ENUM('active','submitted','cancelled') NOT NULL DEFAULT 'active', 
	created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, 
	PRIMARY KEY (action_id), 
	CONSTRAINT fk_refund_action_conversation FOREIGN KEY(conversation_id) REFERENCES conversations (id), 
	CONSTRAINT fk_refund_action_turn FOREIGN KEY(turn_id) REFERENCES workflow_turns (turn_id)
)ENGINE=InnoDB CHARSET=utf8mb4 COMMENT='ch06 待确认退款动作';

CREATE TABLE refund_requests (
	request_id VARCHAR(128) COLLATE utf8mb4_bin NOT NULL, 
	action_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL, 
	conversation_id BIGINT UNSIGNED NOT NULL, 
	refund_no VARCHAR(64) COLLATE utf8mb4_bin NOT NULL, 
	customer_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL, 
	order_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL, 
	item_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL, 
	amount DECIMAL(12, 2) NOT NULL, 
	quantity INTEGER NOT NULL, 
	reason_code VARCHAR(64) COLLATE utf8mb4_bin NOT NULL, 
	payload_digest VARCHAR(64) NOT NULL, 
	status ENUM('pending') NOT NULL DEFAULT 'pending', 
	created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, 
	PRIMARY KEY (request_id), 
	CONSTRAINT fk_refund_request_action FOREIGN KEY(action_id) REFERENCES refund_actions (action_id), 
	CONSTRAINT fk_refund_request_conversation FOREIGN KEY(conversation_id) REFERENCES conversations (id)
)ENGINE=InnoDB CHARSET=utf8mb4 COMMENT='ch06 本地待处理退款申请';

CREATE UNIQUE INDEX uq_refund_action ON refund_requests (action_id);

CREATE UNIQUE INDEX uq_refund_item ON refund_requests (customer_id, order_id, item_id);

CREATE UNIQUE INDEX uq_refund_no ON refund_requests (refund_no);
