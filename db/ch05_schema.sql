CREATE TABLE workflow_turns (
	turn_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL,
	conversation_id BIGINT UNSIGNED NOT NULL,
	status VARCHAR(32) NOT NULL DEFAULT 'running',
	final_message_id BIGINT UNSIGNED,
	PRIMARY KEY (turn_id),
	CONSTRAINT fk_workflow_turn_conversation FOREIGN KEY(conversation_id) REFERENCES conversations (id),
	CONSTRAINT fk_workflow_turn_final_message FOREIGN KEY(final_message_id) REFERENCES messages (id)
)ENGINE=InnoDB CHARSET=utf8mb4 COMMENT='ch05 工作流轮次';

CREATE TABLE workflow_message_keys (
	turn_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL,
	position BIGINT UNSIGNED NOT NULL,
	message_id BIGINT UNSIGNED NOT NULL,
	PRIMARY KEY (turn_id, position),
	CONSTRAINT fk_workflow_message_turn FOREIGN KEY(turn_id) REFERENCES workflow_turns (turn_id),
	CONSTRAINT fk_workflow_message_message FOREIGN KEY(message_id) REFERENCES messages (id)
)ENGINE=InnoDB CHARSET=utf8mb4 COMMENT='ch05 轮次消息幂等键';

CREATE TABLE workflow_actions (
	action_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL,
	conversation_id BIGINT UNSIGNED NOT NULL,
	turn_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL,
	message_id BIGINT UNSIGNED NOT NULL,
	kind ENUM('handoff','create_ticket') NOT NULL,
	description TEXT,
	ticket_type ENUM('售后','投诉','咨询'),
	PRIMARY KEY (action_id),
	CONSTRAINT fk_workflow_action_conversation FOREIGN KEY(conversation_id) REFERENCES conversations (id),
	CONSTRAINT fk_workflow_action_turn FOREIGN KEY(turn_id) REFERENCES workflow_turns (turn_id),
	CONSTRAINT fk_workflow_action_message FOREIGN KEY(message_id) REFERENCES messages (id)
)ENGINE=InnoDB CHARSET=utf8mb4 COMMENT='ch05 无副作用动作建议';

CREATE TABLE workflow_ticket_requests (
	request_key VARCHAR(128) COLLATE utf8mb4_bin NOT NULL,
	conversation_id BIGINT UNSIGNED NOT NULL,
	payload_digest VARCHAR(64) NOT NULL,
	ticket_no VARCHAR(32) NOT NULL,
	PRIMARY KEY (request_key),
	CONSTRAINT fk_workflow_ticket_conversation FOREIGN KEY(conversation_id) REFERENCES conversations (id),
	CONSTRAINT fk_workflow_ticket_ticket FOREIGN KEY(ticket_no) REFERENCES tickets (ticket_no)
)ENGINE=InnoDB CHARSET=utf8mb4 COMMENT='ch05 工单确认幂等键';
