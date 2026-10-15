ALTER TABLE conversations
 ADD COLUMN summary_upto_msg_id BIGINT UNSIGNED NULL,
 ADD COLUMN layer1_from_msg_id BIGINT UNSIGNED NULL,
 ADD COLUMN context_version INT NOT NULL DEFAULT 0,
 ADD COLUMN summary_lease_token VARCHAR(64) NULL,
 ADD COLUMN summary_lease_until DATETIME NULL,
 ADD COLUMN summary_from_id BIGINT UNSIGNED NULL,
 ADD COLUMN summary_to_id BIGINT UNSIGNED NULL,
 ADD COLUMN summary_source_digest VARCHAR(64) NULL;
