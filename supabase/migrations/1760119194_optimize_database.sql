
-- Database optimization migration
-- Add indexes for better query performance

CREATE INDEX IF NOT EXISTS idx_mood_entries_user_created 
  ON mood_entries(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_mood_entries_date 
  ON mood_entries(date);

CREATE INDEX IF NOT EXISTS idx_chat_messages_user_created 
  ON chat_messages(user_id, created_at DESC);

-- Enable Row Level Security policies optimization
ALTER TABLE mood_entries ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- Optimize storage for better performance
ALTER TABLE mood_entries SET (fillfactor = 90);
ALTER TABLE chat_messages SET (fillfactor = 90);
