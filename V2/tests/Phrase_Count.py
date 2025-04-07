#Check Amount of Rationale vs GPT-Phrases

from src.database import get_connection

conn = get_connection()
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM lexicon_terms WHERE source_type = 'rationale'")
print("Rationale phrases:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM lexicon_terms WHERE source_type = 'gpt'")
print("GPT phrases:", cursor.fetchone()[0])
conn.close()
