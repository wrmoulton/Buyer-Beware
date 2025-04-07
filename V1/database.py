import sqlite3
from typing import List, Tuple

class LexiconDatabase:
    def __init__(self, db_path: str = "lexicon.db"):
        """Initialize the database connection and create tables if they don't exist."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()

    def connect(self):
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print(f"Successfully connected to database at {self.db_path}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS lexicon_terms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT UNIQUE NOT NULL,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
            print("Database tables created successfully")
        except sqlite3.Error as e:
            print(f"Error creating tables: {e}")
            raise

    def add_term(self, term: str, embedding: bytes = None):
        """Add a new term to the database."""
        try:
            self.cursor.execute('''
                INSERT OR IGNORE INTO lexicon_terms (term, embedding)
                VALUES (?, ?)
            ''', (term, embedding))
            self.conn.commit()
            print(f"Successfully added term: {term}")
        except sqlite3.Error as e:
            print(f"Error adding term: {e}")
            raise

    def get_all_terms(self) -> List[str]:
        """Retrieve all terms from the database."""
        try:
            self.cursor.execute('SELECT term FROM lexicon_terms')
            return [row[0] for row in self.cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error retrieving terms: {e}")
            return []

    def get_term_embedding(self, term: str) -> bytes:
        """Retrieve the embedding for a specific term."""
        try:
            self.cursor.execute('SELECT embedding FROM lexicon_terms WHERE term = ?', (term,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            print(f"Error retrieving embedding: {e}")
            return None

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# Example usage
if __name__ == "__main__":
    # Initialize the database
    db = LexiconDatabase()
    
    # Example terms to add
    test_terms = ["test_term1", "test_term2", "test_term3"]
    
    # Add terms to the database
    for term in test_terms:
        db.add_term(term)
    
    # Retrieve all terms
    all_terms = db.get_all_terms()
    print("All terms in database:", all_terms)
    
    # Close the database connection
    db.close() 