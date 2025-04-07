import unittest
import os
from src import database

# To Run: python -m unittest discover tests


class TestDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize DB before running tests
        database.initialize_database()

    def setUp(self):
        # Clear the database before each test
        database.clear_database()

    def test_insert_and_fetch_term(self):
        phrase = "test phrase"
        embedding = [0.1, 0.2, 0.3]
        cluster_id = 1
        source_post_id = "post123"

        database.insert_term(phrase, embedding, cluster_id, source_post_id)
        terms = database.get_all_terms()

        self.assertEqual(len(terms), 1)
        self.assertEqual(terms[0][1], phrase)

    def test_insert_without_embedding(self):
        phrase = "bare phrase"
        database.insert_term(phrase)
        terms = database.get_all_terms()

        self.assertEqual(len(terms), 1)
        self.assertEqual(terms[0][1], phrase)
        self.assertIsNone(terms[0][2])  # Embedding should be None

if __name__ == '__main__':
    unittest.main()
