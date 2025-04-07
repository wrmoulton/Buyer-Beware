import json
from typing import List
from src.database import insert_term

def load_json_posts(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)  # Loads as a dict with keys like "synthetic_1"

def extract_rationale_spans(tokens: List[str], rationales: List[List[int]]) -> List[str]:
    """Returns phrases where at least one annotator marked a token as rationale."""
    final_mask = [max(r[i] for r in rationales) for i in range(len(tokens))]

    spans = []
    current_span = []

    for token, keep in zip(tokens, final_mask):
        if keep:
            current_span.append(token)
        elif current_span:
            spans.append(" ".join(current_span))
            current_span = []

    if current_span:
        spans.append(" ".join(current_span))

    return spans

def parse_and_store_from_file(json_path: str):
    data = load_json_posts(json_path)  # dict of posts
    count = 0

    for post_id, post in data.items():
        tokens = post.get("post_tokens", [])
        rationales = post.get("rationales", [])

        if not tokens or not rationales:
            continue

        phrases = extract_rationale_spans(tokens, rationales)

        for phrase in phrases:
            insert_term(phrase, source_post_id=post_id)
            count += 1

    print(f"Inserted {count} rationale phrases from {len(data)} posts.")
