import re
import os
import pandas as pd
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

class ChapterAwareEmbeddingStore:
    """Store for chapter-aware embeddings with narrative constraints."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = []
        self.metadata = []

    def add_chunk(self, text: str, metadata: Dict[str, Any]):
        """Add a chunk with its metadata."""
        self.chunks.append(text)
        embedding = self.model.encode(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata)

    def search_similar(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """Find most similar chunks to query."""
        query_embedding = self.model.encode(query)
        similarities = []

        for i, emb in enumerate(self.embeddings):
            sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities.append((sim, i))

        similarities.sort(reverse=True)
        results = []
        for sim, idx in similarities[:top_k]:
            results.append((self.chunks[idx], self.metadata[idx], sim))

        return results

    def save(self, filepath: str):
        """Save the store to disk."""
        data = {
            'chunks': self.chunks,
            'embeddings': np.array(self.embeddings),
            'metadata': self.metadata
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """Load the store from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.chunks = data['chunks']
        self.embeddings = data['embeddings']
        self.metadata = data['metadata']

def extract_narrative_constraints(text: str) -> Dict[str, Any]:
    """Extract narrative constraints from text."""
    constraints = {
        'promises': [],
        'temporal_markers': [],
        'character_relations': [],
        'locations': [],
        'events': []
    }

    # Promises and commitments
    promise_patterns = [
        r'(?:will|shall|must)\s+(.+?)(?:\s+(?:by|in|on|before)\s+\w+)',
        r'(?:promise|swear|vow)\s+to\s+(.+?)',
    ]
    for pattern in promise_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        constraints['promises'].extend(matches)

    # Temporal markers
    temporal_patterns = [
        r'(?:before|after|during|when|while|since|until)\s+(.+?)(?:\s+(?:he|she|they|it)\s+\w+)',
        r'\b(?:yesterday|tomorrow|today|morning|evening|night|day|week|month|year)s?\b'
    ]
    for pattern in temporal_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        constraints['temporal_markers'].extend(matches)

    # Character relations
    relation_patterns = [
        r'(?:father|mother|son|daughter|sister|brother|wife|husband)\s+of\s+(.+?)',
        r'(?:friend|enemy|lover|companion)\s+of\s+(.+?)'
    ]
    for pattern in relation_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        constraints['character_relations'].extend(matches)

    # Locations
    location_patterns = [
        r'\b(?:in|at|on|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    ]
    for pattern in location_patterns:
        matches = re.findall(pattern, text)
        constraints['locations'].extend(matches)

    # Events (actions)
    event_patterns = [
        r'(?:arrived|departed|met|found|saw|heard|said|did|went|came)\s+(.+?)(?:\s+(?:and|but|then|when))'
    ]
    for pattern in event_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        constraints['events'].extend(matches)

    return constraints

def process_book(file_path: str, book_name: str) -> List[Dict[str, Any]]:
    """Process a book and return chapter-aware chunks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Chapter/scene detection patterns
    chapter_patterns = [
        (r'^CHAPTER\s+([IVXLCDM]+)\.?\s*$', 'roman_caps'),
        (r'^Chapter\s+(\d+)\.?\s+.*$', 'arabic_with_title'),
        (r'^([IVXLCDM]+)\.?\s*$', 'roman_standalone'),
        (r'^CHAPTER\s+([IVXLCDM]+)[\.\:\s]+(.+)$', 'roman_with_title'),
    ]

    chunks = []
    lines = text.split('\n')
    current_chunk = []
    current_metadata = {'book': book_name, 'chapter': None, 'type': None}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        chapter_match = None
        for pattern, chapter_type in chapter_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                # Save previous chunk if exists
                if current_chunk and current_metadata['chapter']:
                    chunk_text = '\n'.join(current_chunk)
                    constraints = extract_narrative_constraints(chunk_text)
                    chunk_data = {
                        'text': chunk_text,
                        'metadata': current_metadata.copy(),
                        'constraints': constraints
                    }
                    chunks.append(chunk_data)

                # Start new chunk
                current_chunk = [line]
                if chapter_type in ['roman_caps', 'roman_standalone', 'roman_with_title']:
                    current_metadata['chapter'] = match.group(1)
                elif chapter_type == 'arabic_with_title':
                    current_metadata['chapter'] = match.group(1)
                current_metadata['type'] = chapter_type
                break
        else:
            # Continue current chunk
            current_chunk.append(line)

    # Save last chunk
    if current_chunk and current_metadata['chapter']:
        chunk_text = '\n'.join(current_chunk)
        constraints = extract_narrative_constraints(chunk_text)
        chunk_data = {
            'text': chunk_text,
            'metadata': current_metadata.copy(),
            'constraints': constraints
        }
        chunks.append(chunk_data)

    return chunks

def main():
    """Main processing function."""
    books = {
        'In Search of the Castaways': './Dataset/Books/In search of the castaways.txt',
        'The Count of Monte Cristo': './Dataset/Books/The Count of Monte Cristo.txt'
    }

    stores = {}

    for book_name, file_path in books.items():
        print(f"Processing {book_name}...")
        chunks = process_book(file_path, book_name)

        store = ChapterAwareEmbeddingStore()
        for chunk in chunks:
            store.add_chunk(chunk['text'], {**chunk['metadata'], **chunk['constraints']})

        stores[book_name] = store
        print(f"Created {len(chunks)} chunks for {book_name}")

        # Save the store
        store.save(f"./{book_name.replace(' ', '_')}_embeddings.pkl")

    print("Processing complete.")

if __name__ == "__main__":
    main()                # Finalize current chunk
