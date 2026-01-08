# =====================================================
# Pathway Pipeline: Chunking + Embedding + Characters
# =====================================================

import pathway as pw
from sentence_transformers import SentenceTransformer
import re

# -----------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------

BOOKS_DIR = "./Dataset/Books/*.txt"

CHARACTER_LIST = [
    "Edmond", "Dantès", "Monte Cristo",
    "Fernand", "Mercedes", "Danglars"
]

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Load embedding model once
embedding_model = SentenceTransformer(MODEL_NAME)

# -----------------------------------------------------
# Utility: text chunking
# -----------------------------------------------------

def chunk_text(text: str):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

# -----------------------------------------------------
# STEP 1: Read novels
# -----------------------------------------------------

novels = pw.io.fs.read(
    "./Dataset/Books/*.txt",
    format="plaintext_by_file",
    with_metadata=True,
    mode="static"   # 🔑 IMPORTANT
)
novels = novels.select(
    book_name=pw.apply(
    lambda p: p.split("/")[-1].replace(".txt", ""),
    pw.apply(str, pw.this._metadata["path"])
),
    text=pw.this.data
)

# -----------------------------------------------------
# STEP 2: Chunk novels
# -----------------------------------------------------

chunks = novels.select(
    book_name=pw.this.book_name,
    chunk_text=pw.apply(chunk_text, pw.this.text)
)

# Explode list → one row per chunk
chunks = chunks.flatten(pw.this.chunk_text)

# -----------------------------------------------------
# STEP 3: Extract character mentions
# -----------------------------------------------------

def extract_characters(text: str):
    text_lower = text.lower()
    found = []

    for character in CHARACTER_LIST:
        pattern = r"\b" + re.escape(character.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(character)

    return list(set(found))

chunks = chunks.with_columns(
    characters=pw.apply(
        extract_characters,
        pw.this.chunk_text
    )
)

# -----------------------------------------------------
# STEP 4: Compute embeddings
# -----------------------------------------------------

def embed_text(text: str):
    return embedding_model.encode(text).tolist()

chunks = chunks.with_columns(
    embedding=pw.apply(embed_text, pw.this.chunk_text)
)

# -----------------------------------------------------
# STEP 5: Final vector store
# -----------------------------------------------------

vector_store = chunks.select(
    book_name=pw.this.book_name,
    chunk_text=pw.this.chunk_text,
    embedding=pw.this.embedding,
    characters=pw.this.characters
)

# -----------------------------------------------------
# STEP 6: Debug output
# -----------------------------------------------------

pw.debug.compute_and_print(vector_store)

# -----------------------------------------------------
# STEP 7: Run Pathway
# -----------------------------------------------------

chunk_counts = vector_store.groupby(
    vector_store.book_name
).reduce(
    book_name=pw.this.book_name,
    num_chunks=pw.reducers.count()
)

pw.debug.compute_and_print(chunk_counts)
pw.run()
