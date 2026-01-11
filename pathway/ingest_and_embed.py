# =====================================================
# Pathway Pipeline: Chunking + Embedding + Characters
# =====================================================

import pathway as pw
from sentence_transformers import SentenceTransformer
import re
from collections import Counter

# -----------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------

BOOKS_DIR = "./Dataset/Books/*.txt"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

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
# Utility: extract character names (one time per book)
# -----------------------------------------------------

def extract_characters(full_text):
    names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', full_text)
    counts = Counter(names)

    blacklist = {
        "The","Chapter","Monsieur","Madame","Count",
        "Volume","Book","Part"
    }

    return sorted([
        n for n,c in counts.items()
        if c > 6 and n not in blacklist
    ])

# -----------------------------------------------------
# STEP 1: Read novels
# -----------------------------------------------------

novels = pw.io.fs.read(
    BOOKS_DIR,
    format="plaintext_by_file",
    with_metadata=True,
    mode="static"
)

novels = novels.select(
    book_name=pw.apply(
        lambda p: p.split("/")[-1].replace(".txt",""),
        pw.apply(str, pw.this._metadata["path"])
    ),
    text=pw.this.data
)

# -----------------------------------------------------
# STEP 2: Extract characters per book
# -----------------------------------------------------

novels = novels.with_columns(
    characters=pw.apply(extract_characters, pw.this.text)
)

# -----------------------------------------------------
# STEP 3: Chunk novels
# -----------------------------------------------------

chunks = novels.select(
    book_name=pw.this.book_name,
    characters=pw.this.characters,
    chunk_text=pw.apply(chunk_text, pw.this.text)
)

chunks = chunks.flatten(pw.this.chunk_text)

# -----------------------------------------------------
# STEP 4: Embeddings
# -----------------------------------------------------

def embed_text(text: str):
    return embedding_model.encode(text).tolist()

chunks = chunks.with_columns(
    embedding=pw.apply(embed_text, pw.this.chunk_text)
)

# -----------------------------------------------------
# STEP 5: Character presence per chunk
# -----------------------------------------------------

def character_presence(chunk, characters):
    text = chunk.lower()
    presence = {}

    for name in characters:
        found = 0
        for part in name.lower().split():
            if re.search(r'\b' + re.escape(part) + r'\b', text):
                found = 1
                break
        presence[name.replace(" ","_")] = found

    return presence

chunks = chunks.with_columns(
    char_map=pw.apply(character_presence, pw.this.chunk_text, pw.this.characters)
)

# Expand char_map into actual Pathway columns
def explode_chars(row):
    return row["char_map"]

chunks = chunks.select(
    book_name=pw.this.book_name,
    chunk_text=pw.this.chunk_text,
    embedding=pw.this.embedding,
    **pw.apply(explode_chars, pw.this)
)

# -----------------------------------------------------
# STEP 6: Final vector store
# -----------------------------------------------------

vector_store = chunks

pw.io.fs.write(vector_store,"./vector_store",format="json")

pw.debug.compute_and_print(vector_store)

# -----------------------------------------------------
# STEP 7: Chunk counts
# -----------------------------------------------------

chunk_counts = vector_store.groupby(
    vector_store.book_name
).reduce(
    book_name=pw.this.book_name,
    num_chunks=pw.reducers.count()
)

pw.debug.compute_and_print(chunk_counts)

# -----------------------------------------------------
# STEP 8: Run
# -----------------------------------------------------

pw.run()
