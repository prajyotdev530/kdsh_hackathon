# =====================================================
# Pathway Pipeline: Chunking + Embedding (Judge-Safe)
# =====================================================

import pathway as pw
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------

BOOKS_DIR = "./Dataset/Books/*.txt"

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

# Load embedding model once
embedding_model = SentenceTransformer(MODEL_NAME)

# -----------------------------------------------------
# Utility: text chunking (word-based with overlap)
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
# STEP 1: Read novels (one row per book)
# -----------------------------------------------------

novels = pw.io.fs.read(
    BOOKS_DIR,
    format="plaintext_by_file",
    with_metadata=True,
    mode="static"   # important for offline datasets
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

# Explode list -> one row per chunk
chunks = chunks.flatten(pw.this.chunk_text)

# -----------------------------------------------------
# STEP 3: Compute embeddings
# -----------------------------------------------------

def embed_text(text: str):
    return embedding_model.encode(text).tolist()

chunks = chunks.with_columns(
    embedding=pw.apply(embed_text, pw.this.chunk_text)
)

# -----------------------------------------------------
# STEP 4: Final vector store
# -----------------------------------------------------

vector_store = chunks.select(
    book_name=pw.this.book_name,
    chunk_text=pw.this.chunk_text,
    embedding=pw.this.embedding
)

pw.io.fs.write(
    vector_store,"./vector_store",format="json"
)

# -----------------------------------------------------
# STEP 5: Debug output (optional)
# -----------------------------------------------------

pw.debug.compute_and_print(vector_store)

# -----------------------------------------------------
# STEP 6: Chunk statistics (sanity check)
# -----------------------------------------------------

chunk_counts = vector_store.groupby(
    vector_store.book_name
).reduce(
    book_name=pw.this.book_name,
    num_chunks=pw.reducers.count()
)

pw.debug.compute_and_print(chunk_counts)

# -----------------------------------------------------
# STEP 7: Run Pathway
# -----------------------------------------------------

pw.run()