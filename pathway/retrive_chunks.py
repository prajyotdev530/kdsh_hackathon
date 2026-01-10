# =====================================================
# Claim + Contradiction Chunk Retrieval (Pathway)
# =====================================================

import json
import numpy as np
import pathway as pw
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------

def make_join_key(row_id, claim_id):
    return f"{row_id}::{claim_id}"

VECTOR_STORE_PATH = "./vector_store"
TRAIN_PATH = "./Dataset/train_with_claims_and_contradictions.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CLAIMS = 4

model = SentenceTransformer(MODEL_NAME)

# -----------------------------------------------------
# UTILITIES
# -----------------------------------------------------

def embed(text):
    if text is None:
        return [0.0] * 384  # MiniLM embedding size

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if text == "":
        return [0.0] * 384

    return model.encode(text).tolist()

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# -----------------------------------------------------
# LOAD VECTOR STORE
# -----------------------------------------------------

vector_store = pw.io.fs.read(
    VECTOR_STORE_PATH,
    format="json",
    mode="static",
    schema=pw.schema_from_types(
        book_name=str,
        chunk_text=str,
        embedding=list[float],
    ),
)

# -----------------------------------------------------
# LOAD TRAIN DATA
# -----------------------------------------------------

train = pw.io.fs.read(
    TRAIN_PATH,
    format="csv",
    mode="static",
    schema=pw.schema_from_csv(TRAIN_PATH),
)

# Rename CSV id immediately
train = train.select(
    row_id=pw.this.id,
    book_name=pw.this.book_name,
    claims=pw.this.claims,
    contradictions=pw.this.contradictions,
)

# -----------------------------------------------------
# PARSE CLAIMS (max 4)
# -----------------------------------------------------

def parse_claims(claims_json):
    claims = json.loads(claims_json)[:MAX_CLAIMS]
    return [{"claim_id": c["claim_id"], "claim_text": c["claim_text"]} for c in claims]

train = train.select(
    row_id=pw.this.row_id,
    book_name=pw.this.book_name,
    claims=pw.apply(parse_claims, pw.this.claims),
    contradictions=pw.apply(json.loads, pw.this.contradictions),
)

train = train.flatten(pw.this.claims)

# -----------------------------------------------------
# EXTRACT CLAIM FIELDS
# -----------------------------------------------------

train = train.select(
    row_id=pw.this.row_id,
    book_name=pw.this.book_name,
    claim_id=pw.apply(lambda c: c["claim_id"], pw.this.claims),
    claim_text=pw.apply(lambda c: c["claim_text"], pw.this.claims),
    contradictions=pw.this.contradictions,
  claim_embedding=pw.apply(
    embed,
    pw.apply(lambda c: c["claim_text"], pw.this.claims),
),
)

# -----------------------------------------------------
# CLAIM → BEST CHUNK (argmax)
# -----------------------------------------------------

claim_join = train.join(
    vector_store,
    train.book_name == vector_store.book_name,
    how=pw.JoinMode.INNER,
)

claim_join = claim_join.select(
    row_id=pw.this.row_id,
    claim_id=pw.this.claim_id,
    claim_text=pw.this.claim_text,
    chunk_text=pw.this.chunk_text,
    score=pw.apply(
        cosine_sim,
        pw.this.claim_embedding,
        pw.this.embedding,
    ),
)

claim_best_chunk = (
    claim_join.groupby(
        pw.this.row_id,
        pw.this.claim_id,
        pw.this.claim_text,   # ✅ include here
    )
    .reduce(
        row_id=pw.this.row_id,
        claim_id=pw.this.claim_id,
        claim_text=pw.this.claim_text,
        claim_chunk=pw.reducers.argmax(
            pw.this.chunk_text,
            pw.this.score,
        ),
    )
)

# -----------------------------------------------------
# CONTRADICTIONS → BEST CHUNK EACH
# -----------------------------------------------------
def explode_contradictions(contradictions, claim_id):
    if contradictions is None:
        return []

    data = json.loads(str(contradictions))  # ✅ critical fix
    return data.get(str(claim_id), [])

contra = train.select(
    row_id=pw.this.row_id,
    book_name=pw.this.book_name,
    claim_id=pw.this.claim_id,
    contradiction_text=pw.apply(
        explode_contradictions,
        pw.this.contradictions,
        pw.this.claim_id,
    ),
)

contra = contra.flatten(pw.this.contradiction_text)

contra = contra.select(
    row_id=pw.this.row_id,
    book_name=pw.this.book_name,
    claim_id=pw.this.claim_id,
    contradiction_text=pw.this.contradiction_text,
    contra_embedding=pw.apply(embed, pw.this.contradiction_text),
)

contra_join = contra.join(
    vector_store,
    contra.book_name == vector_store.book_name,
    how=pw.JoinMode.INNER,
)

contra_join = contra_join.select(
    row_id=pw.this.row_id,
    claim_id=pw.this.claim_id,
    contradiction_text=pw.this.contradiction_text,
    chunk_text=pw.this.chunk_text,
    score=pw.apply(
        cosine_sim,
        pw.this.contra_embedding,
        pw.this.embedding,
    ),
)

contra_best_chunks = (
    contra_join.groupby(
        pw.this.row_id,
        pw.this.claim_id,
        pw.this.contradiction_text,
    )
    .reduce(
        row_id=pw.this.row_id,
        claim_id=pw.this.claim_id,
        contradiction_text=pw.this.contradiction_text,
       contra_chunk=pw.reducers.argmax(
    pw.this.chunk_text,
    pw.this.score,
),
    )
)

claim_best_chunk = claim_best_chunk.select(
   join_key=pw.apply(make_join_key, pw.this.row_id, pw.this.claim_id),
    row_id=pw.this.row_id,
    claim_id=pw.this.claim_id,
    claim_text=pw.this.claim_text,
    claim_chunk=pw.this.claim_chunk,
)

contra_best_chunks = contra_best_chunks.select(
    join_key=pw.apply(make_join_key, pw.this.row_id, pw.this.claim_id),
    row_id=pw.this.row_id,
    claim_id=pw.this.claim_id,
    contradiction_text=pw.this.contradiction_text,
    contra_chunk=pw.this.contra_chunk,
)

# -----------------------------------------------------
# FINAL AGGREGATION
# -----------------------------------------------------
# -----------------------------------------------------
# FINAL AGGREGATION
# -----------------------------------------------------

final = claim_best_chunk.join(
    contra_best_chunks,
    claim_best_chunk.join_key == contra_best_chunks.join_key,
    how=pw.JoinMode.LEFT,
)

# 🔑 DISAMBIGUATE COLUMNS AFTER JOIN
final = final.select(
    row_id=claim_best_chunk.row_id,
    claim_id=claim_best_chunk.claim_id,
    claim_text=claim_best_chunk.claim_text,
    claim_chunk=claim_best_chunk.claim_chunk,
    contradiction_text=contra_best_chunks.contradiction_text,
    contra_chunk=contra_best_chunks.contra_chunk,
)

# -----------------------------------------------------
# OUTPUT
# -----------------------------------------------------

pw.debug.compute_and_print(final)
pw.run()
