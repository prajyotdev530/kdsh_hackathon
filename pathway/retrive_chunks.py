# =====================================================
# Claim + Contradiction Chunk Retrieval (Pathway)
# =====================================================

import json
import pathway as pw
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------

VECTOR_STORE_PATH = "./vector_store"
TRAIN_PATH = "./Dataset/train_with_claims.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CLAIMS = 4
CLAIM_TOP_K = 2
CONTRADICTION_TOP_K = 1

model = SentenceTransformer(MODEL_NAME)

# -----------------------------------------------------
# UTILITIES
# -----------------------------------------------------

def embed(text: str):
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
    mode="static"
)

# -----------------------------------------------------
# LOAD TRAIN DATA
# -----------------------------------------------------

train = pw.io.fs.read(
    TRAIN_PATH,
    format="csv",
    mode="static"
)

# -----------------------------------------------------
# EXPAND CLAIMS (limit to 4)
# -----------------------------------------------------

def parse_claims(claims_json):
    claims = json.loads(claims_json)
    return claims[:MAX_CLAIMS]

train = train.with_columns(
    claims=pw.apply(parse_claims, pw.this.claims),
    contradictions=pw.apply(json.loads, pw.this.contradictions)
)

train = train.flatten(pw.this.claims)

# -----------------------------------------------------
# EMBED CLAIMS
# -----------------------------------------------------

train = train.with_columns(
    claim_id=pw.this.claims["claim_id"],
    claim_text=pw.this.claims["claim_text"],
    claim_embedding=pw.apply(embed, pw.this.claims["claim_text"])
)

# -----------------------------------------------------
# FILTER VECTOR STORE BY BOOK
# -----------------------------------------------------

joined = train.join(
    vector_store,
    train.book_name == vector_store.book_name,
    how="inner"
)

# -----------------------------------------------------
# CLAIM → TOP 2 CHUNKS
# -----------------------------------------------------

joined = joined.with_columns(
    claim_score=pw.apply(
        cosine_sim,
        pw.this.claim_embedding,
        pw.this.embedding
    )
)

claim_top_chunks = (
    joined.groupby(train.id, train.claim_id)
    .reduce(
        id=pw.this.id,
        claim_id=pw.this.claim_id,
        claim_text=pw.this.claim_text,
        top_claim_chunks=pw.reducers.top_k(
            pw.this.chunk_text,
            by=pw.this.claim_score,
            k=CLAIM_TOP_K
        )
    )
)

# -----------------------------------------------------
# CONTRADICTION → TOP 1 CHUNK EACH
# -----------------------------------------------------

def explode_contradictions(contradictions, claim_id):
    return contradictions.get(str(claim_id), [])

train_contra = train.select(
    id=pw.this.id,
    book_name=pw.this.book_name,
    claim_id=pw.this.claim_id,
    contradiction_text=pw.apply(
        explode_contradictions,
        pw.this.contradictions,
        pw.this.claim_id
    )
)

train_contra = train_contra.flatten(pw.this.contradiction_text)

train_contra = train_contra.with_columns(
    contra_embedding=pw.apply(embed, pw.this.contradiction_text)
)

contra_joined = train_contra.join(
    vector_store,
    train_contra.book_name == vector_store.book_name,
    how="inner"
)

contra_joined = contra_joined.with_columns(
    contra_score=pw.apply(
        cosine_sim,
        pw.this.contra_embedding,
        pw.this.embedding
    )
)

contra_top_chunks = (
    contra_joined.groupby(train_contra.id, train_contra.claim_id, train_contra.contradiction_text)
    .reduce(
        id=pw.this.id,
        claim_id=pw.this.claim_id,
        contradiction_text=pw.this.contradiction_text,
        top_contra_chunk=pw.reducers.argmax(
            pw.this.chunk_text,
            by=pw.this.contra_score
        )
    )
)

# -----------------------------------------------------
# FINAL AGGREGATION PER CLAIM
# -----------------------------------------------------

final = claim_top_chunks.join(
    contra_top_chunks,
    on=["id", "claim_id"],
    how="left"
)

final = final.groupby(final.id, final.claim_id).reduce(
    id=pw.this.id,
    claim_id=pw.this.claim_id,
    claim_text=pw.this.claim_text,
    claim_chunks=pw.this.top_claim_chunks,
    contradiction_chunks=pw.reducers.collect(
        pw.struct(
            text=pw.this.contradiction_text,
            chunk=pw.this.top_contra_chunk
        )
    )
)

# -----------------------------------------------------
# OUTPUT
# -----------------------------------------------------

pw.debug.compute_and_print(final)
pw.run()