import csv
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

# =====================================================
# CONFIG
# =====================================================

INPUT_CSV = "trained_with_claims.csv"
OUTPUT_CSV = "claims_with_supporting_chunks.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

NOVEL_PATHS = {
    "count_of_monte_cristo": "./Dataset/Books/count_of_monte_cristo.txt",
    "jane_eyre": "./Dataset/Books/jane_eyre.txt",
    "wuthering_heights": "./Dataset/Books/wuthering_heights.txt",
}

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# =====================================================
# STEP 1: LOAD AND CHUNK NOVEL
# =====================================================

def load_and_chunk_novel(
    novel_path: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict]:

    if not Path(novel_path).exists():
        print(f"Novel not found: {novel_path}")
        return []

    print(f"Loading novel: {Path(novel_path).name}")

    with open(novel_path, "r", encoding="utf-8") as f:
        text = f.read()

    words = text.split()
    chunks = []
    chunk_texts = []

    start = 0
    chunk_id = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_texts.append(" ".join(words[start:end]))
        start += (chunk_size - overlap)
        chunk_id += 1

    embeddings = embedding_model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=True
    )

    start = 0
    for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
        end = min(start + chunk_size, len(words))
        chunks.append({
            "id": i,
            "text": chunk_text,
            "embedding": embedding / np.linalg.norm(embedding),
            "start_word": start,
            "end_word": end
        })
        start += (chunk_size - overlap)

    print(f"Created {len(chunks)} chunks\n")
    return chunks


# =====================================================
# STEP 2: RETRIEVE TOP-2 SUPPORTING CHUNKS
# =====================================================

def retrieve_top_2_supporting_chunks(
    claim_text: str,
    keywords: List[str],
    chunks: List[Dict],
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[Dict]:

    if not chunks:
        return []

    claim_embedding = embedding_model.encode(claim_text)
    claim_embedding /= np.linalg.norm(claim_embedding)

    chunk_embeddings = np.array([c["embedding"] for c in chunks])
    semantic_scores = np.dot(chunk_embeddings, claim_embedding)
    semantic_scores = (semantic_scores + 1) / 2

    keyword_scores = np.zeros(len(chunks))
    for i, chunk in enumerate(chunks):
        text = chunk["text"].lower()
        keyword_scores[i] = sum(
            1 for kw in keywords if kw.lower() in text
        )

    if keyword_scores.max() > 0:
        keyword_scores /= keyword_scores.max()

    combined_scores = (
        semantic_weight * semantic_scores +
        keyword_weight * keyword_scores
    )

    top_indices = np.argsort(combined_scores)[::-1][:2]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            "rank": rank,
            "chunk_id": chunks[idx]["id"],
            "text": chunks[idx]["text"][:500],
            "semantic_score": round(float(semantic_scores[idx]), 3),
            "keyword_score": round(float(keyword_scores[idx]), 3),
            "combined_score": round(float(combined_scores[idx]), 3),
            "keywords_found": int(keyword_scores[idx] * len(keywords))
        })

    return results


# =====================================================
# STEP 3: MAIN PIPELINE
# =====================================================

def process_csv():

    print("=" * 70)
    print("RETRIEVING SUPPORTING CHUNKS")
    print("=" * 70)

    if not Path(INPUT_CSV).exists():
        print(f"Input CSV not found: {INPUT_CSV}")
        return

    results = []
    novel_cache = {}

    with open(INPUT_CSV, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        for idx, row in enumerate(reader, 1):
            print(f"\nProcessing row {idx}, ID={row.get('id')}")

            book_name = row.get("book_name", "").lower().replace(" ", "_")
            novel_path = NOVEL_PATHS.get(book_name)

            if not novel_path:
                row["supporting_chunks"] = "{}"
                results.append(row)
                continue

            if novel_path not in novel_cache:
                novel_cache[novel_path] = load_and_chunk_novel(novel_path)

            chunks = novel_cache[novel_path]

            try:
                claims = json.loads(row.get("claims", "[]"))
                claim_results = {}

                for claim in claims:
                    claim_id = str(claim["claim_id"])
                    claim_text = claim["claim_text"]
                    keywords = claim.get("keywords", [])

                    print(f"  Claim {claim_id}: {claim_text[:60]}...")

                    supporting = retrieve_top_2_supporting_chunks(
                        claim_text,
                        keywords,
                        chunks
                    )

                    claim_results[claim_id] = supporting

                row["supporting_chunks"] = json.dumps(
                    claim_results,
                    ensure_ascii=False,
                    indent=2
                )

            except Exception as e:
                print(f"Error processing row: {e}")
                row["supporting_chunks"] = "{}"

            results.append(row)

    if results:
        fieldnames = list(results[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print("Processing complete")
    print(f"Output saved to {OUTPUT_CSV}")


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    process_csv()
