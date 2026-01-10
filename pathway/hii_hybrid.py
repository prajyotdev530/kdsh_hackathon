import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Set
import re

# =====================================================
# CONFIG
# =====================================================
VECTOR_STORE_PATH = "./vector_store"
INPUT_PATH = "./Dataset/train_with_claims_and_contradictions.csv"
OUTPUT_PATH = "./Dataset/train_with_chunks_improved.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CLAIMS = 4

# Retrieval strategy configs
TOP_K_CLAIM = 5  # Get more candidates initially
TOP_K_CONTRADICTION = 3
FINAL_CHUNKS_PER_CLAIM = 3  # Best chunks to support the claim
FINAL_CHUNKS_PER_CONTRADICTION = 2  # Best chunks per contradiction
DIVERSITY_THRESHOLD = 0.85  # Avoid too similar chunks

model = SentenceTransformer(MODEL_NAME)

# =====================================================
# UTILITIES
# =====================================================
def embed(text):
    """Generate embeddings for text."""
    if text is None or text == "":
        return [0.0] * 384
    return model.encode(str(text).strip()).tolist()

def cosine_sim(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)

def keyword_overlap_score(text: str, keywords: List[str]) -> float:
    """Calculate keyword overlap score."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / len(keywords)

def extract_entities(text: str) -> Set[str]:
    """Extract potential named entities (capitalized words/phrases)."""
    # Simple entity extraction: capitalized words that aren't at sentence start
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    return set(words)

def entity_overlap_score(chunk_text: str, claim_text: str) -> float:
    """Calculate entity overlap between chunk and claim."""
    chunk_entities = extract_entities(chunk_text)
    claim_entities = extract_entities(claim_text)
    
    if not claim_entities:
        return 0.0
    
    overlap = len(chunk_entities & claim_entities)
    return overlap / len(claim_entities)

def hybrid_score(semantic_sim: float, keyword_score: float, entity_score: float,
                 weights: Tuple[float, float, float] = (0.6, 0.25, 0.15)) -> float:
    """Combine multiple signals for better ranking."""
    return (weights[0] * semantic_sim + 
            weights[1] * keyword_score + 
            weights[2] * entity_score)

def is_diverse(new_chunk: str, selected_chunks: List[str], threshold: float = DIVERSITY_THRESHOLD) -> bool:
    """Check if new chunk is sufficiently different from already selected chunks."""
    if not selected_chunks:
        return True
    
    new_emb = embed(new_chunk)
    for chunk in selected_chunks:
        chunk_emb = embed(chunk)
        if cosine_sim(new_emb, chunk_emb) > threshold:
            return False
    return True

def diversified_top_k(scored_chunks: List[Tuple[str, float]], k: int, 
                      diversity_threshold: float = DIVERSITY_THRESHOLD) -> List[str]:
    """Select top k chunks with diversity constraint."""
    selected = []
    for chunk, score in scored_chunks:
        if len(selected) >= k:
            break
        if is_diverse(chunk, selected, diversity_threshold):
            selected.append(chunk)
    
    # If we don't have enough diverse chunks, fill with highest scoring ones
    if len(selected) < k:
        for chunk, score in scored_chunks:
            if chunk not in selected:
                selected.append(chunk)
                if len(selected) >= k:
                    break
    
    return selected

def query_expansion(claim_text: str, keywords: List[str]) -> str:
    """Create expanded query for better retrieval."""
    # Combine claim with keywords
    expanded = claim_text
    if keywords:
        expanded += " " + " ".join(keywords)
    return expanded

# =====================================================
# ADVANCED RETRIEVAL FUNCTIONS
# =====================================================
def retrieve_for_claim(claim_text: str, keywords: List[str], 
                       book_vectors: List[Dict], top_k: int = FINAL_CHUNKS_PER_CLAIM) -> List[str]:
    """
    Retrieve best chunks to support a claim using hybrid scoring.
    """
    # Create expanded query
    search_text = query_expansion(claim_text, keywords)
    claim_emb = embed(search_text)
    
    # Score all chunks
    scored_chunks = []
    for vec in book_vectors:
        chunk_text = vec['chunk_text']
        
        # Semantic similarity
        sem_score = cosine_sim(claim_emb, vec['embedding'])
        
        # Keyword overlap
        kw_score = keyword_overlap_score(chunk_text, keywords)
        
        # Entity overlap
        entity_score = entity_overlap_score(chunk_text, claim_text)
        
        # Hybrid score
        final_score = hybrid_score(sem_score, kw_score, entity_score)
        
        scored_chunks.append((chunk_text, final_score))
    
    # Sort by score
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Select diverse top-k chunks
    return diversified_top_k(scored_chunks, top_k)

def retrieve_for_contradiction(contradiction_text: str, claim_text: str,
                               book_vectors: List[Dict], 
                               top_k: int = FINAL_CHUNKS_PER_CONTRADICTION) -> List[str]:
    """
    Retrieve chunks that might prove a contradiction.
    
    Strategy: Find chunks that are semantically related to BOTH the claim
    and the contradiction, as these are most likely to contain evidence.
    """
    # Embed both claim and contradiction
    claim_emb = embed(claim_text)
    contra_emb = embed(contradiction_text)
    
    # Average embedding (finds chunks related to both)
    combined_emb = (np.array(claim_emb) + np.array(contra_emb)) / 2
    
    scored_chunks = []
    for vec in book_vectors:
        chunk_text = vec['chunk_text']
        
        # Similarity to combined query
        combined_score = cosine_sim(combined_emb, vec['embedding'])
        
        # Also check individual similarities
        claim_score = cosine_sim(claim_emb, vec['embedding'])
        contra_score = cosine_sim(contra_emb, vec['embedding'])
        
        # Take minimum of the two (chunk must be relevant to both)
        min_score = min(claim_score, contra_score)
        
        # Weighted combination: prioritize chunks relevant to both
        final_score = 0.6 * combined_score + 0.4 * min_score
        
        scored_chunks.append((chunk_text, final_score))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Select diverse top-k chunks
    return diversified_top_k(scored_chunks, top_k)

def retrieve_context_chunks(claim_chunks: List[str], book_vectors: List[Dict],
                           top_k: int = 2) -> List[str]:
    """
    Retrieve additional context chunks that are near the claim chunks.
    This can help provide broader context for verification.
    """
    if not claim_chunks:
        return []
    
    # Get embeddings of already selected chunks
    selected_embs = [embed(chunk) for chunk in claim_chunks]
    avg_emb = np.mean(selected_embs, axis=0)
    
    scored_chunks = []
    for vec in book_vectors:
        chunk_text = vec['chunk_text']
        
        # Skip if already selected
        if chunk_text in claim_chunks:
            continue
        
        score = cosine_sim(avg_emb, vec['embedding'])
        scored_chunks.append((chunk_text, score))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Get diverse context chunks
    return diversified_top_k(scored_chunks, top_k)

# =====================================================
# LOAD VECTOR STORE
# =====================================================
print("Loading vector store...")
vector_store = {}
book_name_mapping = {}

with open(VECTOR_STORE_PATH, 'r') as f:
    for line in f:
        data = json.loads(line)
        book = data['book_name'].strip('"').replace('\\"', '').strip()
        book_lower = book.lower()
        
        if book_lower not in vector_store:
            vector_store[book_lower] = []
            book_name_mapping[book_lower] = book
        
        vector_store[book_lower].append({
            'chunk_text': data['chunk_text'],
            'embedding': data['embedding']
        })

print(f"Loaded vectors for {len(vector_store)} books:")
for book_lower in vector_store.keys():
    print(f"  '{book_name_mapping[book_lower]}': {len(vector_store[book_lower])} chunks")

# =====================================================
# LOAD INPUT CSV
# =====================================================
print("\nLoading input CSV...")
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} rows")

# =====================================================
# ADD NEW COLUMNS FOR CHUNKS
# =====================================================
for i in range(1, MAX_CLAIMS + 1):
    df[f'claim_{i}_chunks'] = ""
    df[f'claim_{i}_chunks_metadata'] = ""  # Store metadata about retrieval

# =====================================================
# PROCESS EACH ROW
# =====================================================
print("\nProcessing rows...")
for idx, row in df.iterrows():
    book_name = row['book_name']
    book_name_lower = book_name.lower()
    
    # Get vector store for this book
    book_vectors = vector_store.get(book_name_lower, [])
    if not book_vectors:
        print(f"Row {idx}: No vectors found for '{book_name}'")
        continue
    
    # Parse claims and contradictions
    try:
        claims = json.loads(row['claims'])[:MAX_CLAIMS]
        contradictions = json.loads(row['contradictions'])
    except Exception as e:
        print(f"Row {idx}: Error parsing claims/contradictions - {e}")
        continue
    
    # Process each claim
    for claim_obj in claims:
        claim_id = claim_obj['claim_id']
        claim_text = claim_obj['claim_text']
        keywords = claim_obj.get('keywords', [])
        
        # 1. Get chunks supporting the claim
        claim_chunks = retrieve_for_claim(claim_text, keywords, book_vectors, 
                                         FINAL_CHUNKS_PER_CLAIM)
        
        # 2. Get chunks for each contradiction
        contra_texts = contradictions.get(str(claim_id), [])[:3]
        contradiction_chunks = []
        
        for contra_text in contra_texts:
            contra_chunks = retrieve_for_contradiction(contra_text, claim_text, 
                                                      book_vectors, 
                                                      FINAL_CHUNKS_PER_CONTRADICTION)
            contradiction_chunks.extend(contra_chunks)
        
        # 3. Get context chunks (optional - can be disabled if too many chunks)
        # context_chunks = retrieve_context_chunks(claim_chunks, book_vectors, top_k=1)
        
        # Combine all chunks (remove duplicates while preserving order)
        all_chunks = []
        seen = set()
        for chunk in claim_chunks + contradiction_chunks:  # + context_chunks
            if chunk not in seen:
                all_chunks.append(chunk)
                seen.add(chunk)
        
        # Store in the appropriate column
        col_name = f'claim_{claim_id}_chunks'
        df.at[idx, col_name] = json.dumps(all_chunks)
        
        # Store metadata
        metadata = {
            'num_claim_chunks': len(claim_chunks),
            'num_contradiction_chunks': len(contradiction_chunks),
            'total_chunks': len(all_chunks),
            'num_contradictions': len(contra_texts)
        }
        df.at[idx, f'claim_{claim_id}_chunks_metadata'] = json.dumps(metadata)
    
    if (idx + 1) % 5 == 0:
        print(f"Processed {idx + 1}/{len(df)} rows")

print(f"Processed all {len(df)} rows")

# =====================================================
# SAVE OUTPUT
# =====================================================
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to {OUTPUT_PATH}")

# =====================================================
# SHOW STATISTICS
# =====================================================
print("\n" + "="*60)
print("RETRIEVAL STATISTICS")
print("="*60)

for i in range(1, MAX_CLAIMS + 1):
    col = f'claim_{i}_chunks'
    metadata_col = f'claim_{i}_chunks_metadata'
    
    if col in df.columns:
        # Count rows with chunks
        has_chunks = df[col].apply(lambda x: len(json.loads(x)) if x else 0)
        avg_chunks = has_chunks[has_chunks > 0].mean()
        
        print(f"\nClaim {i}:")
        print(f"  Rows with chunks: {(has_chunks > 0).sum()}/{len(df)}")
        print(f"  Average chunks per claim: {avg_chunks:.2f}")
        print(f"  Max chunks: {has_chunks.max()}")
        print(f"  Min chunks (non-zero): {has_chunks[has_chunks > 0].min() if (has_chunks > 0).any() else 0}")

# Show sample
print("\n" + "="*60)
print("SAMPLE OUTPUT")
print("="*60)
if len(df) > 0:
    first_row = df.iloc[0]
    print(f"\nBook: {first_row['book_name']}")
    for i in range(1, MAX_CLAIMS + 1):
        col = f'claim_{i}_chunks'
        metadata_col = f'claim_{i}_chunks_metadata'
        
        if col in df.columns and first_row[col]:
            chunks = json.loads(first_row[col])
            metadata = json.loads(first_row[metadata_col]) if first_row[metadata_col] else {}
            
            print(f"\n  Claim {i}:")
            print(f"    Total chunks: {len(chunks)}")
            print(f"    Metadata: {metadata}")
            print(f"    First chunk preview: {chunks[0][:100]}..." if chunks else "    No chunks")