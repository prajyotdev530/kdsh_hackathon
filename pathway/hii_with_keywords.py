import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# =====================================================
# CONFIG
# =====================================================
VECTOR_STORE_PATH = "./vector_store"
INPUT_PATH = "./Dataset/train_with_claims_and_contradictions.csv"
OUTPUT_PATH = "./Dataset/train_with_chunks.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CLAIMS = 4

model = SentenceTransformer(MODEL_NAME)

# =====================================================
# UTILITIES
# =====================================================
def embed(text):
    if text is None or text == "":
        return [0.0] * 384
    return model.encode(str(text).strip()).tolist()

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)

# =====================================================
# LOAD VECTOR STORE
# =====================================================
print("Loading vector store...")
vector_store = {}
book_name_mapping = {}  # To store lowercase -> original mapping

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

# =====================================================
# PROCESS EACH ROW
# =====================================================
print("\nProcessing rows...")
for idx, row in df.iterrows():
    book_name = row['book_name']
    book_name_lower = book_name.lower()
    
    # Get vector store for this book (case-insensitive)
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
        
        # Combine claim text with keywords for better search
        search_text = claim_text
        if keywords:
            search_text = claim_text + " " + " ".join(keywords)
        
        # Get 2 best chunks for the claim
        claim_emb = embed(search_text)
        claim_scores = []
        for vec in book_vectors:
            score = cosine_sim(claim_emb, vec['embedding'])
            claim_scores.append((vec['chunk_text'], score))
        claim_scores.sort(key=lambda x: x[1], reverse=True)
        top_2_claim_chunks = [chunk for chunk, _ in claim_scores[:2]]
        
        # Get 1 chunk for each of the 3 contradictions
        contra_texts = contradictions.get(str(claim_id), [])[:3]
        contra_chunks = []
        
        for contra_text in contra_texts:
            contra_emb = embed(contra_text)
            contra_scores = []
            for vec in book_vectors:
                score = cosine_sim(contra_emb, vec['embedding'])
                contra_scores.append((vec['chunk_text'], score))
            contra_scores.sort(key=lambda x: x[1], reverse=True)
            if contra_scores:
                contra_chunks.append(contra_scores[0][0])
        
        # Combine: 2 claim chunks + up to 3 contradiction chunks = up to 5 total
        all_chunks = top_2_claim_chunks + contra_chunks
        
        # Store in the appropriate column
        col_name = f'claim_{claim_id}_chunks'
        df.at[idx, col_name] = json.dumps(all_chunks)
    
    if (idx + 1) % 5 == 0:
        print(f"Processed {idx + 1}/{len(df)} rows")

print(f"Processed all {len(df)} rows")

# =====================================================
# SAVE OUTPUT
# =====================================================
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to {OUTPUT_PATH}")

# Show sample
print("\nSample output:")
if len(df) > 0:
    first_row = df.iloc[0]
    print(f"Book: {first_row['book_name']}")
    for i in range(1, MAX_CLAIMS + 1):
        col = f'claim_{i}_chunks'
        if col in df.columns and first_row[col]:
            chunks = json.loads(first_row[col])
            print(f"  Claim {i}: {len(chunks)} chunks")