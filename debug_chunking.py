import pathway as pw
from pathway.xpacks.llm.splitters import TokenCountSplitter

# Configuration
DATA_DIR = "./Dataset/Books"

# 1. Read the files (Mode="static" means read once and stop, not streaming)
# format="string" ensures we read it as text, not raw binary bytes
files = pw.io.fs.read(
    DATA_DIR,
    format="plaintext",
    mode="static",
    with_metadata=True
)

# 2. Define and apply the Splitter
# This is the exact logic that will be used in your final app
splitter = TokenCountSplitter(
    min_tokens=100,
    max_tokens=512,
    encoding_name="cl100k_base" # OpenAI's encoding
)

# 3. Apply the Splitter - apply on the data column (text content)
# The splitter returns a column of list of (chunk_text, metadata) tuples
# Add the split result as a new column to the table
files_with_chunks = files.select(
    _metadata=pw.this._metadata,
    chunks=splitter(pw.this.data)
)

# 4. Flatten the chunks - explode the list into individual rows
flattened = files_with_chunks.flatten(
    pw.this.chunks,
    origin_id="_origin_id"
)

# 5. Extract text and metadata from the chunks column
# Each chunk is a tuple of (text, metadata)
chunks = flattened.select(
    text=flattened.chunks[0],
    metadata=flattened.chunks[1],
    _origin_id=flattened._origin_id
)

# 6. Run and Print
print(f"--- ✂️  Splitting novels in {DATA_DIR} ---")
result = pw.debug.table_to_pandas(chunks)
print(f"Total chunks created: {len(result)}")
print(f"\n--- Sample Chunks ---")
for i in range(min(3, len(result))):
    print(f"\nChunk {i+1} (origin: {result.iloc[i]['_origin_id']}):")
    print(f"Text: {result.iloc[i]['text'][:150]}...")
    print(f"Metadata: {result.iloc[i]['metadata']}")
