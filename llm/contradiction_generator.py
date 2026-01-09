"""
Claim Contradiction Generator (FINAL)
------------------------------------
For each claim in trained_with_claims.csv:
- Generate exactly 3 logical contradictions
- Store contradictions as JSON per row
- Training-safe and reproducible
"""

import csv
import json
from openai import OpenAI
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

INPUT_CSV = "trained_with_claims.csv"
OUTPUT_CSV = "trained_with_claims_and_contradictions.csv"
MODEL = "gpt-4o-mini"
NUM_CONTRADICTIONS = 3

client = OpenAI()

# =====================================================
# LLM CONTRADICTION GENERATION
# =====================================================

def generate_contradictions(claim_text: str) -> list:
    """
    Generate exactly 3 narrative contradictions for a claim.
    """

    prompt = f"""
Generate exactly {NUM_CONTRADICTIONS} narrative contradictions
for the following factual claim.

Rules:
- Each contradiction must logically negate the claim
- Must be plausible sentences from a novel
- No explanations
- Return ONLY valid JSON list of strings

CLAIM:
"{claim_text}"

FORMAT:
[
  "contradiction 1",
  "contradiction 2",
  "contradiction 3"
]
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )

        text = response.choices[0].message.content.strip()
        contradictions = json.loads(text)

        if not isinstance(contradictions, list):
            raise ValueError("LLM output not a list")

    except Exception as e:
        print(f"    ⚠️ LLM failed, using fallback: {e}")
        contradictions = get_fallback_contradictions(claim_text)

    # HARD GUARANTEE: EXACTLY 3
    contradictions = contradictions[:NUM_CONTRADICTIONS]
    while len(contradictions) < NUM_CONTRADICTIONS:
        contradictions.append(
            f"Narrative evidence contradicts the claim that {claim_text}"
        )

    return contradictions

# =====================================================
# FALLBACK CONTRADICTIONS (TRAINING-SAFE)
# =====================================================

def get_fallback_contradictions(claim_text: str) -> list:
    return [
        f"The story shows that the opposite of '{claim_text}' is true.",
        f"Events in the novel contradict the idea that '{claim_text}'.",
        f"The narrative provides evidence against '{claim_text}'."
    ]

# =====================================================
# CSV PIPELINE
# =====================================================

def process_csv():
    if not Path(INPUT_CSV).exists():
        print(f" Input file not found: {INPUT_CSV}")
        return

    print("=" * 70)
    print("CLAIM CONTRADICTION GENERATOR (FINAL)")
    print("=" * 70 + "\n")

    with open(INPUT_CSV, newline="", encoding="utf-8") as infile, \
         open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)

        if "claims" not in reader.fieldnames:
            print("'claims' column missing in input CSV")
            return

        fieldnames = reader.fieldnames + ["contradictions"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(reader, start=1):
            row_id = row.get("id", idx)
            char = row.get("char", "?")

            print(f"[{idx}] 🔄 Processing ID={row_id} | Char={char}")

            try:
                claims = json.loads(row["claims"])
                contradiction_map = {}

                for claim in claims:
                    cid = str(claim["claim_id"])
                    ctext = claim["claim_text"]

                    print(f"     Claim {cid}: {ctext[:50]}...")
                    contradiction_map[cid] = generate_contradictions(ctext)

                row["contradictions"] = json.dumps(
                    contradiction_map,
                    ensure_ascii=False
                )

            except Exception as e:
                print(f"Error processing row: {e}")
                row["contradictions"] = "{}"

            writer.writerow(row)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Output saved to: {OUTPUT_CSV}")

# =====================================================
# SAMPLE VIEWER
# =====================================================

def view_sample(csv_path: str, num_rows: int = 1):
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT")
    print("=" * 70 + "\n")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if i >= num_rows:
                break

            print(f"Row {i+1} | ID={row.get('id')} | Char={row.get('char')}")
            contradictions = json.loads(row["contradictions"])

            for cid, cons in contradictions.items():
                print(f"Claim {cid}:")
                for j, c in enumerate(cons, 1):
                    print(f"{j}. {c}")
            print()

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    process_csv()

    if Path(OUTPUT_CSV).exists():
        view_sample(OUTPUT_CSV, num_rows=1)