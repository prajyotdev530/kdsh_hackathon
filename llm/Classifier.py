# verifier.py
import json
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

# Setup logging for audit trail
logger = logging.getLogger("verifier")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# ---------------------------
# LLM client interface
# ---------------------------
class LLMClient:
    """
    Minimal interface: concrete classes must implement generate_chat(messages, timeout_s).
    messages: list of {"role": "system"/"user"/"assistant", "content": str}
    returns: the assistant text as string
    """
    def generate_chat(self, messages: List[Dict[str,str]], timeout_s: int) -> str:
        raise NotImplementedError

# Example OpenAI client (replace or extend for local models)
try:
    import openai
    class OpenAIChatClient(LLMClient):
        def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 512):
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens

        def generate_chat(self, messages: List[Dict[str,str]], timeout_s: int) -> str:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=timeout_s,
            )
            return resp["choices"][0]["message"]["content"]
except Exception:
    OpenAIChatClient = None  # user can implement their own client


# ---------------------------
# Prompts (strict)
# ---------------------------
SYSTEM_PROMPT = """You are a strict verifier. YOU MUST follow the rules below exactly.
1) You may ONLY use the provided chunks to decide. Do NOT use any external knowledge or world facts.
2) You MUST output ONLY a single JSON object (no other text). The JSON schema is described below.
3) Quotes must be exact substrings taken from the provided chunks. If you cannot quote exact evidence that supports or contradicts the claim, set verdict = "INSUFFICIENT".
4) For SUPPORTED or CONTRADICTED you MUST include at least one evidence object containing chunk_id and quote.
5) Keep "justification" to 1-2 short sentences that reference the provided quotes and DO NOT invent new facts.
6) Confidence must be one of: HIGH, MEDIUM, LOW. If unsure, use MEDIUM.
7) If you cannot produce valid JSON, output a single-line valid JSON with verdict = "INSUFFICIENT" and confidence = "LOW".
Output schema:
{
  "claim_id": <int>,
  "verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT",
  "evidence": [ { "chunk_id": <int>, "quote": "<exact substring>" }, ... ],
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "justification": "<1-2 sentence justification referencing the quoted text>"
}
"""

# stricter retry prompt (if initial response malformed)
RETRY_SYSTEM_PROMPT = SYSTEM_PROMPT + "\nIMPORTANT: This is a retry. If your previous response was not valid JSON, be more concise and ensure the output is EXACT JSON."

USER_PROMPT_TEMPLATE = """Claim (id={claim_id}): "{claim_text}"

Chunks (indexed):
{chunk_listing}

Remember: ONLY use the text above. Output ONLY the JSON described in the system prompt.
"""

# ---------------------------
# Helpers: JSON extraction & validation
# ---------------------------
JSON_OBJ_RE = re.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)  # robust attempt (may not work in all pythons)

def extract_json_from_text(text: str) -> Optional[str]:
    """
    Try to find first JSON object in the text. Fallback to heuristic.
    """
    # quick attempt: find first balanced {...}
    # safer fallback: naive find from first { to last }
    text = text.strip()
    # attempt regex
    m = JSON_OBJ_RE.search(text)
    if m:
        return m.group(0)
    # fallback: naive
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return None

def validate_verifier_json(obj: Dict[str,Any], chunks: List[str]) -> Tuple[bool, str]:
    """
    Validate the parsed JSON object against rules:
    - required fields
    - verdict allowed values
    - evidence quotes are exact substrings of corresponding chunk text
    - evidence present for SUPPORTED/CONTRADICTED
    """
    required = {"claim_id","verdict","evidence","confidence","justification"}
    if not required.issubset(set(obj.keys())):
        return False, f"Missing fields: {required - set(obj.keys())}"
    if obj["verdict"] not in ("SUPPORTED","CONTRADICTED","INSUFFICIENT"):
        return False, f"Invalid verdict: {obj['verdict']}"
    if obj["confidence"] not in ("HIGH","MEDIUM","LOW"):
        return False, f"Invalid confidence: {obj['confidence']}"
    evidence = obj["evidence"]
    if obj["verdict"] in ("SUPPORTED","CONTRADICTED"):
        if not isinstance(evidence, list) or len(evidence) == 0:
            return False, "SUPPORT/CONTRADICTED requires at least one evidence item"
    # verify each evidence quote is exact substring of chunk
    if not isinstance(evidence, list):
        return False, "Evidence must be a list"
    for ev in evidence:
        if "chunk_id" not in ev or "quote" not in ev:
            return False, "Each evidence must have chunk_id and quote"
        cid = ev["chunk_id"]
        if not (0 <= cid < len(chunks)):
            return False, f"chunk_id {cid} out of range"
        quote = ev["quote"]
        if quote not in chunks[cid]:
            return False, f"Quote not exact substring of chunk {cid}: {repr(quote)[:60]}"
    # justification must be short
    if not isinstance(obj["justification"], str) or len(obj["justification"].split()) > 40:
        # allow up to ~40 words
        return False, "Justification too long or not a string"
    return True, "OK"

# ---------------------------
# Verifier logic: call LLM with retries and validation
# ---------------------------
def call_verifier_llm(
    llm: LLMClient,
    claim_id: int,
    claim_text: str,
    chunks: List[str],
    timeout_s: int = 20,
    max_retries: int = 2,
    top_k_limit: int = 5,
) -> Dict[str,Any]:
    """
    Execute the verifier LLM for one claim and its chunks.
    Returns a validated dict matching the schema (or a safe INSUFFICIENT fallback).
    """
    # enforce chunk limit
    chunks = chunks[:top_k_limit]
    chunk_listing = "\n".join([f"[{i}] {chunks[i]}" for i in range(len(chunks))])

    user_prompt = USER_PROMPT_TEMPLATE.format(claim_id=claim_id, claim_text=claim_text.replace('"','\\"'), chunk_listing=chunk_listing)

    attempts = 0
    system_prompt = SYSTEM_PROMPT
    while attempts < max_retries:
        attempts += 1
        if attempts == 2:
            # stricter system prompt on retry
            system_prompt = RETRY_SYSTEM_PROMPT

        messages = [
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": user_prompt}
        ]
        try:
            text = llm.generate_chat(messages, timeout_s=timeout_s)
            logger.info(f"LLM raw response (attempt {attempts}): {text[:400]}")
        except Exception as e:
            logger.exception("LLM call failed")
            text = ""
        # extract JSON
        json_text = extract_json_from_text(text)
        if not json_text:
            logger.warning("No JSON found in LLM response")
            # retry
            continue
        try:
            parsed = json.loads(json_text)
        except Exception as e:
            logger.warning("JSON parse failed: %s", e)
            continue
        # validate against chunks
        ok, reason = validate_verifier_json(parsed, chunks)
        if ok:
            # optional: for CONTRADICTED, do a second confirmation call
            if parsed["verdict"] == "CONTRADICTED":
                confirmed = confirm_contradiction(llm, parsed, claim_text, chunks, timeout_s=timeout_s)
                if not confirmed:
                    # downgrade to INSUFFICIENT with LOW confidence and log
                    logger.info("Contradiction not confirmed by second verifier -> downgrade to INSUFFICIENT")
                    return {
                        "claim_id": claim_id,
                        "verdict": "INSUFFICIENT",
                        "evidence": [],
                        "confidence": "LOW",
                        "justification": "Contradiction not confirmed by second verifier."
                    }
            # success
            parsed.setdefault("claim_id", claim_id)
            parsed["model_attempts"] = attempts
            parsed["timestamp"] = time.time()
            logger.info("Verifier output accepted")
            return parsed
        else:
            logger.warning("Validation failed: %s", reason)
            # retry
            continue

    # exhausted retries -> return safe fallback
    logger.error("Verifier failed after retries; returning safe INSUFFICIENT fallback")
    return {
        "claim_id": claim_id,
        "verdict": "INSUFFICIENT",
        "evidence": [],
        "confidence": "LOW",
        "justification": "Verifier failed to produce valid output after retries."
    }

def confirm_contradiction(llm: LLMClient, parsed_output: Dict[str,Any], claim_text: str, chunks: List[str], timeout_s: int = 20) -> bool:
    """
    Conservative second-check for contradictions. Calls verifier again but with a prompt
    that asks to confirm or refute the contradiction using the same chunks.
    Return True only if second call returns CONTRADICTED with valid evidence.
    """
    # Build a short confirmation user prompt referencing the original evidence
    evidence_snippet = "\n".join([f"[{ev['chunk_id']}] {chunks[ev['chunk_id']]}" for ev in parsed_output.get("evidence", [])])
    user_prompt = (
        f"This is a confirmation check. Original claim (id={parsed_output['claim_id']}): \"{claim_text}\"\n\n"
        f"Original alleged contradictory evidence (quotes shown below):\n{evidence_snippet}\n\n"
        "Using ONLY the provided chunks above, confirm whether the claim is contradicted. "
        "Output the same JSON schema as before."
    )
    messages = [
        {"role":"system", "content": RETRY_SYSTEM_PROMPT},
        {"role":"user", "content": user_prompt}
    ]
    try:
        text = llm.generate_chat(messages, timeout_s=timeout_s)
    except Exception:
        logger.exception("Second verifier call failed")
        return False
    json_text = extract_json_from_text(text)
    if not json_text:
        logger.warning("No JSON in second verifier response")
        return False
    try:
        parsed2 = json.loads(json_text)
    except Exception:
        logger.warning("Second verifier JSON parse failed")
        return False
    ok, reason = validate_verifier_json(parsed2, chunks)
    if not ok:
        logger.warning("Second verifier validation failed: %s", reason)
        return False
    return parsed2.get("verdict") == "CONTRADICTED"

# ---------------------------
# Aggregation helpers
# ---------------------------
def aggregate_backstory_verdict(per_claim_results: List[Dict[str,Any]]) -> Dict[str,Any]:
    """
    Deterministic aggregation:
    - If any STRONG contradiction -> INCONSISTENT
    - Else if all claims are SUPPORTED or SUPPORTED+INSUFFICIENT -> CONSISTENT
    - Else -> UNKNOWN (flag for review)
    A 'strong contradiction' is a CONTRADICTED with confidence HIGH or MEDIUM and with evidence.
    """
    any_strong_contra = False
    any_supported = False
    any_insufficient = False
    low_conflict = False

    for res in per_claim_results:
        v = res["verdict"]
        c = res.get("confidence","MEDIUM")
        if v == "CONTRADICTED" and c in ("HIGH","MEDIUM") and len(res.get("evidence",[]))>0:
            any_strong_contra = True
        if v == "SUPPORTED":
            any_supported = True
        if v == "INSUFFICIENT":
            any_insufficient = True

    if any_strong_contra:
        final = "INCONSISTENT"
    else:
        # if there is at least one supported and no contradictions -> CONSISTENT
        if any_supported and not any_strong_contra:
            final = "CONSISTENT"
        else:
            final = "UNKNOWN"

    return {
        "final_verdict": final,
        "per_claim": per_claim_results,
        "summary": {
            "num_claims": len(per_claim_results),
            "num_supported": sum(1 for r in per_claim_results if r["verdict"]=="SUPPORTED"),
            "num_contradicted": sum(1 for r in per_claim_results if r["verdict"]=="CONTRADICTED"),
            "num_insufficient": sum(1 for r in per_claim_results if r["verdict"]=="INSUFFICIENT"),
        }
    }

# ---------------------------
# Utility: claim validation gate (checks claim present in backstory)
# ---------------------------
def validate_claim_against_backstory(
    llm: LLMClient,
    claim_text: str,
    backstory_text: str,
    timeout_s: int = 20
) -> str:
    """
    Return "VALID" | "INVALID" | "AMBIGUOUS".
    This uses an evidence-locked call where only backstory_text is provided.
    """
    system = (
        "You are a strict claim validator. Use ONLY the provided backstory text. "
        "Decide whether the claim is explicitly stated, reasonably implied, or NOT present. "
        "Output a single word: VALID, INVALID, or AMBIGUOUS. No other text."
    )
    user = f"Backstory:\n{backstory_text}\n\nClaim: \"{claim_text}\"\n\nAnswer with one word: VALID, INVALID, or AMBIGUOUS."
    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    try:
        text = llm.generate_chat(messages, timeout_s=timeout_s)
    except Exception:
        logger.exception("Claim validator LLM call failed")
        return "AMBIGUOUS"
    text = text.strip().upper()
    # normalize
    if text.startswith("VALID"):
        return "VALID"
    if text.startswith("INVALID") or text.startswith("NO") or "NOT" in text:
        return "INVALID"
    return "AMBIGUOUS"

# ---------------------------
# Example integration function (one backstory)
# ---------------------------
def verify_backstory(
    llm: LLMClient,
    backstory_claims: List[Tuple[int,str]],
    backstory_text: str,
    retrieval_fn,  # function(claim_text) -> List[str]  (top-k chunks)
    timeout_s: int = 20
) -> Dict[str,Any]:
    """
    Orchestrates claim validation, verification per claim, and aggregation.
    backstory_claims: list of (claim_id, claim_text)
    retrieval_fn: user-provided function that returns the list of chunks (strings) for a claim
    """
    per_claim_results = []
    for claim_id, claim_text in backstory_claims:
        # 1) Claim validation gate
        status = validate_claim_against_backstory(llm, claim_text, backstory_text, timeout_s=timeout_s)
        logger.info(f"Claim {claim_id} validation status: {status}")
        if status != "VALID":
            # invalid or ambiguous -> mark insufficient (do not allow to flip backstory)
            per_claim_results.append({
                "claim_id": claim_id,
                "verdict": "INSUFFICIENT",
                "evidence": [],
                "confidence": "LOW",
                "justification": f"Claim validation status: {status}"
            })
            continue
        # 2) Retrieve evidence chunks for this claim
        chunks = retrieval_fn(claim_text)
        if not chunks:
            per_claim_results.append({
                "claim_id": claim_id,
                "verdict": "INSUFFICIENT",
                "evidence": [],
                "confidence": "LOW",
                "justification": "No chunks retrieved"
            })
            continue
        # 3) Call verifier LLM
        result = call_verifier_llm(llm, claim_id, claim_text, chunks, timeout_s=timeout_s)
        # annotate with raw chunks for audit
        result["chunks_used"] = chunks
        per_claim_results.append(result)

    # 4) Aggregate
    final = aggregate_backstory_verdict(per_claim_results)
    logger.info(f"Aggregation result: {final['final_verdict']}")
    return final

# ---------------------------
# Example usage (pseudo)
# ---------------------------
if __name__ == "__main__":
    # Example: if you have OpenAI key and want to test:
    if OpenAIChatClient is None:
        print("OpenAIChatClient not available. Implement LLMClient for your setup.")
    else:
        client = OpenAIChatClient(model="gpt-4o-mini", temperature=0.0, max_tokens=512)
        # dummy retrieval fn (replace with your vector retrieval returning strings)
        def retrieval_fn_demo(claim_text):
            return [
                "Alice walked into the room and closed the door behind her.",
                "Later, Alice left without saying goodbye. She was upset.",
                "There was no mention of a red hat in the story."
            ]
        backstory_claims = [(1, "Alice closed the door when she entered the room."), (2, "Alice was wearing a red hat.")]
        backstory_text = "Alice entered, closed the door, and later left. She was upset."
        res = verify_backstory(client, backstory_claims, backstory_text, retrieval_fn_demo)
        print(json.dumps(res, indent=2))
        