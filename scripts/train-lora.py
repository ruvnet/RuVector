#!/usr/bin/env python3
"""LoRA training epochs and drift monitoring for pi.ruv.io brain.

Submits 3 LoRA delta submissions from 3 different contributor keys to
trigger Byzantine-tolerant federated aggregation (min_submissions=3),
checks drift after each, and reports final state.

The server expects LoraSubmission with:
  - down_proj: Vec<f32> of size hidden_dim * rank = 128 * 2 = 256
  - up_proj:   Vec<f32> of size rank * hidden_dim = 2 * 128 = 256
  - rank: 2
  - hidden_dim: 128
  - evidence_count: u64 (>= 5)
"""
import json
import random
import urllib.request
import urllib.error
import time
import hashlib
import sys

BASE = "https://ruvbrain-875130704813.us-central1.run.app"

# Server defaults: Rank-2, 128-dim
RANK = 2
HIDDEN_DIM = 128
PROJ_SIZE = HIDDEN_DIM * RANK  # 256 floats each for down_proj and up_proj

# 3 distinct contributor keys for Byzantine aggregation testing
CONTRIBUTOR_KEYS = [
    "lora-trainer-key-alpha-" + hashlib.sha256(b"contributor-0").hexdigest()[:16],
    "lora-trainer-key-beta-" + hashlib.sha256(b"contributor-1").hexdigest()[:16],
    "lora-trainer-key-gamma-" + hashlib.sha256(b"contributor-2").hexdigest()[:16],
]


def make_headers(key):
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }


def api_get(path, key=None):
    """GET request, returns (data_dict, status_code)."""
    headers = make_headers(key or CONTRIBUTOR_KEYS[0])
    req = urllib.request.Request(f"{BASE}{path}", headers=headers, method="GET")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        body = resp.read().decode()
        return json.loads(body) if body else {}, resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500]
        return {"error": body, "code": e.code}, e.code
    except Exception as e:
        return {"error": str(e)}, 0


def api_post(path, data, key=None):
    """POST request, returns (data_dict, status_code)."""
    headers = make_headers(key or CONTRIBUTOR_KEYS[0])
    payload = json.dumps(data).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=payload, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        body = resp.read().decode()
        return json.loads(body) if body else {}, resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500]
        return {"error": body, "code": e.code}, e.code
    except Exception as e:
        return {"error": str(e)}, 0


def generate_proj_weights(size=PROJ_SIZE, std=0.01):
    """Generate small random weights centered around 0 with given std deviation."""
    return [round(random.gauss(0.0, std), 8) for _ in range(size)]


def weight_stats(weights):
    """Compute min/max/mean/norm for a weight vector."""
    mn = min(weights)
    mx = max(weights)
    mean = sum(weights) / len(weights)
    norm = sum(w * w for w in weights) ** 0.5
    return {"min": round(mn, 6), "max": round(mx, 6), "mean": round(mean, 6), "norm": round(norm, 4)}


def print_separator():
    print("=" * 60)


def main():
    print_separator()
    print("LoRA Training & Drift Monitoring for pi.ruv.io")
    print(f"  Base URL: {BASE}")
    print(f"  Contributors: {len(CONTRIBUTOR_KEYS)}")
    print(f"  Rank: {RANK}, Hidden dim: {HIDDEN_DIM}")
    print(f"  Projection size (each): {PROJ_SIZE}")
    print_separator()

    # --- Step 1: Check current LoRA state ---
    print("\n[1] Checking current LoRA state: GET /v1/lora/latest")
    lora_state, status = api_get("/v1/lora/latest")
    print(f"    Status: {status}")
    print(f"    Response: {json.dumps(lora_state, indent=2)[:500]}")

    current_epoch = 0
    if status == 200 and "epoch" in lora_state:
        current_epoch = lora_state["epoch"]
        print(f"    Current epoch: {current_epoch}")
    else:
        print("    No existing LoRA state found or error, starting from epoch 0")

    # --- Step 2: Check initial drift ---
    print("\n[2] Checking initial drift: GET /v1/drift")
    drift_initial, drift_status = api_get("/v1/drift")
    print(f"    Status: {drift_status}")
    print(f"    Response: {json.dumps(drift_initial, indent=2)[:500]}")

    # --- Step 3: Submit 3 LoRA submissions from 3 different contributors ---
    # Server requires min_submissions=3 before aggregation triggers a new epoch
    print("\n[3] Submitting 3 LoRA deltas from 3 contributors...")
    print("    (Server aggregates after 3 submissions via Byzantine-tolerant federation)")
    errors = []

    for i in range(3):
        contributor_key = CONTRIBUTOR_KEYS[i]
        contributor_label = ["alpha", "beta", "gamma"][i]

        print(f"\n    --- Submission {i+1}/3 (contributor: {contributor_label}) ---")

        # Generate random LoRA delta weights for down_proj and up_proj
        down_proj = generate_proj_weights()
        up_proj = generate_proj_weights()

        print(f"    down_proj stats: {weight_stats(down_proj)}")
        print(f"    up_proj stats:   {weight_stats(up_proj)}")

        # Build LoraSubmission matching the server's expected schema
        payload = {
            "down_proj": down_proj,
            "up_proj": up_proj,
            "rank": RANK,
            "hidden_dim": HIDDEN_DIM,
            "evidence_count": 10 + i * 5,  # >= 5 required
        }

        print(f"    POST /v1/lora/submit (rank={RANK}, hidden_dim={HIDDEN_DIM}, evidence={payload['evidence_count']})")
        result, submit_status = api_post("/v1/lora/submit", payload, key=contributor_key)
        print(f"    Submit status: {submit_status}")
        print(f"    Submit response: {json.dumps(result, indent=2)[:400]}")

        if submit_status not in (200, 201):
            errors.append(f"Submission {i+1}: failed with status {submit_status}: {result}")

        # Brief pause between submissions
        time.sleep(0.5)

        # Check drift after this submission
        print(f"\n    Checking drift after submission {i+1}: GET /v1/drift")
        drift_data, drift_s = api_get("/v1/drift")
        print(f"    Drift status: {drift_s}")
        print(f"    Drift response: {json.dumps(drift_data, indent=2)[:400]}")

    # --- Step 4: Check final LoRA state (should show new epoch after 3 submissions) ---
    print("\n" + "=" * 60)
    print("[4] Final LoRA state: GET /v1/lora/latest")
    final_lora, final_status = api_get("/v1/lora/latest")
    print(f"    Status: {final_status}")
    # Print truncated weights info
    if final_status == 200 and final_lora.get("weights"):
        w = final_lora["weights"]
        summary = {
            "epoch": final_lora.get("epoch"),
            "rank": w.get("rank"),
            "hidden_dim": w.get("hidden_dim"),
            "contributor_count": w.get("contributor_count"),
            "total_evidence": w.get("total_evidence"),
            "down_proj_len": len(w.get("down_proj", [])),
            "up_proj_len": len(w.get("up_proj", [])),
        }
        if w.get("down_proj"):
            summary["down_proj_sample"] = [round(x, 6) for x in w["down_proj"][:5]]
        if w.get("up_proj"):
            summary["up_proj_sample"] = [round(x, 6) for x in w["up_proj"][:5]]
        print(f"    Consensus summary: {json.dumps(summary, indent=2)}")
    else:
        print(f"    Response: {json.dumps(final_lora, indent=2)[:600]}")

    # --- Step 5: Final drift report ---
    print("\n[5] Final drift report: GET /v1/drift")
    final_drift, final_drift_status = api_get("/v1/drift")
    print(f"    Status: {final_drift_status}")
    print(f"    Response: {json.dumps(final_drift, indent=2)[:600]}")

    # --- Step 6: Check status for LoRA info ---
    print("\n[6] Server status: GET /v1/status")
    srv_status, srv_code = api_get("/v1/status")
    if srv_code == 200:
        lora_info = {
            "lora_epoch": srv_status.get("lora_epoch"),
            "lora_pending_submissions": srv_status.get("lora_pending_submissions"),
            "total_memories": srv_status.get("total_memories"),
            "total_contributors": srv_status.get("total_contributors"),
        }
        print(f"    {json.dumps(lora_info, indent=2)}")
    else:
        print(f"    Status: {srv_code} - {json.dumps(srv_status)[:200]}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    final_epoch = final_lora.get("epoch", "?")
    print(f"  Starting epoch: {current_epoch}")
    print(f"  Final epoch:    {final_epoch}")

    if final_lora.get("weights"):
        w = final_lora["weights"]
        print(f"  Contributors:   {w.get('contributor_count', '?')}")
        print(f"  Total evidence: {w.get('total_evidence', '?')}")

    if final_drift_status == 200:
        print(f"  Drift detected: {final_drift.get('is_drifting', 'unknown')}")
        print(f"  Drift trend:    {final_drift.get('trend', 'unknown')}")
        print(f"  Drift CoV:      {final_drift.get('coefficient_of_variation', 'unknown')}")
        print(f"  Delta sparsity: {final_drift.get('delta_sparsity', 'unknown')}")
        print(f"  Window size:    {final_drift.get('window_size', 'unknown')}")
        print(f"  Suggested:      {final_drift.get('suggested_action', 'unknown')}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for err in errors:
            print(f"    - {err[:200]}")
    else:
        print("\n  Errors: None")

    print("=" * 60)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
