#!/usr/bin/env bash
# =============================================================================
# RuVector Training Orchestrator
# Connects pi.ruv.io brain with local discovery files for self-improving knowledge
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BRAIN_URL="${BRAIN_URL:-https://pi.ruv.io}"
AUTH_TOKEN="${AUTH_TOKEN:-ruvector-swarm}"
DISCOVERY_DIR="$(cd "$(dirname "$0")" && pwd)"
CURL_OPTS=(-s --max-time 15 -H "Authorization: Bearer ${AUTH_TOKEN}" -H "Content-Type: application/json")

# ---------------------------------------------------------------------------
# Colors & formatting
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

banner() {
    echo ""
    echo -e "${MAGENTA}${BOLD}"
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║           RuVector Training Orchestrator v1.0               ║"
    echo "  ║       Brain: ${BRAIN_URL}                          ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo -e "${RESET}"
}

info()    { echo -e "  ${BLUE}[INFO]${RESET}    $*"; }
success() { echo -e "  ${GREEN}[OK]${RESET}      $*"; }
warn()    { echo -e "  ${YELLOW}[WARN]${RESET}    $*"; }
error()   { echo -e "  ${RED}[ERROR]${RESET}   $*"; }
header()  { echo -e "\n  ${CYAN}${BOLD}━━━ $* ━━━${RESET}\n"; }
dim()     { echo -e "  ${DIM}$*${RESET}"; }

# Progress bar: progress_bar current total label
progress_bar() {
    local current=$1 total=$2 label="${3:-}"
    local width=40
    local pct=0
    if (( total > 0 )); then
        pct=$(( current * 100 / total ))
    fi
    local filled=$(( current * width / (total > 0 ? total : 1) ))
    local empty=$(( width - filled ))
    local bar=""
    for ((i=0; i<filled; i++)); do bar+="█"; done
    for ((i=0; i<empty; i++)); do bar+="░"; done
    printf "\r  ${GREEN}[%s]${RESET} %3d%% (%d/%d) %s" "$bar" "$pct" "$current" "$total" "$label"
}

# Safe curl wrapper that handles errors
brain_get() {
    local endpoint="$1"
    local result
    result=$(curl "${CURL_OPTS[@]}" "${BRAIN_URL}${endpoint}" 2>/dev/null) || true
    echo "$result"
}

brain_post() {
    local endpoint="$1"
    local data="$2"
    local result
    result=$(curl "${CURL_OPTS[@]}" -X POST -d "$data" "${BRAIN_URL}${endpoint}" 2>/dev/null) || true
    echo "$result"
}

# ---------------------------------------------------------------------------
# 1. Discovery Scanner
# ---------------------------------------------------------------------------
discovery_scan() {
    header "Discovery Scanner"
    info "Scanning ${DISCOVERY_DIR} for discovery files..."

    local scan_result
    scan_result=$(python3 << 'PYEOF'
import json, os, sys
from collections import defaultdict

discovery_dir = os.environ.get("DISCOVERY_DIR", ".")
files = sorted([f for f in os.listdir(discovery_dir) if f.endswith('.json')])
total_entries = 0
domains = defaultdict(int)
categories = defaultdict(int)
tags_all = defaultdict(int)
entries_per_file = {}
all_tags_sets = []
errors = []

for fname in files:
    fpath = os.path.join(discovery_dir, fname)
    try:
        with open(fpath) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            entries_per_file[fname] = len(data)
            total_entries += len(data)
            file_tags = set()
            for entry in data:
                d = entry.get("domain", "unknown")
                domains[d] += 1
                c = entry.get("category", "uncategorized")
                categories[c] += 1
                for t in entry.get("tags", []):
                    tags_all[t] += 1
                    file_tags.add(t)
            all_tags_sets.append(file_tags)
        elif isinstance(data, dict):
            entries_per_file[fname] = 1
            total_entries += 1
            d = data.get("domain", "unknown")
            if d != "unknown":
                domains[d] += 1
    except Exception as e:
        errors.append(f"{fname}: {e}")

# Cross-reference density: average tag overlap between files
overlap_count = 0
pair_count = 0
for i in range(len(all_tags_sets)):
    for j in range(i+1, min(i+20, len(all_tags_sets))):
        if all_tags_sets[i] and all_tags_sets[j]:
            overlap = len(all_tags_sets[i] & all_tags_sets[j])
            union = len(all_tags_sets[i] | all_tags_sets[j])
            if union > 0:
                overlap_count += overlap / union
                pair_count += 1
cross_ref = overlap_count / pair_count if pair_count > 0 else 0

# Novelty gaps: domains with fewest entries
sorted_domains = sorted(domains.items(), key=lambda x: x[1])
novelty_gaps = sorted_domains[:5] if len(sorted_domains) > 5 else sorted_domains

# Top tags
top_tags = sorted(tags_all.items(), key=lambda x: -x[1])[:15]

# Top files by entry count
top_files = sorted(entries_per_file.items(), key=lambda x: -x[1])[:10]

result = {
    "total_files": len(files),
    "total_entries": total_entries,
    "unique_domains": len(domains),
    "domains": dict(sorted(domains.items(), key=lambda x: -x[1])),
    "unique_categories": len(categories),
    "top_categories": dict(sorted(categories.items(), key=lambda x: -x[1])[:10]),
    "cross_ref_density": round(cross_ref, 4),
    "novelty_gaps": dict(novelty_gaps),
    "top_tags": dict(top_tags),
    "top_files": dict(top_files),
    "errors": errors
}
print(json.dumps(result))
PYEOF
)

    local total_files total_entries unique_domains cross_ref
    total_files=$(echo "$scan_result" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_files'])")
    total_entries=$(echo "$scan_result" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_entries'])")
    unique_domains=$(echo "$scan_result" | python3 -c "import sys,json; print(json.load(sys.stdin)['unique_domains'])")
    cross_ref=$(echo "$scan_result" | python3 -c "import sys,json; print(json.load(sys.stdin)['cross_ref_density'])")

    echo -e "  ${WHITE}${BOLD}Scan Results:${RESET}"
    echo -e "  ${CYAN}Files scanned:${RESET}        $total_files"
    echo -e "  ${CYAN}Total entries:${RESET}        $total_entries"
    echo -e "  ${CYAN}Unique domains:${RESET}       $unique_domains"
    echo -e "  ${CYAN}Cross-ref density:${RESET}    $cross_ref"
    echo ""

    echo -e "  ${WHITE}${BOLD}Domain Distribution:${RESET}"
    echo "$scan_result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for d, c in data['domains'].items():
    bar = '█' * min(c // 5, 40)
    print(f'    {d:<25s} {c:>5d}  {bar}')
"
    echo ""

    echo -e "  ${WHITE}${BOLD}Top Tags:${RESET}"
    echo "$scan_result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for t, c in list(data['top_tags'].items())[:10]:
    print(f'    {t:<25s} {c:>5d}')
"
    echo ""

    echo -e "  ${WHITE}${BOLD}Novelty Gaps (underrepresented domains):${RESET}"
    echo "$scan_result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for d, c in data['novelty_gaps'].items():
    print(f'    {d:<25s} {c:>5d} entries  ⚠ needs enrichment')
"

    # Store for later use
    SCAN_RESULT="$scan_result"
    export SCAN_RESULT
}

# ---------------------------------------------------------------------------
# 2. Brain Gap Analysis
# ---------------------------------------------------------------------------
brain_gap_analysis() {
    header "Brain Gap Analysis"
    info "Querying brain at ${BRAIN_URL}..."

    # Query explore endpoint
    local explore_result
    explore_result=$(brain_get "/v1/explore")
    if [ -z "$explore_result" ] || echo "$explore_result" | grep -q '"error"'; then
        warn "Explore endpoint unavailable or returned error"
        explore_result='{"clusters":[]}'
    else
        success "Explore endpoint responded"
    fi

    # Query partition endpoint
    local partition_result
    partition_result=$(brain_get "/v1/partition")
    if [ -z "$partition_result" ] || echo "$partition_result" | grep -q '"error"'; then
        warn "Partition endpoint unavailable or returned error"
        partition_result='{"partitions":[]}'
    else
        success "Partition endpoint responded"
    fi

    # Query drift endpoint
    local drift_result
    drift_result=$(brain_get "/v1/drift")
    if [ -z "$drift_result" ] || echo "$drift_result" | grep -q '"error"'; then
        warn "Drift endpoint unavailable or returned error"
        drift_result='{"drift":0}'
    else
        success "Drift endpoint responded"
    fi

    # Query status
    local status_result
    status_result=$(brain_get "/v1/status")
    if [ -z "$status_result" ] || echo "$status_result" | grep -q '"error"'; then
        warn "Status endpoint unavailable"
        status_result='{"status":"unknown"}'
    else
        success "Status endpoint responded"
    fi

    echo ""
    echo -e "  ${WHITE}${BOLD}Brain Status:${RESET}"
    echo "$status_result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (str, int, float, bool)):
                print(f'    {k:<30s} {v}')
            elif isinstance(v, dict):
                print(f'    {k}:')
                for k2, v2 in v.items():
                    print(f'      {k2:<28s} {v2}')
except: print('    (could not parse status)')
" 2>/dev/null || dim "    (could not parse status)"

    echo ""
    echo -e "  ${WHITE}${BOLD}Explore / Cluster Analysis:${RESET}"
    echo "$explore_result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 0:
                print(f'    {k}: {len(v)} items')
                for item in v[:5]:
                    if isinstance(item, dict):
                        label = item.get('label', item.get('category', item.get('id', str(item)[:60])))
                        score = item.get('coherence', item.get('score', item.get('curiosity', '')))
                        print(f'      - {label}  (score: {score})')
                    else:
                        print(f'      - {str(item)[:80]}')
            elif isinstance(v, (str, int, float)):
                print(f'    {k}: {v}')
except: print('    (could not parse explore data)')
" 2>/dev/null || dim "    (could not parse explore data)"

    echo ""
    echo -e "  ${WHITE}${BOLD}Partition Analysis:${RESET}"
    echo "$partition_result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                print(f'    {k}: {len(v)} partitions')
                for item in v[:8]:
                    if isinstance(item, dict):
                        name = item.get('name', item.get('label', item.get('category', '?')))
                        size = item.get('size', item.get('count', '?'))
                        coh = item.get('coherence', item.get('density', '?'))
                        print(f'      - {name:<30s} size={size}  coherence={coh}')
                    else:
                        print(f'      - {str(item)[:80]}')
            elif isinstance(v, (str, int, float)):
                print(f'    {k}: {v}')
except: print('    (could not parse partition data)')
" 2>/dev/null || dim "    (could not parse partition data)"

    echo ""
    echo -e "  ${WHITE}${BOLD}Drift Analysis:${RESET}"
    echo "$drift_result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (str, int, float, bool)):
                print(f'    {k:<30s} {v}')
except: print('    (could not parse drift data)')
" 2>/dev/null || dim "    (could not parse drift data)"
}

# ---------------------------------------------------------------------------
# 3. Batch Upload Pipeline
# ---------------------------------------------------------------------------
batch_upload() {
    header "Batch Upload Pipeline"
    info "Uploading discovery entries to brain..."

    local json_files
    json_files=$(find "$DISCOVERY_DIR" -maxdepth 1 -name '*.json' -not -name '*.tmp' | sort)
    local file_count
    file_count=$(echo "$json_files" | wc -l)

    local total_uploaded=0
    local total_failed=0
    local total_skipped=0
    local total_to_upload=0

    # First pass: count entries
    for f in $json_files; do
        local count
        count=$(python3 -c "
import json, sys
try:
    with open('$f') as fh:
        data = json.load(fh)
    if isinstance(data, list):
        print(len(data))
    elif isinstance(data, dict) and 'title' in data:
        print(1)
    else:
        print(0)
except:
    print(0)
" 2>/dev/null)
        total_to_upload=$((total_to_upload + count))
    done

    info "Found $total_to_upload entries across $file_count files"
    echo ""

    local current=0
    for f in $json_files; do
        local fname
        fname=$(basename "$f")

        # Process each file
        local upload_result
        upload_result=$(BRAIN_URL="$BRAIN_URL" AUTH_TOKEN="$AUTH_TOKEN" python3 << PYEOF
import json, sys, os, urllib.request, urllib.error, ssl

brain_url = os.environ.get("BRAIN_URL", "https://pi.ruv.io")
auth_token = os.environ.get("AUTH_TOKEN", "ruvector-swarm")
fpath = "$f"
fname = "$fname"

ctx = ssl.create_default_context()

def brain_request(method, endpoint, data=None):
    url = brain_url + endpoint
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

uploaded = 0
failed = 0
skipped = 0

try:
    with open(fpath) as fh:
        data = json.load(fh)

    entries = []
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "title" in data:
        entries = [data]

    for entry in entries:
        title = entry.get("title", "")
        content = entry.get("content", "")
        category = entry.get("category", entry.get("domain", "discovery"))
        tags = entry.get("tags", [])
        domain = entry.get("domain", "general")

        if not title or not content:
            skipped += 1
            continue

        # Get nonce
        challenge = brain_request("GET", "/v1/challenge")
        nonce = challenge.get("nonce", "")
        if not nonce:
            failed += 1
            continue

        # Upload
        payload = {
            "category": category,
            "title": title,
            "content": content[:2000],
            "tags": tags[:10] + [domain, fname.replace(".json","")],
            "nonce": nonce
        }
        result = brain_request("POST", "/v1/memories", payload)
        if "error" in result:
            failed += 1
        else:
            uploaded += 1

except Exception as e:
    print(json.dumps({"file": fname, "uploaded": 0, "failed": 0, "skipped": 0, "error": str(e)}))
    sys.exit(0)

print(json.dumps({"file": fname, "uploaded": uploaded, "failed": failed, "skipped": skipped}))
PYEOF
)

        local file_uploaded file_failed file_skipped
        file_uploaded=$(echo "$upload_result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('uploaded',0))" 2>/dev/null || echo 0)
        file_failed=$(echo "$upload_result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('failed',0))" 2>/dev/null || echo 0)
        file_skipped=$(echo "$upload_result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('skipped',0))" 2>/dev/null || echo 0)

        total_uploaded=$((total_uploaded + file_uploaded))
        total_failed=$((total_failed + file_failed))
        total_skipped=$((total_skipped + file_skipped))
        current=$((current + file_uploaded + file_failed + file_skipped))

        progress_bar "$current" "$total_to_upload" "$fname"
    done

    echo ""
    echo ""
    echo -e "  ${WHITE}${BOLD}Upload Summary:${RESET}"
    echo -e "  ${GREEN}Uploaded:${RESET}  $total_uploaded"
    echo -e "  ${RED}Failed:${RESET}    $total_failed"
    echo -e "  ${YELLOW}Skipped:${RESET}   $total_skipped"
    echo -e "  ${CYAN}Total:${RESET}     $((total_uploaded + total_failed + total_skipped))"
}

# ---------------------------------------------------------------------------
# 4. Training & Optimization Cycle
# ---------------------------------------------------------------------------
training_cycle() {
    header "Training & Optimization Cycle"

    # Step 1: Trigger training
    info "Triggering training..."
    local train_result
    train_result=$(brain_post "/v1/pipeline/optimize" '{"actions":["train"]}')
    if [ -z "$train_result" ]; then
        # Fallback to /v1/train
        train_result=$(brain_post "/v1/train" '{}')
    fi
    if echo "$train_result" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'error' not in d else 1)" 2>/dev/null; then
        success "Training triggered"
    else
        warn "Training response: $(echo "$train_result" | head -c 200)"
    fi
    echo "$train_result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool)):
            print(f'    {k:<30s} {v}')
except: pass
" 2>/dev/null

    echo ""

    # Step 2: Drift check
    info "Running drift check..."
    local drift_result
    drift_result=$(brain_post "/v1/pipeline/optimize" '{"actions":["drift_check"]}')
    if [ -z "$drift_result" ]; then
        drift_result=$(brain_get "/v1/drift")
    fi
    if [ -n "$drift_result" ]; then
        success "Drift check complete"
        echo "$drift_result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool)):
            print(f'    {k:<30s} {v}')
except: pass
" 2>/dev/null
    else
        warn "Drift check unavailable"
    fi

    echo ""

    # Step 3: Domain transfers between related categories
    info "Running cross-domain transfers..."
    local transfers=(
        "space-science:earth-science"
        "academic-research:medical-genomics"
        "economics-finance:academic-research"
        "materials-physics:space-science"
    )
    for pair in "${transfers[@]}"; do
        local src="${pair%%:*}"
        local dst="${pair##*:}"
        local xfer_result
        xfer_result=$(brain_post "/v1/transfer" "{\"source_domain\":\"$src\",\"target_domain\":\"$dst\"}")
        if [ -n "$xfer_result" ] && ! echo "$xfer_result" | grep -q '"error"' 2>/dev/null; then
            success "Transfer: $src -> $dst"
        else
            dim "    Transfer $src -> $dst: $(echo "$xfer_result" | head -c 100)"
        fi
    done

    echo ""

    # Step 4: Attractor analysis
    info "Triggering attractor analysis..."
    local attractor_result
    attractor_result=$(brain_get "/v1/explore")
    if [ -n "$attractor_result" ]; then
        success "Attractor analysis complete"
        echo "$attractor_result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    # Show summary stats
    for k, v in data.items():
        if isinstance(v, list):
            print(f'    {k}: {len(v)} attractors found')
        elif isinstance(v, (str, int, float)):
            print(f'    {k}: {v}')
except: pass
" 2>/dev/null
    fi

    echo ""

    # Step 5: Full optimization
    info "Running full optimization..."
    local opt_result
    opt_result=$(brain_post "/v1/pipeline/optimize" '{"actions":["train","drift_check"]}')
    if [ -n "$opt_result" ]; then
        success "Optimization complete"
        echo "$opt_result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool)):
            print(f'    {k:<30s} {v}')
except: pass
" 2>/dev/null
    fi
}

# ---------------------------------------------------------------------------
# 5. Cross-Domain Discovery Engine
# ---------------------------------------------------------------------------
cross_domain_discovery() {
    header "Cross-Domain Discovery Engine"
    info "Finding novel connections across domains..."

    local domains=("space" "earthquake" "economics" "genomics" "quantum" "climate" "AI" "defense" "energy" "materials")
    local search_results=""

    for domain in "${domains[@]}"; do
        local result
        result=$(brain_get "/v1/memories/search?q=${domain}")
        if [ -n "$result" ]; then
            search_results="${search_results}${domain}|||${result}
"
        fi
        printf "\r  Searching: %-20s" "$domain"
    done
    echo ""
    echo ""

    # Analyze cross-domain similarities
    echo "$search_results" | python3 << 'PYEOF'
import sys, json
from collections import defaultdict

lines = sys.stdin.read().strip().split("\n")
domain_tags = defaultdict(set)
domain_entries = defaultdict(list)

for line in lines:
    if "|||" not in line:
        continue
    domain, raw = line.split("|||", 1)
    try:
        data = json.loads(raw)
        entries = []
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            if "results" in data:
                entries = data["results"]
            elif "memories" in data:
                entries = data["memories"]
            elif "title" in data:
                entries = [data]

        for e in entries:
            if isinstance(e, dict):
                tags = set(e.get("tags", []))
                domain_tags[domain] |= tags
                domain_entries[domain].append(e.get("title", "")[:80])
    except:
        continue

# Find cross-domain overlaps
print("  Cross-Domain Tag Overlaps:")
print("  " + "-" * 60)
domains = list(domain_tags.keys())
connections = []
for i in range(len(domains)):
    for j in range(i+1, len(domains)):
        overlap = domain_tags[domains[i]] & domain_tags[domains[j]]
        if overlap:
            connections.append((domains[i], domains[j], overlap))

connections.sort(key=lambda x: -len(x[2]))
for src, dst, overlap in connections[:15]:
    shared = ", ".join(list(overlap)[:5])
    print(f"    {src:<15s} <-> {dst:<15s}  shared: {shared}")

print()
print("  Domain Knowledge Map:")
print("  " + "-" * 60)
for domain in sorted(domain_entries.keys()):
    entries = domain_entries[domain]
    tag_count = len(domain_tags.get(domain, set()))
    print(f"    {domain:<15s}  {len(entries):>3d} entries  {tag_count:>3d} unique tags")
PYEOF
}

# ---------------------------------------------------------------------------
# 6. Interactive Mode
# ---------------------------------------------------------------------------
interactive_mode() {
    header "Interactive Mode"
    echo -e "  ${DIM}Commands: explore, inject, train, status, gaps, transfer, optimize, scan, upload, quit${RESET}"
    echo ""

    while true; do
        echo -ne "  ${MAGENTA}ruv>${RESET} "
        read -r cmd args || break

        case "$cmd" in
            explore)
                if [ -z "$args" ]; then
                    warn "Usage: explore <topic>"
                    continue
                fi
                info "Searching brain for: $args"
                local result
                result=$(brain_get "/v1/memories/search?q=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$args'))")")
                if [ -n "$result" ]; then
                    echo "$result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    entries = data if isinstance(data, list) else data.get('results', data.get('memories', [data] if 'title' in data else []))
    if not entries:
        print('    No results found.')
    for i, e in enumerate(entries[:10]):
        if isinstance(e, dict):
            title = e.get('title', 'untitled')[:70]
            cat = e.get('category', '?')
            score = e.get('score', e.get('similarity', ''))
            tags = ', '.join(e.get('tags', [])[:5])
            print(f'    {i+1:>2d}. [{cat}] {title}')
            if score: print(f'        score: {score}')
            if tags: print(f'        tags: {tags}')
except Exception as ex:
    print(f'    Parse error: {ex}')
    print(f'    Raw: {sys.stdin.read()[:200]}')
" 2>/dev/null
                else
                    warn "No response from brain"
                fi
                ;;

            inject)
                local title content
                title=$(echo "$args" | awk '{print $1}')
                content=$(echo "$args" | cut -d' ' -f2-)
                if [ -z "$title" ] || [ -z "$content" ]; then
                    warn "Usage: inject <title> <content>"
                    continue
                fi
                info "Injecting: $title"
                # Get nonce
                local nonce_resp
                nonce_resp=$(brain_get "/v1/challenge")
                local nonce
                nonce=$(echo "$nonce_resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('nonce',''))" 2>/dev/null)
                if [ -z "$nonce" ]; then
                    error "Could not get nonce"
                    continue
                fi
                local inject_data
                inject_data=$(python3 -c "
import json
print(json.dumps({
    'category': 'interactive',
    'title': '''$title''',
    'content': '''$content''',
    'tags': ['interactive', 'live-inject'],
    'nonce': '$nonce'
}))
")
                local inject_result
                inject_result=$(brain_post "/v1/memories" "$inject_data")
                if echo "$inject_result" | grep -q '"error"' 2>/dev/null; then
                    error "Injection failed: $(echo "$inject_result" | head -c 200)"
                else
                    success "Injected successfully"
                    echo "$inject_result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool)):
            print(f'    {k}: {v}')
except: pass
" 2>/dev/null
                fi
                ;;

            train)
                info "Triggering training cycle..."
                local train_r
                train_r=$(brain_post "/v1/pipeline/optimize" '{"actions":["train"]}')
                if [ -z "$train_r" ]; then
                    train_r=$(brain_post "/v1/train" '{}')
                fi
                if [ -n "$train_r" ]; then
                    success "Training triggered"
                    echo "$train_r" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for k,v in data.items():
        print(f'    {k}: {v}')
except: pass
" 2>/dev/null
                else
                    warn "No response from training endpoint"
                fi
                ;;

            status)
                local stat_r
                stat_r=$(brain_get "/v1/status")
                if [ -n "$stat_r" ]; then
                    echo "$stat_r" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for k,v in data.items():
        if isinstance(v, dict):
            print(f'    {k}:')
            for k2,v2 in v.items():
                print(f'      {k2:<28s} {v2}')
        else:
            print(f'    {k:<30s} {v}')
except: pass
" 2>/dev/null
                else
                    warn "Brain unreachable"
                fi
                ;;

            gaps)
                info "Analyzing knowledge gaps..."
                local gap_explore
                gap_explore=$(brain_get "/v1/explore")
                local gap_partition
                gap_partition=$(brain_get "/v1/partition")
                echo "$gap_explore" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for k, v in data.items():
        if isinstance(v, list):
            low_coherence = [item for item in v if isinstance(item, dict) and item.get('coherence', 1.0) < 0.5]
            if low_coherence:
                print(f'    Low-coherence clusters ({k}):')
                for item in low_coherence[:5]:
                    label = item.get('label', item.get('category', '?'))
                    coh = item.get('coherence', '?')
                    print(f'      - {label}: coherence={coh}')
            else:
                high_curiosity = sorted(v, key=lambda x: x.get('curiosity', 0) if isinstance(x, dict) else 0, reverse=True)[:5]
                if high_curiosity and isinstance(high_curiosity[0], dict):
                    print(f'    Highest curiosity ({k}):')
                    for item in high_curiosity:
                        label = item.get('label', item.get('category', '?'))
                        cur = item.get('curiosity', '?')
                        print(f'      - {label}: curiosity={cur}')
        elif isinstance(v, (str, int, float)):
            print(f'    {k}: {v}')
except: print('    (could not analyze gaps)')
" 2>/dev/null
                echo "$gap_partition" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for k, v in data.items():
        if isinstance(v, list):
            small = sorted(v, key=lambda x: x.get('size', 999) if isinstance(x, dict) else 999)[:5]
            if small and isinstance(small[0], dict):
                print(f'    Smallest partitions ({k}):')
                for item in small:
                    name = item.get('name', item.get('label', '?'))
                    size = item.get('size', '?')
                    print(f'      - {name}: size={size}')
except: pass
" 2>/dev/null
                ;;

            transfer)
                local src dst
                src=$(echo "$args" | awk '{print $1}')
                dst=$(echo "$args" | awk '{print $2}')
                if [ -z "$src" ] || [ -z "$dst" ]; then
                    warn "Usage: transfer <source_domain> <target_domain>"
                    continue
                fi
                info "Transferring: $src -> $dst"
                local xfer_r
                xfer_r=$(brain_post "/v1/transfer" "{\"source_domain\":\"$src\",\"target_domain\":\"$dst\"}")
                if [ -n "$xfer_r" ]; then
                    success "Transfer complete"
                    echo "$xfer_r" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for k,v in data.items():
        print(f'    {k}: {v}')
except: pass
" 2>/dev/null
                else
                    warn "Transfer failed"
                fi
                ;;

            optimize)
                info "Running full optimization pipeline..."
                training_cycle
                ;;

            scan)
                discovery_scan
                ;;

            upload)
                batch_upload
                ;;

            help|h|\?)
                echo -e "  ${WHITE}${BOLD}Available Commands:${RESET}"
                echo -e "    ${CYAN}explore <topic>${RESET}         Search brain for topic, show related memories"
                echo -e "    ${CYAN}inject <title> <content>${RESET} Inject new knowledge in real-time"
                echo -e "    ${CYAN}train${RESET}                   Trigger training cycle"
                echo -e "    ${CYAN}status${RESET}                  Show brain status"
                echo -e "    ${CYAN}gaps${RESET}                    Show knowledge gaps"
                echo -e "    ${CYAN}transfer <from> <to>${RESET}    Cross-domain transfer"
                echo -e "    ${CYAN}optimize${RESET}                Run full optimization pipeline"
                echo -e "    ${CYAN}scan${RESET}                    Re-scan discovery files"
                echo -e "    ${CYAN}upload${RESET}                  Batch upload all discoveries"
                echo -e "    ${CYAN}help${RESET}                    Show this help"
                echo -e "    ${CYAN}quit${RESET}                    Exit"
                ;;

            quit|exit|q)
                echo ""
                info "Shutting down orchestrator. Knowledge persists in the brain."
                break
                ;;

            "")
                ;;

            *)
                warn "Unknown command: $cmd (type 'help' for available commands)"
                ;;
        esac
        echo ""
    done
}

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
main() {
    banner

    # Check dependencies
    for dep in curl python3; do
        if ! command -v "$dep" &>/dev/null; then
            error "Required dependency not found: $dep"
            exit 1
        fi
    done
    success "Dependencies verified (curl, python3)"

    # Check brain connectivity
    info "Testing brain connectivity..."
    local ping_result
    ping_result=$(brain_get "/v1/status")
    if [ -n "$ping_result" ] && ! echo "$ping_result" | grep -q "Could not resolve"; then
        success "Brain is reachable at ${BRAIN_URL}"
    else
        warn "Brain may be unreachable -- continuing in offline mode where possible"
    fi
    echo ""

    # Parse CLI arguments
    local mode="${1:-interactive}"

    case "$mode" in
        scan)
            discovery_scan
            ;;
        gaps)
            brain_gap_analysis
            ;;
        upload)
            discovery_scan
            batch_upload
            ;;
        train)
            training_cycle
            ;;
        cross)
            cross_domain_discovery
            ;;
        full)
            discovery_scan
            brain_gap_analysis
            batch_upload
            training_cycle
            cross_domain_discovery
            header "Full Pipeline Complete"
            success "All stages finished. Brain has been updated and optimized."
            ;;
        interactive|"")
            discovery_scan
            echo ""
            interactive_mode
            ;;
        help|--help|-h)
            echo -e "  ${WHITE}${BOLD}Usage:${RESET} $0 [mode]"
            echo ""
            echo -e "  ${WHITE}${BOLD}Modes:${RESET}"
            echo -e "    ${CYAN}interactive${RESET}   Interactive command mode (default)"
            echo -e "    ${CYAN}scan${RESET}          Scan discovery files only"
            echo -e "    ${CYAN}gaps${RESET}          Brain gap analysis only"
            echo -e "    ${CYAN}upload${RESET}        Scan + upload all discoveries"
            echo -e "    ${CYAN}train${RESET}         Training & optimization cycle"
            echo -e "    ${CYAN}cross${RESET}         Cross-domain discovery engine"
            echo -e "    ${CYAN}full${RESET}          Run complete pipeline (scan->gaps->upload->train->cross)"
            echo -e "    ${CYAN}help${RESET}          Show this help"
            ;;
        *)
            error "Unknown mode: $mode"
            echo "  Run '$0 help' for usage"
            exit 1
            ;;
    esac
}

main "$@"
