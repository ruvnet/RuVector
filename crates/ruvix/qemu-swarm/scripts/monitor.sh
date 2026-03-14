#!/bin/bash
# monitor.sh - Real-time console aggregation for RuVix QEMU swarm
#
# Usage:
#   ./monitor.sh [options]
#
# Options:
#   -d, --dir DIR         Swarm work directory (default: /tmp/ruvix-swarm)
#   -n, --nodes NUM       Number of nodes to monitor (default: auto-detect)
#   -f, --filter LEVEL    Filter by level: debug, info, warn, error, panic
#   -p, --pattern TEXT    Show only lines containing TEXT
#   -o, --output FILE     Write output to file
#   -t, --timestamps      Show timestamps
#   -c, --no-color        Disable colors
#   --tui                 Enable TUI mode (requires cargo build)
#   -h, --help            Show this help

set -e

# Default values
WORK_DIR="${TMPDIR:-/tmp}/ruvix-swarm"
NODES=""
FILTER=""
PATTERN=""
OUTPUT=""
TIMESTAMPS=false
COLOR=true
TUI=false

# Colors
declare -a COLORS=(
    '\033[0;36m'  # Cyan
    '\033[0;32m'  # Green
    '\033[0;33m'  # Yellow
    '\033[0;35m'  # Magenta
    '\033[0;34m'  # Blue
    '\033[0;31m'  # Red
    '\033[0;37m'  # White
    '\033[0;96m'  # Light Cyan
)
NC='\033[0m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BOLD='\033[1m'

show_help() {
    head -20 "$0" | tail -15 | sed 's/^#//'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            WORK_DIR="$2"
            shift 2
            ;;
        -n|--nodes)
            NODES="$2"
            shift 2
            ;;
        -f|--filter)
            FILTER="$2"
            shift 2
            ;;
        -p|--pattern)
            PATTERN="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -t|--timestamps)
            TIMESTAMPS=true
            shift
            ;;
        -c|--no-color)
            COLOR=false
            shift
            ;;
        --tui)
            TUI=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for required tools
if ! command -v socat &> /dev/null; then
    echo "Error: socat not found. Please install it."
    echo "  macOS: brew install socat"
    echo "  Linux: apt-get install socat"
    exit 1
fi

# Auto-detect nodes if not specified
if [[ -z "$NODES" ]]; then
    NODES=0
    for dir in "$WORK_DIR"/node-*; do
        if [[ -d "$dir" ]]; then
            ((NODES++))
        fi
    done

    if [[ $NODES -eq 0 ]]; then
        echo "Error: No nodes found in $WORK_DIR"
        echo "Run launch-swarm.sh first or specify --dir"
        exit 1
    fi
fi

# TUI mode uses the Rust binary
if [[ "$TUI" == "true" ]]; then
    cargo run --release --bin swarm-monitor -- \
        --socket-dir "$WORK_DIR" \
        --nodes "$NODES" \
        ${FILTER:+--filter "$FILTER"} \
        ${PATTERN:+--pattern "$PATTERN"} \
        ${OUTPUT:+--output "$OUTPUT"} \
        --tui
    exit $?
fi

# Print header
echo ""
echo "  RuVix Swarm Monitor"
echo "  ==================="
echo ""
echo "  Directory: $WORK_DIR"
echo "  Nodes: $NODES"
[[ -n "$FILTER" ]] && echo "  Filter: $FILTER"
[[ -n "$PATTERN" ]] && echo "  Pattern: $PATTERN"
[[ -n "$OUTPUT" ]] && echo "  Output: $OUTPUT"
echo ""
echo "  Press Ctrl+C to exit"
echo ""
echo "-----------------------------------------------------------"

# Initialize output file
if [[ -n "$OUTPUT" ]]; then
    echo "# RuVix Swarm Console Log - $(date)" > "$OUTPUT"
    echo "# Nodes: $NODES" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
fi

# Function to filter lines
should_show() {
    local line="$1"
    local lower="${line,,}"  # lowercase

    # Pattern filter
    if [[ -n "$PATTERN" ]]; then
        if [[ "$line" != *"$PATTERN"* ]]; then
            return 1
        fi
    fi

    # Level filter
    if [[ -n "$FILTER" ]]; then
        case "$FILTER" in
            panic)
                [[ "$lower" == *"panic"* || "$lower" == *"fatal"* || "$lower" == *"crash"* ]] && return 0
                return 1
                ;;
            error)
                [[ "$lower" == *"error"* || "$lower" == *"panic"* || "$lower" == *"fatal"* ]] && return 0
                return 1
                ;;
            warn)
                [[ "$lower" == *"warn"* || "$lower" == *"error"* || "$lower" == *"panic"* ]] && return 0
                return 1
                ;;
            info)
                [[ "$lower" == *"debug"* || "$lower" == *"trace"* ]] && return 1
                return 0
                ;;
            debug)
                [[ "$lower" == *"trace"* ]] && return 1
                return 0
                ;;
        esac
    fi

    return 0
}

# Function to format output
format_line() {
    local node_id="$1"
    local line="$2"
    local lower="${line,,}"

    # Get color for this node
    local color=""
    if [[ "$COLOR" == "true" ]]; then
        color="${COLORS[$((node_id % ${#COLORS[@]}))]}"
    fi

    # Detect severity
    local severity=""
    if [[ "$lower" == *"panic"* || "$lower" == *"crash"* || "$lower" == *"fatal"* ]]; then
        severity="${RED}[PANIC]${NC} "
    elif [[ "$lower" == *"error"* ]]; then
        severity="${RED}[ERROR]${NC} "
    elif [[ "$lower" == *"warn"* ]]; then
        severity="${YELLOW}[WARN]${NC} "
    fi

    # Build output
    local output=""

    if [[ "$TIMESTAMPS" == "true" ]]; then
        output+="$(date '+%H:%M:%S.%3N') "
    fi

    if [[ "$COLOR" == "true" ]]; then
        output+="${color}[N$node_id]${NC} $severity$line"
    else
        output+="[N$node_id] $line"
    fi

    echo -e "$output"

    # Write to file
    if [[ -n "$OUTPUT" ]]; then
        echo "[$(date '+%H:%M:%S')] [N$node_id] $line" >> "$OUTPUT"
    fi
}

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping monitor..."
    # Kill all background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM

# Start monitoring each node
for ((i=0; i<NODES; i++)); do
    socket="$WORK_DIR/node-$i/console.sock"

    # Wait for socket to exist
    if [[ ! -S "$socket" ]]; then
        echo "Waiting for node $i socket..."
        for ((j=0; j<30; j++)); do
            if [[ -S "$socket" ]]; then
                break
            fi
            sleep 0.5
        done
    fi

    if [[ -S "$socket" ]]; then
        # Start socat reader in background
        (
            while true; do
                socat -u UNIX-CONNECT:"$socket" - 2>/dev/null | while IFS= read -r line; do
                    if should_show "$line"; then
                        format_line $i "$line"
                    fi
                done
                # Reconnect on disconnect
                sleep 1
            done
        ) &
    else
        echo "Warning: Socket not found for node $i"
    fi
done

# Wait for all background processes
wait
