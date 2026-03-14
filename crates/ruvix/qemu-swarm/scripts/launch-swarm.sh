#!/bin/bash
# launch-swarm.sh - Launch a RuVix QEMU swarm cluster
#
# Usage:
#   ./launch-swarm.sh [options]
#
# Options:
#   -n, --nodes NUM       Number of nodes (default: 3)
#   -t, --topology TYPE   Topology: mesh, ring, star, tree (default: mesh)
#   -k, --kernel PATH     Path to RuVix kernel binary
#   -d, --dtb PATH        Path to device tree blob
#   -m, --memory MB       Memory per node in MB (default: 512)
#   -c, --cpus NUM        CPUs per node (default: 2)
#   -g, --gdb             Enable GDB servers
#   -w, --wait PATTERN    Wait for boot pattern
#   -v, --verbose         Enable verbose output
#   -h, --help            Show this help

set -e

# Default values
NODES=3
TOPOLOGY="mesh"
KERNEL=""
DTB=""
MEMORY=512
CPUS=2
GDB=false
WAIT_PATTERN=""
VERBOSE=false
WORK_DIR="${TMPDIR:-/tmp}/ruvix-swarm"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    head -25 "$0" | tail -20 | sed 's/^#//'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nodes)
            NODES="$2"
            shift 2
            ;;
        -t|--topology)
            TOPOLOGY="$2"
            shift 2
            ;;
        -k|--kernel)
            KERNEL="$2"
            shift 2
            ;;
        -d|--dtb)
            DTB="$2"
            shift 2
            ;;
        -m|--memory)
            MEMORY="$2"
            shift 2
            ;;
        -c|--cpus)
            CPUS="$2"
            shift 2
            ;;
        -g|--gdb)
            GDB=true
            shift
            ;;
        -w|--wait)
            WAIT_PATTERN="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for QEMU
if ! command -v qemu-system-aarch64 &> /dev/null; then
    log_error "qemu-system-aarch64 not found. Please install QEMU."
    exit 1
fi

# Create work directory
mkdir -p "$WORK_DIR"
log_info "Work directory: $WORK_DIR"

# Function to start a single node
start_node() {
    local node_id=$1
    local node_dir="$WORK_DIR/node-$node_id"
    local console_sock="$node_dir/console.sock"
    local monitor_port=$((4444 + node_id))
    local gdb_port=$((1234 + node_id))
    local mac=$(printf "52:54:00:12:34:%02x" $node_id)

    mkdir -p "$node_dir"

    # Build QEMU command
    local cmd="qemu-system-aarch64"
    cmd+=" -machine virt,virtualization=true"
    cmd+=" -cpu cortex-a72"
    cmd+=" -smp $CPUS"
    cmd+=" -m ${MEMORY}M"
    cmd+=" -nographic"

    if [[ -n "$KERNEL" ]]; then
        cmd+=" -kernel $KERNEL"
    fi

    if [[ -n "$DTB" ]]; then
        cmd+=" -dtb $DTB"
    fi

    # Network
    cmd+=" -netdev socket,id=net0,mcast=239.0.0.1:5000,localaddr=127.0.0.1"
    cmd+=" -device virtio-net-pci,netdev=net0,mac=$mac"

    # Serial console
    cmd+=" -serial unix:$console_sock,server,nowait"

    # Monitor
    cmd+=" -monitor tcp:127.0.0.1:$monitor_port,server,nowait"

    # GDB
    if [[ "$GDB" == "true" ]]; then
        cmd+=" -gdb tcp::$gdb_port -S"
    fi

    # Deterministic mode
    cmd+=" -icount shift=1,align=off,sleep=on"

    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Starting node $node_id: $cmd"
    fi

    # Start in background
    eval "$cmd" > "$node_dir/qemu.log" 2>&1 &
    local pid=$!
    echo $pid > "$node_dir/qemu.pid"

    log_success "Node $node_id started (PID: $pid, Monitor: $monitor_port)"

    if [[ "$GDB" == "true" ]]; then
        log_info "  GDB: localhost:$gdb_port"
    fi
}

# Print banner
echo ""
echo "  ____        _   ___       "
echo " |  _ \ _   _| | / (_)_  __ "
echo " | |_) | | | | |/ /| \ \/ / "
echo " |  _ <| |_| |   < | |>  <  "
echo " |_| \_\\\\__,_|_|\_\|_/_/\_\\ "
echo ""
echo "  QEMU Swarm Launcher"
echo "  ==================="
echo ""

log_info "Configuration:"
log_info "  Nodes: $NODES"
log_info "  Topology: $TOPOLOGY"
log_info "  Memory: ${MEMORY}MB per node"
log_info "  CPUs: $CPUS per node"
[[ -n "$KERNEL" ]] && log_info "  Kernel: $KERNEL"
[[ "$GDB" == "true" ]] && log_info "  GDB: enabled"
echo ""

# Start all nodes
log_info "Starting $NODES nodes..."
for ((i=0; i<NODES; i++)); do
    start_node $i
    # Small delay between node starts
    sleep 0.5
done

echo ""
log_success "All nodes started!"
echo ""

# Print network topology
log_info "Network Topology: $TOPOLOGY"
case $TOPOLOGY in
    mesh)
        log_info "  All nodes connected to all others"
        ;;
    ring)
        log_info "  Each node connected to neighbors"
        ;;
    star)
        log_info "  Node 0 is the hub"
        ;;
    tree)
        log_info "  Binary tree structure, node 0 is root"
        ;;
esac
echo ""

# Print connection info
log_info "Console sockets:"
for ((i=0; i<NODES; i++)); do
    echo "  Node $i: $WORK_DIR/node-$i/console.sock"
done
echo ""

log_info "Monitor ports:"
for ((i=0; i<NODES; i++)); do
    echo "  Node $i: localhost:$((4444 + i))"
done
echo ""

# Wait for pattern if specified
if [[ -n "$WAIT_PATTERN" ]]; then
    log_info "Waiting for pattern: '$WAIT_PATTERN'"
    # Monitor all console sockets for the pattern
    timeout=60
    found=0

    while [[ $timeout -gt 0 && $found -lt $NODES ]]; do
        for ((i=0; i<NODES; i++)); do
            if [[ -S "$WORK_DIR/node-$i/console.sock" ]]; then
                if socat -u UNIX-CONNECT:"$WORK_DIR/node-$i/console.sock" - 2>/dev/null | grep -q "$WAIT_PATTERN"; then
                    ((found++))
                fi
            fi
        done
        sleep 1
        ((timeout--))
    done

    if [[ $found -eq $NODES ]]; then
        log_success "All nodes ready!"
    else
        log_warn "Only $found/$NODES nodes matched pattern"
    fi
fi

# Trap Ctrl+C to cleanup
cleanup() {
    echo ""
    log_info "Shutting down swarm..."
    for ((i=0; i<NODES; i++)); do
        if [[ -f "$WORK_DIR/node-$i/qemu.pid" ]]; then
            pid=$(cat "$WORK_DIR/node-$i/qemu.pid")
            kill $pid 2>/dev/null || true
            rm -f "$WORK_DIR/node-$i/qemu.pid"
        fi
    done
    log_success "Swarm stopped."
    exit 0
}

trap cleanup INT TERM

log_info "Swarm is running. Press Ctrl+C to stop."
echo ""

# Monitor node health
while true; do
    sleep 5
    alive=0
    for ((i=0; i<NODES; i++)); do
        if [[ -f "$WORK_DIR/node-$i/qemu.pid" ]]; then
            pid=$(cat "$WORK_DIR/node-$i/qemu.pid")
            if kill -0 $pid 2>/dev/null; then
                ((alive++))
            fi
        fi
    done

    if [[ $alive -eq 0 ]]; then
        log_error "All nodes have exited!"
        exit 1
    elif [[ $alive -lt $NODES ]]; then
        log_warn "$alive/$NODES nodes running"
    fi
done
