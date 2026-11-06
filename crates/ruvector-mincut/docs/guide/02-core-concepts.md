# Core Concepts

This guide explains the fundamental concepts behind minimum cut algorithms in accessible language. Whether you're new to graph theory or experienced with algorithms, this guide will help you understand what makes RuVector's minimum cut implementation special.

---

## 1. Graph Basics

### What is a Graph?

Think of a graph like a social network or a map:

- **Vertices (Nodes)**: These are the "things" in your system
  - Cities on a map
  - People in a social network
  - Computers in a network
  - Pixels in an image

- **Edges (Links)**: These are the connections between things
  - Roads between cities
  - Friendships between people
  - Network cables between computers
  - Similarity between adjacent pixels

```mermaid
graph LR
    A[Alice] ---|Friend| B[Bob]
    B ---|Friend| C[Carol]
    A ---|Friend| C
    C ---|Friend| D[Dave]
    B ---|Friend| D

    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style C fill:#e1f5ff
    style D fill:#e1f5ff
```

### Weighted vs Unweighted Graphs

**Unweighted Graph**: All connections are equal
- Example: "Is there a friendship?" (yes/no)

**Weighted Graph**: Connections have different strengths or capacities
- Example: "How strong is the friendship?" (1-10 scale)
- Example: "What's the bandwidth of this network cable?" (100 Mbps, 1 Gbps, etc.)

```mermaid
graph LR
    subgraph "Unweighted Graph"
        A1[City A] --- B1[City B]
        B1 --- C1[City C]
        A1 --- C1
    end

    subgraph "Weighted Graph"
        A2[City A] ---|50 km| B2[City B]
        B2 ---|30 km| C2[City C]
        A2 ---|80 km| C2
    end

    style A1 fill:#ffe1e1
    style B1 fill:#ffe1e1
    style C1 fill:#ffe1e1
    style A2 fill:#e1ffe1
    style B2 fill:#e1ffe1
    style C2 fill:#e1ffe1
```

### Directed vs Undirected Graphs

**Undirected**: Connections work both ways
- Example: Roads (usually bidirectional)
- Example: Mutual friendships

**Undirected**: Connections have a direction
- Example: One-way streets
- Example: Twitter follows (Alice follows Bob doesn't mean Bob follows Alice)

```mermaid
graph LR
    subgraph "Undirected (Bidirectional)"
        A1[A] --- B1[B]
        B1 --- C1[C]
    end

    subgraph "Directed (One-way)"
        A2[A] --> B2[B]
        B2 --> C2[C]
        C2 --> A2
    end
```

**RuVector focuses on undirected, weighted graphs** for minimum cut problems.

---

## 2. What is Minimum Cut?

### Definition

A **cut** in a graph divides vertices into two groups. The **minimum cut** is the division that requires removing the fewest (or lowest-weight) edges.

Think of it like this:
- Imagine a network of pipes carrying water
- A cut is choosing which pipes to block to split the network in two
- The minimum cut finds the weakest point - the smallest set of pipes that, if blocked, would separate the network

```mermaid
graph TB
    subgraph "Original Graph"
        A[A] ---|2| B[B]
        A ---|3| C[C]
        B ---|1| D[D]
        C ---|1| D
        B ---|4| C
    end

    subgraph "Minimum Cut (weight = 2)"
        A1[A] ---|2| B1[B]
        A1 ---|3| C1[C]
        B1 -.X.-|1| D1[D]
        C1 -.X.-|1| D1
        B1 ---|4| C1

        style A1 fill:#ffcccc
        style B1 fill:#ffcccc
        style C1 fill:#ffcccc
        style D1 fill:#ccffcc
    end
```

In the example above:
- Cutting edges B-D and C-D (total weight = 1 + 1 = 2) separates the graph
- This is the minimum cut because no smaller cut exists
- The red group {A, B, C} is separated from the green group {D}

### Why Minimum Cut Matters

Minimum cut algorithms solve real-world problems:

#### 1. **Network Reliability**
Find the weakest point in your infrastructure:
- Which network links, if they fail, would split your system?
- What's the minimum bandwidth bottleneck?
- Where should you add redundancy?

#### 2. **Image Segmentation**
Separate objects from backgrounds:
- Each pixel is a vertex
- Similar adjacent pixels have high-weight edges
- Minimum cut finds natural object boundaries

```mermaid
graph LR
    subgraph "Image Pixels"
        P1[Sky] ---|9| P2[Sky]
        P2 ---|9| P3[Sky]
        P3 ---|2| P4[Tree]
        P4 ---|8| P5[Tree]
        P5 ---|8| P6[Tree]
    end

    style P1 fill:#87ceeb
    style P2 fill:#87ceeb
    style P3 fill:#87ceeb
    style P4 fill:#228b22
    style P5 fill:#228b22
    style P6 fill:#228b22
```

#### 3. **Community Detection**
Find natural groupings in social networks:
- Strong connections within communities
- Weak connections between communities
- Minimum cut reveals community boundaries

#### 4. **VLSI Design**
Partition circuits to minimize connections between chips:
- Reduces manufacturing complexity
- Minimizes communication overhead
- Optimizes physical layout

### Global Minimum Cut vs S-T Minimum Cut

There are two types of minimum cut problems:

#### **S-T Minimum Cut (Terminal Cut)**
- You specify two vertices: source (s) and sink (t)
- Find the minimum cut that separates s from t
- Common in flow networks and image segmentation

```mermaid
graph LR
    S[Source S] ---|5| A[A]
    S ---|3| B[B]
    A ---|2| T[Sink T]
    B ---|4| T
    A ---|1| B

    style S fill:#ffcccc
    style A fill:#ffcccc
    style B fill:#ccffcc
    style T fill:#ccffcc
```

#### **Global Minimum Cut (All-Pairs)**
- No specific source/sink specified
- Find the absolute minimum cut across the entire graph
- Harder problem, but more general

**RuVector implements global minimum cut algorithms** - the most general and challenging variant.

---

## 3. Dynamic vs Static Algorithms

### The Static Approach

Traditional algorithms start from scratch every time:

```mermaid
sequenceDiagram
    participant User
    participant Algorithm
    participant Graph

    User->>Graph: Initial graph with 1000 edges
    User->>Algorithm: Compute minimum cut
    Algorithm->>Algorithm: Process all 1000 edges
    Algorithm->>User: Result (takes 10 seconds)

    User->>Graph: Add 1 edge
    User->>Algorithm: Compute minimum cut again
    Algorithm->>Algorithm: Reprocess all 1001 edges from scratch
    Algorithm->>User: Result (takes 10 seconds again!)

    Note over User,Algorithm: Inefficient: Full recomputation every time
```

**Problem**: If you add/remove just one edge, static algorithms recompute everything!

### The Dynamic Approach (Revolutionary!)

Dynamic algorithms maintain the solution incrementally:

```mermaid
sequenceDiagram
    participant User
    participant DynAlg as Dynamic Algorithm
    participant Graph

    User->>Graph: Initial graph with 1000 edges
    User->>DynAlg: Compute minimum cut
    DynAlg->>DynAlg: Process all 1000 edges, build data structures
    DynAlg->>User: Result (takes 10 seconds)

    User->>Graph: Add 1 edge
    User->>DynAlg: Update minimum cut
    DynAlg->>DynAlg: Update only affected parts
    DynAlg->>User: Result (takes 0.1 seconds!)

    Note over User,DynAlg: Efficient: Incremental updates only
```

**Advantage**: Updates are typically much faster than full recomputation!

### Why Dynamic is Revolutionary

Consider a practical scenario:

| Operation | Static Algorithm | Dynamic Algorithm |
|-----------|------------------|-------------------|
| Initial computation (10,000 edges) | 100 seconds | 100 seconds |
| Add 1 edge | 100 seconds | 0.5 seconds |
| Add 100 edges (one at a time) | 10,000 seconds (2.7 hours!) | 50 seconds |
| **Speed improvement** | — | **200× faster** |

```mermaid
graph TD
    A[Change in Graph] --> B{Use Dynamic Algorithm?}
    B -->|Yes| C[Update incrementally]
    B -->|No| D[Recompute from scratch]
    C --> E[Fast Update 0.5s]
    D --> F[Slow Recompute 100s]

    style C fill:#90EE90
    style D fill:#FFB6C6
    style E fill:#90EE90
    style F fill:#FFB6C6
```

### Amortized vs Worst-Case Complexity

Dynamic algorithms have two complexity measures:

#### **Amortized Complexity**
- Average time per operation over many operations
- Usually much better than worst-case
- Example: O(log² n) per edge insertion

#### **Worst-Case Complexity**
- Maximum time for a single operation
- Guarantees for real-time systems
- Example: O(log⁴ n) per edge insertion

These definitions describe algorithm analysis generally. This crate has not
proved a subpolynomial update bound for `DynamicMinCut`; it may rerun the
polynomial exact solver after a graph change.

---

## 4. Algorithm Choices

### DynamicMinCut (Main API)

`MinCutBuilder` constructs an exact `DynamicMinCut`. It caches the selected
partition and runs sparse Stoer-Wagner when a change requires recomputation.
The full solve is polynomial. `MinCutBuilder::approximate(ε)` is a legacy
compatibility option and does not select another algorithm; results remain
exact and report an approximation ratio of 1.0.

### ApproxMinCut (Separate Research API)

`ApproxMinCut` samples a graph using heuristic resistance estimates. Its
reported interval is calculated from ε, but it is not a certified `(1+ε)`
error guarantee for all graphs. Validate correctness and latency on your own
graph family before using it for decisions.

### Other Research Components

`SubpolynomialMinCut`, `MinCutWrapper`, `PolylogConnectivity`, and hierarchy
components expose separate experiments. Their presence does not establish the
paper bounds for the main API or a release-qualified performance claim.

---

## 5. Key Data Structures

Dynamic minimum cut algorithms rely on sophisticated data structures. You don't need to understand these deeply to use RuVector, but knowing they exist helps appreciate the complexity.

### 5.1 Link-Cut Trees

**Purpose**: Maintain connectivity in forests with dynamic edge insertions/deletions

**Operations**:
- `link(u, v)`: Connect two trees
- `cut(u, v)`: Disconnect an edge
- `find_root(v)`: Find root of v's tree
- `path_aggregate(u, v)`: Aggregate values on path from u to v

**Time Complexity**: O(log n) per operation (amortized)

```mermaid
graph TB
    subgraph "Link-Cut Tree Structure"
        R1[Root] --> C1[Child 1]
        R1 --> C2[Child 2]
        C1 --> G1[Grandchild 1]
        C1 --> G2[Grandchild 2]
        C2 --> G3[Grandchild 3]
    end

    subgraph "Operations"
        O1[link: Add edge]
        O2[cut: Remove edge]
        O3[find_root: Query root]
        O4[path_aggregate: Sum on path]
    end

    style R1 fill:#ffcccc
    style C1 fill:#ccffcc
    style C2 fill:#ccffcc
    style G1 fill:#ccccff
    style G2 fill:#ccccff
    style G3 fill:#ccccff
```

**Used in**: All three algorithms for maintaining spanning forests

### 5.2 Euler Tour Trees

**Purpose**: Alternative dynamic connectivity structure with different trade-offs

**Key Idea**: Represent tree as a cyclic sequence (Euler tour)

**Advantages**:
- Efficient subtree operations
- Good for maintaining subtree properties
- Deterministic performance

**Time Complexity**: O(log n) per operation

```mermaid
graph LR
    A[A] --> B[B]
    A --> C[C]
    B --> D[D]
    B --> E[E]

    subgraph "Euler Tour Sequence"
        direction LR
        ET[A → B → D → B → E → B → A → C → A]
    end

    style A fill:#ffcccc
    style B fill:#ccffcc
    style C fill:#ccccff
    style D fill:#ffffcc
    style E fill:#ffccff
```

**Used in**: PolylogConnectivity algorithm for deterministic guarantees

### 5.3 Hierarchical Decomposition

**Purpose**: Partition graph into levels with decreasing density

**Key Idea**:
- Level 0: Original graph
- Level i: Graph with edges of weight ≥ 2^i
- Higher levels are sparser

**Advantages**:
- Focus computation on relevant parts
- Skip unnecessary levels
- Efficient updates

```mermaid
graph TB
    subgraph "Level 0 (All edges)"
        L0A[A] ---|1| L0B[B]
        L0A ---|2| L0C[C]
        L0A ---|4| L0D[D]
        L0B ---|8| L0C
    end

    subgraph "Level 1 (Weight ≥ 2)"
        L1A[A] ---|2| L1C[C]
        L1A ---|4| L1D[D]
        L1B[B] ---|8| L1C
    end

    subgraph "Level 2 (Weight ≥ 4)"
        L2A[A] ---|4| L2D[D]
        L2B[B] ---|8| L2C[C]
    end

    subgraph "Level 3 (Weight ≥ 8)"
        L3B[B] ---|8| L3C[C]
    end

    style L0A fill:#ffcccc
    style L1A fill:#ccffcc
    style L2A fill:#ccccff
    style L3B fill:#ffffcc
```

**Used in**: Approximate and PolylogConnectivity algorithms for hierarchical graph processing

### 5.4 Local k-Cut Hierarchy

**Purpose**: Maintain minimum cuts of varying connectivity

**Key Idea**:
- Store cuts of different sizes (1-cut, 2-cut, ..., k-cut)
- Update only affected levels
- Query appropriate level for minimum cut

**Advantages**:
- Efficient querying of different cut sizes
- Incremental updates
- Supports connectivity curve analysis

```mermaid
graph TB
    H1[1-Cut: λ=2] --> H2[2-Cut: λ=5]
    H2 --> H3[3-Cut: λ=7]
    H3 --> H4[4-Cut: λ=9]

    style H1 fill:#ffcccc
    style H2 fill:#ccffcc
    style H3 fill:#ccccff
    style H4 fill:#ffffcc
```

**Used in**: All algorithms for maintaining cut hierarchies

---

## 6. Which Algorithm Should I Use?

For exact global cuts, start with `MinCutBuilder` and `DynamicMinCut`:

```rust
use ruvector_mincut::MinCutBuilder;

let mincut = MinCutBuilder::new()
    .with_edges(vec![(1, 2, 1.0), (2, 3, 1.0), (3, 1, 1.0)])
    .build()?;
assert!(mincut.min_cut().is_exact);
```

Measure updates and memory on the graph family you intend to use. A changed
cut can trigger a polynomial-time solve. `.approximate(ε)` does not reduce
this cost. Research modules should be independently evaluated before their
outputs are used in reliability, medical, or security decisions.

---

## Summary

- `DynamicMinCut` maintains an exact cut and may recompute it after updates.
- `ApproxMinCut` is a separate, experimental sampled-graph API.
- Paper complexity bounds and historical benchmark observations are not
  guarantees of the current main API.
