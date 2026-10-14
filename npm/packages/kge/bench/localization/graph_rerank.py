"""Inductive typed-code-graph reranker for the localization benchmark.

This is an experimental benchmark sidecar. It does not call the native KGE
runtime and makes no production API claim. Repository ASTs are parsed as text;
source files are never imported or executed.
"""
import ast
import math
from collections import defaultdict

RELATIONS = ("import", "call", "inherit")
FEATURES = ("bm25_rank", "swerank_rank") + tuple(f"{r}_{hop}" for r in RELATIONS for hop in (1, 2))


def _module(path):
    value = path[:-3] if path.endswith(".py") else path
    if value.endswith("/__init__"):
        value = value[:-9]
    return value.replace("/", ".")


def _local_module(module, paths, package):
    if module in paths:
        return module
    prefix = package + "." if package else ""
    candidate = prefix + module if module else prefix.rstrip(".")
    if candidate in paths:
        return candidate
    return None


def build_graph(sources):
    """Return a graph from {relative_python_path: complete_source_text}."""
    if not isinstance(sources, dict) or any(not isinstance(path, str) or not path.endswith(".py")
                                             or not isinstance(source, str)
                                             for path, source in sources.items()):
        raise ValueError("expected complete Python source text keyed by path")
    paths_by_module = {_module(path): path for path in sources}
    names = defaultdict(set)
    parsed = {}
    for path, source in sources.items():
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, TypeError):
            continue
        parsed[path] = tree
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names[node.name].add(path)

    edges = {relation: defaultdict(set) for relation in RELATIONS}
    for path, tree in parsed.items():
        package = _module(path).rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = _local_module(alias.name, paths_by_module, package)
                    if target and paths_by_module[target] != path:
                        edges["import"][path].add(paths_by_module[target])
            elif isinstance(node, ast.ImportFrom):
                prefix = package
                if node.level:
                    parts = package.split(".") if package else []
                    keep = max(0, len(parts) - node.level + 1)
                    prefix = ".".join(parts[:keep])
                imported = ((".".join(x for x in (prefix, node.module or "") if x))
                            if node.level else (node.module or ""))
                candidates = [imported] + [".".join(x for x in (imported, a.name) if x)
                                            for a in node.names]
                for candidate in candidates:
                    target = _local_module(candidate, paths_by_module, "")
                    if target and paths_by_module[target] != path:
                        edges["import"][path].add(paths_by_module[target])
            elif isinstance(node, ast.ClassDef):
                for base in node.bases:
                    name = base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else None
                    if name:
                        for target in names.get(name, ()):
                            if target != path:
                                edges["inherit"][path].add(target)
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
                if name:
                    for target in names.get(name, ()):
                        if target != path:
                            edges["call"][path].add(target)
    return {rel: {src: tuple(sorted(dst)) for src, dst in sorted(adj.items())}
            for rel, adj in edges.items()}


def _neighbors(graph, relation, node, reverse=False):
    if not reverse:
        return graph[relation].get(node, ())
    return tuple(src for src, dsts in graph[relation].items() if node in dsts)


def graph_features(graph, rankings, candidates):
    """Build fixed-width features from BM25 and SWERank retrieval rankings."""
    if len(candidates) != len(set(candidates)):
        raise ValueError("duplicate candidate path")
    if set(rankings) != {"bm25", "swerank"}:
        raise ValueError("both retrieval rankings are required")
    positions = {}
    seed_weights = defaultdict(float)
    for arm, ranking in rankings.items():
        positions[arm] = {row["path"]: i + 1 for i, row in enumerate(ranking)}
        if not set(candidates) <= set(positions[arm]):
            raise ValueError("retrieval ranking omits candidates")
        for i, row in enumerate(ranking[:min(10, len(ranking))]):
            seed_weights[row["path"]] = max(seed_weights[row["path"]], 1.0 / math.log2(i + 2))
    result = {}
    for path in candidates:
        values = [1.0 / positions[arm][path] for arm in ("bm25", "swerank")]
        for relation in RELATIONS:
            one = set(_neighbors(graph, relation, path)) | set(_neighbors(graph, relation, path, True))
            score1 = sum(seed_weights.get(other, 0.0) for other in one)
            two = set()
            for middle in one:
                two.update(_neighbors(graph, relation, middle))
                two.update(_neighbors(graph, relation, middle, True))
            two.discard(path)
            score2 = sum(seed_weights.get(other, 0.0) for other in two)
            values.extend((score1, score2))
        result[path] = tuple(values)
    return result


def _solve(matrix, vector):
    """Small deterministic Gaussian elimination with pivoting."""
    n = len(vector)
    a = [list(row) + [float(vector[i])] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            a[pivot][col] = 1e-12
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [v / scale for v in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if factor:
                a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [a[i][-1] for i in range(n)]


def fit_pairwise(examples, l2=1.0):
    """Fit ridge weights to positive-minus-negative feature differences."""
    if l2 <= 0 or not math.isfinite(l2):
        raise ValueError("L2 must be finite and positive")
    dim = len(FEATURES)
    gram = [[0.0] * dim for _ in range(dim)]
    rhs = [0.0] * dim
    pair_count = 0
    for example in examples:
        features = example["features"]
        positive = set(example["gold"])
        if not positive or not positive <= set(features):
            raise ValueError("gold file missing from feature candidates")
        negatives = sorted(set(features) - positive)
        for good in sorted(positive):
            for bad in negatives:
                delta = [a - b for a, b in zip(features[good], features[bad])]
                if len(delta) != dim:
                    raise ValueError("wrong feature width")
                for i in range(dim):
                    rhs[i] += delta[i]
                    for j in range(dim):
                        gram[i][j] += delta[i] * delta[j]
                pair_count += 1
    if pair_count == 0:
        raise ValueError("no positive-negative training pairs")
    for i in range(dim):
        gram[i][i] += l2
    return _solve(gram, rhs)


def score_ranking(ranking, features, weights):
    if len(weights) != len(FEATURES):
        raise ValueError("wrong model width")
    candidates = [row["path"] for row in ranking]
    if set(candidates) != set(features):
        raise ValueError("feature and retrieval candidate universes differ")
    scored = []
    for row in ranking:
        vector = features[row["path"]]
        score = sum(a * b for a, b in zip(weights, vector))
        if not math.isfinite(score):
            raise ValueError("nonfinite graph score")
        scored.append({"path": row["path"], "score": score})
    return sorted(scored, key=lambda row: (-row["score"], row["path"]))


def select_l2_loro(examples, l2_values=(0.01, 0.1, 1.0, 10.0)):
    """Select L2 by leave-one-repository-out macro MRR; ties favor smaller L2."""
    repos = sorted({example["repo"] for example in examples})
    values = sorted(set(float(value) for value in l2_values))
    if len(repos) < 2 or not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("need multiple repositories and positive finite L2 values")
    results = []
    for l2 in values:
        fold_scores = []
        for heldout in repos:
            train = [example for example in examples if example["repo"] != heldout]
            test = [example for example in examples if example["repo"] == heldout]
            weights = fit_pairwise(train, l2=l2)
            reciprocal_ranks = []
            for example in test:
                ranking = score_ranking(example["ranking"], example["features"], weights)
                paths = [row["path"] for row in ranking]
                gold = set(example["gold"])
                if not gold or not gold <= set(paths):
                    raise ValueError("validation gold missing from candidate ranking")
                reciprocal_ranks.append(1.0 / min(paths.index(path) + 1 for path in gold))
            fold_scores.append(sum(reciprocal_ranks) / len(reciprocal_ranks))
        results.append({"l2": l2, "macroMrr": sum(fold_scores) / len(fold_scores),
                        "repositoryMrr": fold_scores})
    winner = min(results, key=lambda row: (-row["macroMrr"], row["l2"]))
    return {"l2": winner["l2"], "macroMrr": winner["macroMrr"], "results": results}
