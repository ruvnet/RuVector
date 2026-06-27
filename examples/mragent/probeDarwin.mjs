// Probe the @metaharness/darwin surface so we wire to the real exports.
// Mirrors examples/sonic-ct/probeDarwin.mjs. Exits 0 (skip) when the optional
// package is absent — the example must run without it (ADR-150).
try {
  const mod = await import("@metaharness/darwin");
  const names = Object.keys(mod).sort();
  console.log("exports", names);
  for (const required of ["mapLimit", "paretoFront"]) {
    if (typeof mod[required] !== "function") {
      throw new Error(`Missing required export: ${required}`);
    }
  }
  const paretoNames = names.filter((n) => /pareto/i.test(n));
  console.log("pareto exports", paretoNames);
  const fnNames = names.filter((n) => typeof mod[n] === "function");
  console.log("function exports:", fnNames);
  if (typeof mod.evolve === "function") console.log("evolve.length (arity):", mod.evolve.length);
} catch (e) {
  if (e.code === "ERR_MODULE_NOT_FOUND" || e.code === "MODULE_NOT_FOUND") {
    console.warn("[probe] @metaharness/darwin not installed — skipping (optional dependency).");
    process.exit(0);
  }
  throw e;
}
