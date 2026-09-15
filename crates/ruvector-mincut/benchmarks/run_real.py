"""Bounded measurements: preserve timeout evidence, never report it as a speedup."""
import json, pathlib, subprocess, sys, time
root, executable, label = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
records = []
timer = ['/usr/bin/time', '-f', 'peak_rss_kib=%M'] if pathlib.Path('/usr/bin/time').is_file() else []
for item in json.loads((root / 'manifest.json').read_text()):
    start = time.monotonic()
    try:
        p = subprocess.run(timer + [executable, str(root/item['file']), str(item['oracle_cut'])], text=True, capture_output=True, timeout=90)
        row = dict(name=item['name'], label=label, status='ok' if p.returncode == 0 else 'error', output=p.stdout, diagnostics=p.stderr)
        if p.returncode: raise RuntimeError(row)
    except subprocess.TimeoutExpired as e:
        row = dict(name=item['name'], label=label, status='timeout', limit_seconds=90)
    row['wall_seconds'] = time.monotonic()-start
    records.append(row)
    print(json.dumps(row), flush=True)
(root / f'{label}.json').write_text(json.dumps(records, indent=2)+'\n')
