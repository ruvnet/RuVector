"""Bounded measurements: preserve timeout evidence, never report it as a speedup."""
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

root, executable, label = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
records = []
timer = ['/usr/bin/time', '-f', 'peak_rss_kib=%M'] if pathlib.Path('/usr/bin/time').is_file() else []
for item in json.loads((root / 'manifest.json').read_text()):
    start = time.monotonic()
    process = subprocess.Popen(timer + [executable, str(root/item['file']), str(item['oracle_cut'])],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        output, diagnostics = process.communicate(timeout=90)
        row = dict(name=item['name'], label=label, status='ok' if process.returncode == 0 else 'error',
            output=output, diagnostics=diagnostics)
        if process.returncode: raise RuntimeError(row)
    except subprocess.TimeoutExpired:
        # Kill the whole process group, including the timed child, before the next case.
        os.killpg(process.pid, signal.SIGKILL)
        output, diagnostics = process.communicate()
        row = dict(name=item['name'], label=label, status='timeout', limit_seconds=90,
            output=output, diagnostics=diagnostics)
    row['wall_seconds'] = time.monotonic()-start
    records.append(row)
    print(json.dumps(row), flush=True)
(root / f'{label}.json').write_text(json.dumps(records, indent=2)+'\n')
