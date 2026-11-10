"""Execute the published Python quickstarts against installed distributions."""
from pathlib import Path
import re
import json
import subprocess
import sys
import tempfile

base = Path(__file__).resolve().parent
count = 0
for name in json.loads((base / 'manifest.json').read_text())['packages']:
    readme = base / name / 'README.md'
    examples = re.findall(r'```python\n(.*?)```', readme.read_text(), re.S)
    if not examples:
        raise ValueError(f'{name} has no executable Python quickstart')
    for code in examples:
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, '-c', code], cwd=directory,
                           stdin=subprocess.DEVNULL, check=True, timeout=30)
        count += 1
        print(f'{readme.parent.name}: quickstart passed')
print(f'{count} quickstarts passed across all manifest packages')
