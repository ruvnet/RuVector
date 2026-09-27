#!/usr/bin/env python3
"""Update a running board over its serial port using the firmware's OTA command.

python3 tools/ota.py --port /dev/ttyUSB0 ruvector_decision.bin
Sends the application image (not the merged factory image) in acknowledged
1 KiB blocks, waits for the restart, and reports whether the new slot was
committed by its boot self-test or rolled back. Use --status to only query.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from transport import _drain, negotiate_baud

def lines(s, deadline):
    while time.monotonic() < deadline:
        raw = s.readline()
        if raw.startswith(b'{'):
            try: yield json.loads(raw)
            except json.JSONDecodeError: continue

def wait_for(s, predicate, timeout):
    deadline = time.monotonic()+timeout
    seen = []
    for msg in lines(s, deadline):
        seen.append(msg)
        if 'error' in msg: raise RuntimeError(msg)
        if predicate(msg): return msg, seen
    raise TimeoutError(f'no matching reply; saw {seen[-3:]}')

def query(s, command, key, timeout=5):
    s.reset_input_buffer(); s.write((command+'\n').encode()); _drain(s)
    return wait_for(s, lambda m: key in m, timeout)[0]

def update(port, image, boot_timeout=20, baud=None):
    import serial
    data = Path(image).read_bytes()
    if not data or data[0] != 0xE9: raise ValueError('not an ESP application image (expected magic 0xE9)')
    if data[0x8000:0x8002] == b'\xaa\x50':
        raise ValueError('this is a factory image with a partition table; send the app image (*-ota.bin / ruvector_decision.bin)')
    digest = hashlib.sha256(data).hexdigest()
    with serial.Serial(port, 115200, timeout=1) as s:
        time.sleep(0.2)
        before = query(s, 'ota status', 'ota')
        if baud and baud != 115200:

            negotiate_baud(s, baud)
        ready = query(s, f'ota begin {len(data)} {digest}', 'ota')
        if ready.get('ota') != 'ready': raise RuntimeError(ready)
        block = int(ready['block']); sent = 0; t0 = time.monotonic()
        while sent < len(data):
            chunk = data[sent:sent+block]; s.write(chunk); _drain(s); sent += len(chunk)
            if sent < len(data):
                wait_for(s, lambda m: m.get('ota_ack') == sent, 10)
            print(f'\r{sent}/{len(data)} bytes', end='', file=sys.stderr)
        print(file=sys.stderr)
        done, _ = wait_for(s, lambda m: m.get('ota') == 'done', 30)
        transfer_s = time.monotonic()-t0
        # The board restarts at 115200; collect the boot report and the decision.
        if baud and baud != 115200: s.baudrate = 115200
        boot, seen = wait_for(s, lambda m: m.get('event') == 'ready' or m.get('ota') == 'rollback', boot_timeout)
        outcome = 'rollback' if boot.get('ota') == 'rollback' else None
        if outcome is None:
            try: outcome = wait_for(s, lambda m: m.get('ota') in ('committed', 'rollback'), 3)[0]['ota']
            except TimeoutError: outcome = 'no_decision'
        if outcome == 'rollback':
            try: wait_for(s, lambda m: m.get('event') == 'ready', boot_timeout)
            except (TimeoutError, RuntimeError): pass
        time.sleep(0.5)
        after = query(s, 'ota status', 'ota')
    return {'image': str(image), 'bytes': len(data), 'sha256': digest, 'transfer_s': round(transfer_s, 1),
            'before': before, 'done': done, 'outcome': outcome, 'after': after,
            'boot_selftest': boot.get('selftest_pass'), 'switched': after['running'] != before['running']}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', nargs='?'); p.add_argument('--port', required=True)
    p.add_argument('--status', action='store_true')
    p.add_argument('--baud', type=int, default=115200, help='negotiate a faster link for the transfer (230400/460800/921600)')
    a = p.parse_args()
    if a.status or not a.image:
        import serial
        with serial.Serial(a.port, 115200, timeout=1) as s: print(json.dumps(query(s, 'ota status', 'ota'), indent=2))
        return
    report = update(a.port, a.image, baud=a.baud)
    print(json.dumps(report, indent=2))
    if report['outcome'] != 'committed': sys.exit(1)
if __name__ == '__main__': main()
