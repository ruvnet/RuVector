"""Exclusive E59 work guardian; acquiring this lock alone authorizes no science."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl, json, os, signal, stat, sys, time

BASE = Path('/dev/shm/ruvector-sota-2026-10-22/epoch59-9b677a916983/guardian27')
LOCK = Path('/tmp/ruvector-sota-2026-10-22/exclusive.work.lock')
BOOT = '4632fb06-b0ef-4df5-b72b-0e875bbad810'
NAMESPACE = 'pid:[4026532200]'

def durable_new(path, value):
    raw = (json.dumps(value, sort_keys=True) + '\n').encode()
    out = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            n = os.write(out, raw[offset:])
            if n <= 0: raise OSError('zero write')
            offset += n
        os.fsync(out)
    finally: os.close(out)
    directory = os.open(BASE, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(directory)
    finally: os.close(directory)

def main():
    assert not sys.flags.optimize and len(sys.argv) == 1
    assert Path(__file__).resolve() == BASE / 'lock_supervisor.py'
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == BOOT
    assert os.readlink('/proc/self/ns/pid') == NAMESPACE
    assert sorted(os.sched_getaffinity(0)) == [0, 1]
    assert os.getsid(0) == os.getpid() and os.getpgrp() == os.getpid(), 'detached session required'
    for name in ('owner.json', 'release.request', 'released.json', 'heartbeat.txt'):
        assert not os.path.lexists(BASE / name), 'existing guardian27 evidence: STOP'
    fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW)
    st = os.fstat(fd)
    assert fd == 3 and stat.S_ISREG(st.st_mode) and (st.st_dev, st.st_ino) == (33, 13626)
    assert (LOCK.stat().st_dev, LOCK.stat().st_ino) == (33, 13626)
    try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        print('ACTIVE_SCOPED_WORK_LOCK_STOP', flush=True)
        return 73
    stop = {'requested': False}
    def request_stop(_sig, _frame): stop['requested'] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.time()
    expiry = started + 7200
    ticks = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    owner = {
        'run_id': 'epoch59-9b677a916983-generation27',
        'epoch': 59, 'generation': 27, 'pid': os.getpid(), 'startticks': ticks,
        'lock': str(LOCK), 'lock_device': 33, 'lock_inode': 13626, 'lock_fd': 3,
        'launch_mode': 'DETACHED_BOUNDED_SESSION', 'session_id': os.getsid(0), 'process_group': os.getpgrp(),
        'parent_pid_at_start': os.getppid(),
        'started_unix': started, 'expires_unix': expiry,
        'started_at': datetime.fromtimestamp(started, timezone.utc).isoformat(),
        'expires_at': datetime.fromtimestamp(expiry, timezone.utc).isoformat(),
        'CPU_affinity': [0, 1], 'kernel_boot_id': BOOT, 'PID_namespace': NAMESPACE,
        'scope': 'ruvnet/RuVector:benchmarks/ruvector-sota-2026-10-22',
        'purpose': 'BOUNDED_RESEARCH_E59_TYPED_NLI_AND_NATIVE_EVIDENCE',
        'scientific_invocation_authorized_by_lock_alone': False,
    }
    try:
        durable_new(BASE / 'owner.json', owner)
        print(json.dumps(owner, sort_keys=True), flush=True)
        while time.time() < expiry and not stop['requested'] and not os.path.lexists(BASE / 'release.request'):
            assert (os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (33, 13626)
            assert (LOCK.stat().st_dev, LOCK.stat().st_ino) == (33, 13626)
            heartbeat = os.open(BASE / 'heartbeat.txt', os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            try: os.write(heartbeat, (str(time.time()) + '\n').encode())
            finally: os.close(heartbeat)
            time.sleep(1)
        durable_new(BASE / 'released.json', {
            'epoch': 59, 'generation': 27, 'pid': os.getpid(), 'startticks': ticks,
            'released_at': datetime.now(timezone.utc).isoformat(),
            'reason': 'signal' if stop['requested'] else ('expiry' if time.time() >= expiry else 'release.request'),
            'scientific_invocations_by_guardian': 0,
        })
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    return 0

if __name__ == '__main__': raise SystemExit(main())
