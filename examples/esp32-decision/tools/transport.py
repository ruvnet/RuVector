"""Bounded JSON transport for the same firmware over native pipes or UART."""
import json
import queue
import subprocess
import threading
import time

def _drain(stream):
    """Finish a write before waiting for the reply.

    pyserial's Windows backend implements flush() as "sleep 50 ms while bytes
    are queued", which added ~50 ms stalls to 1-5% of fast-link queries on a
    physical C6. Its blocking write() already waits for the overlapped write to
    complete, so nothing more is needed there. Every other stream (pipes,
    POSIX tcdrain, test adapters) keeps its own flush()."""
    if type(stream).__module__!='serial.serialwin32' or getattr(stream,'write_timeout',None)==0:
        stream.flush()

class Device:
    def __init__(self,command=None,port=None,baud=None):
        self.proc=None;self.serial=None;self.pending=bytearray()
        self._query_lock=threading.Lock()
        if command:
            self.proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
            self.lines=queue.Queue(maxsize=256)
            def read():
                while True:
                    data=self.proc.stdout.readline(16385)
                    self.lines.put(data)
                    if not data:return
            threading.Thread(target=read,daemon=True).start()
            self.meta=self.response()
        else:
            import serial
            self.serial=serial.Serial(port,115200,timeout=.25,write_timeout=5)
            time.sleep(2)
            self.serial.reset_input_buffer()
            self.meta=self.query('meta')
            if baud and baud!=BAUD_DEFAULT:
                with self._query_lock:
                    negotiate_baud(self.serial,baud);self.pending.clear()
    def response(self,timeout=30):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if self.proc:
                try:line=self.lines.get(timeout=max(.01,deadline-time.monotonic()))
                except queue.Empty:break
                if not line:raise RuntimeError('firmware closed stdout')
                if len(line)>16384:raise ValueError('oversized response')
            else:
                self.pending.extend(self.serial.read_until(b'\n',16385))
                if len(self.pending)>16384:raise ValueError('oversized response')
                if not self.pending.endswith(b'\n'):continue
                line=bytes(self.pending);self.pending.clear()
            if line.startswith(b'{'):return json.loads(line)
        raise TimeoutError('firmware response deadline')
    def query(self,text):
        if '\r' in text or '\n' in text:
            raise ValueError('firmware command must be a single line')
        with self._query_lock:
            stream=self.proc.stdin if self.proc else self.serial
            stream.write((text+'\n').encode());_drain(stream)
            return self.response()
    def close(self):
        if self.proc:
            self.proc.terminate()
            try:self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
            self.proc.stdin.close();self.proc.stdout.close()
        if self.serial:self.serial.close()

BAUD_DEFAULT=115200
BAUD_RATES=(115200,230400,460800,921600)

def _json_lines(ser,deadline):
    buf=bytearray()
    while time.monotonic()<deadline:
        buf.extend(ser.read_until(b'\n',4097))
        if len(buf)>4096:buf.clear();continue
        if not buf.endswith(b'\n'):continue
        line=bytes(buf);buf.clear()
        if line.startswith(b'{'):
            try:yield json.loads(line)
            except json.JSONDecodeError:continue

def negotiate_baud(ser,rate,timeout=3.0):
    """Switch a running board to `rate`; it reverts to 115200 unless confirmed.

    The confirmation is preceded by a bare newline so any bytes garbled during
    the switch end up in their own (rejected) line rather than in `baud ok`."""
    if rate not in BAUD_RATES:raise ValueError('unsupported baud rate')
    ser.reset_input_buffer();ser.write(f'baud {rate}\n'.encode());_drain(ser)
    reply=next(_json_lines(ser,time.monotonic()+timeout),None)
    if not reply or reply.get('baud')!=rate:raise RuntimeError(f'baud switch refused: {reply}')
    time.sleep(0.05);ser.baudrate=rate;time.sleep(0.05);ser.reset_input_buffer()
    ser.write(b'\nbaud ok\n');_drain(ser)
    for msg in _json_lines(ser,time.monotonic()+timeout):
        if msg.get('confirmed') and msg.get('baud')==rate:return rate
    raise TimeoutError('baud confirmation not received; board reverts to 115200')

def stable(answer):return {k:v for k,v in answer.items() if not k.endswith('_us')}
