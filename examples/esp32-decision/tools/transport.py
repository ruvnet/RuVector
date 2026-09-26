"""Bounded JSON transport for the same firmware over native pipes or UART."""
import json
import queue
import subprocess
import threading
import time

class Device:
    def __init__(self,command=None,port=None):
        self.proc=None;self.serial=None;self.pending=bytearray()
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
        stream=self.proc.stdin if self.proc else self.serial
        stream.write((text+'\n').encode());stream.flush()
        return self.response()
    def close(self):
        if self.proc:
            self.proc.terminate()
            try:self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
            self.proc.stdin.close();self.proc.stdout.close()
        if self.serial:self.serial.close()

def stable(answer):return {k:v for k,v in answer.items() if not k.endswith('_us')}
