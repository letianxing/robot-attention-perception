"""Bounded numeric decision traces; no images, raw audio, or biometric vectors."""
import json,os,queue,threading
from pathlib import Path

class DecisionTrace:
    def __init__(self,path,max_bytes=10*1024*1024,backups=2):
        self.path=Path(path);self.max_bytes=max_bytes;self.backups=backups
        self.items=queue.Queue(maxsize=128);self.stop=threading.Event();self.dropped=0;self.error='';self.written=0
        self.thread=threading.Thread(target=self._run,daemon=True,name='attention-decision-trace');self.thread.start()
    def submit(self,record):
        try:self.items.put_nowait(record)
        except queue.Full:self.dropped+=1
    def status(self):return {'path':str(self.path),'dropped':self.dropped,'error':self.error,'written':self.written}
    def close(self):self.stop.set();self.thread.join(timeout=2)
    def _run(self):
        stream=None;size=0
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            stream=self.path.open('ab');size=self.path.stat().st_size
            while not self.stop.is_set() or not self.items.empty():
                try:item=self.items.get(timeout=.1)
                except queue.Empty:continue
                data=(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n').encode()
                if size+len(data)>self.max_bytes:
                    stream.close()
                    for i in range(self.backups,0,-1):
                        source=self.path if i==1 else Path(str(self.path)+f'.{i-1}')
                        if source.exists():os.replace(source,Path(str(self.path)+f'.{i}'))
                    stream=self.path.open('wb');size=0
                stream.write(data);stream.flush();size+=len(data);self.written+=1
        except Exception as exc:self.error=str(exc)
        finally:
            if stream:stream.close()
