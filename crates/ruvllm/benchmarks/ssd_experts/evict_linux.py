"""Evict only this benchmark's private model file and count resident pages."""
import ctypes, json, mmap, os, sys
from pathlib import Path
path=Path(sys.argv[1])
page=os.sysconf('SC_PAGESIZE')
c=ctypes.CDLL(None,use_errno=True)
with path.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_COPY) as view:
    addr=ctypes.addressof(ctypes.c_char.from_buffer(view))
    pages=(len(view)+page-1)//page
    def present():
        vec=(ctypes.c_ubyte*pages)()
        if c.mincore(ctypes.c_void_p(addr),ctypes.c_size_t(len(view)),vec)!=0: raise OSError(ctypes.get_errno(),'mincore')
        return sum(x&1 for x in vec)
    before=present()
    os.fsync(f.fileno())
    os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    after=present()
    print(json.dumps(dict(bytes=len(view),pages=pages,before=before,after=after,page_size=page)))
