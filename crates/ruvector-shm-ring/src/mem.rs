//! Mappings the ring reads and writes through atomics.
//!
//! [`Mapping::anonymous`] is private to this process. [`allocate`] creates an
//! object another process can map, and hands back a read-only descriptor so
//! a reader that asks for write access is refused by the kernel.

use std::io;
use std::ptr::NonNull;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

#[cfg(unix)]
use std::os::fd::AsRawFd;

/// A view of ring memory. Words are reached only through [`AtomicU64`].
#[derive(Debug)]
pub struct Mapping {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    ptr: NonNull<u8>,
    len: usize,
    writable: bool,
    kind: Kind,
}

#[derive(Debug)]
enum Kind {
    Heap,
    #[cfg(unix)]
    Mapped,
}

// SAFETY: the bytes are shared memory (or a private buffer) and every access
// goes through AtomicU64. The pointer stays valid until the last Mapping drops.
unsafe impl Send for Inner {}
// SAFETY: as above. Atomic operations are the only way words are touched.
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            match self.kind {
                Kind::Heap => {
                    let layout = heap_layout(self.len).expect("len was checked at allocation");
                    std::alloc::dealloc(self.ptr.as_ptr(), layout);
                }
                #[cfg(unix)]
                Kind::Mapped => {
                    libc::munmap(self.ptr.as_ptr().cast(), self.len);
                }
            }
        }
    }
}

impl Mapping {
    /// Private zeroed buffer of `len` bytes. It is not visible to other processes.
    pub fn anonymous(len: usize) -> io::Result<Self> {
        if len == 0 || !len.is_multiple_of(8) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "mapping length must be a non-zero multiple of 8",
            ));
        }
        let layout = heap_layout(len)?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).ok_or_else(|| io::Error::from(io::ErrorKind::OutOfMemory))?;
        Ok(Self {
            inner: Arc::new(Inner {
                ptr,
                len,
                writable: true,
                kind: Kind::Heap,
            }),
        })
    }

    /// Map `len` bytes of `fd`.
    ///
    /// Refuses an object shorter than `len`. A writable map of a read-only
    /// descriptor fails with the kernel's permission error.
    #[cfg(unix)]
    pub fn map(fd: std::os::fd::BorrowedFd<'_>, len: usize, writable: bool) -> io::Result<Self> {
        if len == 0 || !len.is_multiple_of(8) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "mapping length must be a non-zero multiple of 8",
            ));
        }
        let mut st = std::mem::MaybeUninit::<libc::stat>::zeroed();
        let rc = unsafe { libc::fstat(fd.as_raw_fd(), st.as_mut_ptr()) };
        if rc < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: fstat wrote a complete stat on success.
        let st = unsafe { st.assume_init() };
        if st.st_size < 0 || (st.st_size as u64) < len as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("object holds {} bytes, need {len}", st.st_size),
            ));
        }
        let prot = if writable {
            libc::PROT_READ | libc::PROT_WRITE
        } else {
            libc::PROT_READ
        };
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                prot,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let ptr = NonNull::new(p.cast::<u8>()).ok_or_else(io::Error::last_os_error)?;
        Ok(Self {
            inner: Arc::new(Inner {
                ptr,
                len,
                writable,
                kind: Kind::Mapped,
            }),
        })
    }

    /// Another handle to the same bytes.
    pub fn share(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Mapped length in bytes.
    pub fn len(&self) -> usize {
        self.inner.len
    }

    /// Whether the mapping contains no bytes.
    ///
    /// A usable ring is never empty. Allocation and
    /// [`Geometry::fit`](crate::Geometry::fit) both refuse a zero length.
    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    /// Whether this view was mapped with write permission.
    pub fn writable(&self) -> bool {
        self.inner.writable
    }

    pub(crate) fn word(&self, off: usize) -> &AtomicU64 {
        assert!(
            off.is_multiple_of(8) && off + 8 <= self.inner.len,
            "word {off} is outside a {}-byte mapping",
            self.inner.len
        );
        // SAFETY: `off` is in range and 8-aligned. The allocation is at least
        // 8-aligned (64 for the heap, page-aligned for mmap) and lives as long
        // as `self`. Every access to this address is atomic.
        unsafe { &*self.inner.ptr.as_ptr().add(off).cast::<AtomicU64>() }
    }
}

fn heap_layout(len: usize) -> io::Result<std::alloc::Layout> {
    std::alloc::Layout::from_size_align(len, 64)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "mapping length is too large"))
}

/// A shared-memory object with a read-write descriptor and a read-only one.
///
/// [`allocate`] is the only constructor, and it returns this only on Linux and macOS.
#[cfg(unix)]
#[derive(Debug)]
pub struct SharedMemory {
    /// Read-write descriptor. The writer maps this.
    pub read_write: std::os::fd::OwnedFd,
    /// Read-only descriptor of the same object. Hand this to a reader.
    pub read_only: std::os::fd::OwnedFd,
}

/// Create a zeroed shared-memory object of `len` bytes.
///
/// On Linux the object is a memfd sealed against shrinking and growing, and
/// the read-only descriptor is a reopen of that memfd. On macOS it is a POSIX
/// shared-memory object whose name is removed before this function returns.
/// Other targets return [`io::ErrorKind::Unsupported`].
#[cfg(unix)]
pub fn allocate(len: usize) -> io::Result<SharedMemory> {
    if len == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "shared-memory length must be non-zero",
        ));
    }
    let (read_write, read_only) = platform::create(len)?;
    Ok(SharedMemory {
        read_write,
        read_only,
    })
}

#[cfg(unix)]
mod platform {
    use std::io;
    use std::os::fd::{FromRawFd, OwnedFd};

    fn check(rc: libc::c_int) -> io::Result<libc::c_int> {
        if rc < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(rc)
        }
    }

    #[cfg(target_os = "linux")]
    pub(super) fn create(len: usize) -> io::Result<(OwnedFd, OwnedFd)> {
        let flags = libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING;
        use std::os::fd::AsRawFd;
        let fd = check(unsafe { libc::memfd_create(c"rvr".as_ptr(), flags) })?;
        let read_write = unsafe { OwnedFd::from_raw_fd(fd) };
        let size = libc::off_t::try_from(len).map_err(|_| io::ErrorKind::InvalidInput)?;
        check(unsafe { libc::ftruncate(read_write.as_raw_fd(), size) })?;
        let seals = libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_SEAL;
        check(unsafe { libc::fcntl(read_write.as_raw_fd(), libc::F_ADD_SEALS, seals) })?;
        let path = std::ffi::CString::new(format!("/proc/self/fd/{}", read_write.as_raw_fd()))
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "fd path"))?;
        let ro = check(unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) })?;
        Ok((read_write, unsafe { OwnedFd::from_raw_fd(ro) }))
    }

    #[cfg(target_os = "macos")]
    pub(super) fn create(len: usize) -> io::Result<(OwnedFd, OwnedFd)> {
        use std::os::fd::AsRawFd;
        use std::sync::atomic::{AtomicU32, Ordering};

        static NEXT: AtomicU32 = AtomicU32::new(0);
        let mut last = io::Error::from(io::ErrorKind::AlreadyExists);
        for _ in 0..8 {
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let tick = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0);
            let name = format!(
                "/r{:x}{:04x}{:04x}",
                std::process::id(),
                tick & 0xffff,
                n & 0xffff
            );
            if name.len() >= 31 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "shared-memory name is too long",
                ));
            }
            let c_name = std::ffi::CString::new(name)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "shared-memory name"))?;
            let fd = unsafe {
                libc::shm_open(
                    c_name.as_ptr(),
                    libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                    0o600,
                )
            };
            if fd < 0 {
                last = io::Error::last_os_error();
                continue;
            }
            let read_write = unsafe { OwnedFd::from_raw_fd(fd) };
            let opened: io::Result<OwnedFd> = (|| {
                let size = libc::off_t::try_from(len).map_err(|_| io::ErrorKind::InvalidInput)?;
                check(unsafe { libc::ftruncate(read_write.as_raw_fd(), size) })?;
                let ro = check(unsafe { libc::shm_open(c_name.as_ptr(), libc::O_RDONLY, 0) })?;
                Ok(unsafe { OwnedFd::from_raw_fd(ro) })
            })();
            unsafe { libc::shm_unlink(c_name.as_ptr()) };
            let read_only = opened?;
            cloexec(&read_write)?;
            cloexec(&read_only)?;
            return Ok((read_write, read_only));
        }
        Err(last)
    }

    #[cfg(target_os = "macos")]
    fn cloexec(fd: &OwnedFd) -> io::Result<()> {
        use std::os::fd::AsRawFd;
        check(unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) }).map(|_| ())
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub(super) fn create(_len: usize) -> io::Result<(OwnedFd, OwnedFd)> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "shared-memory rings are implemented on Linux and macOS",
        ))
    }
}
