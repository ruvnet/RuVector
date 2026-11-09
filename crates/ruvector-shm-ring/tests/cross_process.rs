//! A second process maps the read-only descriptor and reads committed records.

#[cfg(unix)]
#[test]
fn a_second_process_reads_committed_records() {
    use std::os::fd::{AsFd, AsRawFd};
    use std::os::unix::process::CommandExt;
    use std::process::Command;

    use ruvector_shm_ring::{allocate, Geometry, Mapping, RingWriter};

    if std::env::var_os("RUVECTOR_SHM_RING_CHILD").is_some() {
        child();
        return;
    }

    let geometry = Geometry::fit(64, 64 * 1024).unwrap();
    let shared = allocate(geometry.bytes()).unwrap();
    let map = Mapping::map(shared.read_write.as_fd(), geometry.bytes(), true).unwrap();
    let mut writer = RingWriter::format(map, geometry, 1).unwrap();
    for i in 1..=10u64 {
        writer.write(i, 7, &i.to_le_bytes()).unwrap();
    }

    let raw = shared.read_only.as_raw_fd();
    let exe = std::env::current_exe().unwrap();
    let mut cmd = Command::new(exe);
    cmd.arg("a_second_process_reads_committed_records")
        .arg("--exact")
        .env("RUVECTOR_SHM_RING_CHILD", "1")
        .env("RUVECTOR_SHM_RING_LEN", geometry.bytes().to_string())
        .env("RUVECTOR_SHM_RING_SLOTS", geometry.slot_count.to_string())
        .env(
            "RUVECTOR_SHM_RING_SLOT_SIZE",
            geometry.slot_size.to_string(),
        )
        .env("RUVECTOR_SHM_RING_MAX", geometry.max_record.to_string());
    unsafe {
        cmd.pre_exec(move || {
            let rc = libc::dup2(raw, 42);
            if rc < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let output = cmd.output().unwrap();
    assert!(
        output.status.success(),
        "status {:?}\nstdout {}\nstderr {}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    // Keep the writer mapping alive until the child has finished reading.
    assert_eq!(writer.head(), 10);
}

#[cfg(unix)]
fn child() {
    use std::os::fd::{AsFd, FromRawFd, OwnedFd};

    use ruvector_shm_ring::{Geometry, Mapping, RingReader};

    let len: usize = std::env::var("RUVECTOR_SHM_RING_LEN")
        .unwrap()
        .parse()
        .unwrap();
    let geometry = Geometry::from_parts(
        std::env::var("RUVECTOR_SHM_RING_SLOTS")
            .unwrap()
            .parse()
            .unwrap(),
        std::env::var("RUVECTOR_SHM_RING_SLOT_SIZE")
            .unwrap()
            .parse()
            .unwrap(),
        std::env::var("RUVECTOR_SHM_RING_MAX")
            .unwrap()
            .parse()
            .unwrap(),
    )
    .unwrap();
    let fd = unsafe { OwnedFd::from_raw_fd(42) };
    let map = Mapping::map(fd.as_fd(), len, false).expect("child map");
    let reader = RingReader::attach(map, geometry).expect("child attach");
    let mut next = 1u64;
    for i in 1..=10u64 {
        let record = reader.read(&mut next).expect("record");
        assert_eq!(record.seq, i);
        assert_eq!(record.t_ns, i);
        assert_eq!(record.w_ns, 7);
        assert_eq!(record.data, i.to_le_bytes());
    }
    assert!(reader.read(&mut next).is_none());
}

#[cfg(not(unix))]
#[test]
fn shared_memory_is_unavailable() {
    // `allocate` is unix-only. The in-process ring is covered by the library tests.
}
