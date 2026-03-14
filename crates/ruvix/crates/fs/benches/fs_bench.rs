//! Benchmarks for the RuVix filesystem layer.

#![cfg(feature = "alloc")]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use ruvix_fs::{FileSystem, FileType, Path, PathBuf, RamFs};

fn bench_path_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("path_parsing");

    group.bench_function("parse_short", |b| {
        b.iter(|| {
            let path = Path::new(black_box("/foo/bar"));
            black_box(path.file_name())
        });
    });

    group.bench_function("parse_long", |b| {
        let long_path = "/a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s/t/u/v/w/x/y/z";
        b.iter(|| {
            let path = Path::new(black_box(long_path));
            black_box(path.file_name())
        });
    });

    group.bench_function("components_iterate", |b| {
        let path = Path::new("/usr/local/bin/program");
        b.iter(|| {
            let count: usize = path.components().count();
            black_box(count)
        });
    });

    group.bench_function("pathbuf_push", |b| {
        b.iter(|| {
            let mut path = PathBuf::from("/home");
            path.push(Path::new("user"));
            path.push(Path::new("documents"));
            path.push(Path::new("file.txt"));
            black_box(path)
        });
    });

    group.finish();
}

fn bench_ramfs_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("ramfs");

    group.bench_function("create_file", |b| {
        let mut fs = RamFs::new();
        fs.mount().unwrap();
        let root = fs.root().unwrap();
        let mut counter = 0u64;

        b.iter(|| {
            counter += 1;
            let name = format!("file_{}.txt", counter);
            let id = fs.create(root, &name, FileType::Regular, 0o644).unwrap();
            black_box(id)
        });
    });

    group.bench_function("create_directory", |b| {
        let mut fs = RamFs::new();
        fs.mount().unwrap();
        let root = fs.root().unwrap();
        let mut counter = 0u64;

        b.iter(|| {
            counter += 1;
            let name = format!("dir_{}", counter);
            let id = fs.create(root, &name, FileType::Directory, 0o755).unwrap();
            black_box(id)
        });
    });

    group.bench_function("lookup_file", |b| {
        let mut fs = RamFs::new();
        fs.mount().unwrap();
        let root = fs.root().unwrap();

        // Create some files
        for i in 0..100 {
            let name = format!("file_{}.txt", i);
            fs.create(root, &name, FileType::Regular, 0o644).unwrap();
        }

        b.iter(|| {
            let id = fs.lookup(root, black_box("file_50.txt")).unwrap();
            black_box(id)
        });
    });

    group.bench_function("stat", |b| {
        let mut fs = RamFs::new();
        fs.mount().unwrap();
        let root = fs.root().unwrap();
        let file_id = fs.create(root, "test.txt", FileType::Regular, 0o644).unwrap();

        b.iter(|| {
            let stat = fs.stat(black_box(file_id)).unwrap();
            black_box(stat)
        });
    });

    group.finish();
}

fn bench_ramfs_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("ramfs_io");

    // 1KB write
    group.throughput(Throughput::Bytes(1024));
    group.bench_function("write_1kb", |b| {
        let mut fs = RamFs::new();
        fs.mount().unwrap();
        let root = fs.root().unwrap();
        let file_id = fs.create(root, "data.bin", FileType::Regular, 0o644).unwrap();
        let data = [0xABu8; 1024];

        b.iter(|| {
            let written = fs.write(file_id, 0, black_box(&data)).unwrap();
            black_box(written)
        });
    });

    // 1KB read
    group.throughput(Throughput::Bytes(1024));
    group.bench_function("read_1kb", |b| {
        let mut fs = RamFs::new();
        fs.mount().unwrap();
        let root = fs.root().unwrap();
        let file_id = fs.create(root, "data.bin", FileType::Regular, 0o644).unwrap();
        let data = [0xABu8; 1024];
        fs.write(file_id, 0, &data).unwrap();

        let mut buf = [0u8; 1024];
        b.iter(|| {
            let read = fs.read(file_id, 0, black_box(&mut buf)).unwrap();
            black_box(read)
        });
    });

    // 64KB write
    group.throughput(Throughput::Bytes(65536));
    group.bench_function("write_64kb", |b| {
        let mut fs = RamFs::new();
        fs.mount().unwrap();
        let root = fs.root().unwrap();
        let file_id = fs.create(root, "large.bin", FileType::Regular, 0o644).unwrap();
        let data = vec![0xABu8; 65536];

        b.iter(|| {
            let written = fs.write(file_id, 0, black_box(&data)).unwrap();
            black_box(written)
        });
    });

    // 64KB read
    group.throughput(Throughput::Bytes(65536));
    group.bench_function("read_64kb", |b| {
        let mut fs = RamFs::new();
        fs.mount().unwrap();
        let root = fs.root().unwrap();
        let file_id = fs.create(root, "large.bin", FileType::Regular, 0o644).unwrap();
        let data = vec![0xABu8; 65536];
        fs.write(file_id, 0, &data).unwrap();

        let mut buf = vec![0u8; 65536];
        b.iter(|| {
            let read = fs.read(file_id, 0, black_box(&mut buf)).unwrap();
            black_box(read)
        });
    });

    group.finish();
}

fn bench_ramfs_directory(c: &mut Criterion) {
    let mut group = c.benchmark_group("ramfs_directory");

    group.bench_function("readdir_10", |b| {
        let mut fs = RamFs::new();
        fs.mount().unwrap();
        let root = fs.root().unwrap();

        for i in 0..10 {
            let name = format!("file_{}.txt", i);
            fs.create(root, &name, FileType::Regular, 0o644).unwrap();
        }

        b.iter(|| {
            let entries = fs.readdir(root, 0).unwrap();
            black_box(entries)
        });
    });

    group.bench_function("readdir_100", |b| {
        let mut fs = RamFs::new();
        fs.mount().unwrap();
        let root = fs.root().unwrap();

        for i in 0..100 {
            let name = format!("file_{}.txt", i);
            fs.create(root, &name, FileType::Regular, 0o644).unwrap();
        }

        b.iter(|| {
            let entries = fs.readdir(root, 0).unwrap();
            black_box(entries)
        });
    });

    group.bench_function("lookup_path_depth_5", |b| {
        let mut fs = RamFs::new();
        fs.mount().unwrap();
        let root = fs.root().unwrap();

        let mut current = root;
        for name in ["a", "b", "c", "d", "e"] {
            current = fs.create(current, name, FileType::Directory, 0o755).unwrap();
        }
        fs.create(current, "file.txt", FileType::Regular, 0o644).unwrap();

        b.iter(|| {
            let id = fs.lookup_path(Path::new(black_box("/a/b/c/d/e/file.txt"))).unwrap();
            black_box(id)
        });
    });

    group.finish();
}

fn bench_ramfs_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("ramfs_mixed");

    group.bench_function("create_write_read_delete", |b| {
        let mut fs = RamFs::new();
        fs.mount().unwrap();
        let root = fs.root().unwrap();
        let data = [0xABu8; 256];
        let mut buf = [0u8; 256];
        let mut counter = 0u64;

        b.iter(|| {
            counter += 1;
            let name = format!("temp_{}.txt", counter);

            // Create
            let file_id = fs.create(root, &name, FileType::Regular, 0o644).unwrap();

            // Write
            fs.write(file_id, 0, &data).unwrap();

            // Read
            fs.read(file_id, 0, &mut buf).unwrap();

            // Delete
            fs.unlink(root, &name).unwrap();

            black_box(buf)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_path_parsing,
    bench_ramfs_operations,
    bench_ramfs_io,
    bench_ramfs_directory,
    bench_ramfs_mixed_workload,
);

criterion_main!(benches);
