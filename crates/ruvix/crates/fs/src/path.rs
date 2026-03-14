//! Path parsing utilities for the RuVix filesystem.
//!
//! This module provides path manipulation utilities compatible with `no_std`
//! environments. Paths are always forward-slash separated and support
//! both absolute (starting with '/') and relative paths.

use crate::error::{FsError, FsResult};
use crate::{MAX_NAME_LEN, MAX_PATH_LEN};
use core::fmt;

/// A borrowed path slice, similar to `std::path::Path`.
#[derive(Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Path {
    inner: str,
}

impl Path {
    /// Create a new `Path` from a string slice.
    ///
    /// # Safety
    ///
    /// The path must be valid UTF-8.
    #[must_use]
    pub fn new<S: AsRef<str> + ?Sized>(s: &S) -> &Self {
        // SAFETY: Path is a transparent wrapper around str
        unsafe { &*(s.as_ref() as *const str as *const Self) }
    }

    /// Returns the path as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.inner
    }

    /// Returns `true` if the path is absolute (starts with '/').
    #[must_use]
    pub fn is_absolute(&self) -> bool {
        self.inner.starts_with('/')
    }

    /// Returns `true` if the path is relative (doesn't start with '/').
    #[must_use]
    pub fn is_relative(&self) -> bool {
        !self.is_absolute()
    }

    /// Returns `true` if the path is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the length of the path in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns the parent directory of this path, if any.
    #[must_use]
    pub fn parent(&self) -> Option<&Path> {
        let s = self.inner.trim_end_matches('/');
        if s.is_empty() {
            return None;
        }

        match s.rfind('/') {
            Some(0) => Some(Path::new("/")),
            Some(idx) => Some(Path::new(&s[..idx])),
            None => {
                if self.is_absolute() {
                    Some(Path::new("/"))
                } else {
                    None
                }
            }
        }
    }

    /// Returns the final component of the path, if any.
    #[must_use]
    pub fn file_name(&self) -> Option<&str> {
        let s = self.inner.trim_end_matches('/');
        if s.is_empty() {
            return None;
        }

        match s.rfind('/') {
            Some(idx) => Some(&s[idx + 1..]),
            None => Some(s),
        }
    }

    /// Returns an iterator over the components of the path.
    #[must_use]
    pub fn components(&self) -> PathIter<'_> {
        PathIter::new(&self.inner)
    }

    /// Joins this path with another path component.
    ///
    /// If `other` is absolute, it replaces `self`.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn join<P: AsRef<Path>>(&self, other: P) -> PathBuf {
        let other = other.as_ref();
        if other.is_absolute() {
            return PathBuf::from(other.as_str());
        }

        let mut buf = PathBuf::from(self.as_str());
        buf.push(other);
        buf
    }

    /// Returns `true` if the path starts with the given prefix.
    #[must_use]
    pub fn starts_with<P: AsRef<Path>>(&self, prefix: P) -> bool {
        let prefix = prefix.as_ref();
        self.inner.starts_with(prefix.as_str())
    }

    /// Returns `true` if the path ends with the given suffix.
    #[must_use]
    pub fn ends_with<P: AsRef<Path>>(&self, suffix: P) -> bool {
        let suffix = suffix.as_ref();
        self.inner.ends_with(suffix.as_str())
    }

    /// Strips the prefix from the path.
    #[must_use]
    pub fn strip_prefix<P: AsRef<Path>>(&self, prefix: P) -> Option<&Path> {
        let prefix = prefix.as_ref();
        self.inner.strip_prefix(prefix.as_str()).map(Path::new)
    }

    /// Validates the path for filesystem use.
    pub fn validate(&self) -> FsResult<()> {
        if self.len() > MAX_PATH_LEN {
            return Err(FsError::PathTooLong);
        }

        for component in self.components() {
            match component {
                PathComponent::Normal(name) => {
                    if name.len() > MAX_NAME_LEN {
                        return Err(FsError::NameTooLong);
                    }
                    if name.contains('\0') {
                        return Err(FsError::InvalidFilename);
                    }
                }
                PathComponent::RootDir | PathComponent::CurDir | PathComponent::ParentDir => {}
            }
        }

        Ok(())
    }
}

impl AsRef<Path> for Path {
    fn as_ref(&self) -> &Path {
        self
    }
}

impl AsRef<Path> for str {
    fn as_ref(&self) -> &Path {
        Path::new(self)
    }
}

impl AsRef<str> for Path {
    fn as_ref(&self) -> &str {
        &self.inner
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.inner)
    }
}

/// An owned path buffer, similar to `std::path::PathBuf`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct PathBuf {
    inner: alloc::string::String,
}

#[cfg(feature = "alloc")]
impl PathBuf {
    /// Create a new empty `PathBuf`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: alloc::string::String::new(),
        }
    }

    /// Create a `PathBuf` from a string.
    #[must_use]
    pub fn from<S: Into<alloc::string::String>>(s: S) -> Self {
        Self { inner: s.into() }
    }

    /// Returns a `Path` reference to this buffer.
    #[must_use]
    pub fn as_path(&self) -> &Path {
        Path::new(&self.inner)
    }

    /// Push a path component onto this path.
    pub fn push<P: AsRef<Path>>(&mut self, path: P) {
        let path = path.as_ref();
        if path.is_absolute() {
            self.inner.clear();
            self.inner.push_str(path.as_str());
            return;
        }

        if !self.inner.is_empty() && !self.inner.ends_with('/') {
            self.inner.push('/');
        }
        self.inner.push_str(path.as_str());
    }

    /// Pop the last component from this path.
    pub fn pop(&mut self) -> bool {
        match self.as_path().parent() {
            Some(parent) => {
                self.inner.truncate(parent.len());
                true
            }
            None => false,
        }
    }

    /// Set the file name of this path.
    pub fn set_file_name<S: AsRef<str>>(&mut self, name: S) {
        if self.as_path().file_name().is_some() {
            let _ = self.pop();
        }
        self.push(Path::new(name.as_ref()));
    }

    /// Clear the path buffer.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Returns the capacity of the underlying buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Reserves capacity for at least `additional` more bytes.
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Returns the path as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.inner
    }

    /// Consumes the `PathBuf`, returning the underlying `String`.
    #[must_use]
    pub fn into_string(self) -> alloc::string::String {
        self.inner
    }
}

#[cfg(feature = "alloc")]
impl AsRef<Path> for PathBuf {
    fn as_ref(&self) -> &Path {
        self.as_path()
    }
}

#[cfg(feature = "alloc")]
impl AsRef<str> for PathBuf {
    fn as_ref(&self) -> &str {
        &self.inner
    }
}

#[cfg(feature = "alloc")]
impl core::ops::Deref for PathBuf {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        self.as_path()
    }
}

#[cfg(feature = "alloc")]
impl fmt::Display for PathBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.inner)
    }
}

/// A path component returned by `PathIter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathComponent<'a> {
    /// The root directory `/`.
    RootDir,
    /// The current directory `.`.
    CurDir,
    /// The parent directory `..`.
    ParentDir,
    /// A normal component (file or directory name).
    Normal(&'a str),
}

impl<'a> PathComponent<'a> {
    /// Returns the component as a string slice.
    #[must_use]
    pub const fn as_str(&self) -> &'a str {
        match self {
            Self::RootDir => "/",
            Self::CurDir => ".",
            Self::ParentDir => "..",
            Self::Normal(s) => s,
        }
    }
}

impl<'a> fmt::Display for PathComponent<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Iterator over path components.
#[derive(Debug, Clone)]
pub struct PathIter<'a> {
    path: &'a str,
    pos: usize,
    front_done: bool,
}

impl<'a> PathIter<'a> {
    /// Create a new path iterator.
    #[must_use]
    pub const fn new(path: &'a str) -> Self {
        Self {
            path,
            pos: 0,
            front_done: false,
        }
    }
}

impl<'a> Iterator for PathIter<'a> {
    type Item = PathComponent<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // Handle root directory
        if !self.front_done && self.path.starts_with('/') {
            self.front_done = true;
            self.pos = 1;
            return Some(PathComponent::RootDir);
        }
        self.front_done = true;

        // Skip leading slashes
        while self.pos < self.path.len() && self.path.as_bytes()[self.pos] == b'/' {
            self.pos += 1;
        }

        if self.pos >= self.path.len() {
            return None;
        }

        // Find the end of this component
        let start = self.pos;
        while self.pos < self.path.len() && self.path.as_bytes()[self.pos] != b'/' {
            self.pos += 1;
        }

        let component = &self.path[start..self.pos];
        match component {
            "." => Some(PathComponent::CurDir),
            ".." => Some(PathComponent::ParentDir),
            _ => Some(PathComponent::Normal(component)),
        }
    }
}

/// Normalize a path by removing `.` and resolving `..`.
#[cfg(feature = "alloc")]
#[must_use]
pub fn normalize_path(path: &Path) -> PathBuf {
    let mut result = alloc::vec::Vec::new();
    let is_absolute = path.is_absolute();

    for component in path.components() {
        match component {
            PathComponent::RootDir => {}
            PathComponent::CurDir => {}
            PathComponent::ParentDir => {
                if !result.is_empty() && result.last() != Some(&"..") {
                    result.pop();
                } else if !is_absolute {
                    result.push("..");
                }
            }
            PathComponent::Normal(name) => {
                result.push(name);
            }
        }
    }

    let mut path_buf = PathBuf::new();
    if is_absolute {
        path_buf.push(Path::new("/"));
    }
    for (i, component) in result.iter().enumerate() {
        if i > 0 || !is_absolute {
            if i > 0 {
                path_buf.inner.push('/');
            }
            path_buf.inner.push_str(component);
        } else {
            path_buf.inner.push_str(component);
        }
    }

    if path_buf.is_empty() && is_absolute {
        path_buf.inner.push('/');
    }

    path_buf
}

/// Split a path into parent and filename.
#[must_use]
pub fn split_path(path: &Path) -> (Option<&Path>, Option<&str>) {
    (path.parent(), path.file_name())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_is_absolute() {
        assert!(Path::new("/").is_absolute());
        assert!(Path::new("/foo").is_absolute());
        assert!(Path::new("/foo/bar").is_absolute());
        assert!(!Path::new("foo").is_absolute());
        assert!(!Path::new("foo/bar").is_absolute());
        assert!(!Path::new("").is_absolute());
    }

    #[test]
    fn test_path_parent() {
        assert_eq!(Path::new("/foo/bar").parent(), Some(Path::new("/foo")));
        assert_eq!(Path::new("/foo").parent(), Some(Path::new("/")));
        assert_eq!(Path::new("/").parent(), None);
        assert_eq!(Path::new("foo/bar").parent(), Some(Path::new("foo")));
        assert_eq!(Path::new("foo").parent(), None);
    }

    #[test]
    fn test_path_file_name() {
        assert_eq!(Path::new("/foo/bar").file_name(), Some("bar"));
        assert_eq!(Path::new("/foo").file_name(), Some("foo"));
        assert_eq!(Path::new("/").file_name(), None);
        assert_eq!(Path::new("foo").file_name(), Some("foo"));
        assert_eq!(Path::new("").file_name(), None);
    }

    #[test]
    fn test_path_components() {
        let components: alloc::vec::Vec<_> = Path::new("/foo/bar/baz").components().collect();
        assert_eq!(
            components,
            alloc::vec![
                PathComponent::RootDir,
                PathComponent::Normal("foo"),
                PathComponent::Normal("bar"),
                PathComponent::Normal("baz"),
            ]
        );

        let components: alloc::vec::Vec<_> = Path::new("foo/./bar/../baz").components().collect();
        assert_eq!(
            components,
            alloc::vec![
                PathComponent::Normal("foo"),
                PathComponent::CurDir,
                PathComponent::Normal("bar"),
                PathComponent::ParentDir,
                PathComponent::Normal("baz"),
            ]
        );
    }

    #[test]
    fn test_path_validate() {
        assert!(Path::new("/foo/bar").validate().is_ok());
        assert!(Path::new("foo/bar").validate().is_ok());

        // Create a path that's too long
        let long_path = "a".repeat(MAX_PATH_LEN + 1);
        assert_eq!(Path::new(&long_path).validate(), Err(FsError::PathTooLong));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_pathbuf_push() {
        let mut path = PathBuf::from("/foo");
        path.push(Path::new("bar"));
        assert_eq!(path.as_str(), "/foo/bar");

        path.push(Path::new("baz"));
        assert_eq!(path.as_str(), "/foo/bar/baz");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_pathbuf_push_absolute() {
        let mut path = PathBuf::from("/foo/bar");
        path.push(Path::new("/baz"));
        assert_eq!(path.as_str(), "/baz");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_pathbuf_pop() {
        let mut path = PathBuf::from("/foo/bar/baz");
        assert!(path.pop());
        assert_eq!(path.as_str(), "/foo/bar");
        assert!(path.pop());
        assert_eq!(path.as_str(), "/foo");
        assert!(path.pop());
        assert_eq!(path.as_str(), "/");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_normalize_path() {
        assert_eq!(normalize_path(Path::new("/foo/./bar")).as_str(), "/foo/bar");
        assert_eq!(normalize_path(Path::new("/foo/../bar")).as_str(), "/bar");
        assert_eq!(normalize_path(Path::new("/foo/bar/..")).as_str(), "/foo");
        assert_eq!(normalize_path(Path::new("foo/./bar")).as_str(), "foo/bar");
        assert_eq!(normalize_path(Path::new("/../foo")).as_str(), "/foo");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_path_join() {
        let path = Path::new("/foo");
        let joined = path.join(Path::new("bar"));
        assert_eq!(joined.as_str(), "/foo/bar");

        let joined = path.join(Path::new("/bar"));
        assert_eq!(joined.as_str(), "/bar");
    }

    #[test]
    fn test_split_path() {
        let (parent, name) = split_path(Path::new("/foo/bar"));
        assert_eq!(parent, Some(Path::new("/foo")));
        assert_eq!(name, Some("bar"));

        let (parent, name) = split_path(Path::new("/"));
        assert!(parent.is_none());
        assert!(name.is_none());
    }
}
