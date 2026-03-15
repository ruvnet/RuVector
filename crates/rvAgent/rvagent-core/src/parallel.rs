//! Parallel execution utilities (ADR-103 A2).
//!
//! When an LLM response contains multiple tool calls these utilities execute
//! them concurrently while preserving the original input ordering in the
//! output vector.

use std::future::Future;
use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::task::JoinSet;

/// Execute multiple async operations concurrently, collecting results.
///
/// The output vector preserves the ordering of the input `items` — i.e.
/// `result[i]` corresponds to `items[i]` — regardless of the order in which
/// the futures complete.
///
/// # Panics
///
/// Panics if any spawned task panics.
pub async fn parallel_execute<T, F, Fut>(items: Vec<T>, f: F) -> Vec<Fut::Output>
where
    T: Send + 'static,
    F: Fn(T) -> Fut + Send + Sync + 'static,
    Fut: Future + Send + 'static,
    Fut::Output: Send + 'static,
{
    let len = items.len();
    if len == 0 {
        return Vec::new();
    }

    let f = Arc::new(f);
    let mut set = JoinSet::new();

    for (idx, item) in items.into_iter().enumerate() {
        let f = Arc::clone(&f);
        set.spawn(async move {
            let result = f(item).await;
            (idx, result)
        });
    }

    let mut indexed: Vec<(usize, Fut::Output)> = Vec::with_capacity(len);
    while let Some(res) = set.join_next().await {
        indexed.push(res.expect("spawned task panicked"));
    }

    indexed.sort_by_key(|(idx, _)| *idx);
    indexed.into_iter().map(|(_, v)| v).collect()
}

/// Execute multiple async operations with a concurrency limit.
///
/// At most `max_concurrent` operations run at any given time. Output ordering
/// matches the input ordering, same as [`parallel_execute`].
///
/// # Panics
///
/// Panics if any spawned task panics or if `max_concurrent` is 0.
pub async fn parallel_execute_limited<T, F, Fut>(
    items: Vec<T>,
    f: F,
    max_concurrent: usize,
) -> Vec<Fut::Output>
where
    T: Send + 'static,
    F: Fn(T) -> Fut + Send + Sync + 'static,
    Fut: Future + Send + 'static,
    Fut::Output: Send + 'static,
{
    assert!(max_concurrent > 0, "max_concurrent must be > 0");

    let len = items.len();
    if len == 0 {
        return Vec::new();
    }

    let f = Arc::new(f);
    let sem = Arc::new(Semaphore::new(max_concurrent));
    let mut set = JoinSet::new();

    for (idx, item) in items.into_iter().enumerate() {
        let f = Arc::clone(&f);
        let sem = Arc::clone(&sem);
        set.spawn(async move {
            let _permit = sem
                .acquire()
                .await
                .expect("semaphore closed unexpectedly");
            let result = f(item).await;
            (idx, result)
        });
    }

    let mut indexed: Vec<(usize, Fut::Output)> = Vec::with_capacity(len);
    while let Some(res) = set.join_next().await {
        indexed.push(res.expect("spawned task panicked"));
    }

    indexed.sort_by_key(|(idx, _)| *idx);
    indexed.into_iter().map(|(_, v)| v).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_empty() {
        let result: Vec<i32> =
            parallel_execute(Vec::<i32>::new(), |x| async move { x * 2 }).await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_single() {
        let result = parallel_execute(vec![5], |x| async move { x + 1 }).await;
        assert_eq!(result, vec![6]);
    }
}
