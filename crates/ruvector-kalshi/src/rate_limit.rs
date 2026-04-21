//! Async token-bucket rate limiter for the Kalshi REST client.
//!
//! Kalshi enforces per-endpoint call limits. Hitting 429 in hot-path loops
//! kills strategies (signed-request retries pile up, timestamps drift).
//! The limiter is a simple leaky bucket refilled at a fixed rate.
//!
//! Not using a dedicated crate — this is 40 lines and avoids pulling in
//! a whole governor tree for one call site.

use std::time::Duration;

use tokio::sync::Mutex;
use tokio::time::Instant;

#[derive(Debug)]
pub struct RateLimiter {
    capacity: u32,
    refill_per_sec: f64,
    state: Mutex<State>,
}

#[derive(Debug)]
struct State {
    tokens: f64,
    last_refill: Instant,
}

impl RateLimiter {
    /// `capacity` tokens maximum, refilled at `refill_per_sec` tokens/sec.
    /// A fresh limiter starts full.
    pub fn new(capacity: u32, refill_per_sec: f64) -> Self {
        Self {
            capacity,
            refill_per_sec: refill_per_sec.max(0.0),
            state: Mutex::new(State {
                tokens: capacity as f64,
                last_refill: Instant::now(),
            }),
        }
    }

    /// Block the async task until one token is available, then consume it.
    pub async fn acquire(&self) {
        loop {
            let wait = {
                let mut st = self.state.lock().await;
                self.refill(&mut st);
                if st.tokens >= 1.0 {
                    st.tokens -= 1.0;
                    return;
                }
                // How long to sleep for the next token.
                if self.refill_per_sec <= 0.0 {
                    Duration::from_secs(3600) // effectively forever
                } else {
                    let need = 1.0 - st.tokens;
                    Duration::from_secs_f64(need / self.refill_per_sec)
                }
            };
            tokio::time::sleep(wait).await;
        }
    }

    /// Non-blocking try. Returns `true` if a token was consumed.
    pub async fn try_acquire(&self) -> bool {
        let mut st = self.state.lock().await;
        self.refill(&mut st);
        if st.tokens >= 1.0 {
            st.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn refill(&self, st: &mut State) {
        let now = Instant::now();
        let elapsed = now.duration_since(st.last_refill).as_secs_f64();
        st.tokens = (st.tokens + elapsed * self.refill_per_sec).min(self.capacity as f64);
        st.last_refill = now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn initial_bucket_is_full() {
        let rl = RateLimiter::new(3, 1.0);
        assert!(rl.try_acquire().await);
        assert!(rl.try_acquire().await);
        assert!(rl.try_acquire().await);
        assert!(!rl.try_acquire().await);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn tokens_refill_over_time() {
        let rl = RateLimiter::new(1, 10.0); // 1 token cap, 10/sec
        assert!(rl.try_acquire().await);
        assert!(!rl.try_acquire().await);
        // After 150ms at 10/s we have 1.5 tokens (capped at 1).
        tokio::time::advance(Duration::from_millis(150)).await;
        assert!(rl.try_acquire().await);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn acquire_waits_when_empty() {
        let rl = RateLimiter::new(1, 20.0); // 1 token cap, 20/sec
        rl.acquire().await; // consume initial
        let t0 = Instant::now();
        rl.acquire().await; // must wait ~50ms
        let dt = t0.elapsed();
        assert!(dt >= Duration::from_millis(40), "waited {dt:?}");
    }
}
