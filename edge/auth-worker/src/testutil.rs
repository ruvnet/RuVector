//! Test doubles shared by the unit tests (cfg(test) only).

use ruvector_edge_authz::{Clock, Rng, StoreError};
use std::cell::Cell;
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

struct NoopWake;

impl Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}

/// Drive a future whose ports are all immediately ready.
pub(crate) fn block_on<F: Future>(fut: F) -> F::Output {
    let waker = Waker::from(Arc::new(NoopWake));
    let mut cx = Context::from_waker(&waker);
    let mut fut = pin!(fut);
    for _ in 0..10_000 {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return v;
        }
    }
    panic!("future did not complete");
}

/// Settable clock.
pub(crate) struct FixedClock(pub Cell<u64>);

impl FixedClock {
    pub(crate) fn at(t: u64) -> Self {
        FixedClock(Cell::new(t))
    }
    pub(crate) fn advance(&self, secs: u64) {
        self.0.set(self.0.get() + secs);
    }
}

impl Clock for FixedClock {
    fn now_unix(&self) -> u64 {
        self.0.get()
    }
}

/// Deterministic, never-repeating byte stream (test only).
#[derive(Default)]
pub(crate) struct SeqRng(Cell<u64>);

impl Rng for SeqRng {
    fn fill(&self, buf: &mut [u8]) -> Result<(), StoreError> {
        for b in buf.iter_mut() {
            let n = self.0.get().wrapping_add(1);
            self.0.set(n);
            *b = (n.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as u8;
        }
        Ok(())
    }
}
