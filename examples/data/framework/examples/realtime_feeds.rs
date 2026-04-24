//! Real-Time News Feed Integration Example
//!
//! Demonstrates RSS/Atom feed parsing and aggregation from multiple sources.
//!
//! Usage:
//! ```bash
//! cargo run --example realtime_feeds
//! ```

use ruvector_data_framework::realtime::{FeedSource, NewsAggregator, RealTimeEngine};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🌐 RuVector Real-Time Feed Integration Demo\n");

    // Example 1: News Aggregator with default sources
    println!("📰 Example 1: Fetching from multiple news sources...");
    let mut aggregator = NewsAggregator::new();
    aggregator.add_default_sources();

    match aggregator.fetch_latest(20).await {
        Ok(vectors) => {
            println!("✅ Fetched {} articles", vectors.len());
            for (i, vector) in vectors.iter().take(5).enumerate() {
                println!(
                    "  {}. {} - {:?} ({})",
                    i + 1,
                    vector
                        .metadata
                        .get("title")
                        .map(|s| s.as_str())
                        .unwrap_or("Untitled"),
                    vector.domain,
                    vector.timestamp.format("%Y-%m-%d %H:%M")
                );
            }
        }
        Err(e) => {
            println!("⚠️  Error fetching news: {}", e);
        }
    }

    println!("\n📡 Example 2: Real-Time Engine with callbacks...");

    // Example 2: Real-time engine with callback
    let mut engine = RealTimeEngine::new(Duration::from_secs(60));

    // Add feed sources
    engine.add_feed(FeedSource::Rss {
        url: "https://earthobservatory.nasa.gov/feeds/image-of-the-day.rss".to_string(),
        category: "climate".to_string(),
    });

    engine.add_feed(FeedSource::Rss {
        url: "https://finance.yahoo.com/news/rssindex".to_string(),
        category: "finance".to_string(),
    });

    // Set callback for new data
    engine.set_callback(|vectors| {
        println!("🔔 Received {} new items:", vectors.len());
        for vector in vectors.iter().take(3) {
            println!(
                "   - {} ({:?})",
                vector
                    .metadata
                    .get("title")
                    .map(|s| s.as_str())
                    .unwrap_or("Untitled"),
                vector.domain
            );
        }
    });

    println!("   Starting real-time monitoring (Ctrl+C to stop)...");

    // Start the engine
    if let Err(e) = engine.start().await {
        eprintln!("❌ Failed to start engine: {}", e);
        return Ok(());
    }

    println!("   Engine running. Checking feeds every 60 seconds...");

    // Run for 3 minutes as demo
    tokio::time::sleep(Duration::from_secs(180)).await;

    // Stop the engine
    engine.stop().await;
    println!("   Engine stopped.");

    println!("\n📊 Example 3: Feed statistics...");
    println!("   Total sources configured: 5 (default)");
    println!("   Domains covered: Climate, Finance, Research, General News");
    println!("   Update interval: 60 seconds");
    println!("   Deduplication: ✅ Enabled");

    println!("\n✨ Demo complete!");
    println!("\nNext steps:");
    println!("  1. Integrate with DiscoveryEngine for pattern detection");
    println!("  2. Add custom RSS feeds with FeedSource::Rss");
    println!("  3. Implement REST polling with FeedSource::RestPolling");
    println!("  4. Connect to RuVector's HNSW index for semantic search");

    Ok(())
}
