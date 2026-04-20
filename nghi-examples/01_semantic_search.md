# 01. Semantic Search: Building a Blog Search Engine

Traditional search engines match keywords. If you search for "canine", a keyword search won't find "dog" unless you explicitly tell it to.
**Semantic Search** solves this by converting text into vectors (embeddings) where similar meanings are close together.

## The Scenario
You have a blog with 1000 articles. You want users to be able to search for "how to stay healthy" and find articles about "nutrition", "exercise", and "sleep", even if they don't contain the exact words "stay healthy".

## The Implementation

### 1. Define the Data
```rust
struct BlogPost {
    id: u64,
    title: String,
    content: String,
    // The vector representation of the content
    embedding: Vec<f32>,
}
```

### 2. Generating Embeddings (Conceptual)
You would use an external model (like OpenAI's `text-embedding-3-small` or a local BERT model) to turn text into numbers.

```rust
fn get_embedding(text: &str) -> Vec<f32> {
    // Call API or local model
    // Returns e.g., [0.1, -0.5, 0.8, ...]
    vec![0.1, 0.2] // simplified
}
```

### 3. Inserting into Ruvector
```rust
use ruvector_core::{VectorStore, entry::Entry};

fn index_posts(store: &mut VectorStore, posts: Vec<BlogPost>) {
    for post in posts {
        let embedding = get_embedding(&post.content);
        
        let entry = Entry::new(post.id, embedding)
            .with_metadata("title", post.title)
            .with_metadata("url", format!("/blog/{}", post.id));
            
        store.insert(entry).unwrap();
    }
}
```

### 4. Searching
When a user types a query, we convert that query into a vector and search for the nearest neighbors.

```rust
fn search_blog(store: &VectorStore, query: &str) {
    let query_vec = get_embedding(query);
    
    // Find top 5 most similar posts
    let results = store.search(&query_vec, 5).unwrap();
    
    for result in results {
        println!("Found post ID: {} (Score: {})", result.id, result.score);
        // Fetch full post details from database using result.id
    }
}
```

## Why Ruvector?
*   **Speed**: It uses HNSW to find the nearest neighbors without checking every single post.
*   **Persistence**: It saves the index to disk so you don't have to rebuild it every time.

## Next
Let's see how this applies to images. Go to **[02_image_similarity.md](./02_image_similarity.md)**.
