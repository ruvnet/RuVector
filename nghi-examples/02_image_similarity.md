# 02. Image Similarity: E-commerce Recommendation

"Visual Search" is powerful. If a user is looking at a red dress, they might want to see other red dresses, regardless of the brand or description.

## The Scenario
You run an online store. When a user views a product, you want to show a "Visually Similar Items" section.

## The Implementation

### 1. The Feature Vector
Instead of text, we use a Convolutional Neural Network (CNN) like ResNet or CLIP to extract features from images.
Two images that look similar will have vectors that are close in distance.

### 2. Metadata Filtering
Ruvector allows attaching metadata. This is crucial for e-commerce. You might want to find similar images *but only* within the "Shoes" category.

```rust
use ruvector_core::{VectorStore, entry::Entry};

fn index_product(store: &mut VectorStore, product_id: u64, image_path: &str, category: &str) {
    let image_vec = extract_image_features(image_path);
    
    let entry = Entry::new(product_id, image_vec)
        .with_metadata("category", category)
        .with_metadata("price", 99.99); // Example
        
    store.insert(entry).unwrap();
}
```

### 3. Filtered Search
```rust
fn find_similar_shoes(store: &VectorStore, current_shoe_vec: &[f32]) {
    // Search for nearest neighbors
    // BUT filter so we only get items where category == "shoes"
    let filter = Filter::new("category", "shoes");
    
    let results = store.search_with_filter(current_shoe_vec, 10, filter).unwrap();
    
    for result in results {
        println!("Recommended Product ID: {}", result.id);
    }
}
```

## Why Ruvector?
*   **Filtering**: Efficiently combines vector search with metadata filtering (pre-filtering or post-filtering).
*   **Scalability**: Can handle millions of product images.

## Next
What if we want to recommend things based on user behavior, not just similarity? Go to **[03_recommendation_system.md](./03_recommendation_system.md)**.
