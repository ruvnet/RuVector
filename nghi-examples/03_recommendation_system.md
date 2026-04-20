# 03. Recommendation System: Movie Recommendations

Recommendation systems are everywhere (Netflix, YouTube, Spotify).
One common technique is **Collaborative Filtering** using vector embeddings.

## The Scenario
You want to recommend movies to a user.
Assumption: We have trained a model (like Matrix Factorization or Two-Tower Neural Network) that maps both **Users** and **Movies** into the *same* vector space.

*   If a User vector is close to a Movie vector, the user is likely to enjoy that movie.

## The Implementation

### 1. The Vectors
*   **Movie Vector**: Represents the movie's genre, style, actors, etc.
*   **User Vector**: Represents the user's taste (based on watch history).

### 2. Indexing Movies
We index all movies into `ruvector`.

```rust
struct Movie {
    id: u64,
    title: String,
    vector: Vec<f32>,
}

fn index_movies(store: &mut VectorStore, movies: Vec<Movie>) {
    for movie in movies {
        let entry = Entry::new(movie.id, movie.vector)
            .with_metadata("title", movie.title);
        store.insert(entry).unwrap();
    }
}
```

### 3. Generating Recommendations
When a user visits the homepage, we look up their User Vector (computed offline or in real-time) and search for the nearest movies.

```rust
fn recommend_movies(store: &VectorStore, user_id: u64) {
    // 1. Fetch user vector from a Feature Store (e.g., Redis)
    let user_vector = fetch_user_vector(user_id);
    
    // 2. Search for nearest movies in Ruvector
    let recommendations = store.search(&user_vector, 10).unwrap();
    
    println!("Recommended for User {}:", user_id);
    for rec in recommendations {
        println!("- Movie ID: {} (Score: {})", rec.id, rec.score);
    }
}
```

### 4. "More Like This"
This is even simpler. If a user is watching "The Matrix", we just search for movies close to "The Matrix" vector.
```rust
fn more_like_this(store: &VectorStore, movie_id: u64) {
    // 1. Get the vector of the current movie
    let movie_vector = store.get_vector(movie_id).unwrap();
    
    // 2. Search
    let results = store.search(&movie_vector, 5).unwrap();
    // ...
}
```

## Why Ruvector?
*   **Low Latency**: Recommendations need to be served in milliseconds.
*   **Updates**: As new movies are added, they can be inserted immediately.

## Next
The hottest topic in AI right now: RAG. Go to **[04_rag_pipeline.md](./04_rag_pipeline.md)**.
