# 04. RAG Pipeline: Chatbot with Custom Knowledge

**RAG (Retrieval-Augmented Generation)** allows LLMs (like GPT-4) to answer questions about your private data.

## The Scenario
You are building a chatbot for your company's internal documentation.
The LLM doesn't know about your "Project X" because it wasn't trained on it.
You need to provide that information in the prompt.

## The Pipeline

1.  **Ingestion**: Chunk your documents and store them in `ruvector`.
2.  **Retrieval**: When a user asks a question, find relevant chunks.
3.  **Generation**: Send the question + relevant chunks to the LLM.

## The Implementation

### 1. Ingestion (Chunking)
```rust
struct Chunk {
    id: u64,
    text: String,
    vector: Vec<f32>,
}

fn ingest_docs(store: &mut VectorStore, docs: Vec<String>) {
    for (i, doc) in docs.iter().enumerate() {
        // Split doc into chunks of ~500 words
        let chunks = split_text(doc);
        
        for chunk_text in chunks {
            let vector = get_embedding(&chunk_text);
            let id = generate_unique_id();
            
            let entry = Entry::new(id, vector)
                .with_metadata("content", chunk_text); // Store text in metadata!
                
            store.insert(entry).unwrap();
        }
    }
}
```

### 2. Retrieval & Generation
```rust
async fn ask_chatbot(store: &VectorStore, question: &str) -> String {
    // 1. Embed the question
    let question_vec = get_embedding(question);
    
    // 2. Retrieve relevant context
    let results = store.search(&question_vec, 3).unwrap();
    
    let mut context_str = String::new();
    for result in results {
        // We stored the text in metadata, so we can retrieve it
        let text = result.metadata.get("content").unwrap();
        context_str.push_str(&format!("- {}\n", text));
    }
    
    // 3. Construct Prompt
    let prompt = format!(
        "Answer the question based on the context below:\n\nContext:\n{}\n\nQuestion: {}", 
        context_str, question
    );
    
    // 4. Call LLM
    let answer = call_openai_api(&prompt).await;
    answer
}
```

## Why Ruvector?
*   **Metadata Storage**: You can store the actual text chunk alongside the vector, avoiding a separate database lookup.
*   **Hybrid Search**: You can combine vector search with keyword filtering (e.g., "only search documents from 2024").

## Conclusion
You've seen 4 powerful ways to use `ruvector`.
From simple search to complex AI pipelines, the core concept remains the same: **Represent things as vectors, and find the nearest neighbors.**

**Go build something amazing!**
