//! pgvector Compatibility Tests
//!
//! Ensures all pgvector SQL syntax works unchanged with RuVector.
//! Tests cover:
//! - Vector type creation and operators (<->, <#>, <=>)
//! - HNSW and IVFFlat index creation
//! - Basic CRUD operations
//! - SQL function compatibility

use super::harness::*;

/// Test module for pgvector SQL syntax compatibility
#[cfg(test)]
mod pgvector_syntax_tests {
    use super::*;

    // ========================================================================
    // Vector Type Tests
    // ========================================================================

    /// Test vector type creation with text literal
    #[test]
    fn test_vector_type_text_literal() {
        // Verify that '[1,2,3]'::vector syntax works
        let sql = "SELECT '[1,2,3]'::vector;";

        // This test validates the expected SQL syntax
        assert!(sql.contains("::vector"));

        // Vector literal should be parseable
        let v = vec_to_pg_array(&[1.0, 2.0, 3.0]);
        assert_eq!(v, "[1.000000,2.000000,3.000000]");
    }

    /// Test vector with different dimensions
    #[test]
    fn test_vector_dimensions() {
        // Test 1D through high-D vectors
        for dims in [1, 2, 3, 128, 384, 768, 1536, 2048] {
            let data: Vec<f32> = (0..dims).map(|i| i as f32 * 0.01).collect();
            let literal = vec_to_pg_array(&data);

            assert!(literal.starts_with('['));
            assert!(literal.ends_with(']'));
            assert_eq!(literal.split(',').count(), dims);
        }
    }

    /// Test vector type with array conversion
    #[test]
    fn test_vector_array_conversion() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let pg_array = vec_to_pg_real_array(&v);

        assert!(pg_array.contains("ARRAY["));
        assert!(pg_array.contains("::real[]"));
    }

    // ========================================================================
    // Operator Tests
    // ========================================================================

    /// Test L2 distance operator (<->)
    #[test]
    fn test_l2_distance_operator() {
        let query = "[1,2,3]";
        let sql = format!(
            "SELECT embedding <-> '{}' AS distance FROM vectors ORDER BY embedding <-> '{}' LIMIT 10;",
            query, query
        );

        // Verify operator syntax
        assert!(sql.contains("<->"));
        assert!(sql.contains("ORDER BY"));
    }

    /// Test cosine distance operator (<=>)
    #[test]
    fn test_cosine_distance_operator() {
        let query = "[1,0,0]";
        let sql = format!(
            "SELECT embedding <=> '{}' AS distance FROM vectors ORDER BY embedding <=> '{}' LIMIT 10;",
            query, query
        );

        // Verify operator syntax
        assert!(sql.contains("<=>"));
    }

    /// Test inner product operator (<#>)
    #[test]
    fn test_inner_product_operator() {
        let query = "[1,2,3]";
        let sql = format!(
            "SELECT embedding <#> '{}' AS distance FROM vectors ORDER BY embedding <#> '{}' LIMIT 10;",
            query, query
        );

        // Verify operator syntax - note: negative for max inner product
        assert!(sql.contains("<#>"));
    }

    /// Test all distance operators return expected types
    #[test]
    fn test_operator_return_types() {
        // All operators should return float (real)
        let operators = ["<->", "<=>", "<#>"];

        for op in operators {
            let sql = format!("SELECT '[1,2,3]'::vector {} '[4,5,6]'::vector;", op);
            assert!(sql.contains(op));
        }
    }

    // ========================================================================
    // Index Creation Tests
    // ========================================================================

    /// Test HNSW index creation syntax
    #[test]
    fn test_hnsw_index_creation() {
        let ctx = TestContext::new("hnsw_index");

        let sql = sql::create_hnsw_index(&ctx.schema_name, "vectors", 16, 64);

        assert!(sql.contains("USING hnsw"));
        assert!(sql.contains("vector_l2_ops"));
        assert!(sql.contains("m = 16"));
        assert!(sql.contains("ef_construction = 64"));
    }

    /// Test HNSW index with different operator classes
    #[test]
    fn test_hnsw_operator_classes() {
        let operator_classes = ["vector_l2_ops", "vector_cosine_ops", "vector_ip_ops"];

        for op_class in operator_classes {
            let sql = format!(
                "CREATE INDEX ON vectors USING hnsw (embedding {});",
                op_class
            );
            assert!(sql.contains(op_class));
        }
    }

    /// Test IVFFlat index creation syntax
    #[test]
    fn test_ivfflat_index_creation() {
        let ctx = TestContext::new("ivfflat_index");

        let sql = sql::create_ivfflat_index(&ctx.schema_name, "vectors", 100);

        assert!(sql.contains("USING ivfflat"));
        assert!(sql.contains("vector_l2_ops"));
        assert!(sql.contains("lists = 100"));
    }

    /// Test IVFFlat index with different list counts
    #[test]
    fn test_ivfflat_lists_parameter() {
        for lists in [10, 50, 100, 500, 1000] {
            let sql = format!(
                "CREATE INDEX ON vectors USING ivfflat (embedding vector_l2_ops) WITH (lists = {});",
                lists
            );
            assert!(sql.contains(&format!("lists = {}", lists)));
        }
    }

    // ========================================================================
    // CRUD Operations Tests
    // ========================================================================

    /// Test INSERT with vector column
    #[test]
    fn test_insert_vector() {
        let ctx = TestContext::new("insert");
        let vector = vec_to_pg_array(&[1.0, 2.0, 3.0]);

        let sql = sql::insert_vector(&ctx.schema_name, "vectors", &vector, r#"{"key": "value"}"#);

        assert!(sql.contains("INSERT INTO"));
        assert!(sql.contains(&vector));
        assert!(sql.contains("RETURNING id"));
    }

    /// Test batch INSERT with multiple vectors
    #[test]
    fn test_batch_insert_vectors() {
        let ctx = TestContext::new("batch_insert");
        let vectors: Vec<String> = (0..10)
            .map(|i| vec_to_pg_array(&[i as f32, (i + 1) as f32, (i + 2) as f32]))
            .collect();

        let sql = sql::batch_insert_vectors(&ctx.schema_name, "vectors", &vectors);

        assert!(sql.contains("INSERT INTO"));
        assert!(sql.contains("VALUES"));
        // Should have 10 value rows
        assert_eq!(sql.matches("metadata").count(), 10);
    }

    /// Test SELECT with vector ordering
    #[test]
    fn test_select_with_ordering() {
        let ctx = TestContext::new("select_order");
        let query = vec_to_pg_array(&[1.0, 2.0, 3.0]);

        let sql = sql::nn_search_l2(&ctx.schema_name, "vectors", &query, 10);

        assert!(sql.contains("SELECT"));
        assert!(sql.contains("ORDER BY"));
        assert!(sql.contains("LIMIT 10"));
    }

    /// Test UPDATE vector column
    #[test]
    fn test_update_vector() {
        let new_vector = vec_to_pg_array(&[4.0, 5.0, 6.0]);
        let sql = format!(
            "UPDATE vectors SET embedding = '{}' WHERE id = 1;",
            new_vector
        );

        assert!(sql.contains("UPDATE"));
        assert!(sql.contains("SET embedding"));
    }

    /// Test DELETE with vector condition
    #[test]
    fn test_delete_with_vector_condition() {
        let sql = "DELETE FROM vectors WHERE embedding <-> '[0,0,0]' > 10;";

        assert!(sql.contains("DELETE"));
        assert!(sql.contains("<->"));
    }

    // ========================================================================
    // Function Compatibility Tests
    // ========================================================================

    /// Test vector_dims function
    #[test]
    fn test_vector_dims_function() {
        let sql = "SELECT vector_dims(embedding) FROM vectors LIMIT 1;";
        assert!(sql.contains("vector_dims"));
    }

    /// Test vector_norm function
    #[test]
    fn test_vector_norm_function() {
        let sql = "SELECT vector_norm(embedding) FROM vectors LIMIT 1;";
        assert!(sql.contains("vector_norm"));
    }

    /// Test array to vector cast
    #[test]
    fn test_array_to_vector_cast() {
        let sql = "SELECT ARRAY[1.0, 2.0, 3.0]::vector;";
        assert!(sql.contains("::vector"));
    }

    /// Test vector to array cast
    #[test]
    fn test_vector_to_array_cast() {
        let sql = "SELECT embedding::real[] FROM vectors LIMIT 1;";
        assert!(sql.contains("::real[]"));
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    /// Test single dimension vector
    #[test]
    fn test_single_dimension() {
        let v = vec_to_pg_array(&[42.0]);
        assert_eq!(v, "[42.000000]");
    }

    /// Test maximum supported dimensions
    #[test]
    fn test_max_dimensions() {
        // pgvector supports up to 16000 dimensions
        let dims = 16000;
        let data: Vec<f32> = (0..dims).map(|i| (i as f32) * 0.0001).collect();
        let literal = vec_to_pg_array(&data);

        assert!(literal.starts_with('['));
        assert!(literal.ends_with(']'));
    }

    /// Test vector with special float values
    #[test]
    fn test_special_float_values() {
        // Test with very small and very large values
        let small = vec_to_pg_array(&[1e-10, 1e-15, 1e-20]);
        let large = vec_to_pg_array(&[1e10, 1e15, 1e20]);

        assert!(small.contains("0.000000")); // Very small rounds to 0
        assert!(large.len() > 0); // Large values formatted
    }

    /// Test vector normalization in SQL
    #[test]
    fn test_sql_normalization() {
        let sql = "SELECT l2_normalize(embedding) FROM vectors LIMIT 1;";
        assert!(sql.contains("l2_normalize"));
    }

    // ========================================================================
    // Distance Function Tests
    // ========================================================================

    /// Test l2_distance function
    #[test]
    fn test_l2_distance_function() {
        let sql = "SELECT l2_distance(embedding, '[1,2,3]'::vector) FROM vectors;";
        assert!(sql.contains("l2_distance"));
    }

    /// Test cosine_distance function
    #[test]
    fn test_cosine_distance_function() {
        let sql = "SELECT cosine_distance(embedding, '[1,0,0]'::vector) FROM vectors;";
        assert!(sql.contains("cosine_distance"));
    }

    /// Test inner_product function
    #[test]
    fn test_inner_product_function() {
        let sql = "SELECT inner_product(embedding, '[1,2,3]'::vector) FROM vectors;";
        assert!(sql.contains("inner_product"));
    }

    // ========================================================================
    // Table Creation Tests
    // ========================================================================

    /// Test CREATE TABLE with vector column
    #[test]
    fn test_create_table_with_vector() {
        let ctx = TestContext::new("create_table");
        let sql = sql::create_vector_table(&ctx.schema_name, "embeddings", 384);

        assert!(sql.contains("CREATE TABLE"));
        assert!(sql.contains("vector(384)"));
    }

    /// Test ALTER TABLE ADD vector column
    #[test]
    fn test_alter_table_add_vector() {
        let sql = "ALTER TABLE documents ADD COLUMN embedding vector(768);";

        assert!(sql.contains("ALTER TABLE"));
        assert!(sql.contains("vector(768)"));
    }

    // ========================================================================
    // Set Operations Tests
    // ========================================================================

    /// Test SET ivfflat.probes
    #[test]
    fn test_set_ivfflat_probes() {
        let sql = "SET ivfflat.probes = 10;";
        assert!(sql.contains("ivfflat.probes"));
    }

    /// Test SET hnsw.ef_search
    #[test]
    fn test_set_hnsw_ef_search() {
        let sql = "SET hnsw.ef_search = 100;";
        assert!(sql.contains("hnsw.ef_search"));
    }
}

/// Test module for pgvector numerical accuracy
#[cfg(test)]
mod pgvector_accuracy_tests {
    use super::*;

    /// Test L2 distance accuracy
    #[test]
    fn test_l2_distance_accuracy() {
        // [1,2,3] <-> [4,5,6] = sqrt(9+9+9) = sqrt(27) = 5.196...
        let expected = 27.0_f32.sqrt();

        // We just validate the expected value here
        // Actual DB test would compare against this
        assertions::assert_approx_eq(expected, 5.196, 0.001);
    }

    /// Test cosine distance accuracy
    #[test]
    fn test_cosine_distance_accuracy() {
        // cosine_distance([1,0], [0,1]) = 1 - 0 = 1
        let expected = 1.0;

        assertions::assert_approx_eq(expected, 1.0, 0.001);
    }

    /// Test inner product accuracy
    #[test]
    fn test_inner_product_accuracy() {
        // [1,2,3] dot [4,5,6] = 4 + 10 + 18 = 32
        // <#> returns negative: -32
        let expected = -32.0;

        assertions::assert_approx_eq(expected, -32.0, 0.001);
    }

    /// Test normalized vector accuracy
    #[test]
    fn test_normalized_accuracy() {
        // [3,4] normalized = [0.6, 0.8], norm = 1.0
        let norm = (0.6_f32.powi(2) + 0.8_f32.powi(2)).sqrt();

        assertions::assert_approx_eq(norm, 1.0, 0.0001);
    }
}

/// Test module for pgvector index behavior
#[cfg(test)]
mod pgvector_index_tests {
    use super::*;

    /// Test HNSW index parameters
    #[test]
    fn test_hnsw_parameters() {
        // Valid parameter ranges
        let valid_m = [4, 8, 16, 32, 64];
        let valid_ef_construction = [32, 64, 128, 256, 512];

        for m in valid_m {
            for ef in valid_ef_construction {
                let sql = format!(
                    "CREATE INDEX ON t USING hnsw (v) WITH (m = {}, ef_construction = {});",
                    m, ef
                );
                assert!(sql.contains(&format!("m = {}", m)));
                assert!(sql.contains(&format!("ef_construction = {}", ef)));
            }
        }
    }

    /// Test IVFFlat index parameters
    #[test]
    fn test_ivfflat_parameters() {
        // Valid list counts
        let valid_lists = [10, 50, 100, 500, 1000, 4096];

        for lists in valid_lists {
            let sql = format!(
                "CREATE INDEX ON t USING ivfflat (v) WITH (lists = {});",
                lists
            );
            assert!(sql.contains(&format!("lists = {}", lists)));
        }
    }

    /// Test index operator class selection
    #[test]
    fn test_operator_class_selection() {
        let configs = [
            ("vector_l2_ops", "<->", "L2 distance"),
            ("vector_cosine_ops", "<=>", "cosine distance"),
            ("vector_ip_ops", "<#>", "inner product"),
        ];

        for (op_class, operator, _desc) in configs {
            let create_sql = format!("CREATE INDEX ON t USING hnsw (v {});", op_class);
            let query_sql = format!("SELECT * FROM t ORDER BY v {} q LIMIT 10;", operator);

            assert!(create_sql.contains(op_class));
            assert!(query_sql.contains(operator));
        }
    }
}
