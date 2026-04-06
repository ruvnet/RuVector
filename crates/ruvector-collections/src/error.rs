//! Error types for collection management

use thiserror::Error;

/// Result type for collection operations
pub type Result<T> = std::result::Result<T, CollectionError>;

/// Errors that can occur during collection management
#[derive(Debug, Error)]
pub enum CollectionError {
    /// Collection was not found
    #[error("Collection not found: {name}")]
    CollectionNotFound {
        /// Name of the missing collection
        name: String,
    },

    /// Collection already exists
    #[error("Collection already exists: {name}")]
    CollectionAlreadyExists {
        /// Name of the existing collection
        name: String,
    },

    /// Alias was not found
    #[error("Alias not found: {alias}")]
    AliasNotFound {
        /// Name of the missing alias
        alias: String,
    },

    /// Alias already exists
    #[error("Alias already exists: {alias}")]
    AliasAlreadyExists {
        /// Name of the existing alias
        alias: String,
    },

    /// Invalid collection configuration
    #[error("Invalid configuration: {message}")]
    InvalidConfiguration {
        /// Error message
        message: String,
    },

    /// Alias points to non-existent collection
    #[error("Alias '{alias}' points to non-existent collection '{collection}'")]
    InvalidAlias {
        /// Alias name
        alias: String,
        /// Target collection name
        collection: String,
    },

    /// Cannot delete collection with active aliases
    #[error("Cannot delete collection '{collection}' because it has active aliases: {aliases:?}")]
    CollectionHasAliases {
        /// Collection name
        collection: String,
        /// List of aliases
        aliases: Vec<String>,
    },

    /// Invalid collection name
    #[error("Invalid collection name: {name} - {reason}")]
    InvalidName {
        /// Collection name
        name: String,
        /// Reason for invalidity
        reason: String,
    },

    /// Core database error
    #[error("Database error: {0}")]
    DatabaseError(#[from] ruvector_core::error::RuvectorError),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl From<serde_json::Error> for CollectionError {
    fn from(err: serde_json::Error) -> Self {
        CollectionError::SerializationError(err.to_string())
    }
}

impl From<bincode::error::EncodeError> for CollectionError {
    fn from(err: bincode::error::EncodeError) -> Self {
        CollectionError::SerializationError(err.to_string())
    }
}

impl From<bincode::error::DecodeError> for CollectionError {
    fn from(err: bincode::error::DecodeError) -> Self {
        CollectionError::SerializationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_not_found_display() {
        let err = CollectionError::CollectionNotFound {
            name: "missing".to_string(),
        };
        assert_eq!(err.to_string(), "Collection not found: missing");
    }

    #[test]
    fn test_collection_already_exists_display() {
        let err = CollectionError::CollectionAlreadyExists {
            name: "dup".to_string(),
        };
        assert_eq!(err.to_string(), "Collection already exists: dup");
    }

    #[test]
    fn test_alias_not_found_display() {
        let err = CollectionError::AliasNotFound {
            alias: "no_alias".to_string(),
        };
        assert_eq!(err.to_string(), "Alias not found: no_alias");
    }

    #[test]
    fn test_alias_already_exists_display() {
        let err = CollectionError::AliasAlreadyExists {
            alias: "dup_alias".to_string(),
        };
        assert_eq!(err.to_string(), "Alias already exists: dup_alias");
    }

    #[test]
    fn test_invalid_configuration_display() {
        let err = CollectionError::InvalidConfiguration {
            message: "bad param".to_string(),
        };
        assert_eq!(err.to_string(), "Invalid configuration: bad param");
    }

    #[test]
    fn test_invalid_alias_display() {
        let err = CollectionError::InvalidAlias {
            alias: "a".to_string(),
            collection: "c".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Alias 'a' points to non-existent collection 'c'"
        );
    }

    #[test]
    fn test_collection_has_aliases_display() {
        let err = CollectionError::CollectionHasAliases {
            collection: "main".to_string(),
            aliases: vec!["a1".to_string(), "a2".to_string()],
        };
        let msg = err.to_string();
        assert!(msg.contains("main"));
        assert!(msg.contains("a1"));
        assert!(msg.contains("a2"));
    }

    #[test]
    fn test_invalid_name_display() {
        let err = CollectionError::InvalidName {
            name: "bad!".to_string(),
            reason: "special chars".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Invalid collection name: bad! - special chars"
        );
    }

    #[test]
    fn test_serialization_error_display() {
        let err = CollectionError::SerializationError("parse fail".to_string());
        assert_eq!(err.to_string(), "Serialization error: parse fail");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err: CollectionError = io_err.into();
        match err {
            CollectionError::IoError(e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("Expected IoError, got: {:?}", other),
        }
    }

    #[test]
    fn test_serde_json_error_conversion() {
        let json_err = serde_json::from_str::<serde_json::Value>("{{bad json").unwrap_err();
        let err: CollectionError = json_err.into();
        match err {
            CollectionError::SerializationError(msg) => {
                assert!(!msg.is_empty());
            }
            other => panic!("Expected SerializationError, got: {:?}", other),
        }
    }

    #[test]
    fn test_error_is_debug() {
        let err = CollectionError::CollectionNotFound {
            name: "test".to_string(),
        };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("CollectionNotFound"));
    }
}
