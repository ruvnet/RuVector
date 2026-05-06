use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("gRPC transport: {0}")]
    Transport(#[from] tonic::transport::Error),
    #[error("gRPC status: {0}")]
    Status(#[from] tonic::Status),
    #[error("connect timeout for {node}")]
    ConnectTimeout { node: String },
    #[error("all nodes unreachable")]
    AllNodesDown,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
