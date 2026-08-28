import Foundation

public enum RuVectorEdgeMLError: Error, LocalizedError, Equatable, Sendable {
    case invalidConfiguration(String)
    case invalidInput(String)
    case modelShapeMismatch(String)
    case resourceUnavailable(String)
    case cancelled
    case timeLimitExceeded
    case numericalFailure

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message), .invalidInput(let message),
             .modelShapeMismatch(let message), .resourceUnavailable(let message):
            return message
        case .cancelled:
            return "The edge-ML operation was cancelled."
        case .timeLimitExceeded:
            return "The bounded edge-ML operation exceeded its time limit."
        case .numericalFailure:
            return "The edge-ML operation produced a non-finite value."
        }
    }
}
