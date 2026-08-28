import CoreML
import XCTest
@testable import RuVectorAppleML

final class CoreMLModelSessionTests: XCTestCase {
    func testBoundedInputValidationRejectsUnsupportedContainerTypes() throws {
        let dictionary = try MLFeatureValue(dictionary: [
            NSNumber(value: 1): NSNumber(value: 2),
        ])
        let sequence = MLFeatureValue(sequence: MLSequence(int64s: [1, 2]))

        XCTAssertThrowsError(try CoreMLModelSession.validateInputFeatures(["dictionary": dictionary]))
        XCTAssertThrowsError(try CoreMLModelSession.validateInputFeatures(["sequence": sequence]))
    }

    func testBoundedInputValidationAcceptsSupportedScalars() {
        XCTAssertNoThrow(try CoreMLModelSession.validateInputFeatures([
            "count": MLFeatureValue(int64: 7),
            "score": MLFeatureValue(double: 0.75),
            "label": MLFeatureValue(string: "present"),
        ]))
    }

    func testBoundedInputValidationRejectsNonFiniteScalar() {
        XCTAssertThrowsError(try CoreMLModelSession.validateInputFeatures([
            "score": MLFeatureValue(double: .nan),
        ]))
    }

    func testOutputNamesAreSortedAndOversizedSetsAreRejected() throws {
        XCTAssertEqual(
            try CoreMLModelSession.validatedOutputNames(Set(["z", "a", "m"])),
            ["a", "m", "z"]
        )
        let tooMany = Set((0...256).map { "output-\($0)" })
        XCTAssertThrowsError(try CoreMLModelSession.validatedOutputNames(tooMany))
    }
}
