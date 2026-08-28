Pod::Spec.new do |s|
  s.name         = 'RuVectorAppleML'
  s.version      = '2.3.0'
  s.summary      = 'Bounded Apple-native training and inference primitives for RuVector applications'
  s.description  = <<-DESC
    A reusable, semantics-neutral Apple runtime for bounded temporal models.
    It provides MPSGraph training, Accelerate inference, Core ML compute policy,
    and thermal-aware execution decisions. Applications retain model governance.
  DESC
  s.homepage     = 'https://github.com/ruvnet/ruvector'
  s.license      = { :type => 'MIT', :file => 'LICENSE' }
  s.author       = { 'RuVector' => 'opensource@ruv.net' }
  s.source       = { :git => 'https://github.com/ruvnet/ruvector.git', :tag => "v#{s.version}" }
  s.platform     = :ios, '16.0'
  s.swift_version = '5.9'
  s.static_framework = true
  s.source_files = 'apple/RuVectorAppleML/Sources/RuVectorAppleML/**/*.swift'
  s.frameworks   = 'Accelerate', 'CoreML', 'CryptoKit', 'Metal', 'MetalPerformanceShadersGraph'
  s.resource_bundles = {
    'RuVectorAppleMLPrivacy' => ['apple/RuVectorAppleML/Sources/RuVectorAppleML/PrivacyInfo.xcprivacy']
  }
end
