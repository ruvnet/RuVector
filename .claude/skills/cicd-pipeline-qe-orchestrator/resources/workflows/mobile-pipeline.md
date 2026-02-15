# Mobile App CI/CD Pipeline Workflow

Complete quality orchestration workflow for mobile applications (iOS and Android).

## Pipeline Configuration

**Service Type**: Native Mobile Application
**Tech Stack**: React Native (iOS + Android)
**Deployment**: App Store, Google Play
**Frequency**: Bi-weekly releases

---

## Phase 1: Commit (Pre-Merge)

### Quality Strategy
- Unit tests for business logic
- Component snapshot tests
- Fast feedback (<5 minutes)

### Agent Orchestration

```javascript
// Fast mobile-specific validation
Task("Unit Tests", "Test business logic and utilities", "qe-test-generator")
Task("Snapshot Tests", "Validate component snapshots", "qe-test-executor")
Task("Code Review", "Mobile-specific code review", "code-review-swarm")
```

### Quality Gates
- [ ] Unit tests passing
- [ ] No snapshot failures
- [ ] No critical code review issues
- [ ] Execution time < 5 minutes

---

## Phase 2: Build (iOS + Android)

### Quality Strategy
- Platform-specific build validation
- Integration tests on simulators/emulators
- Test data generation for offline testing

### Agent Orchestration

```javascript
// Parallel builds and tests
Task("iOS Build", "Build and test on iOS simulator", "qe-test-executor")
Task("Android Build", "Build and test on Android emulator", "qe-test-executor")
Task("Integration Tests", "Run E2E tests on both platforms", "qe-test-executor")
Task("Test Data", "Generate offline test data", "qe-test-data-architect")
```

### Quality Gates
- [ ] iOS build successful
- [ ] Android build successful
- [ ] Integration tests passing (both platforms)
- [ ] No platform-specific regressions

---

## Phase 3: Device Testing

### Quality Strategy
- Real device testing (multiple models)
- Performance on low-end devices
- Compatibility across OS versions
- Visual regression testing

### Agent Orchestration

```javascript
// Device farm testing
Task("Device Testing", "Test on 20+ real devices (BrowserStack/Firebase)", "qe-test-executor")
Task("Performance Test", "Test on low-end devices", "qe-performance-tester")
Task("Visual Testing", "Visual regression across devices", "qe-visual-tester")
Task("Compatibility", "Test iOS 14-17, Android 10-14", "qe-test-executor")
```

### Skills for Mobile Testing
```javascript
// Invoke mobile testing skill
Skill("mobile-testing")  // Gestures, sensors, permissions, fragmentation
```

### Quality Gates
- [ ] All device tests passing
- [ ] Performance acceptable on low-end devices (60fps)
- [ ] No visual regressions
- [ ] Tested on 90% of user devices (analytics-based)

---

## Phase 4: Beta Testing

### Quality Strategy
- Internal beta (TestFlight/Internal Testing)
- Crash analytics monitoring
- Accessibility validation
- Localization testing

### Agent Orchestration

```javascript
// Beta validation
Task("Crash Analysis", "Monitor crash reports from beta", "qe-production-intelligence")
Task("Accessibility", "WCAG compliance for mobile", "qe-visual-tester")
Task("Localization", "Test 10+ languages", "qe-test-executor")
```

### Skills for Beta Testing
```javascript
Skill("accessibility-testing")   // Screen readers, voice control
Skill("localization-testing")    // i18n, RTL languages
```

### Quality Gates
- [ ] Crash-free rate > 99.5%
- [ ] Accessibility score > 90%
- [ ] All target languages validated
- [ ] Beta user feedback positive

---

## Phase 5: Production Release

### Quality Strategy
- Staged rollout (1% → 10% → 50% → 100%)
- App store optimization (screenshots, metadata)
- Production monitoring
- Review response monitoring

### Agent Orchestration

```javascript
// Production monitoring
Task("Rollout Monitoring", "Monitor staged rollout metrics", "qe-production-intelligence")
Task("Quality Metrics", "Track production quality", "qe-quality-analyzer")
```

### Quality Gates (Per Stage)
- [ ] Crash-free rate maintained
- [ ] ANR rate < 0.1% (Android)
- [ ] App Store rating > 4.5 stars
- [ ] No increase in 1-star reviews

---

## Skill Integration

### Skills Used by Phase

**Commit Phase**:
- `shift-left-testing`
- `tdd-london-chicago`

**Build Phase**:
- `test-automation-strategy`
- `test-data-management`

**Device Testing Phase**:
- `mobile-testing` ⭐ (primary)
- `compatibility-testing`
- `visual-testing-advanced`
- `performance-testing`

**Beta Testing Phase**:
- `accessibility-testing`
- `localization-testing`
- `exploratory-testing-advanced`

**Production Phase**:
- `shift-right-testing`
- `quality-metrics`

---

## Complete Pipeline Code

```javascript
// Full mobile pipeline orchestration

// PHASE 1: Commit
Task("Unit Tests", "Business logic tests", "qe-test-generator")
Task("Snapshot Tests", "Component snapshots", "qe-test-executor")

// PHASE 2: Build (Parallel iOS + Android)
Task("iOS Build", "Build and test iOS", "qe-test-executor")
Task("Android Build", "Build and test Android", "qe-test-executor")
Task("Integration Tests", "E2E on both platforms", "qe-test-executor")

// PHASE 3: Device Testing
Task("Device Farm", "Test on 20+ devices", "qe-test-executor")
Task("Performance", "Low-end device testing", "qe-performance-tester")
Task("Visual Testing", "Cross-device visuals", "qe-visual-tester")

// Apply mobile testing skill
Skill("mobile-testing")

// PHASE 4: Beta
Task("Crash Monitoring", "Monitor beta crashes", "qe-production-intelligence")
Task("Accessibility", "Mobile a11y validation", "qe-visual-tester")
Task("Localization", "Multi-language testing", "qe-test-executor")

// Apply accessibility and localization skills
Skill("accessibility-testing")
Skill("localization-testing")

// PHASE 5: Production (Staged Rollout)
Task("Rollout Monitor", "Track staged rollout", "qe-production-intelligence")
Task("Quality Metrics", "Production quality tracking", "qe-quality-analyzer")
```

---

## Timing Budget

| Phase | Target Time | Max Time |
|-------|-------------|----------|
| Commit | 5 min | 10 min |
| Build (iOS + Android) | 20 min | 40 min |
| Device Testing | 60 min | 120 min |
| Beta Testing | 3 days | 7 days |
| Production Rollout | 5 days (staged) | 10 days |
| **Total** | **~9 days** | **~18 days** |

---

## Device Coverage Strategy

### Minimum Device Coverage

**iOS**:
- iPhone SE (low-end)
- iPhone 12/13 (mid-range)
- iPhone 15 Pro (high-end)
- iPad (tablet)

**Android**:
- Samsung Galaxy A series (low-end)
- Google Pixel (mid-range)
- Samsung Galaxy S series (high-end)
- Various tablets

### OS Version Coverage

**iOS**: Last 3 versions (iOS 15, 16, 17)
**Android**: Last 4 versions (Android 11, 12, 13, 14)

---

## Performance Benchmarks

### Target Metrics

| Metric | Target | Max |
|--------|--------|-----|
| Cold start | 1.5s | 3s |
| Hot start | 0.5s | 1s |
| Frame rate | 60fps | 50fps |
| Memory (idle) | 50MB | 100MB |
| Memory (active) | 150MB | 300MB |
| APK/IPA size | 30MB | 50MB |

### Testing

```javascript
Task("Performance Benchmarks",
     "Test cold/hot start, frame rate, memory on low-end devices",
     "qe-performance-tester")
```

---

## Accessibility Requirements

### WCAG Mobile Criteria

- [ ] Touch target size (44x44 minimum)
- [ ] Screen reader support (TalkBack/VoiceOver)
- [ ] Color contrast (4.5:1 minimum)
- [ ] Keyboard navigation (external keyboard)
- [ ] Gesture alternatives
- [ ] Orientation support

### Testing

```javascript
Task("Mobile Accessibility",
     "Validate WCAG mobile criteria with TalkBack and VoiceOver",
     "qe-visual-tester")

Skill("accessibility-testing")
```

---

## Localization Coverage

### Priority Languages
1. English (en)
2. Spanish (es)
3. French (fr)
4. German (de)
5. Chinese Simplified (zh-CN)
6. Japanese (ja)
7. Arabic (ar) - RTL testing

### Testing

```javascript
Task("Localization Testing",
     "Test all 7 languages including RTL for Arabic",
     "qe-test-executor")

Skill("localization-testing")
```

---

## Staged Rollout Strategy

### Rollout Phases

**Phase 1 (1%)**: Internal team + power users (24 hours)
**Phase 2 (10%)**: Early adopters (48 hours)
**Phase 3 (50%)**: Broader audience (48 hours)
**Phase 4 (100%)**: All users

### Monitoring Per Phase

```javascript
// Monitor each rollout phase
Task("Phase 1 Monitoring", "Track 1% rollout metrics", "qe-production-intelligence")
// Check gates, then proceed to next phase
Task("Phase 2 Monitoring", "Track 10% rollout metrics", "qe-production-intelligence")
// Repeat for each phase...
```

### Rollback Triggers

- Crash rate > 1%
- ANR rate > 0.5% (Android)
- App rating drops below 4.0
- Critical bug reported by > 10 users

---

## App Store Optimization

### Pre-Release Checklist

- [ ] App store screenshots (all device sizes)
- [ ] App preview videos
- [ ] Metadata in all languages
- [ ] Privacy policy updated
- [ ] What's New section written
- [ ] Keywords optimized

### Visual Testing

```javascript
Task("ASO Visual Check",
     "Validate app store screenshots and previews",
     "qe-visual-tester")
```

---

**Use Cases**: Consumer apps, Enterprise mobile apps, Mobile games
**Team Size**: 5-15 developers
**Deploy Frequency**: Bi-weekly releases
**Platforms**: iOS (App Store), Android (Google Play)
**Testing**: Device farms (BrowserStack, Firebase Test Lab)
