---
name: mobile-testing
description: "Comprehensive mobile testing for iOS and Android platforms including gestures, sensors, permissions, device fragmentation, and performance. Use when testing native apps, hybrid apps, or mobile web, ensuring quality across 1000+ device variants."
category: specialized-testing
priority: high
tokenEstimate: 1000
agents: [qe-test-executor, qe-performance-tester, qe-visual-tester]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [mobile, ios, android, appium, gestures, device-fragmentation, sensors]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/mobile-testing.yaml

---

# Mobile Testing

<default_to_action>
When testing mobile applications:
1. DEFINE device coverage matrix (Tier 1: 60%, Tier 2: 30%, Tier 3: 10%)
2. TEST platform differences (iOS ≠ Android: back button, permissions, UI)
3. VALIDATE touch gestures (tap, swipe, pinch, long-press)
4. TEST mobile-specific scenarios (offline, low battery, interruptions)
5. USE real devices for critical paths, emulators for fast feedback

**Quick Mobile Checklist:**
- Test on latest iOS + Android flagship devices
- Test offline mode and network transitions
- Verify push notifications work
- Test gesture interactions (swipe, pinch)
- Check permissions flow (camera, location, notifications)

**Critical Success Factors:**
- Emulators for 80% of testing, real devices for 20% critical paths
- Test on devices your users actually use (analytics)
- Device fragmentation is Android's biggest challenge
</default_to_action>

## Quick Reference Card

### When to Use
- Native app development (iOS/Android)
- Hybrid apps (React Native, Flutter)
- Mobile web / PWAs
- App store submission preparation

### iOS vs Android Differences
| Aspect | iOS | Android |
|--------|-----|---------|
| OS Versions | 2-3 supported | 10+ in use |
| Devices | ~40 models | 1000+ variants |
| Back Button | Gesture/nav | Hardware/software |
| Permissions | Single prompt | Runtime granular |
| App Store | Strict review | Google Play + sideload |

### Device Coverage Tiers
| Tier | Coverage | Devices |
|------|----------|---------|
| **Tier 1** | 60% users | iPhone 15, Galaxy S24, iPad |
| **Tier 2** | 30% users | iPhone 14/13, Pixel 8 |
| **Tier 3** | 10% users | Older devices, other manufacturers |

### Mobile Performance Goals
| Metric | Target |
|--------|--------|
| App launch | < 2 seconds |
| Screen transition | < 300ms |
| Frame rate | 60 FPS |
| Battery drain | < 5%/hour background |

---

## Touch Gesture Testing

```javascript
// Appium gesture examples
// Tap
await driver.touchAction({ action: 'tap', x: 100, y: 200 });

// Swipe (scroll down)
await driver.touchAction([
  { action: 'press', x: 200, y: 400 },
  { action: 'moveTo', x: 200, y: 100 },
  { action: 'release' }
]);

// Pinch to zoom
const finger1 = [
  { action: 'press', x: 100, y: 200 },
  { action: 'moveTo', x: 50, y: 150 },
  { action: 'release' }
];
const finger2 = [
  { action: 'press', x: 200, y: 200 },
  { action: 'moveTo', x: 250, y: 250 },
  { action: 'release' }
];
await driver.multiTouchAction([finger1, finger2]);

// Long press
await driver.touchAction({
  action: 'longPress',
  x: 100, y: 200,
  duration: 2000
});
```

---

## Mobile-Specific Scenarios

```javascript
// Offline mode testing
test('app works offline', async () => {
  await driver.toggleAirplaneMode();

  await driver.findElement('view-saved-items').click();
  const items = await driver.findElements('saved-item');
  expect(items.length).toBeGreaterThan(0);

  const banner = await driver.findElement('offline-banner');
  expect(banner.getText()).toContain('No internet');

  await driver.toggleAirplaneMode(); // Restore
});

// Location testing
test('location-based features', async () => {
  await driver.setGeoLocation({
    latitude: 37.7749,
    longitude: -122.4194,
    altitude: 0
  });

  const stores = await driver.findElement('stores-list');
  expect(stores.getText()).toContain('San Francisco');
});

// Permission testing (Android)
test('camera permission flow', async () => {
  await driver.findElement('take-photo').click();

  // Handle permission dialog
  await driver.findElement(
    'com.android.packageinstaller:id/permission_allow_button'
  ).click();

  expect(await driver.findElement('camera-view')).toBeDefined();
});
```

---

## Agent-Driven Mobile Testing

```typescript
// Cross-platform mobile testing
await Task("Mobile Test Suite", {
  platforms: ['iOS', 'Android'],
  deviceTiers: [1, 2],
  tests: 'regression-suite',
  parallelDevices: 5,
  deviceFarm: 'browserstack'
}, "qe-test-executor");

// Device farm integration
await Task("Device Farm Execution", {
  service: 'browserstack',
  devices: [
    'iPhone 15 - iOS 17',
    'Samsung Galaxy S24 - Android 14'
  ],
  recordVideo: true,
  captureNetworkLogs: true
}, "qe-test-executor");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/mobile-testing/
├── device-matrix/*      - Device coverage strategy
├── platform-tests/*     - iOS/Android specific tests
├── gesture-library/*    - Reusable gesture patterns
└── performance/*        - Mobile performance metrics
```

### Fleet Coordination
```typescript
const mobileFleet = await FleetManager.coordinate({
  strategy: 'mobile-testing',
  agents: [
    'qe-test-executor',       // Cross-platform execution
    'qe-performance-tester',  // Mobile performance
    'qe-visual-tester'        // Screen size validation
  ],
  topology: 'parallel'
});
```

---

## Related Skills
- [accessibility-testing](../accessibility-testing/) - VoiceOver, TalkBack
- [performance-testing](../performance-testing/) - Mobile performance
- [compatibility-testing](../compatibility-testing/) - Device compatibility

---

## Remember

**Mobile is not a smaller desktop - it's a different platform.** 60%+ of web traffic is mobile. Device fragmentation (1000+ Android devices), touch gestures, sensors, permissions, offline scenarios - all require specific testing.

**Test on real devices for critical flows.** Emulators catch 80% of bugs but real devices needed for actual performance, sensor behavior, and platform quirks.

**With Agents:** `qe-test-executor` orchestrates testing across device farms, manages platform differences, and tests 10+ devices in parallel. Reduces mobile testing from days to hours.
