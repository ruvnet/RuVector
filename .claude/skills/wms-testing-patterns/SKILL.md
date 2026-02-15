---
name: wms-testing-patterns
description: "Warehouse Management System testing patterns for inventory operations, pick/pack/ship workflows, wave management, EDI X12/EDIFACT compliance, RF/barcode scanning, and WMS-ERP integration. Use when testing WMS platforms (Blue Yonder, Manhattan, SAP EWM)."
category: enterprise-integration
priority: high
tokenEstimate: 1800
agents: [qe-middleware-validator, qe-contract-validator, qe-sap-idoc-tester, qe-odata-contract-tester]
implementation_status: optimized
optimization_version: 2.0
last_optimized: 2026-02-04
dependencies: [api-testing-patterns, contract-testing, enterprise-integration-testing]
quick_reference_card: true
tags: [wms, warehouse, inventory, edi, pick-pack-ship, blue-yonder, manhattan, sap-ewm, rf-scanning]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/wms-testing-patterns.yaml
---

# WMS Testing Patterns

<default_to_action>
When testing Warehouse Management Systems:
1. VALIDATE inventory accuracy (receipt, putaway, cycle count, adjustments)
2. TEST pick/pack/ship workflows end-to-end (wave release -> shipment confirm)
3. VERIFY EDI message processing (856 ASN, 940 Order, 945 Confirmation, 943/944 Stock)
4. ASSERT RF/barcode scanning flows (scan -> validate -> update -> confirm)
5. EXERCISE allocation and replenishment logic (FIFO, FEFO, lot control)
6. TEST WMS-ERP integration (inventory sync, order status, goods receipt)
7. VALIDATE wave management (wave planning, release, short-pick handling)

**Quick Pattern Selection:**
- Inventory discrepancies -> Cycle count and adjustment tests
- Order fulfillment issues -> Pick/pack/ship workflow tests
- EDI failures -> Message format and acknowledgment tests
- Scanning problems -> RF device simulation tests
- Allocation errors -> Lot control and FIFO/FEFO tests
- Integration gaps -> WMS-ERP sync boundary tests

**Critical Success Factors:**
- WMS accuracy directly impacts customer experience and financial reporting
- Always test both normal and exception flows (short picks, damaged goods, returns)
- EDI testing must cover both syntax validation AND business rule validation
</default_to_action>

## Quick Reference Card

### When to Use
- Testing WMS platforms (Blue Yonder/JDA, Manhattan Associates, SAP EWM, Oracle WMS)
- Validating inventory transaction accuracy (receipts, picks, adjustments)
- Testing EDI document exchange (X12 856/940/945/943/944, EDIFACT DESADV/ORDERS)
- Verifying RF/mobile scanning workflows
- Testing wave management and allocation logic
- Validating WMS-to-ERP integration (SAP MM/WM, Oracle)

### Testing Levels
| Level | Purpose | Dependencies | Speed |
|-------|---------|--------------|-------|
| Unit Logic | Allocation/replenishment rules | None | Fast |
| API Contract | WMS REST/SOAP endpoint contracts | API stubs | Fast |
| EDI Validation | Document syntax + business rules | EDI parser | Medium |
| Integration | WMS-ERP inventory sync | Full stack | Slower |
| E2E Workflow | Complete order fulfillment cycle | WMS + ERP + TMS | Slow |

### Critical Test Scenarios
| Scenario | Must Test | Example |
|----------|----------|---------|
| Receiving | Inbound accuracy | PO receipt with over/under delivery tolerance |
| Putaway | Location assignment | Directed putaway by product attributes |
| Picking | Wave/batch pick | Multi-order wave with priority allocation |
| Packing | Pack verification | Cartonization and weight/dim validation |
| Shipping | Carrier assignment | Rate shopping and label generation |
| Cycle Count | Inventory accuracy | Blind count vs. guided count reconciliation |
| Returns | Reverse logistics | RMA receipt, inspection, disposition |
| EDI 856 | ASN generation | Ship-confirm triggers correct ASN to customer |
| EDI 940 | Warehouse order | Inbound order creates correct work orders |
| Short Pick | Exception handling | Partial allocation with backorder creation |

### Tools
- **WMS Platforms**: Blue Yonder WMS, Manhattan WMOS, SAP EWM, Oracle WMS Cloud
- **EDI Testing**: Bots EDI, SPS Commerce Test, Cleo Clarify
- **RF Simulation**: Custom RF emulators, Zebra SDK test harnesses
- **Integration**: SAP PI/PO, MuleSoft, Dell Boomi
- **Monitoring**: Splunk, WMS dashboards, EDI tracking portals

### Agent Coordination
- `qe-middleware-validator`: WMS-ERP integration flows and message transformation validation
- `qe-contract-validator`: EDI document contracts and WMS API contracts
- `qe-sap-idoc-tester`: WMS-SAP IDoc validation (WMMBID01, SHPCON)
- `qe-odata-contract-tester`: SAP EWM OData service testing

---

## Inventory Transaction Testing

### Receipt and Putaway
```javascript
describe('Inbound Receipt - PO with Lot Control', () => {
  it('receives goods against PO with lot tracking', async () => {
    const receipt = await wms.receive({
      po: 'PO-4500001234',
      item: 'MAT-100',
      quantity: 100,
      lot: 'LOT-2026-001',
      expiryDate: '2027-06-30',
      uom: 'EA'
    });

    expect(receipt.status).toBe('RECEIVED');
    expect(receipt.quantityReceived).toBe(100);
    expect(receipt.lot).toBe('LOT-2026-001');

    // Verify putaway task was created
    const putawayTask = await wms.getTask(receipt.putawayTaskId);
    expect(putawayTask.type).toBe('DIRECTED_PUTAWAY');
    expect(putawayTask.suggestedLocation).toMatch(/^BIN-/);
  });

  it('rejects over-receipt beyond tolerance', async () => {
    // PO line is for 100 EA, tolerance is 10%
    const result = await wms.receive({
      po: 'PO-4500001234',
      item: 'MAT-100',
      quantity: 115, // 15% over, exceeds 10% tolerance
      lot: 'LOT-2026-002'
    });

    expect(result.status).toBe('REJECTED');
    expect(result.reason).toContain('Over-receipt tolerance exceeded');
    expect(result.maxAllowed).toBe(110);
  });

  it('accepts under-receipt and creates remaining open quantity', async () => {
    const result = await wms.receive({
      po: 'PO-4500001234',
      item: 'MAT-100',
      quantity: 80,
      lot: 'LOT-2026-003'
    });

    expect(result.status).toBe('PARTIAL_RECEIPT');
    expect(result.quantityReceived).toBe(80);
    expect(result.remainingOpen).toBe(20);
  });
});

describe('Directed Putaway', () => {
  it('assigns location based on product attributes', async () => {
    // Hazmat product should go to hazmat zone
    const putaway = await wms.executePutaway({
      item: 'MAT-HAZ-001',
      attributes: { hazmat: true, temperature: 'ambient' },
      quantity: 50,
      lot: 'LOT-HAZ-001'
    });

    expect(putaway.location).toMatch(/^HAZ-/); // Hazmat zone prefix
    expect(putaway.status).toBe('PUTAWAY_COMPLETE');
  });

  it('falls back to overflow when primary zone is full', async () => {
    // Fill the primary zone first
    await wms.fillZone('ZONE-A', 100); // 100% capacity

    const putaway = await wms.executePutaway({
      item: 'MAT-200',
      quantity: 10,
      preferredZone: 'ZONE-A'
    });

    expect(putaway.location).toMatch(/^OVF-/); // Overflow zone
    expect(putaway.overflowReason).toBe('PRIMARY_ZONE_FULL');
  });
});
```

### Cycle Count and Adjustments
```javascript
describe('Cycle Count - Blind Count', () => {
  it('auto-approves variance within threshold', async () => {
    // System shows 100 EA, physical count is 98
    const count = await wms.submitCycleCount({
      location: 'BIN-A-01-01',
      item: 'MAT-100',
      countedQuantity: 98,
      countType: 'BLIND', // Counter does not see system quantity
      varianceThreshold: 5 // percent
    });

    expect(count.systemQuantity).toBe(100);
    expect(count.variance).toBe(-2);
    expect(count.variancePercent).toBeCloseTo(2.0);
    expect(count.autoApproved).toBe(true);
    expect(count.adjustmentPosted).toBe(true);
  });

  it('requires supervisor approval when variance exceeds threshold', async () => {
    const count = await wms.submitCycleCount({
      location: 'BIN-A-01-02',
      item: 'MAT-200',
      countedQuantity: 80,
      countType: 'BLIND',
      varianceThreshold: 5 // 20% variance exceeds 5% threshold
    });

    expect(count.variancePercent).toBeCloseTo(20.0);
    expect(count.autoApproved).toBe(false);
    expect(count.adjustmentPosted).toBe(false);
    expect(count.status).toBe('PENDING_SUPERVISOR_APPROVAL');
  });

  it('triggers recount when variance is critical', async () => {
    const count = await wms.submitCycleCount({
      location: 'BIN-A-01-03',
      item: 'MAT-300',
      countedQuantity: 0, // System shows 50
      countType: 'BLIND',
      varianceThreshold: 5,
      criticalThreshold: 50
    });

    expect(count.variancePercent).toBe(100);
    expect(count.status).toBe('RECOUNT_REQUIRED');
    expect(count.recountAssigned).toBe(true);
    expect(count.recountAssignee).not.toBe(count.originalCounter);
  });
});
```

---

## Pick/Pack/Ship Workflow Testing

### Wave Management
```javascript
describe('Wave Planning and Release', () => {
  it('creates wave from eligible orders with batch-pick strategy', async () => {
    // Create 5 orders for wave planning
    const orders = await Promise.all(
      Array.from({ length: 5 }, (_, i) => wms.createOrder({
        orderId: `SO-100${i}`,
        lines: [
          { item: 'MAT-100', qty: 10 },
          { item: 'MAT-200', qty: 5 }
        ],
        priority: i === 0 ? 'RUSH' : 'STANDARD',
        shipBy: '2026-02-05'
      }))
    );

    const wave = await wms.planWave({
      strategy: 'BATCH_PICK',
      maxOrders: 10,
      cutoffTime: '2026-02-05T14:00:00Z'
    });

    expect(wave.orderCount).toBe(5);
    expect(wave.pickTasks).toBeGreaterThan(0);
    expect(wave.status).toBe('PLANNED');

    // Rush order should be first in pick sequence
    const firstTask = wave.pickTasks[0];
    expect(firstTask.orderId).toBe('SO-1000');
    expect(firstTask.priority).toBe('RUSH');
  });

  it('handles short-pick by creating backorder', async () => {
    // Only 5 EA available, order needs 10
    await wms.setInventory('BIN-B-01-01', 'MAT-500', 5);

    const order = await wms.createOrder({
      orderId: 'SO-SHORT',
      lines: [{ item: 'MAT-500', qty: 10 }]
    });

    const wave = await wms.planWave({ strategy: 'SINGLE_ORDER' });
    await wms.releaseWave(wave.id);

    // Attempt to pick
    const pickResult = await wms.confirmPick({
      taskId: wave.pickTasks[0].id,
      pickedQty: 5 // Only 5 available
    });

    expect(pickResult.status).toBe('SHORT_PICK');
    expect(pickResult.pickedQuantity).toBe(5);
    expect(pickResult.shortQuantity).toBe(5);
    expect(pickResult.backorderCreated).toBe(true);
    expect(pickResult.backorderId).toBeDefined();
  });
});
```

### Allocation Logic
```javascript
describe('FIFO Allocation', () => {
  it('allocates oldest lot first', async () => {
    // Set up two lots with different receipt dates
    await wms.receiveInventory('MAT-FIFO', 50, { lot: 'LOT-OLD', receiptDate: '2026-01-01' });
    await wms.receiveInventory('MAT-FIFO', 50, { lot: 'LOT-NEW', receiptDate: '2026-02-01' });

    const allocation = await wms.allocate({
      item: 'MAT-FIFO',
      quantity: 30,
      method: 'FIFO'
    });

    expect(allocation.allocatedLot).toBe('LOT-OLD');
    expect(allocation.quantity).toBe(30);
  });

  it('splits allocation across lots when oldest is insufficient', async () => {
    await wms.receiveInventory('MAT-SPLIT', 20, { lot: 'LOT-A', receiptDate: '2026-01-01' });
    await wms.receiveInventory('MAT-SPLIT', 50, { lot: 'LOT-B', receiptDate: '2026-01-15' });

    const allocation = await wms.allocate({
      item: 'MAT-SPLIT',
      quantity: 40,
      method: 'FIFO'
    });

    expect(allocation.splits).toHaveLength(2);
    expect(allocation.splits[0]).toEqual({ lot: 'LOT-A', quantity: 20 });
    expect(allocation.splits[1]).toEqual({ lot: 'LOT-B', quantity: 20 });
  });
});

describe('FEFO Allocation (First Expired, First Out)', () => {
  it('allocates lot with earliest expiry first', async () => {
    await wms.receiveInventory('MAT-FEFO', 50, { lot: 'LOT-EXP-LATE', expiry: '2027-12-31' });
    await wms.receiveInventory('MAT-FEFO', 50, { lot: 'LOT-EXP-SOON', expiry: '2026-06-30' });

    const allocation = await wms.allocate({
      item: 'MAT-FEFO',
      quantity: 30,
      method: 'FEFO'
    });

    expect(allocation.allocatedLot).toBe('LOT-EXP-SOON');
  });

  it('excludes expired lots from allocation', async () => {
    await wms.receiveInventory('MAT-EXPIRED', 100, { lot: 'LOT-PAST', expiry: '2025-12-31' });
    await wms.receiveInventory('MAT-EXPIRED', 50, { lot: 'LOT-VALID', expiry: '2027-12-31' });

    const allocation = await wms.allocate({
      item: 'MAT-EXPIRED',
      quantity: 30,
      method: 'FEFO'
    });

    expect(allocation.allocatedLot).toBe('LOT-VALID');
    expect(allocation.excludedLots).toContain('LOT-PAST');
    expect(allocation.excludeReason).toBe('EXPIRED');
  });
});
```

### Packing and Shipping
```javascript
describe('Pack Verification', () => {
  it('verifies pack contents match pick list', async () => {
    const packResult = await wms.packVerify({
      orderId: 'SO-1001',
      cartonId: 'CTN-001',
      scannedItems: [
        { barcode: 'MAT-100-LOT001', qty: 10 },
        { barcode: 'MAT-200-LOT002', qty: 5 }
      ]
    });

    expect(packResult.status).toBe('VERIFIED');
    expect(packResult.allItemsScanned).toBe(true);
    expect(packResult.weightValidation).toBe('WITHIN_TOLERANCE');
  });

  it('rejects pack with missing items', async () => {
    const packResult = await wms.packVerify({
      orderId: 'SO-1001',
      cartonId: 'CTN-002',
      scannedItems: [
        { barcode: 'MAT-100-LOT001', qty: 10 }
        // Missing MAT-200
      ]
    });

    expect(packResult.status).toBe('INCOMPLETE');
    expect(packResult.missingItems).toContainEqual({ item: 'MAT-200', expectedQty: 5 });
  });
});

describe('Shipping and Carrier Assignment', () => {
  it('rate-shops across carriers and selects cheapest', async () => {
    const shipment = await wms.shipConfirm({
      orderId: 'SO-1001',
      cartons: ['CTN-001'],
      carrierSelection: 'RATE_SHOP',
      eligibleCarriers: ['UPS', 'FEDEX', 'USPS']
    });

    expect(shipment.status).toBe('SHIPPED');
    expect(shipment.carrier).toBeDefined();
    expect(shipment.trackingNumber).toBeDefined();
    expect(shipment.rateShopResults).toHaveLength(3);
    expect(shipment.selectedRate).toBeLessThanOrEqual(
      Math.min(...shipment.rateShopResults.map(r => r.rate))
    );
  });

  it('generates shipping label and triggers EDI 856', async () => {
    const shipment = await wms.shipConfirm({
      orderId: 'SO-1002',
      cartons: ['CTN-003'],
      carrier: 'UPS',
      service: 'GROUND'
    });

    expect(shipment.label).toBeDefined();
    expect(shipment.label.format).toBe('ZPL');
    expect(shipment.edi856Generated).toBe(true);
    expect(shipment.edi856.transactionSetId).toMatch(/^856/);
  });
});
```

---

## EDI Document Testing

### EDI 856 - Advanced Ship Notice
```javascript
describe('EDI 856 ASN Generation', () => {
  it('generates valid 856 with correct hierarchical levels', async () => {
    const asn = await edi.generate856({
      shipmentId: 'SHP-001',
      carrier: { scac: 'UPSN', proNumber: 'PRO-12345' },
      orders: [{
        orderId: 'SO-1001',
        items: [
          { sku: 'MAT-100', qty: 10, lot: 'LOT-001', upc: '012345678901' }
        ]
      }]
    });

    // Validate hierarchical structure: Shipment -> Order -> Item
    expect(asn.segments.filter(s => s.id === 'HL')).toHaveLength(3);
    expect(asn.getSegment('BSN', 'BSN01')).toBe('00'); // Original
    expect(asn.getSegment('TD5', 'TD504')).toBe('UPSN');
    expect(asn.getSegment('SN1', 'SN102')).toBe('10');

    // Validate segment count and control numbers
    expect(asn.getSegment('CTT', 'CTT01')).toBe('1'); // 1 transaction
    expect(asn.getSegment('SE', 'SE01')).toBeDefined(); // Segment count
  });

  it('validates 997 functional acknowledgment for sent 856', async () => {
    const asn = await edi.send856('SHP-001');
    const ack = await edi.receive997({ timeout: 30000 });

    expect(ack.transactionSetId).toBe('997');
    expect(ack.acknowledgmentCode).toBe('A'); // Accepted
    expect(ack.referencedTransactionSet).toBe('856');
  });

  it('handles 997 rejection and triggers resend', async () => {
    const asn = await edi.send856('SHP-INVALID');
    const ack = await edi.receive997({ timeout: 30000 });

    expect(ack.acknowledgmentCode).toBe('R'); // Rejected
    expect(ack.errors).toContainEqual(
      expect.objectContaining({ segmentId: 'SN1', errorCode: '7' })
    );

    // Verify resend was triggered
    const resendLog = await edi.getResendLog('SHP-INVALID');
    expect(resendLog.attempts).toBeGreaterThanOrEqual(1);
  });
});
```

### EDI 940 - Warehouse Shipping Order
```javascript
describe('EDI 940 Inbound Processing', () => {
  it('creates WMS order from valid 940 document', async () => {
    const edi940 = loadEdiDocument('testdata/edi-940-standard.edi');
    const result = await edi.process940(edi940);

    expect(result.orderCreated).toBe(true);
    expect(result.wmsOrderId).toBeDefined();
    expect(result.lineCount).toBe(5);
    expect(result.allocationTriggered).toBe(true);
    expect(result.priorityMapped).toBe('STANDARD');
  });

  it('rejects 940 with unknown SKU and generates 997 rejection', async () => {
    const edi940 = loadEdiDocument('testdata/edi-940-invalid-sku.edi');
    const result = await edi.process940(edi940);

    expect(result.orderCreated).toBe(false);
    expect(result.errors).toContainEqual(
      expect.objectContaining({ segment: 'LIN', error: 'Unknown SKU: INVALID-SKU' })
    );
    expect(result.ack997.code).toBe('R');
  });

  it('detects duplicate 940 by BSN reference number', async () => {
    const edi940 = loadEdiDocument('testdata/edi-940-standard.edi');

    // Process first time - success
    await edi.process940(edi940);

    // Process duplicate - should detect
    const result = await edi.process940(edi940);

    expect(result.orderCreated).toBe(false);
    expect(result.duplicate).toBe(true);
    expect(result.originalOrderId).toBeDefined();
  });
});
```

### EDI 945 - Warehouse Shipping Advice
```javascript
describe('EDI 945 Outbound Generation', () => {
  it('generates 945 on shipment confirmation', async () => {
    await wms.shipConfirm({ orderId: 'SO-1005', carrier: 'FEDEX' });

    const edi945 = await edi.getLatest945('SO-1005');

    expect(edi945.transactionSetId).toBe('945');
    expect(edi945.getSegment('W06', 'W0601')).toBe('N'); // Normal shipment
    expect(edi945.getSegment('W12', 'W1202')).toBeDefined(); // Quantity shipped
    expect(edi945.getSegment('N1', 'N102')).toBeDefined(); // Ship-to name
  });
});
```

---

## RF/Barcode Scanning Flow Testing

```javascript
describe('RF Directed Pick Workflow', () => {
  let rfSession;

  beforeEach(async () => {
    rfSession = await rfSimulator.createSession({
      device: 'Zebra-MC9300',
      mode: 'directed-pick'
    });
  });

  it('completes directed pick with valid scans', async () => {
    // Step 1: Login
    const login = await rfSession.scan('badge', 'USER-PICKER-01');
    expect(login.screen).toBe('MAIN_MENU');

    // Step 2: Select pick task
    const taskSelect = await rfSession.selectMenu('PICK');
    expect(taskSelect.screen).toBe('PICK_TASK_ASSIGNED');
    expect(taskSelect.taskId).toBeDefined();

    // Step 3: Scan source location
    const locationScan = await rfSession.scan('location', 'BIN-A-01-01');
    expect(locationScan.screen).toBe('SCAN_ITEM');
    expect(locationScan.expectedItem).toBe('MAT-100');

    // Step 4: Scan item barcode
    const itemScan = await rfSession.scan('item', 'MAT-100-LOT001');
    expect(itemScan.screen).toBe('ENTER_QUANTITY');
    expect(itemScan.maxQty).toBe(10);

    // Step 5: Enter quantity
    const qtyConfirm = await rfSession.enterQuantity(10);
    expect(qtyConfirm.screen).toBe('SCAN_DESTINATION');

    // Step 6: Scan staging lane
    const destScan = await rfSession.scan('location', 'STAGE-LANE-01');
    expect(destScan.screen).toBe('PICK_COMPLETE');
    expect(destScan.taskStatus).toBe('COMPLETED');
  });

  it('rejects wrong item scan with mismatch alert', async () => {
    await rfSession.scan('badge', 'USER-PICKER-01');
    await rfSession.selectMenu('PICK');
    await rfSession.scan('location', 'BIN-A-01-01');

    const wrongItem = await rfSession.scan('item', 'MAT-999-WRONG');

    expect(wrongItem.screen).toBe('ITEM_MISMATCH');
    expect(wrongItem.alert).toBe('MISMATCH');
    expect(wrongItem.expectedItem).toBe('MAT-100');
    expect(wrongItem.scannedItem).toBe('MAT-999');
  });

  it('warns on over-pick quantity', async () => {
    await rfSession.scan('badge', 'USER-PICKER-01');
    await rfSession.selectMenu('PICK');
    await rfSession.scan('location', 'BIN-A-01-01');
    await rfSession.scan('item', 'MAT-100-LOT001');

    const overPick = await rfSession.enterQuantity(999);

    expect(overPick.screen).toBe('OVER_PICK_WARNING');
    expect(overPick.maxAllowed).toBe(10);
    expect(overPick.requiresSupervisor).toBe(true);
  });

  it('handles wrong location scan gracefully', async () => {
    await rfSession.scan('badge', 'USER-PICKER-01');
    await rfSession.selectMenu('PICK');

    const wrongLocation = await rfSession.scan('location', 'BIN-Z-99-99');

    expect(wrongLocation.screen).toBe('WRONG_LOCATION');
    expect(wrongLocation.expectedLocation).toBe('BIN-A-01-01');
    expect(wrongLocation.action).toBe('RESCAN');
  });
});

describe('RF Receiving Workflow', () => {
  it('receives goods by scanning PO and items', async () => {
    const rfSession = await rfSimulator.createSession({ mode: 'receiving' });

    await rfSession.scan('badge', 'USER-RECEIVER-01');
    const poScan = await rfSession.scan('document', 'PO-4500001234');
    expect(poScan.screen).toBe('SCAN_ITEM');
    expect(poScan.expectedItems).toHaveLength(3);

    const itemScan = await rfSession.scan('item', 'MAT-100');
    expect(itemScan.screen).toBe('ENTER_QUANTITY');

    const receipt = await rfSession.enterQuantity(100);
    expect(receipt.screen).toBe('SCAN_LOT');

    const lotScan = await rfSession.scan('lot', 'LOT-2026-050');
    expect(lotScan.screen).toBe('RECEIPT_CONFIRMED');
    expect(lotScan.quantityReceived).toBe(100);
  });
});
```

---

## WMS-ERP Integration Testing

### SAP EWM to S/4HANA Sync
```javascript
describe('WMS-ERP Goods Receipt Sync', () => {
  it('syncs goods receipt from EWM to S/4HANA within SLA', async () => {
    const startTime = Date.now();

    // Confirm inbound delivery in EWM
    await ewm.confirmInboundDelivery({
      deliveryId: 'IBD-001',
      items: [{ material: 'MAT-100', quantity: 100, batch: 'BATCH-001' }]
    });

    // Wait for sync to S/4HANA
    const materialDoc = await s4hana.waitForMaterialDocument({
      reference: 'IBD-001',
      timeout: 30000
    });

    const elapsed = Date.now() - startTime;

    expect(materialDoc.created).toBe(true);
    expect(materialDoc.type).toBe('WE'); // Goods receipt
    expect(materialDoc.quantity).toBe(100);
    expect(materialDoc.batch).toBe('BATCH-001');
    expect(materialDoc.plant).toBe('1000');
    expect(elapsed).toBeLessThan(30000); // 30s SLA
  });

  it('validates inventory quantities match between EWM and S/4HANA', async () => {
    const ewmStock = await ewm.getStock('MAT-100', 'WH-01');
    const s4Stock = await s4hana.getStock('MAT-100', '1000', '0001');

    expect(ewmStock.available).toBe(s4Stock.unrestricted);
    expect(ewmStock.allocated).toBe(s4Stock.reserved);
    expect(ewmStock.blocked).toBe(s4Stock.qualityInspection);
  });

  it('handles sync failure with IDoc error and retry', async () => {
    // Simulate S/4HANA unavailability
    await s4hana.simulateDowntime(10000);

    await ewm.confirmInboundDelivery({
      deliveryId: 'IBD-FAIL',
      items: [{ material: 'MAT-200', quantity: 50 }]
    });

    // Check IDoc status
    const idoc = await sap.getIdoc({ reference: 'IBD-FAIL', type: 'WMMBID01' });
    expect(idoc.status).toBe('03'); // Error status

    // Restore S/4HANA and verify auto-retry
    await s4hana.restoreService();
    const retried = await sap.waitForIdocStatus(idoc.id, '53', { timeout: 60000 });
    expect(retried.status).toBe('53'); // Successfully processed
  });
});
```

### Manhattan WMOS to Oracle EBS Sync
```javascript
describe('Manhattan WMOS to Oracle EBS', () => {
  it('posts inventory adjustment to Oracle via REST API', async () => {
    const adjustment = await wmos.postAdjustment({
      item: 'ITEM-500',
      location: 'LOC-A-01',
      adjustmentQty: -5,
      reason: 'DAMAGED'
    });

    const oracleTransaction = await oracle.waitForTransaction({
      reference: adjustment.transactionId,
      timeout: 15000
    });

    expect(oracleTransaction.transactionType).toBe('ADJUSTMENT');
    expect(oracleTransaction.quantity).toBe(-5);
    expect(oracleTransaction.reasonCode).toBe('DAMAGED');
    expect(oracleTransaction.accountPosted).toBe(true);
  });
});
```

---

## Returns and Reverse Logistics

```javascript
describe('Returns Processing', () => {
  it('processes RMA receipt with inspection and disposition', async () => {
    const rma = await wms.createRMA({
      rmaNumber: 'RMA-001',
      originalOrder: 'SO-1001',
      items: [{ sku: 'MAT-100', qty: 2, reason: 'DEFECTIVE' }]
    });

    // Receive return
    const receipt = await wms.receiveReturn({
      rmaId: rma.id,
      scannedItems: [{ barcode: 'MAT-100', qty: 2 }]
    });

    expect(receipt.status).toBe('RECEIVED_PENDING_INSPECTION');

    // Inspect returned items
    const inspection = await wms.inspectReturn({
      rmaId: rma.id,
      results: [
        { item: 'MAT-100', unit: 1, condition: 'RESTOCK', grade: 'A' },
        { item: 'MAT-100', unit: 2, condition: 'SCRAP', grade: 'F' }
      ]
    });

    expect(inspection.disposition.restock).toBe(1);
    expect(inspection.disposition.scrap).toBe(1);
    expect(inspection.restockLocation).toBeDefined();
  });
});
```

---

## Replenishment Testing

```javascript
describe('Replenishment Triggers', () => {
  it('triggers min/max replenishment when pick face drops below min', async () => {
    // Set pick face to minimum threshold
    await wms.setInventory('PICK-A-01', 'MAT-100', 5); // Min = 10, Max = 50

    const replenishment = await wms.checkReplenishment('PICK-A-01', 'MAT-100');

    expect(replenishment.triggered).toBe(true);
    expect(replenishment.type).toBe('MIN_MAX');
    expect(replenishment.replenishQty).toBe(45); // Fill to max (50 - 5)
    expect(replenishment.sourceLocation).toMatch(/^BULK-/);
  });

  it('triggers demand-based replenishment from wave allocation', async () => {
    // Wave needs 100 EA, pick face only has 20
    await wms.setInventory('PICK-B-01', 'MAT-200', 20);

    const wave = await wms.planWave({
      strategy: 'SINGLE_ORDER',
      orders: [{ orderId: 'SO-REPLEN', lines: [{ item: 'MAT-200', qty: 100 }] }]
    });

    const replenishment = await wms.getReplenishmentTasks(wave.id);

    expect(replenishment).toHaveLength(1);
    expect(replenishment[0].type).toBe('DEMAND');
    expect(replenishment[0].quantity).toBe(80); // 100 needed - 20 available
    expect(replenishment[0].priority).toBe('URGENT');
  });
});
```

---

## Best Practices

### Do This
- Test the complete lifecycle: receive -> putaway -> allocate -> pick -> pack -> ship -> EDI
- Validate inventory accuracy at every state transition (on-hand, allocated, in-transit)
- Test all allocation methods for your WMS (FIFO, FEFO, LIFO, zone priority)
- Simulate RF device failures mid-transaction (battery death, WiFi drop, scan timeout)
- Validate EDI acknowledgments (997) for every outbound document
- Test short-pick, over-pick, and wrong-pick scenarios explicitly
- Verify WMS-ERP sync latency against SLA thresholds
- Test cycle count variance at multiple thresholds (auto-approve, supervisor, recount)

### Avoid This
- Testing only happy-path receiving without over/under tolerance scenarios
- Skipping RF error flows (wrong scan, timeout, connectivity loss)
- Ignoring EDI segment-level validation (trusting document-level pass)
- Testing allocation logic without expired lot scenarios
- Assuming WMS-ERP sync is instantaneous; always test with timing assertions
- Deploying without testing wave planning with mixed priority orders
- Ignoring cartonization and weight validation in pack verification

---

## Agent-Assisted WMS Testing

```typescript
// Validate EDI document contracts
await Task("EDI Contract Validation", {
  documents: ['856-ASN', '940-Order', '945-Advice', '943-StockTransfer'],
  validateSegments: true,
  checkAcknowledgments: true,
  testErrorScenarios: ['invalid-sku', 'duplicate-bsn', 'missing-required']
}, "qe-contract-validator");

// Test WMS-ERP integration flows
await Task("WMS-ERP Integration Test", {
  source: 'SAP-EWM',
  target: 'SAP-S4HANA',
  flows: ['goods-receipt', 'goods-issue', 'stock-transfer', 'inventory-adjustment'],
  slaThreshold: 30000,
  validateIdocs: true
}, "qe-middleware-validator");

// Validate SAP IDoc structures
await Task("IDoc Structure Validation", {
  idocTypes: ['WMMBID01', 'SHPCON', 'MBGMCR01'],
  validateSegments: true,
  testMandatoryFields: true,
  crossReferenceWithEWM: true
}, "qe-sap-idoc-tester");

// Test EWM OData services
await Task("EWM OData Service Testing", {
  services: [
    '/sap/opu/odata/sap/API_WAREHOUSE_ORDER',
    '/sap/opu/odata/sap/API_PHYSICAL_INVENTORY_DOC'
  ],
  testCrud: true,
  validateNavigationProperties: true,
  testFilterAndPagination: true
}, "qe-odata-contract-tester");
```

---

## Agent Coordination Hints

### Primary Agents
- **qe-middleware-validator**: WMS-ERP integration flow testing and message transformation validation
- **qe-contract-validator**: EDI document contract testing and WMS API contracts

### Supporting Agents
- **qe-message-broker-tester**: When WMS uses message queues for async processing
- **qe-odata-contract-tester**: When WMS exposes OData services (SAP EWM)
- **qe-sap-idoc-tester**: When WMS-SAP integration uses IDocs (WMMBID01, SHPCON)

### Coordination Pattern
```
1. qe-contract-validator -> Validate EDI document schemas and segments
2. qe-middleware-validator -> Test WMS message flows and transformations
3. qe-message-broker-tester -> Verify async queue processing and DLQ
4. qe-odata-contract-tester -> Test WMS OData APIs (if SAP EWM)
5. qe-sap-idoc-tester -> Validate IDoc structure and content
```

### Memory Namespace
```
aqe/wms-testing/
  inventory/         - Receipt, putaway, adjustment test results
  fulfillment/       - Pick/pack/ship workflow results
  edi/               - EDI document validation results
  rf-scanning/       - RF workflow simulation results
  allocation/        - FIFO/FEFO/lot control test results
  integration/       - WMS-ERP sync test results
  returns/           - RMA and reverse logistics results
```

### Fleet Coordination
```typescript
const wmsFleet = await FleetManager.coordinate({
  strategy: 'wms-testing',
  agents: [
    'qe-contract-validator',     // EDI contracts and API contracts
    'qe-middleware-validator',    // WMS-ERP message flows
    'qe-sap-idoc-tester',        // IDoc validation
    'qe-odata-contract-tester'   // SAP EWM OData testing
  ],
  topology: 'mesh'
});

await wmsFleet.execute({
  workflows: [
    { name: 'inbound', stages: ['receive', 'putaway', 'inventory-sync'] },
    { name: 'outbound', stages: ['wave', 'pick', 'pack', 'ship', 'edi-856'] },
    { name: 'returns', stages: ['rma-receipt', 'inspection', 'disposition'] }
  ],
  testEdi: true,
  testRfScanning: true,
  testAllocation: true
});
```

### QCSD Integration
- **Ideation**: Flag `HAS_WMS` triggers WMS-specific quality criteria (inventory accuracy, EDI compliance)
- **Refinement**: SFDIPOT includes warehouse operations product factors (throughput, accuracy, latency)
- **Development**: Coverage analysis includes EDI document parsing paths and allocation algorithms
- **Verification**: Pipeline gates include WMS integration health checks and EDI acknowledgment validation

---

## Related Skills
- [api-testing-patterns](../api-testing-patterns/) - REST/GraphQL API testing fundamentals
- [contract-testing](../contract-testing/) - Consumer-driven contract testing with Pact
- [enterprise-integration-testing](../enterprise-integration-testing/) - Orchestrating all enterprise agents
- [middleware-testing-patterns](../middleware-testing-patterns/) - ESB and message broker testing
- [observability-testing-patterns](../observability-testing-patterns/) - Monitoring and alerting validation

---

## Remember

Warehouse Management Systems are the physical-digital bridge. When inventory is wrong, customers get wrong shipments, financial reports are inaccurate, and replenishment fails. Test every inventory state transition, every EDI document field, and every RF scanning exception. A 2% inventory variance might seem small until it compounds across 100,000 SKUs in 50 warehouses.

**With Agents:** Agents validate EDI document structures segment-by-segment, test WMS-ERP sync within SLA thresholds, and simulate RF scanning workflows at scale. Use agents to cover the combinatorial explosion of allocation methods, lot control scenarios, and EDI error conditions that manual testing cannot reach.
