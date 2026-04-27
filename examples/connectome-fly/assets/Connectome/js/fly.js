// Connectome OS — Embodied Fly simulation (procedural, Three.js)
// A stylized articulated fly body driven by motor signals from the LIF engine.
// Intentionally abstract — not a realistic render. Six legs oscillate at a tripod gait,
// wings beat at ~200 Hz (rendered at 20 Hz equivalent), antennae twitch on sensory bursts.

(function () {
  const W = window;

  function create(containerEl) {
    if (!W.THREE) return null;
    const THREE = W.THREE;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x07110d, 0.018);

    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
    camera.position.set(3.2, 2.2, 5.8);
    camera.lookAt(0, 0.4, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, devicePixelRatio));
    renderer.setClearColor(0x000000, 0);
    containerEl.appendChild(renderer.domElement);
    renderer.domElement.style.cssText = 'position:absolute; inset:0; width:100%; height:100%; display:block;';

    // Lighting
    const ambient = new THREE.AmbientLight(0x9fffb0, 0.35);
    scene.add(ambient);
    const key = new THREE.DirectionalLight(0xb8ff3c, 1.1);
    key.position.set(4, 6, 5);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x7cffbf, 0.3);
    fill.position.set(-4, 2, -3);
    scene.add(fill);

    // Ground grid — operational control-surface aesthetic
    const gridGeo = new THREE.PlaneGeometry(20, 20, 40, 40);
    const gridMat = new THREE.LineBasicMaterial({ color: 0x1b2924, transparent: true, opacity: 0.5 });
    const grid = new THREE.LineSegments(new THREE.EdgesGeometry(gridGeo), gridMat);
    grid.rotation.x = -Math.PI / 2;
    grid.position.y = -0.01;
    scene.add(grid);

    // A second circular coherence ring on the floor
    const ringGeo = new THREE.RingGeometry(2.2, 2.26, 96);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0xb8ff3c, transparent: true, opacity: 0.25, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.005;
    scene.add(ring);

    // Fly body root group
    const fly = new THREE.Group();
    fly.position.y = 0.8;
    scene.add(fly);

    // Shared materials
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0x1a2320, metalness: 0.35, roughness: 0.6, emissive: 0x0a1410, emissiveIntensity: 0.3,
    });
    const signalMat = new THREE.MeshStandardMaterial({
      color: 0xb8ff3c, emissive: 0xb8ff3c, emissiveIntensity: 0.9, metalness: 0, roughness: 0.3,
    });
    const legMat = new THREE.MeshStandardMaterial({
      color: 0x2a3632, metalness: 0.2, roughness: 0.7,
    });
    const wingMat = new THREE.MeshBasicMaterial({
      color: 0xb8ff3c, transparent: true, opacity: 0.08, side: THREE.DoubleSide,
    });
    const wingEdgeMat = new THREE.LineBasicMaterial({ color: 0xb8ff3c, transparent: true, opacity: 0.6 });

    // Thorax (ellipsoid)
    const thoraxGeo = new THREE.SphereGeometry(0.55, 28, 20);
    thoraxGeo.scale(1.0, 0.85, 1.3);
    const thorax = new THREE.Mesh(thoraxGeo, bodyMat);
    thorax.position.set(0, 0, 0);
    fly.add(thorax);

    // Abdomen (segmented, behind)
    const abdomen = new THREE.Group();
    for (let i = 0; i < 5; i++) {
      const s = 0.5 - i * 0.06;
      const seg = new THREE.Mesh(new THREE.SphereGeometry(s * 0.6, 20, 16), bodyMat);
      seg.position.z = 0.6 + i * 0.25;
      seg.scale.z = 0.85;
      abdomen.add(seg);
    }
    fly.add(abdomen);

    // Head
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.42, 24, 18), bodyMat);
    head.position.set(0, 0.1, -0.75);
    head.scale.set(1.1, 0.9, 0.95);
    fly.add(head);

    // Compound eyes (signal glow)
    const eyeGeo = new THREE.SphereGeometry(0.22, 20, 16);
    const leftEye = new THREE.Mesh(eyeGeo, signalMat.clone());
    leftEye.position.set(-0.28, 0.1, -0.85);
    leftEye.scale.set(0.9, 1.1, 1.0);
    leftEye.material.emissiveIntensity = 0.5;
    fly.add(leftEye);
    const rightEye = leftEye.clone();
    rightEye.material = leftEye.material.clone();
    rightEye.position.x = 0.28;
    fly.add(rightEye);

    // Antennae
    const antennae = [];
    for (const side of [-1, 1]) {
      const a = new THREE.Group();
      a.position.set(side * 0.12, 0.3, -0.95);
      const stemGeo = new THREE.CylinderGeometry(0.015, 0.02, 0.4, 6);
      const stem = new THREE.Mesh(stemGeo, legMat);
      stem.position.y = 0.2;
      a.add(stem);
      const tip = new THREE.Mesh(new THREE.SphereGeometry(0.04, 10, 8), signalMat.clone());
      tip.material.emissiveIntensity = 0.4;
      tip.position.y = 0.4;
      a.add(tip);
      fly.add(a);
      antennae.push({ group: a, tip });
    }

    // Legs — 3 pairs. Tripod gait: L1/R2/L3 phase A, R1/L2/R3 phase B
    const legs = [];
    // leg anchor positions on thorax (x, y, z)
    const anchors = [
      [-0.45, -0.25, -0.35, 'L1', 0],  // front-left
      [ 0.45, -0.25, -0.35, 'R1', 1],
      [-0.5, -0.3,  0.0,   'L2', 1],  // mid-left
      [ 0.5, -0.3,  0.0,   'R2', 0],
      [-0.45, -0.25, 0.35, 'L3', 0],  // rear-left
      [ 0.45, -0.25, 0.35, 'R3', 1],
    ];
    for (const [x, y, z, name, phase] of anchors) {
      const root = new THREE.Group();
      root.position.set(x, y, z);
      fly.add(root);
      // Coxa → femur → tibia (3 segments)
      const l1 = new THREE.Group();
      l1.rotation.z = x < 0 ? -0.5 : 0.5;
      root.add(l1);
      const femur = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.05, 0.55, 8), legMat);
      femur.position.y = -0.275;
      l1.add(femur);

      const l2 = new THREE.Group();
      l2.position.y = -0.55;
      l1.add(l2);
      const tibia = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.035, 0.5, 8), legMat);
      tibia.position.y = -0.25;
      l2.add(tibia);

      // tarsal tip — glows on ground contact
      const tip = new THREE.Mesh(new THREE.SphereGeometry(0.04, 10, 8), signalMat.clone());
      tip.material.emissiveIntensity = 0.2;
      tip.position.y = -0.5;
      l2.add(tip);

      legs.push({ name, phase, root, l1, l2, tip });
    }

    // Wings
    const wings = [];
    for (const side of [-1, 1]) {
      const root = new THREE.Group();
      root.position.set(side * 0.3, 0.45, -0.05);
      fly.add(root);

      // Wing blade — elongated shape
      const shape = new THREE.Shape();
      shape.moveTo(0, 0);
      shape.quadraticCurveTo(side * 0.4, 0.1, side * 1.3, 0.05);
      shape.quadraticCurveTo(side * 1.4, -0.1, side * 1.1, -0.25);
      shape.quadraticCurveTo(side * 0.5, -0.2, 0, 0);
      const wingGeo = new THREE.ShapeGeometry(shape);
      const wing = new THREE.Mesh(wingGeo, wingMat);
      const wingEdge = new THREE.LineSegments(new THREE.EdgesGeometry(wingGeo), wingEdgeMat);
      root.add(wing);
      root.add(wingEdge);
      wings.push({ side, root });
    }

    // Motor signal path — floating particles flowing from head to legs
    const pathCount = 80;
    const pathGeo = new THREE.BufferGeometry();
    const positions = new Float32Array(pathCount * 3);
    const pathPhases = new Float32Array(pathCount);
    for (let i = 0; i < pathCount; i++) {
      pathPhases[i] = Math.random();
      positions[i * 3] = 0;
      positions[i * 3 + 1] = 0;
      positions[i * 3 + 2] = 0;
    }
    pathGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const pathMat = new THREE.PointsMaterial({
      color: 0xb8ff3c, size: 0.04, transparent: true, opacity: 0.9,
      blending: THREE.AdditiveBlending,
    });
    const pathPoints = new THREE.Points(pathGeo, pathMat);
    scene.add(pathPoints);

    // HUD overlay: stats
    const stats = {
      stepHz: 50, gaitPhase: 0, wingHz: 200, sensoryBurst: 0,
    };

    function resize() {
      const r = containerEl.getBoundingClientRect();
      renderer.setSize(r.width, r.height, false);
      camera.aspect = r.width / Math.max(1, r.height);
      camera.updateProjectionMatrix();
    }
    resize();
    W.addEventListener('resize', resize);

    // --- Orbit controls: drag to rotate, wheel/pinch to zoom -------------
    let userInteracted = false;
    let targetAz = 0, targetEl = 0.38;
    let az = 0, el = 0.38;
    let targetRadius = 5.8;
    const R_MIN = 2.0, R_MAX = 14;
    let radius = 5.8;

    const dom = renderer.domElement;
    dom.style.cursor = 'grab';
    dom.style.touchAction = 'none';

    let dragging = false;
    let dragStart = null;
    const pointers = new Map();
    let pinchPrev = 0;

    function localPos(e) {
      const r = dom.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, h: r.height };
    }
    function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

    dom.addEventListener('pointerdown', (e) => {
      dom.setPointerCapture(e.pointerId);
      const p = localPos(e);
      pointers.set(e.pointerId, { x: p.x, y: p.y });
      if (pointers.size === 1) {
        dragging = true;
        userInteracted = true;
        dragStart = { x: p.x, y: p.y, az: targetAz, el: targetEl };
        dom.style.cursor = 'grabbing';
      } else if (pointers.size === 2) {
        dragging = false;
        const pts = [...pointers.values()];
        pinchPrev = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      }
    });
    dom.addEventListener('pointermove', (e) => {
      if (!pointers.has(e.pointerId)) return;
      const p = localPos(e);
      pointers.set(e.pointerId, { x: p.x, y: p.y });
      if (pointers.size === 2) {
        const pts = [...pointers.values()];
        const d = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        targetRadius = clamp(targetRadius - (d - pinchPrev) * 0.02, R_MIN, R_MAX);
        pinchPrev = d;
        return;
      }
      if (dragging && dragStart) {
        const dx = p.x - dragStart.x;
        const dy = p.y - dragStart.y;
        const s = 2.4 / Math.min(p.w, p.h);
        targetAz = dragStart.az + dx * s;
        targetEl = clamp(dragStart.el + dy * s, -0.3, 1.3);
      }
    });
    function endPtr(e) {
      if (pointers.has(e.pointerId)) pointers.delete(e.pointerId);
      if (pointers.size < 2) pinchPrev = 0;
      if (pointers.size === 0) { dragging = false; dom.style.cursor = 'grab'; }
    }
    dom.addEventListener('pointerup', endPtr);
    dom.addEventListener('pointercancel', endPtr);
    dom.addEventListener('wheel', (e) => {
      e.preventDefault();
      userInteracted = true;
      const factor = e.deltaMode === 1 ? 0.4 : 0.0025;
      targetRadius = clamp(targetRadius + e.deltaY * factor, R_MIN, R_MAX);
    }, { passive: false });
    dom.addEventListener('dblclick', () => {
      userInteracted = false;
      targetAz = 0; targetEl = 0.38; targetRadius = 5.8;
    });

    let running = true;
    let t0 = performance.now();
    function frame() {
      if (!running) return;
      const t = (performance.now() - t0) / 1000;

      // Body hover / breathing
      fly.position.y = 0.85 + Math.sin(t * 1.8) * 0.06;
      fly.rotation.y = Math.sin(t * 0.2) * 0.1;

      // Legs — tripod gait at ~6 Hz
      const gaitHz = 5.5;
      stats.gaitPhase = (t * gaitHz) % 1;
      legs.forEach((leg) => {
        const ph = (t * gaitHz + leg.phase * 0.5) * Math.PI * 2;
        const lift = Math.max(0, Math.sin(ph)) * 0.25;
        const swing = Math.cos(ph) * 0.4;
        leg.l1.rotation.x = swing * 0.3;
        leg.l2.rotation.x = -lift * 1.2 + 0.6;
        // tip glows on ground contact (lift near 0)
        const contact = 1 - Math.min(1, lift * 4);
        leg.tip.material.emissiveIntensity = 0.15 + contact * 0.9;
      });

      // Wings — fast blur-like oscillation
      wings.forEach((wing) => {
        const ph = t * 60;
        wing.root.rotation.z = wing.side * (0.2 + Math.sin(ph) * 0.7);
        wing.root.rotation.x = Math.sin(ph * 0.5) * 0.15;
      });

      // Antennae twitch
      antennae.forEach((a, i) => {
        a.group.rotation.x = Math.sin(t * 3 + i) * 0.1 + Math.sin(t * 11 + i) * 0.04;
        a.group.rotation.z = Math.sin(t * 4 + i * 1.3) * 0.08;
      });

      // Eyes — sensory-tied pulse
      const pulse = 0.4 + (0.5 + 0.5 * Math.sin(t * 2.3)) * 0.4;
      leftEye.material.emissiveIntensity = pulse;
      rightEye.material.emissiveIntensity = pulse * 0.95;

      // Motor signal particles: head → body → legs
      const posAttr = pathGeo.attributes.position;
      for (let i = 0; i < pathCount; i++) {
        let p = (pathPhases[i] + t * 0.35) % 1;
        // path: head (-0.75) → thorax (0) → leg anchor
        const legIdx = i % 6;
        const anchor = anchors[legIdx];
        let x, y, z;
        if (p < 0.5) {
          // head → thorax
          const u = p * 2;
          x = (1 - u) * 0 + u * 0;
          y = (1 - u) * 0.1 + u * (-0.1) + Math.sin(p * 20 + i) * 0.02;
          z = (1 - u) * (-0.75) + u * 0;
        } else {
          // thorax → leg anchor
          const u = (p - 0.5) * 2;
          x = (1 - u) * 0 + u * anchor[0];
          y = (1 - u) * (-0.1) + u * (anchor[1] - 0.3);
          z = (1 - u) * 0 + u * anchor[2];
        }
        posAttr.array[i * 3] = x;
        posAttr.array[i * 3 + 1] = y + fly.position.y - 0.5;
        posAttr.array[i * 3 + 2] = z;
      }
      posAttr.needsUpdate = true;

      // Ring pulse
      const ringPhase = (Math.sin(t * 0.8) + 1) * 0.5;
      ring.material.opacity = 0.1 + ringPhase * 0.25;
      ring.scale.setScalar(1 + ringPhase * 0.05);

      // Camera — user orbit if they've interacted, else gentle auto-orbit
      if (!userInteracted) {
        targetAz = t * 0.15;
        targetEl = 0.38 + Math.sin(t * 0.3) * 0.06;
        targetRadius = 5.8;
      }
      az += (targetAz - az) * 0.12;
      el += (targetEl - el) * 0.12;
      radius += (targetRadius - radius) * 0.12;
      const ce = Math.cos(el), se = Math.sin(el);
      camera.position.x = Math.cos(az) * ce * radius;
      camera.position.z = Math.sin(az) * ce * radius;
      camera.position.y = 0.6 + se * radius;
      camera.lookAt(0, 0.6, 0);

      renderer.render(scene, camera);
      requestAnimationFrame(frame);
    }
    frame();

    return {
      pause: () => { running = false; },
      play: () => { if (!running) { running = true; t0 = performance.now() - 0.001; frame(); } },
      resize,
      reset: () => { userInteracted = false; targetRadius = 5.8; targetEl = 0.38; },
      setSensoryBurst: (v) => { stats.sensoryBurst = v; },
      dispose: () => {
        running = false;
        renderer.dispose();
        if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
      },
      el: renderer.domElement,
    };
  }

  W.FlyScene = { create };
})();
