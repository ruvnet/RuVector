// Connectome OS — 3D graph scene
// Builds a connectome-like node/edge layout with SBM-style clustering,
// highlights a mincut boundary, pulses signal along edges.

(function () {
  const canvas = document.getElementById('three-canvas');
  if (!canvas) return;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 1000);
  camera.position.set(0, 0, 7.4);

  // Fog for atmospheric depth
  scene.fog = new THREE.Fog(0x050a08, 8, 18);

  // --- Build SBM-style graph -------------------------------------------
  const MODULES = 4;
  const NODES_PER_MOD = 52;
  const TOTAL = MODULES * NODES_PER_MOD;

  // Module centers on a shallow 3D plane
  const modCenters = [];
  for (let m = 0; m < MODULES; m++) {
    const a = (m / MODULES) * Math.PI * 2 + 0.3;
    modCenters.push(new THREE.Vector3(
      Math.cos(a) * 1.7,
      Math.sin(a) * 1.1,
      (m % 2 === 0 ? 0.25 : -0.25)
    ));
  }

  // Node positions + module index
  const nodePos = [];
  const nodeMod = [];
  for (let m = 0; m < MODULES; m++) {
    for (let i = 0; i < NODES_PER_MOD; i++) {
      const c = modCenters[m];
      // Gaussian cluster
      let x = c.x + gauss() * 0.55;
      let y = c.y + gauss() * 0.45;
      let z = c.z + gauss() * 0.3;
      nodePos.push(new THREE.Vector3(x, y, z));
      nodeMod.push(m);
    }
  }

  // Edges — high intra-module density, low inter-module.
  // Inter-module edges cross the "cut boundary"
  const edges = [];     // {a, b, boundary, w}
  const boundaryEdges = [];
  const rng = mulberry32(0xC0DEBEEF);

  for (let i = 0; i < TOTAL; i++) {
    for (let j = i + 1; j < TOTAL; j++) {
      const sameMod = nodeMod[i] === nodeMod[j];
      const pIntra = 0.028;
      const pInter = 0.0025;
      const p = sameMod ? pIntra : pInter;
      if (rng() < p) {
        const isBoundary = !sameMod;
        edges.push({ a: i, b: j, boundary: isBoundary, w: 0.3 + rng() * 0.6 });
      }
    }
  }

  // Which module pair is currently "selected" as cut boundary
  let CUT_FROM = 0, CUT_TO = 1;

  // --- Nodes as instanced points ----------------------------------------
  const nodeGeom = new THREE.BufferGeometry();
  const posArr = new Float32Array(TOTAL * 3);
  const colArr = new Float32Array(TOTAL * 3);
  const sizeArr = new Float32Array(TOTAL);
  for (let i = 0; i < TOTAL; i++) {
    posArr[i * 3] = nodePos[i].x;
    posArr[i * 3 + 1] = nodePos[i].y;
    posArr[i * 3 + 2] = nodePos[i].z;
    sizeArr[i] = 8 + Math.random() * 6;
  }
  nodeGeom.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
  nodeGeom.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
  nodeGeom.setAttribute('size', new THREE.BufferAttribute(sizeArr, 1));

  const nodeMat = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uPixelRatio: { value: renderer.getPixelRatio() }
    },
    vertexShader: `
      attribute float size;
      attribute vec3 color;
      varying vec3 vColor;
      uniform float uPixelRatio;
      void main() {
        vColor = color;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * uPixelRatio * (1.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      void main() {
        vec2 uv = gl_PointCoord - 0.5;
        float d = length(uv);
        if (d > 0.5) discard;
        float core = smoothstep(0.5, 0.0, d);
        float halo = smoothstep(0.5, 0.15, d) * 0.6;
        vec3 col = vColor * (core + halo);
        gl_FragColor = vec4(col, core);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  });

  const nodePoints = new THREE.Points(nodeGeom, nodeMat);
  scene.add(nodePoints);

  // --- Edges via LineSegments -------------------------------------------
  const edgePosArr = new Float32Array(edges.length * 2 * 3);
  const edgeColArr = new Float32Array(edges.length * 2 * 3);
  const edgeGeom = new THREE.BufferGeometry();
  edgeGeom.setAttribute('position', new THREE.BufferAttribute(edgePosArr, 3));
  edgeGeom.setAttribute('color', new THREE.BufferAttribute(edgeColArr, 3));
  const edgeMat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });
  const edgeLines = new THREE.LineSegments(edgeGeom, edgeMat);
  scene.add(edgeLines);

  // --- Signal pulses along boundary edges --------------------------------
  const MAX_PULSES = 120;
  const pulseGeom = new THREE.BufferGeometry();
  const pulsePos = new Float32Array(MAX_PULSES * 3);
  const pulseCol = new Float32Array(MAX_PULSES * 3);
  const pulseSize = new Float32Array(MAX_PULSES);
  pulseGeom.setAttribute('position', new THREE.BufferAttribute(pulsePos, 3));
  pulseGeom.setAttribute('color', new THREE.BufferAttribute(pulseCol, 3));
  pulseGeom.setAttribute('size', new THREE.BufferAttribute(pulseSize, 1));
  const pulseMat = nodeMat.clone();
  const pulsePoints = new THREE.Points(pulseGeom, pulseMat);
  scene.add(pulsePoints);

  const pulses = []; // {edgeIdx, t, speed}

  // --- Rim/glow ring ----------------------------------------------------
  const ringGeom = new THREE.RingGeometry(2.6, 2.62, 128);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x7CFF7A, transparent: true, opacity: 0.05, side: THREE.DoubleSide
  });
  const ring = new THREE.Mesh(ringGeom, ringMat);
  scene.add(ring);

  // --- Selected node highlight ------------------------------------------
  let hoverIdx = -1;
  const hoverDot = new THREE.Mesh(
    new THREE.SphereGeometry(0.05, 16, 16),
    new THREE.MeshBasicMaterial({ color: 0xB8FF3C, transparent: true, opacity: 0 })
  );
  scene.add(hoverDot);
  const hoverRing = new THREE.Mesh(
    new THREE.RingGeometry(0.09, 0.11, 32),
    new THREE.MeshBasicMaterial({ color: 0xB8FF3C, transparent: true, opacity: 0, side: THREE.DoubleSide })
  );
  scene.add(hoverRing);

  // --- Colors -----------------------------------------------------------
  const C_BASE   = new THREE.Color(0x4b5a52);
  const C_ACTIVE = new THREE.Color(0xAEB8B1);
  const C_CUT_A  = new THREE.Color(0xB8FF3C); // lime
  const C_CUT_B  = new THREE.Color(0x7CFF7A); // green
  const C_DIM    = new THREE.Color(0x2a3531);

  function recolor() {
    for (let i = 0; i < TOTAL; i++) {
      const m = nodeMod[i];
      let c;
      if (m === CUT_FROM) c = C_CUT_A;
      else if (m === CUT_TO) c = C_CUT_B;
      else c = C_BASE;
      colArr[i * 3] = c.r;
      colArr[i * 3 + 1] = c.g;
      colArr[i * 3 + 2] = c.b;
    }
    nodeGeom.attributes.color.needsUpdate = true;

    // Edges
    for (let e = 0; e < edges.length; e++) {
      const { a, b } = edges[e];
      const pa = nodePos[a], pb = nodePos[b];
      edgePosArr[e * 6]     = pa.x; edgePosArr[e * 6 + 1] = pa.y; edgePosArr[e * 6 + 2] = pa.z;
      edgePosArr[e * 6 + 3] = pb.x; edgePosArr[e * 6 + 4] = pb.y; edgePosArr[e * 6 + 5] = pb.z;

      const mA = nodeMod[a], mB = nodeMod[b];
      const isCutBoundary =
        (mA === CUT_FROM && mB === CUT_TO) ||
        (mA === CUT_TO && mB === CUT_FROM);

      let c;
      if (isCutBoundary) { c = C_CUT_A; edges[e].boundaryActive = true; }
      else if (mA === mB) { c = C_DIM; edges[e].boundaryActive = false; }
      else { c = C_DIM; edges[e].boundaryActive = false; }
      edgeColArr[e * 6]     = c.r * (isCutBoundary ? 1.0 : 0.4);
      edgeColArr[e * 6 + 1] = c.g * (isCutBoundary ? 1.0 : 0.4);
      edgeColArr[e * 6 + 2] = c.b * (isCutBoundary ? 1.0 : 0.4);
      edgeColArr[e * 6 + 3] = edgeColArr[e * 6];
      edgeColArr[e * 6 + 4] = edgeColArr[e * 6 + 1];
      edgeColArr[e * 6 + 5] = edgeColArr[e * 6 + 2];
    }
    edgeGeom.attributes.position.needsUpdate = true;
    edgeGeom.attributes.color.needsUpdate = true;

    // Cache indices of boundary edges for pulses
    boundaryEdges.length = 0;
    for (let e = 0; e < edges.length; e++) if (edges[e].boundaryActive) boundaryEdges.push(e);
  }

  recolor();

  // --- Interaction: drag-rotate + wheel-zoom + pinch ---------------------
  let targetRotX = -0.1, targetRotY = 0.0;
  let rotX = -0.1, rotY = 0.0;
  let targetZoom = 7.4;     // camera.position.z target
  const ZOOM_MIN = 3.2, ZOOM_MAX = 14;
  let zoom = 7.4;
  let mouseNDC = new THREE.Vector2(0, 0);
  let mouseClient = { x: 0, y: 0 };
  let autoDrift = true;

  const group = new THREE.Group();
  scene.add(group);
  group.add(nodePoints); group.add(edgeLines); group.add(pulsePoints);
  group.add(hoverDot); group.add(hoverRing); group.add(ring);

  // Drag state
  let dragging = false;
  let dragStart = null; // {x, y, rotX, rotY}
  let pointers = new Map(); // pointerId -> {x, y}
  let pinchPrevDist = 0;

  function getPos(e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top, rect: r };
  }

  canvas.style.cursor = 'grab';
  canvas.style.touchAction = 'none';

  let tapCandidate = null; // { x, y, t, idx }
  canvas.addEventListener('pointerdown', (e) => {
    const p = getPos(e);
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: p.x, y: p.y });
    if (pointers.size === 1) {
      tapCandidate = { x: p.x, y: p.y, t: performance.now() };
      dragging = true;
      autoDrift = false;
      dragStart = { x: p.x, y: p.y, rotX: targetRotX, rotY: targetRotY };
      canvas.style.cursor = 'grabbing';
      // Hide focus tip while dragging
      if (focusTip) focusTip.style.display = 'none';
      hoverDot.material.opacity = 0;
      hoverRing.material.opacity = 0;
    } else if (pointers.size === 2) {
      dragging = false;
      const pts = [...pointers.values()];
      pinchPrevDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
    }
  });

  canvas.addEventListener('pointermove', (e) => {
    const p = getPos(e);
    const r = p.rect;
    // Update ndc/client for hover
    mouseNDC.set(((p.x) / r.width) * 2 - 1, -((p.y) / r.height) * 2 + 1);
    mouseClient.x = p.x; mouseClient.y = p.y;

    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, { x: p.x, y: p.y });

    if (pointers.size === 2) {
      const pts = [...pointers.values()];
      const d = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      const delta = d - pinchPrevDist;
      targetZoom = clamp(targetZoom - delta * 0.015, ZOOM_MIN, ZOOM_MAX);
      pinchPrevDist = d;
      return;
    }

    if (dragging && dragStart) {
      const dx = p.x - dragStart.x;
      const dy = p.y - dragStart.y;
      // Scale rotation by size so feel is consistent
      const scale = 2.4 / Math.min(r.width, r.height);
      targetRotY = dragStart.rotY + dx * scale;
      targetRotX = dragStart.rotX + dy * scale;
      // Clamp pitch
      targetRotX = clamp(targetRotX, -1.2, 1.2);
    } else {
      doPick();
    }
  });

  function endPointer(e) {
    // Detect tap = small movement + short duration → pick
    if (tapCandidate && pointers.has(e.pointerId)) {
      const p = getPos(e);
      const dx = p.x - tapCandidate.x;
      const dy = p.y - tapCandidate.y;
      const dt = performance.now() - tapCandidate.t;
      if (dx * dx + dy * dy < 36 && dt < 500) {
        // Re-pick at pointer location
        const r = p.rect;
        mouseNDC.set((p.x / r.width) * 2 - 1, -(p.y / r.height) * 2 + 1);
        mouseClient.x = p.x; mouseClient.y = p.y;
        doPick();
        if (hoverIdx >= 0) {
          const types = ['Projection', 'Kenyon', 'Optic', 'Descending'];
          let deg = 0, bdeg = 0;
          for (let i = 0; i < edges.length; i++) {
            if (edges[i].a === hoverIdx || edges[i].b === hoverIdx) {
              deg++;
              if (edges[i].boundaryActive) bdeg++;
            }
          }
          window.dispatchEvent(new CustomEvent('neuron-pick', {
            detail: {
              idx: hoverIdx,
              module: nodeMod[hoverIdx],
              type: types[nodeMod[hoverIdx]],
              degree: deg,
              boundary: bdeg
            }
          }));
        }
      }
      tapCandidate = null;
    }
    if (pointers.has(e.pointerId)) pointers.delete(e.pointerId);
    if (pointers.size < 2) pinchPrevDist = 0;
    if (pointers.size === 0) {
      dragging = false;
      canvas.style.cursor = 'grab';
    }
  }
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);
  canvas.addEventListener('pointerleave', (e) => {
    if (!dragging && pointers.size === 0) {
      hoverIdx = -1;
      if (focusTip) focusTip.style.display = 'none';
      hoverDot.material.opacity = 0;
      hoverRing.material.opacity = 0;
    }
  });

  // Wheel zoom
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    autoDrift = false;
    const factor = e.deltaMode === 1 ? 0.4 : 0.0025; // line vs pixel
    targetZoom = clamp(targetZoom + e.deltaY * factor, ZOOM_MIN, ZOOM_MAX);
  }, { passive: false });
  canvas.addEventListener('dblclick', () => {
    targetRotX = -0.1; targetRotY = 0.0; targetZoom = 7.4; autoDrift = true;
  });

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  const raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 0.06;

  function doPick() {
    raycaster.setFromCamera(mouseNDC, camera);
    const hits = raycaster.intersectObject(nodePoints, false);
    hoverIdx = hits.length ? hits[0].index : -1;
    updateFocusTip();
  }

  const focusTip = document.getElementById('focus-tip');
  function updateFocusTip() {
    if (!focusTip) return;
    if (hoverIdx < 0) {
      focusTip.style.display = 'none';
      hoverDot.material.opacity = 0;
      hoverRing.material.opacity = 0;
      return;
    }
    const p = nodePos[hoverIdx];
    const m = nodeMod[hoverIdx];
    hoverDot.position.copy(p);
    hoverRing.position.copy(p);
    hoverDot.material.opacity = 0.9;
    hoverRing.material.opacity = 0.7;

    // Count degree
    let deg = 0, bdeg = 0;
    for (let e = 0; e < edges.length; e++) {
      if (edges[e].a === hoverIdx || edges[e].b === hoverIdx) {
        deg++;
        if (edges[e].boundaryActive) bdeg++;
      }
    }

    const types = ['Projection', 'Kenyon', 'Optic', 'Descending'];
    focusTip.innerHTML = `
      <div class="title">N-${String(hoverIdx).padStart(5,'0')}</div>
      <div class="kv"><span class="k">type</span><span>${types[m]}</span></div>
      <div class="kv"><span class="k">module</span><span>M${m}</span></div>
      <div class="kv"><span class="k">degree</span><span>${deg}</span></div>
      <div class="kv"><span class="k">boundary</span><span>${bdeg}</span></div>
      <div class="kv"><span class="k">firing</span><span style="color:var(--signal)">${(2 + Math.random() * 30).toFixed(1)} Hz</span></div>
    `;
    focusTip.style.display = 'block';
    const r = canvas.getBoundingClientRect();
    let x = mouseClient.x + 16, y = mouseClient.y + 16;
    if (x + 240 > r.width) x = mouseClient.x - 240;
    if (y + 160 > r.height) y = mouseClient.y - 160;
    focusTip.style.left = x + 'px';
    focusTip.style.top = y + 'px';
  }

  // --- Animation loop ----------------------------------------------------
  const clock = new THREE.Clock();
  let spawnAccum = 0;

  function step() {
    const dt = clock.getDelta();
    const t = clock.elapsedTime;

    // Auto drift — only when user hasn't interacted
    if (autoDrift) targetRotY += 0.00008;
    rotX += (targetRotX - rotX) * 0.12;
    rotY += (targetRotY - rotY) * 0.12;
    group.rotation.x = rotX;
    group.rotation.y = rotY;

    // Zoom
    zoom += (targetZoom - zoom) * 0.14;
    camera.position.z = zoom;

    nodeMat.uniforms.uTime.value = t;
    pulseMat.uniforms.uTime.value = t;

    // Spawn pulses along boundary edges
    spawnAccum += dt;
    const spawnEvery = 0.05;
    while (spawnAccum > spawnEvery) {
      spawnAccum -= spawnEvery;
      if (pulses.length < MAX_PULSES && boundaryEdges.length > 0) {
        const eIdx = boundaryEdges[(Math.random() * boundaryEdges.length) | 0];
        pulses.push({ edgeIdx: eIdx, t: 0, speed: 0.6 + Math.random() * 0.9 });
      }
    }

    // Advance pulses
    for (let i = pulses.length - 1; i >= 0; i--) {
      const p = pulses[i];
      p.t += dt * p.speed;
      if (p.t >= 1) { pulses.splice(i, 1); }
    }

    // Fill pulse buffers
    for (let i = 0; i < MAX_PULSES; i++) {
      if (i < pulses.length) {
        const p = pulses[i];
        const ed = edges[p.edgeIdx];
        const pa = nodePos[ed.a], pb = nodePos[ed.b];
        const x = pa.x + (pb.x - pa.x) * p.t;
        const y = pa.y + (pb.y - pa.y) * p.t;
        const z = pa.z + (pb.z - pa.z) * p.t;
        pulsePos[i * 3] = x; pulsePos[i * 3 + 1] = y; pulsePos[i * 3 + 2] = z;
        const fade = Math.sin(p.t * Math.PI);
        pulseCol[i * 3] = C_CUT_A.r * fade;
        pulseCol[i * 3 + 1] = C_CUT_A.g * fade;
        pulseCol[i * 3 + 2] = C_CUT_A.b * fade;
        pulseSize[i] = 14 * fade;
      } else {
        pulsePos[i * 3] = 0; pulsePos[i * 3 + 1] = 0; pulsePos[i * 3 + 2] = -100;
        pulseSize[i] = 0;
      }
    }
    pulseGeom.attributes.position.needsUpdate = true;
    pulseGeom.attributes.color.needsUpdate = true;
    pulseGeom.attributes.size.needsUpdate = true;

    // Ring breathe
    ring.material.opacity = 0.04 + Math.sin(t * 0.8) * 0.02;
    ring.rotation.z = t * 0.04;

    renderer.render(scene, camera);
    requestAnimationFrame(step);
  }

  function resize() {
    const r = canvas.getBoundingClientRect();
    renderer.setSize(r.width, r.height, false);
    camera.aspect = r.width / r.height;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();
  step();

  // --- Public API --------------------------------------------------------
  window.ConnectomeScene = {
    setCut(fromMod, toMod) {
      CUT_FROM = fromMod; CUT_TO = toMod;
      recolor();
    },
    reset() {
      targetRotX = -0.1; targetRotY = 0.0;
      targetZoom = 7.4;
      autoDrift = true;
    },
    stats() {
      return {
        nodes: TOTAL,
        edges: edges.length,
        boundary: boundaryEdges.length,
        modules: MODULES
      };
    },
    pulseBurst(count = 30) {
      for (let i = 0; i < count; i++) {
        if (pulses.length < MAX_PULSES && boundaryEdges.length > 0) {
          const eIdx = boundaryEdges[(Math.random() * boundaryEdges.length) | 0];
          pulses.push({ edgeIdx: eIdx, t: 0, speed: 1.4 + Math.random() });
        }
      }
    }
  };

  // Utils
  function gauss() {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function mulberry32(seed) {
    return function () {
      let t = (seed += 0x6D2B79F5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
})();
