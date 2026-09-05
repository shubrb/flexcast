// <ercot-3d> — the ERCOT 345 kV network as a dark instrument panel.
// Real CREZ/backbone routes as luminous tubes over a charcoal Texas plate (geometry from
// app/tx3d.js — no network fetch), stations as discs, glowing pools where a zone is tight,
// pulses along the lines that matter, real plants as small models. Live state arrives via
// setData(); clicks emit 'nodepick' (zone id).
(function () {
  // three.js ships with the app (offline-safe); unpkg is only a fallback.
  let threeP = null;
  const loadThree = () => (threeP = threeP ||
    import(new URL('./vendor/three.module.min.js', document.baseURI).href)
      .catch(() => import('https://unpkg.com/three@0.160.0/build/three.module.js')));
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  const IDLE = 0x6c6c78, SEL = 0x0a84ff, HOT = 0xff3b30, PAD = 0xa8a8b2, PAD_RIM = 0x3a3a44, WHITE = 0xf5f5f7;
  const title = s => (s || '').toLowerCase().replace(/\b[a-z]/g, c => c.toUpperCase());

  class Ercot3D extends HTMLElement {
    connectedCallback() {
      this.style.cssText = 'display:block;position:absolute;inset:0;width:100%;height:100%';
      this._yaw = -0.25; this._pitch = 0.95; this._dist = 62;
      this._labels = document.createElement('div');
      this._labels.style.cssText = 'position:absolute;inset:0;pointer-events:none;font-family:-apple-system,BlinkMacSystemFont,sans-serif';
      this._status = document.createElement('div');
      this._status.style.cssText = 'position:absolute;right:14px;bottom:12px;max-width:40%;font-family:"IBM Plex Mono",monospace;font-size:11px;color:rgba(245,245,247,0.42);pointer-events:none;text-align:right;line-height:1.5';
      this._status.textContent = 'Raising the 345 kV backbone…';
      this.appendChild(this._labels);
      this.appendChild(this._status);
      loadThree().then(T => this.boot(T)).catch(e => console.warn('three failed', e));
    }

    disconnectedCallback() {
      cancelAnimationFrame(this._raf);
      if (this._ro) this._ro.disconnect();
      if (this._renderer) this._renderer.dispose();
    }

    boot(T) {
      this.T = T;
      const scene = (this._scene = new T.Scene());
      this._cam = new T.PerspectiveCamera(36, 1, 0.3, 900);
      const rend = (this._renderer = new T.WebGLRenderer({ antialias: true, alpha: true }));
      rend.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      rend.domElement.style.cssText = 'display:block;width:100%;height:100%;cursor:grab;touch-action:none';
      this.insertBefore(rend.domElement, this._labels);

      scene.add(new T.HemisphereLight(0xffffff, 0x33333a, 2.8));
      const key = new T.DirectionalLight(0xffffff, 1.1);
      key.position.set(-50, 90, 50);
      scene.add(key);

      this._world = new T.Group();
      scene.add(this._world);
      this._ray = new T.Raycaster();
      this._pointer = new T.Vector2(-9, -9);
      this.bindInput(rend.domElement);

      this._ro = new ResizeObserver(() => this.resize());
      this._ro.observe(this);
      this.resize();
      this.buildWorld();

      this._clock = 0;
      const tick = () => {
        this._raf = requestAnimationFrame(tick);
        this._clock += 0.016;
        this.place();
        this.animate();
        this.hoverTest();
        rend.render(scene, this._cam);
        this.drawLabels();
      };
      tick();
    }

    resize() {
      const w = this.clientWidth, h = this.clientHeight;
      if (!w || !h || !this._renderer) return;
      if (this._cw === w && this._ch === h) return;
      this._cw = w; this._ch = h;
      this._renderer.setSize(w, h, false);
      this._cam.aspect = w / h;
      this._cam.updateProjectionMatrix();
      const fit = 60 * Math.max(1, 1.5 / (w / h));
      if (!this._userZoomed) this._dist = fit;
      this._fitDist = fit;
    }

    bindInput(el) {
      let px = 0, py = 0, moved = 0;
      el.addEventListener('pointerdown', e => {
        this._dragging = true; moved = 0; px = e.clientX; py = e.clientY;
        el.setPointerCapture(e.pointerId); el.style.cursor = 'grabbing';
      });
      el.addEventListener('pointermove', e => {
        const r = el.getBoundingClientRect();
        this._pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
        if (!this._dragging) return;
        const dx = e.clientX - px, dy = e.clientY - py;
        moved += Math.abs(dx) + Math.abs(dy);
        this._yaw -= dx * 0.006;
        this._pitch = clamp(this._pitch - dy * 0.005, 0.35, 1.45);
        px = e.clientX; py = e.clientY;
      });
      el.addEventListener('pointerup', () => {
        this._dragging = false; el.style.cursor = 'grab';
        if (moved < 6) this.clickTest();
      });
      el.addEventListener('pointerleave', () => this._pointer.set(-9, -9));
      el.addEventListener('dblclick', () => {
        this._yaw = -0.25; this._pitch = 0.95; this._dist = this._fitDist || 62;
      });
      el.addEventListener('wheel', e => {
        e.preventDefault();
        this._userZoomed = true;
        this._dist = clamp(this._dist + e.deltaY * 0.07, 18, 190);
      }, { passive: false });
    }

    place() {
      const c = this._cam, d = this._dist;
      const t = this._focus || new this.T.Vector3(0, 1, 0);
      c.position.set(t.x + Math.sin(this._yaw) * Math.cos(this._pitch) * d, Math.sin(this._pitch) * d + 1, t.z + Math.cos(this._yaw) * Math.cos(this._pitch) * d);
      c.lookAt(t);
    }

    // ---- construction --------------------------------------------------------

    buildWorld() {
      const M = window.ERCOT_NET, P = window.ERCOT_PHYS;
      if (!M || !P) return setTimeout(() => this.buildWorld(), 80);
      M.loadGeo().then(async geo => {
        this._geo = geo; this._M = M;
        this._zonePos = M.NODES.map(n => ({ id: n.id, name: n.name, cap: n.cap, x: geo.pos3[n.id][0], z: -geo.pos3[n.id][1] }));
        this.buildPlate(geo);
        this.buildZones();
        this.buildPlants(geo, P);

        let routes = null, stations = null, meta = null;
        if (window.ERCOT_HIFLD_LOCAL) {
          const d = window.ERCOT_HIFLD_LOCAL;
          routes = d.routes.map(cc => ({ pts: cc.map(c => { const p = geo.project3(c[0], c[1]); return [p[0], -p[1]]; }), name: '345 kV line' }));
          stations = d.stations.map(st => { const p = geo.project3(st.lon, st.lat); return { name: st.n, x: p[0], z: -p[1], lines: st.l }; });
          meta = { source: 'HIFLD 345 kV', routes: routes.length, stations: stations.length, real: true };
        }
        try {
          if (!routes && window.ERCOT_HIFLD) {
            const d = await window.ERCOT_HIFLD.load();
            routes = d.routes.map(r => ({ pts: r.coords.map(c => { const p = geo.project3(c[0], c[1]); return [p[0], -p[1]]; }), name: r.sub1 && r.sub2 ? title(r.sub1) + ' – ' + title(r.sub2) : '345 kV line' }));
            stations = d.stations.map(s => { const p = geo.project3(s.lon, s.lat); return { name: s.name, x: p[0], z: -p[1], lines: s.lines }; });
            meta = { source: 'HIFLD 345 kV', routes: routes.length, stations: stations.length, real: true };
          }
        } catch (e) { console.warn('HIFLD unavailable, using curated backbone', e); }
        if (!routes) {
          const byId = {}; P.SUBSTATIONS.forEach(s => { byId[s.id] = s; });
          const pt = s => { const p = geo.project3(s.lon, s.lat); return [p[0], -p[1]]; };
          routes = P.CORRIDORS.map(([a, b]) => ({ pts: [pt(byId[a]), pt(byId[b])], name: byId[a].name + ' – ' + byId[b].name }));
          stations = P.SUBSTATIONS.map(s => { const p = pt(s); return { name: s.name, x: p[0], z: p[1], lines: 3 }; });
          meta = { source: 'curated 345 kV backbone', routes: routes.length, stations: stations.length, real: false };
        }
        this.buildTubes(routes);
        this.buildStations(stations);
        this._meta = meta;
        this._status.textContent = meta.source + ' · ' + meta.routes.toLocaleString() + ' lines · ' + meta.stations.toLocaleString() + ' stations';
        this._ready = true;
        this.dispatchEvent(new CustomEvent('gridready', { detail: meta, bubbles: true }));
        if (this._pending) this.setData(this._pending);
      });
    }

    zoneAt(x, z) {
      let best = null, bd = 1e9;
      this._zonePos.forEach(n => { const d = (n.x - x) * (n.x - x) + (n.z - z) * (n.z - z); if (d < bd) { bd = d; best = n.id; } });
      return best;
    }

    buildPlate(geo) {
      const T = this.T;
      const shape = new T.Shape(geo.outline.map(p => new T.Vector2(p[0], -p[1])));
      const g = new T.ExtrudeGeometry(shape, { depth: 1.4, bevelEnabled: false });
      g.rotateX(-Math.PI / 2);
      const plate = new T.Mesh(g, new T.MeshLambertMaterial({ color: 0x202027 }));
      plate.position.y = -1.4;
      this._world.add(plate);
      const edge = new T.LineSegments(new T.EdgesGeometry(g, 24), new T.LineBasicMaterial({ color: 0x5a5a66 }));
      edge.position.y = -1.39;
      this._world.add(edge);
      const seg = [];
      geo.counties.forEach(ring => { for (let i = 1; i < ring.length; i++) seg.push(ring[i - 1][0], 0.01, -ring[i - 1][1], ring[i][0], 0.01, -ring[i][1]); });
      const cg = new T.BufferGeometry();
      cg.setAttribute('position', new T.Float32BufferAttribute(seg, 3));
      this._world.add(new T.LineSegments(cg, new T.LineBasicMaterial({ color: 0x34343d })));
    }

    // glowing pools per zone: red where power is tight, blue for the selected zone
    buildZones() {
      const T = this.T;
      this._glows = {};
      this._zonePos.forEach(n => {
        const r = 3 + Math.sqrt(n.cap) / 80;
        const mk = (rad, color) => new T.Mesh(new T.CircleGeometry(rad, 56), new T.MeshBasicMaterial({ color, transparent: true, opacity: 0, depthWrite: false, blending: T.AdditiveBlending }));
        const outer = mk(r * 1.9, HOT), inner = mk(r, HOT);
        const ring = new T.Mesh(new T.RingGeometry(r - 0.2, r, 72), new T.MeshBasicMaterial({ color: SEL, transparent: true, opacity: 0, depthWrite: false }));
        [outer, inner, ring].forEach((m, i) => { m.rotation.x = -Math.PI / 2; m.position.set(n.x, 0.02 + i * 0.005, n.z); this._world.add(m); });
        this._glows[n.id] = { outer, inner, ring, r };
      });
    }

    // one luminous tube per real route, merged per zone so live state can recolor it
    buildTubes(routes) {
      const T = this.T;
      this._routes = routes.map(r => {
        const pts = r.pts.map(p => new T.Vector3(p[0], 0.28, p[1]));
        const path = new T.CurvePath();
        for (let i = 1; i < pts.length; i++) path.add(new T.LineCurve3(pts[i - 1], pts[i]));
        const mid = pts[Math.floor(pts.length / 2)];
        return { path, zone: this.zoneAt(mid.x, mid.z), name: r.name, len: path.getLength() };
      });
      const byZone = {};
      this._routes.forEach(r => { (byZone[r.zone] = byZone[r.zone] || []).push(r); });
      this._tubes = {};
      Object.keys(byZone).forEach(zone => {
        const pos = [], nor = [], idx = [];
        let off = 0;
        byZone[zone].forEach(r => {
          const g = new T.TubeGeometry(r.path, Math.max(6, Math.round(r.len * 3)), 0.11, 6, false);
          const p = g.getAttribute('position').array, n = g.getAttribute('normal').array, ix = g.getIndex().array;
          for (let i = 0; i < p.length; i++) { pos.push(p[i]); nor.push(n[i]); }
          for (let i = 0; i < ix.length; i++) idx.push(ix[i] + off);
          off += p.length / 3;
          g.dispose();
        });
        const geo = new T.BufferGeometry();
        geo.setAttribute('position', new T.Float32BufferAttribute(pos, 3));
        geo.setAttribute('normal', new T.Float32BufferAttribute(nor, 3));
        geo.setIndex(idx);
        const mesh = new T.Mesh(geo, new T.MeshBasicMaterial({ color: IDLE }));
        this._world.add(mesh);
        this._tubes[zone] = mesh;
      });
      this._pulses = [];
      this._pulseGeo = new T.SphereGeometry(0.26, 10, 8);
    }

    // stations as discs sized by how many 345 kV lines land there
    buildStations(stations) {
      const T = this.T;
      this._stations = stations.map(s => ({ ...s, zone: this.zoneAt(s.x, s.z), scale: 0.95 + Math.min(1.7, Math.sqrt(s.lines) * 0.45) }));
      const n = this._stations.length, o = new T.Object3D();
      const mat = new T.MeshBasicMaterial({ color: 0xffffff });
      const rims = new T.InstancedMesh(new T.CylinderGeometry(0.5, 0.5, 0.08, 28), mat, n);
      const pads = new T.InstancedMesh(new T.CylinderGeometry(0.38, 0.38, 0.16, 28), mat, n);
      this._stations.forEach((s, i) => {
        const k = s.scale;
        o.position.set(s.x, 0.04 * k, s.z); o.rotation.set(0, 0, 0); o.scale.set(k, k, k); o.updateMatrix();
        rims.setMatrixAt(i, o.matrix); rims.setColorAt(i, new T.Color(PAD_RIM));
        o.position.set(s.x, 0.1 * k, s.z); o.updateMatrix();
        pads.setMatrixAt(i, o.matrix); pads.setColorAt(i, new T.Color(PAD));
      });
      [rims, pads].forEach(m => { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; this._world.add(m); });
      this._pads = pads; this._rims = rims;
      this._hits = [pads].concat(this._plantHits || []);
    }

    buildPlants(geo, P) {
      const T = this.T;
      this._plants = []; this._spinners = []; this._plantHits = [];
      const grey = new T.MeshLambertMaterial({ color: 0x8a8a94 });
      const white = new T.MeshLambertMaterial({ color: WHITE });
      const glass = new T.MeshLambertMaterial({ color: 0x3b6fb6 });
      const hidden = () => new T.MeshBasicMaterial({ visible: false });
      P.PLANTS.forEach(pl => {
        const p = geo.project3(pl.lon, pl.lat);
        const g = new T.Group();
        g.position.set(p[0], 0, -p[1]);
        let hit;
        if (pl.kind === 'wind') {
          for (let i = 0; i < 7; i++) {
            const t = new T.Group();
            t.position.set((i % 4) * 0.7 - 1.05, 0, Math.floor(i / 4) * 0.75);
            const mast = new T.Mesh(new T.CylinderGeometry(0.025, 0.045, 1.05, 6), white); mast.position.y = 0.52; t.add(mast);
            const nac = new T.Mesh(new T.BoxGeometry(0.18, 0.08, 0.08), white); nac.position.y = 1.05; t.add(nac);
            const rotor = new T.Group(); rotor.position.set(0.09, 1.05, 0);
            for (let b = 0; b < 3; b++) {
              const blade = new T.Mesh(new T.BoxGeometry(0.018, 0.52, 0.06), white); blade.position.y = 0.26;
              const arm = new T.Group(); arm.rotation.z = (b / 3) * Math.PI * 2; arm.add(blade); rotor.add(arm);
            }
            rotor.rotation.y = Math.PI / 2; t.add(rotor);
            this._spinners.push({ rotor, speed: 1.6 + (i % 3) * 0.25 });
            g.add(t);
          }
          hit = new T.Mesh(new T.BoxGeometry(3, 1.3, 1.6), hidden()); hit.position.set(0, 0.65, 0.35);
        } else if (pl.kind === 'solar') {
          for (let r = 0; r < 3; r++) for (let c = 0; c < 5; c++) {
            const panel = new T.Mesh(new T.BoxGeometry(0.42, 0.03, 0.22), glass);
            panel.position.set(c * 0.5 - 1, 0.14, r * 0.36 - 0.36); panel.rotation.x = -0.5; g.add(panel);
          }
          hit = new T.Mesh(new T.BoxGeometry(2.7, 0.5, 1.4), hidden()); hit.position.y = 0.25;
        } else if (pl.kind === 'nuclear') {
          [-0.45, 0.45].forEach(x => {
            const base = new T.Mesh(new T.CylinderGeometry(0.33, 0.36, 0.36, 20), white); base.position.set(x, 0.18, 0); g.add(base);
            const dome = new T.Mesh(new T.SphereGeometry(0.33, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2), white); dome.position.set(x, 0.36, 0); g.add(dome);
          });
          hit = new T.Mesh(new T.BoxGeometry(1.8, 0.8, 1), hidden()); hit.position.y = 0.4;
        } else if (pl.kind === 'gas') {
          const hall = new T.Mesh(new T.BoxGeometry(1.25, 0.36, 0.72), grey); hall.position.y = 0.18; g.add(hall);
          [-0.38, 0.04, 0.46].forEach(x => { const stack = new T.Mesh(new T.CylinderGeometry(0.065, 0.08, 0.98, 10), white); stack.position.set(x, 0.49, -0.47); g.add(stack); });
          hit = new T.Mesh(new T.BoxGeometry(1.6, 1.05, 1.4), hidden()); hit.position.y = 0.52;
        } else {
          [-0.62, 0, 0.62].forEach(z => { const hall = new T.Mesh(new T.BoxGeometry(1.9, 0.3, 0.42), white); hall.position.set(0, 0.15, z); g.add(hall); });
          for (let i = 0; i < 8; i++) { const cab = new T.Mesh(new T.BoxGeometry(0.17, 0.14, 0.28), grey); cab.position.set(i * 0.22 - 0.77, 0.07, 1.12); g.add(cab); }
          hit = new T.Mesh(new T.BoxGeometry(2.2, 0.8, 2.2), hidden()); hit.position.set(0, 0.4, 0.3);
        }
        hit.userData = { plant: true, zone: pl.zone, name: pl.name, sub: pl.kind === 'campus' ? pl.mw + ' MW AI campus' : pl.mw.toLocaleString() + ' MW ' + pl.kind };
        g.add(hit);
        this._plantHits.push(hit);
        this._world.add(g);
        this._plants.push({ zone: pl.zone, kind: pl.kind, name: pl.name, group: g, pos: [p[0], -p[1]] });
      });
      this._hits = this._plantHits.slice();
    }

    // ---- live state ----------------------------------------------------------

    setData(data) {
      if (!this._ready) { this._pending = data; return; }
      const T = this.T;
      this._data = data;
      const byZone = {};
      (data.nodes || []).forEach(n => { byZone[n.id] = n; });
      this._byZone = byZone;
      const selNode = (data.nodes || []).find(n => n.selected);
      this._sel = selNode ? selNode.id : null;
      const tone = z => { const n = byZone[z]; if (!n) return 'idle'; return n.selected ? 'sel' : (n.hot ? 'hot' : 'idle'); };
      this._tone = tone;
      const hex = k => k === 'hot' ? HOT : (k === 'sel' ? SEL : IDLE);

      Object.keys(this._tubes).forEach(z => this._tubes[z].material.color.setHex(hex(tone(z))));
      const hotC = new T.Color(HOT), selC = new T.Color(SEL), padC = new T.Color(PAD), rimC = new T.Color(PAD_RIM);
      this._stations.forEach((s, i) => {
        const k = tone(s.zone);
        this._pads.setColorAt(i, k === 'hot' ? hotC : (k === 'sel' ? selC : padC));
        this._rims.setColorAt(i, k === 'idle' ? rimC : (k === 'hot' ? hotC : selC));
      });
      this._pads.instanceColor.needsUpdate = true;
      this._rims.instanceColor.needsUpdate = true;

      Object.keys(this._glows).forEach(z => {
        const k = tone(z), g = this._glows[z];
        const c = k === 'sel' ? SEL : HOT;
        g.outer.material.color.setHex(c); g.inner.material.color.setHex(c);
        g.outer.material.opacity = k === 'idle' ? 0 : 0.14;
        g.inner.material.opacity = k === 'idle' ? 0 : 0.28;
        g.ring.material.color.setHex(c);
        g.ring.material.opacity = k === 'idle' ? 0 : 0.95;
      });
      this._plants.forEach(p => p.group.scale.setScalar(p.zone === this._sel ? 1.2 : 1));
      this.buildPulses();
    }

    // pulses travel along the lines of the selected zone and of every tight zone
    buildPulses() {
      const T = this.T;
      (this._pulses || []).forEach(p => this._world.remove(p.mesh));
      this._pulses = [];
      const budget = 120;
      const active = this._routes.filter(r => this._tone(r.zone) !== 'idle').sort((a, b) => b.len - a.len).slice(0, budget);
      const per = Math.max(1, Math.min(3, Math.floor(budget / Math.max(1, active.length))));
      active.forEach(r => {
        const k = this._tone(r.zone);
        for (let i = 0; i < per; i++) {
          const mesh = new T.Mesh(this._pulseGeo, new T.MeshBasicMaterial({ color: k === 'hot' ? 0xffb3ad : 0xbfe0ff }));
          this._world.add(mesh);
          this._pulses.push({ mesh, r, off: (i + Math.random()) / per, speed: (k === 'hot' ? 0.16 : 0.1) / Math.max(1, r.len / 6) });
        }
      });
    }

    animate() {
      if (this._spinners) this._spinners.forEach(s => { s.rotor.rotation.x = this._clock * s.speed; });
      if (this._pulses) this._pulses.forEach(p => {
        const t = (this._clock * p.speed + p.off) % 1;
        const pt = p.r.path.getPoint(t);
        p.mesh.position.set(pt.x, 0.34, pt.z);
      });
      if (this._glows && this._tone) {
        const breathe = 0.5 + 0.5 * Math.sin(this._clock * 2.2);
        Object.keys(this._glows).forEach(z => {
          const k = this._tone(z), g = this._glows[z];
          if (k === 'idle') return;
          g.outer.material.opacity = 0.1 + 0.1 * breathe;
          if (k === 'sel') g.ring.material.opacity = 0.6 + 0.4 * breathe;
        });
      }
    }

    // ---- interaction ---------------------------------------------------------

    pick() {
      if (!this._hits || !this._hits.length) return null;
      this._ray.setFromCamera(this._pointer, this._cam);
      const hit = this._ray.intersectObjects(this._hits, false)[0];
      if (!hit) return null;
      if (hit.object === this._pads) {
        const s = this._stations[hit.instanceId];
        return s ? { name: s.name, sub: '345 kV station · ' + s.lines + (s.lines === 1 ? ' line' : ' lines'), zone: s.zone, point: hit.point } : null;
      }
      return { ...hit.object.userData, point: hit.point };
    }

    hoverTest() {
      const h = this.pick();
      const name = h ? h.name : null;
      if (name !== this._hoverName) {
        this._hoverName = name;
        this._hover = h;
        if (this._renderer) this._renderer.domElement.style.cursor = this._dragging ? 'grabbing' : (name ? 'pointer' : 'grab');
      } else if (h) this._hover.point = h.point;
    }

    clickTest() {
      const h = this.pick();
      if (h) this.dispatchEvent(new CustomEvent('nodepick', { detail: h.zone, bubbles: true }));
    }

    drawLabels() {
      if (!this._ready) return;
      const T = this.T, w = this.clientWidth, h = this.clientHeight;
      const items = [];
      const campus = this._plants.find(p => p.kind === 'campus');
      if (campus) items.push({ name: 'Abilene AI Campus', sub: (this._byZone && this._byZone.west ? this._byZone.west.priceLabel : '500 MW') + (this._sel === 'west' ? ' · selected' : ''), pos: campus.pos, y: 1.4, bg: this._sel === 'west' ? '#0a84ff' : '#f5f5f7', fg: this._sel === 'west' ? '#fff' : '#1d1d1f' });
      if (this._hover && this._hover.name !== 'Abilene AI Campus') items.push({ name: this._hover.name, sub: this._hover.sub, world: this._hover.point, bg: '#f5f5f7', fg: '#1d1d1f' });
      if (this._tone) {
        this._zonePos.forEach(n => {
          const k = this._tone(n.id);
          if (k === 'idle' || n.id === 'west') return;
          const z = this._byZone[n.id];
          items.push({ name: n.name, sub: (z ? z.priceLabel : '') + (k === 'hot' ? ' · tight' : ' · selected'), pos: [n.x, n.z], y: 0.4, bg: k === 'hot' ? '#ff3b30' : '#0a84ff', fg: '#fff' });
        });
      }
      if (this._labels.childElementCount !== items.length) {
        this._labels.innerHTML = items.map(() =>
          '<div style="position:absolute;transform:translate(-50%,-100%);white-space:nowrap;text-align:center;padding:6px 11px 6px;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,0.35)"><div style="font-size:13px;font-weight:600;letter-spacing:-0.01em;line-height:1.2"></div><div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;line-height:1.3;margin-top:2px;opacity:0.8"></div></div>'
        ).join('');
      }
      const v = new T.Vector3(), placed = [];
      items.forEach((it, i) => {
        const el = this._labels.children[i];
        if (!el) return;
        if (it.world) v.copy(it.world).add(new T.Vector3(0, 0.9, 0)); else v.set(it.pos[0], it.y, it.pos[1]);
        v.project(this._cam);
        const lx = clamp((v.x * 0.5 + 0.5) * w, 60, w - 60), ly = clamp((-v.y * 0.5 + 0.5) * h, 26, h - 10);
        const bw = Math.max(100, (it.name.length + (it.sub || '').length * 0.6) * 7.4), bh = 40;
        const collides = placed.some(p => Math.abs(p.x - lx) < (p.w + bw) / 2 && Math.abs(p.y - ly) < bh);
        if (!collides) placed.push({ x: lx, y: ly, w: bw });
        el.style.left = lx.toFixed(1) + 'px';
        el.style.top = ly.toFixed(1) + 'px';
        el.style.opacity = v.z > 1 || collides ? 0 : 1;
        el.style.background = it.bg;
        el.style.color = it.fg;
        el.children[0].textContent = it.name;
        el.children[1].textContent = it.sub || '';
      });
    }
  }

  if (!customElements.get('ercot-3d')) customElements.define('ercot-3d', Ercot3D);
})();
