from __future__ import annotations

import json
from typing import Any

import streamlit.components.v1 as components


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _frame_payload(frame: dict[str, Any], *, debug: bool = False) -> dict[str, Any]:
    zone_temps = frame.get("zone_temps_c")
    if not isinstance(zone_temps, list) or not zone_temps:
        fallback_temp = _safe_float(frame.get("temp_c", 960.0), 960.0)
        zone_temps = [fallback_temp]

    safe_zone_temps = [_safe_float(temp, 960.0) for temp in zone_temps]
    safe_zone_temps = [temp for temp in safe_zone_temps if temp == temp]
    if not safe_zone_temps:
        safe_zone_temps = [960.0]
    while len(safe_zone_temps) < 3:
        safe_zone_temps.append(safe_zone_temps[-1])
    safe_zone_temps = safe_zone_temps[:3]

    tick = int(_safe_float(frame.get("tick", 0), 0.0))
    time_hr = _safe_float(frame.get("time_hr", 0.0), 0.0)
    recenter_token = max(0, int(_safe_float(frame.get("recenter_token", 0), 0.0)))

    conversion = _clamp(_safe_float(frame.get("conversion", 0.0), 0.0), 0.0, 1.0)
    confidence = _clamp(_safe_float(frame.get("confidence", 0.85), 0.85), 0.0, 1.0)
    health = _clamp(
        _safe_float(
            frame.get("health", 1.0 - _safe_float(frame.get("fouling_index", 0.0), 0.0)), 1.0
        ),
        0.0,
        1.0,
    )
    methane_fraction = _clamp(
        _safe_float(
            frame.get("methane_fraction", frame.get("ch4_fraction", 1.0 - conversion)),
            1.0 - conversion,
        ),
        0.0,
        1.0,
    )
    hydrogen_fraction = _clamp(
        _safe_float(
            frame.get("hydrogen_fraction", frame.get("h2_fraction", conversion)),
            conversion,
        ),
        0.0,
        1.0,
    )

    return {
        "frame_seq": f"{tick}:{time_hr:.5f}:{recenter_token}",
        "tick": tick,
        "time_hr": time_hr,
        "temp_c": _safe_float(frame.get("temp_c", safe_zone_temps[1]), safe_zone_temps[1]),
        "zone_temps_c": safe_zone_temps,
        "conversion": conversion,
        "h2_rate": max(0.0, _safe_float(frame.get("h2_rate", 0.0), 0.0)),
        "carbon_rate": max(0.0, _safe_float(frame.get("carbon_rate", 0.0), 0.0)),
        "fouling_index": max(0.0, _safe_float(frame.get("fouling_index", 0.0), 0.0)),
        "deltaP_kpa": max(0.0, _safe_float(frame.get("deltaP_kpa", 0.0), 0.0)),
        "power_kw": max(0.0, _safe_float(frame.get("power_kw", 0.0), 0.0)),
        "q_required_kw": max(
            0.0,
            _safe_float(frame.get("q_required_kw", frame.get("power_kw", 0.0)), 0.0),
        ),
        "is_out_of_distribution": bool(frame.get("is_out_of_distribution", False)),
        "ood_score": max(0.0, _safe_float(frame.get("ood_score", 0.0), 0.0)),
        "confidence": confidence,
        "uncertainty_std": max(0.0, _safe_float(frame.get("uncertainty_std", 0.0), 0.0)),
        "methane_fraction": methane_fraction,
        "hydrogen_fraction": hydrogen_fraction,
        "health": health,
        "thermal_view": bool(frame.get("thermal_view", False)),
        "is_playing": bool(frame.get("is_playing", False)),
        "validation_rmse": _safe_optional_float(frame.get("validation_rmse")),
        "validation_mae": _safe_optional_float(frame.get("validation_mae")),
        "recenter_token": recenter_token,
        "debug_mode": bool(frame.get("debug_mode", debug)),
    }


def render_reactor_3d(
    frame: dict[str, Any],
    height: int = 520,
    *,
    debug: bool = False,
) -> None:
    payload_json = json.dumps(_frame_payload(frame, debug=debug))

    update_channel_html = f"""
<script>
try {{
  const frame = {payload_json};
  window.localStorage.setItem("synora.reactor3d.frame.v1", JSON.stringify(frame));
  if (window.parent) {{
    window.parent.__synoraReactor3DLatestFrame = frame;
  }}
}} catch (err) {{
  // ignore channel update errors
}}
</script>
"""
    components.html(update_channel_html, height=0, scrolling=False)

    html_template = """
<div id="synora-reactor3d-root" style="width:100%;height:__HEIGHT__px;position:relative;background:#050a12;border:1px solid rgba(71,95,125,0.55);overflow:hidden;"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.159.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.159.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function () {
  const CONTAINER_ID = "synora-reactor3d-root";
  const ENGINE_KEY = "__synoraReactor3DEngine";
  const CAMERA_STORAGE_KEY = "synora.reactor3d.camera.v6";
  const FRAME_STORAGE_KEY = "synora.reactor3d.frame.v1";
  const container = document.getElementById(CONTAINER_ID);
  if (!container) return;

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }
  function fract(v) {
    return v - Math.floor(v);
  }
  function lerp(a, b, t) {
    return a + ((b - a) * t);
  }
  function seeded(i, salt) {
    return fract(Math.sin((i + 1.0) * 12.9898 + salt * 78.233) * 43758.5453123);
  }
  function hasFiniteNumber(v) {
    return typeof v === "number" && Number.isFinite(v);
  }

  function readLatestFrame() {
    try {
      const raw = window.localStorage.getItem(FRAME_STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") {
          return parsed;
        }
      }
    } catch (err) {
      // ignore storage parse errors
    }
    try {
      const host = window.parent || window;
      if (host.__synoraReactor3DLatestFrame && typeof host.__synoraReactor3DLatestFrame === "object") {
        return host.__synoraReactor3DLatestFrame;
      }
    } catch (err) {
      // ignore parent access errors
    }
    return {};
  }

  function makeThermalTexture(three, zoneTemps, confidence) {
    const canvas = document.createElement("canvas");
    canvas.width = 192;
    canvas.height = 1024;
    const ctx = canvas.getContext("2d");
    const grad = ctx.createLinearGradient(0, canvas.height, 0, 0);

    function colorForTemp(temp) {
      const x = clamp((temp - 820.0) / 420.0, 0.0, 1.0);
      const cool = [13, 84, 185];
      const warm = [250, 125, 29];
      const white = [255, 241, 191];
      if (x < 0.70) {
        const t = x / 0.70;
        return "rgb(" +
          Math.round(cool[0] + (warm[0] - cool[0]) * t) + "," +
          Math.round(cool[1] + (warm[1] - cool[1]) * t) + "," +
          Math.round(cool[2] + (warm[2] - cool[2]) * t) + ")";
      }
      const t = (x - 0.70) / 0.30;
      return "rgb(" +
        Math.round(warm[0] + (white[0] - warm[0]) * t) + "," +
        Math.round(warm[1] + (white[1] - warm[1]) * t) + "," +
        Math.round(warm[2] + (white[2] - warm[2]) * t) + ")";
    }

    grad.addColorStop(0.00, colorForTemp(zoneTemps[0]));
    grad.addColorStop(0.42, colorForTemp((zoneTemps[0] + zoneTemps[1]) * 0.5));
    grad.addColorStop(0.74, colorForTemp((zoneTemps[1] + zoneTemps[2]) * 0.5));
    grad.addColorStop(1.00, colorForTemp(zoneTemps[2]));
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const haze = clamp((1.0 - confidence) * 0.18, 0.0, 0.22);
    ctx.fillStyle = "rgba(16, 18, 20, " + haze.toFixed(3) + ")";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const tex = new three.CanvasTexture(canvas);
    tex.needsUpdate = true;
    tex.minFilter = three.LinearFilter;
    tex.magFilter = three.LinearFilter;
    tex.wrapS = three.RepeatWrapping;
    tex.wrapT = three.RepeatWrapping;
    return tex;
  }

  function createEngine(three, mountEl) {
    const width = Math.max(380, mountEl.clientWidth || 960);
    const height = Math.max(320, mountEl.clientHeight || 760);
    const Y_SHIFT = 1.86;

    mountEl.style.pointerEvents = "auto";
    mountEl.style.userSelect = "none";

    const scene = new three.Scene();
    scene.background = new three.Color(0x040914);
    scene.fog = new three.Fog(0x040914, 17.0, 120.0);

    const camera = new three.PerspectiveCamera(34, width / height, 0.1, 200);
    camera.position.set(9.0, 6.2, 11.8);

    const renderer = new three.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height, false);
    renderer.outputColorSpace = three.SRGBColorSpace;
    renderer.toneMapping = three.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";
    renderer.domElement.style.pointerEvents = "auto";
    renderer.domElement.style.touchAction = "none";
    mountEl.appendChild(renderer.domElement);

    const OrbitControlsCtor = three.OrbitControls || window.OrbitControls;
    const controls = OrbitControlsCtor ? new OrbitControlsCtor(camera, renderer.domElement) : null;
    if (controls) {
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.enableRotate = true;
      controls.enablePan = true;
      controls.enableZoom = true;
      controls.screenSpacePanning = true;
      controls.maxPolarAngle = Math.PI * 0.495;
      controls.mouseButtons = {
        LEFT: three.MOUSE.ROTATE,
        MIDDLE: three.MOUSE.DOLLY,
        RIGHT: three.MOUSE.PAN,
      };
    }

    const ambient = new three.AmbientLight(0x9eb8d6, 0.46);
    scene.add(ambient);
    const keyLight = new three.DirectionalLight(0xc5e7ff, 1.26);
    keyLight.position.set(10, 15, 9);
    scene.add(keyLight);
    const rimLight = new three.DirectionalLight(0x2a78d3, 0.95);
    rimLight.position.set(-10, 2, -8);
    scene.add(rimLight);

    const root = new three.Group();
    scene.add(root);

    function addTube(points, radius, material) {
      const curve = new three.CatmullRomCurve3(points);
      const tube = new three.Mesh(
        new three.TubeGeometry(curve, 64, radius, 14, false),
        material
      );
      root.add(tube);
      return { curve, tube };
    }

    const skid = new three.Mesh(
      new three.BoxGeometry(13.2, 0.18, 7.2),
      new three.MeshStandardMaterial({
        color: 0x121b2b,
        metalness: 0.64,
        roughness: 0.36,
      })
    );
    skid.position.set(0.45, 0.0, 0.0);
    root.add(skid);

    const reactorShadow = new three.Mesh(
      new three.CircleGeometry(2.25, 48),
      new three.MeshBasicMaterial({ color: 0x010203, transparent: true, opacity: 0.24 })
    );
    reactorShadow.rotation.x = -Math.PI / 2.0;
    reactorShadow.position.set(2.2, 0.10, 0.0);
    root.add(reactorShadow);

    const solidsShadow = new three.Mesh(
      new three.CircleGeometry(1.55, 40),
      new three.MeshBasicMaterial({ color: 0x010203, transparent: true, opacity: 0.20 })
    );
    solidsShadow.rotation.x = -Math.PI / 2.0;
    solidsShadow.position.set(-2.85, 0.10, -0.18);
    root.add(solidsShadow);

    const reactorGroup = new three.Group();
    reactorGroup.position.set(2.2, 3.96, 0.0);
    root.add(reactorGroup);

    const shellMat = new three.MeshPhysicalMaterial({
      color: 0xa7c7e8,
      metalness: 0.22,
      roughness: 0.05,
      transmission: 0.97,
      thickness: 0.28,
      ior: 1.50,
      transparent: true,
      opacity: 0.30,
      envMapIntensity: 1.08,
      attenuationDistance: 120.0,
      attenuationColor: new three.Color(0xeaf6ff),
      clearcoat: 0.82,
      clearcoatRoughness: 0.16,
      side: three.DoubleSide,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: 1.0,
      polygonOffsetUnits: 1.0,
    });

    const vesselShell = new three.Mesh(
      new three.CylinderGeometry(0.90, 0.86, 7.2, 96, 1, true),
      shellMat
    );
    reactorGroup.add(vesselShell);

    const vesselHead = new three.Mesh(
      new three.SphereGeometry(0.90, 64, 32, 0.0, Math.PI * 2.0, 0.0, Math.PI / 2.0),
      shellMat.clone()
    );
    vesselHead.position.y = 3.62;
    reactorGroup.add(vesselHead);

    const shellEdge = new three.LineSegments(
      new three.EdgesGeometry(new three.CylinderGeometry(0.90, 0.86, 7.2, 36)),
      new three.LineBasicMaterial({ color: 0xb7d6ff, transparent: true, opacity: 0.14 })
    );
    reactorGroup.add(shellEdge);

    const skirt = new three.Mesh(
      new three.CylinderGeometry(0.98, 0.72, 0.62, 56),
      new three.MeshStandardMaterial({ color: 0x607086, metalness: 0.90, roughness: 0.27 })
    );
    skirt.position.y = -3.25;
    reactorGroup.add(skirt);

    const tubeMat = new three.MeshPhysicalMaterial({
      color: 0x74b8e6,
      metalness: 0.72,
      roughness: 0.22,
      transmission: 0.52,
      thickness: 0.10,
      transparent: true,
      opacity: 0.44,
      clearcoat: 0.34,
      clearcoatRoughness: 0.16,
      depthWrite: false,
      side: three.DoubleSide,
    });

    const nozzleMat = new three.MeshStandardMaterial({
      color: 0x7e8fa7,
      metalness: 0.86,
      roughness: 0.30,
    });

    const topOutlet = addTube(
      [
        new three.Vector3(2.95, 5.66 + Y_SHIFT, 0.0),
        new three.Vector3(3.55, 5.66 + Y_SHIFT, 0.0),
        new three.Vector3(4.25, 5.48 + Y_SHIFT, 0.0),
        new three.Vector3(5.95, 5.48 + Y_SHIFT, 0.0),
      ],
      0.24,
      tubeMat.clone()
    );
    const feedPipe = addTube(
      [
        new three.Vector3(5.95, -1.12 + Y_SHIFT, 0.0),
        new three.Vector3(4.95, -1.12 + Y_SHIFT, 0.0),
        new three.Vector3(4.02, -1.02 + Y_SHIFT, 0.0),
        new three.Vector3(3.00, -1.02 + Y_SHIFT, 0.0),
      ],
      0.28,
      tubeMat.clone()
    );
    const solidsDuct = addTube(
      [
        new three.Vector3(1.40, 3.42 + Y_SHIFT, 0.0),
        new three.Vector3(0.45, 3.44 + Y_SHIFT, 0.0),
        new three.Vector3(-0.55, 3.24 + Y_SHIFT, -0.06),
        new three.Vector3(-1.84, 3.08 + Y_SHIFT, -0.14),
      ],
      0.22,
      tubeMat.clone()
    );
    addTube(
      [
        new three.Vector3(-1.88, -0.96 + Y_SHIFT, 1.18),
        new three.Vector3(-0.60, -0.96 + Y_SHIFT, 1.18),
        new three.Vector3(0.75, -0.92 + Y_SHIFT, 0.86),
        new three.Vector3(1.95, -1.00 + Y_SHIFT, 0.46),
        new three.Vector3(2.62, -1.02 + Y_SHIFT, 0.14),
        new three.Vector3(3.00, -1.02 + Y_SHIFT, 0.00),
      ],
      0.20,
      tubeMat.clone()
    );
    addTube(
      [
        new three.Vector3(-1.88, -0.96 + Y_SHIFT, -1.08),
        new three.Vector3(-0.55, -0.96 + Y_SHIFT, -1.08),
        new three.Vector3(0.80, -0.92 + Y_SHIFT, -0.76),
        new three.Vector3(1.95, -0.98 + Y_SHIFT, -0.48),
        new three.Vector3(2.58, -1.01 + Y_SHIFT, -0.10),
        new three.Vector3(2.86, -1.02 + Y_SHIFT, -0.03),
        new three.Vector3(3.00, -1.02 + Y_SHIFT, 0.00),
      ],
      0.20,
      tubeMat.clone()
    );

    function addFlange(position, radius) {
      const flange = new three.Mesh(
        new three.TorusGeometry(radius, 0.018, 12, 28),
        new three.MeshStandardMaterial({ color: 0x8fa1ba, metalness: 0.9, roughness: 0.3 })
      );
      flange.position.copy(position);
      flange.rotation.y = Math.PI / 2.0;
      root.add(flange);
    }
    addFlange(new three.Vector3(2.95, 5.66 + Y_SHIFT, 0.0), 0.24);
    addFlange(new three.Vector3(3.00, -1.02 + Y_SHIFT, 0.0), 0.28);
    addFlange(new three.Vector3(1.40, 3.42 + Y_SHIFT, 0.0), 0.22);
    addFlange(new three.Vector3(-1.84, 3.08 + Y_SHIFT, -0.14), 0.22);
    addFlange(new three.Vector3(-1.88, -0.96 + Y_SHIFT, 1.18), 0.18);
    addFlange(new three.Vector3(-1.88, -0.96 + Y_SHIFT, -1.08), 0.18);
    addFlange(new three.Vector3(2.62, -1.02 + Y_SHIFT, 0.14), 0.18);
    addFlange(new three.Vector3(3.00, -1.02 + Y_SHIFT, 0.00), 0.18);

    const solidsTower = new three.Group();
    solidsTower.position.set(-2.85, 1.34, -0.18);
    root.add(solidsTower);

    const solidsLowerBody = new three.Mesh(
      new three.BoxGeometry(1.96, 2.72, 1.90),
      new three.MeshStandardMaterial({
        color: 0x2f3f52,
        metalness: 0.54,
        roughness: 0.56,
      })
    );
    solidsLowerBody.position.y = 0.62;
    solidsTower.add(solidsLowerBody);

    const solidsBridge = new three.Mesh(
      new three.BoxGeometry(1.24, 0.82, 1.18),
      new three.MeshStandardMaterial({
        color: 0x3f5268,
        metalness: 0.52,
        roughness: 0.50,
      })
    );
    solidsBridge.position.y = 2.15;
    solidsTower.add(solidsBridge);

    const solidsUpperBody = new three.Mesh(
      new three.BoxGeometry(1.70, 1.40, 1.56),
      new three.MeshStandardMaterial({
        color: 0x43586f,
        metalness: 0.54,
        roughness: 0.46,
      })
    );
    solidsUpperBody.position.y = 3.18;
    solidsTower.add(solidsUpperBody);

    for (let i = 0; i < 6; i += 1) {
      const rib = new three.Mesh(
        new three.BoxGeometry(1.98, 0.09, 0.10),
        new three.MeshStandardMaterial({ color: 0x5f738b, metalness: 0.48, roughness: 0.58 })
      );
      rib.position.set(0.0, (-0.60 + i * 0.52), 1.00);
      solidsTower.add(rib);
    }

    const solidsPipeColor = tubeMat.clone();
    solidsPipeColor.opacity = 0.42;
    addTube(
      [
        new three.Vector3(-3.92, -0.40 + Y_SHIFT, 0.84),
        new three.Vector3(-3.38, -0.40 + Y_SHIFT, 0.84),
        new three.Vector3(-2.62, -0.12 + Y_SHIFT, 0.60),
        new three.Vector3(-2.10, 0.62 + Y_SHIFT, 0.28),
      ],
      0.12,
      solidsPipeColor.clone()
    );
    addTube(
      [
        new three.Vector3(-3.95, -0.28 + Y_SHIFT, 0.26),
        new three.Vector3(-3.40, -0.28 + Y_SHIFT, 0.26),
        new three.Vector3(-2.70, 0.14 + Y_SHIFT, 0.18),
        new three.Vector3(-2.15, 0.88 + Y_SHIFT, 0.10),
      ],
      0.12,
      solidsPipeColor.clone()
    );
    addTube(
      [
        new three.Vector3(-3.95, -0.24 + Y_SHIFT, -0.40),
        new three.Vector3(-3.42, -0.24 + Y_SHIFT, -0.40),
        new three.Vector3(-2.72, 0.08 + Y_SHIFT, -0.22),
        new three.Vector3(-2.15, 0.82 + Y_SHIFT, -0.12),
      ],
      0.12,
      solidsPipeColor.clone()
    );
    addTube(
      [
        new three.Vector3(-3.72, 0.66 + Y_SHIFT, -0.82),
        new three.Vector3(-3.72, 1.82 + Y_SHIFT, -0.82),
        new three.Vector3(-3.34, 2.54 + Y_SHIFT, -0.62),
        new three.Vector3(-3.02, 3.18 + Y_SHIFT, -0.38),
      ],
      0.11,
      solidsPipeColor.clone()
    );
    addTube(
      [
        new three.Vector3(-3.28, 0.66 + Y_SHIFT, 0.90),
        new three.Vector3(-3.28, 1.90 + Y_SHIFT, 0.90),
        new three.Vector3(-3.00, 2.64 + Y_SHIFT, 0.72),
        new three.Vector3(-2.72, 3.24 + Y_SHIFT, 0.46),
      ],
      0.11,
      solidsPipeColor.clone()
    );
    addTube(
      [
        new three.Vector3(-2.42, 0.74 + Y_SHIFT, 0.18),
        new three.Vector3(-2.22, 1.82 + Y_SHIFT, 0.18),
        new three.Vector3(-2.18, 2.58 + Y_SHIFT, 0.12),
        new three.Vector3(-2.26, 3.22 + Y_SHIFT, 0.08),
      ],
      0.11,
      solidsPipeColor.clone()
    );

    const thermalUniforms = {
      uZoneTemps: { value: new three.Vector3(920.0, 980.0, 1040.0) },
      uPower: { value: 0.0 },
      uHealth: { value: 1.0 },
      uThermalView: { value: 0.0 },
      uTime: { value: 0.0 },
      uOpacity: { value: 0.74 },
    };
    const thermalMat = new three.ShaderMaterial({
      uniforms: thermalUniforms,
      transparent: true,
      depthWrite: false,
      side: three.DoubleSide,
      blending: three.AdditiveBlending,
      vertexShader: `
varying float vY;
varying vec3 vPos;
void main() {
  vY = position.y;
  vPos = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`,
      fragmentShader: `
uniform vec3 uZoneTemps;
uniform float uPower;
uniform float uHealth;
uniform float uThermalView;
uniform float uTime;
uniform float uOpacity;
varying float vY;
varying vec3 vPos;

float tempNorm(float t) {
  return clamp((t - 820.0) / 420.0, 0.0, 1.0);
}

vec3 gradientColor(float t) {
  vec3 deepBlue = vec3(0.05, 0.17, 0.64);
  vec3 cyan = vec3(0.02, 0.86, 1.00);
  vec3 yellow = vec3(1.00, 0.84, 0.16);
  vec3 red = vec3(1.00, 0.30, 0.08);
  if (t < 0.38) {
    return mix(deepBlue, cyan, t / 0.38);
  }
  if (t < 0.74) {
    return mix(cyan, yellow, (t - 0.38) / 0.36);
  }
  return mix(yellow, red, (t - 0.74) / 0.26);
}

void main() {
  float yNorm = clamp((vY + 3.25) / 6.5, 0.0, 1.0);
  float t0 = tempNorm(uZoneTemps.x);
  float t1 = tempNorm(uZoneTemps.y);
  float t2 = tempNorm(uZoneTemps.z);
  float lower = mix(t0, t1, smoothstep(0.0, 0.52, yNorm));
  float upper = mix(t1, t2, smoothstep(0.48, 1.0, yNorm));
  float tBlend = mix(lower, upper, yNorm);

  float shimmer = 0.84 + 0.18 * sin((vPos.y * 5.6) + (vPos.x * 12.5) + uTime * 2.1);
  vec3 col = gradientColor(tBlend) * shimmer;

  float thermalModeBoost = mix(1.30, 2.10, clamp(uThermalView, 0.0, 1.0));
  float powerBoost = mix(1.04, 2.20, clamp(uPower, 0.0, 1.4));
  float healthFactor = mix(0.72, 1.0, clamp(uHealth, 0.0, 1.0));
  float alpha = uOpacity * (0.72 + (0.28 * tBlend));

  gl_FragColor = vec4(col * thermalModeBoost * powerBoost * healthFactor, alpha);
}
`,
    });
    const thermalVolume = new three.Mesh(
      new three.CylinderGeometry(0.78, 0.72, 6.35, 88, 1, true),
      thermalMat
    );
    reactorGroup.add(thermalVolume);

    const moltenMat = new three.MeshStandardMaterial({
      color: 0xffa12e,
      emissive: 0xff5717,
      emissiveIntensity: 1.46,
      transparent: true,
      opacity: 0.70,
      roughness: 0.18,
      metalness: 0.02,
      depthWrite: false,
    });
    const moltenPool = new three.Mesh(new three.CylinderGeometry(0.62, 0.58, 1.2, 56), moltenMat);
    moltenPool.position.y = -2.52;
    reactorGroup.add(moltenPool);

    const moltenSurface = new three.Mesh(
      new three.CircleGeometry(0.60, 56),
      new three.MeshBasicMaterial({
        color: 0xffe17f,
        transparent: true,
        opacity: 0.56,
        blending: three.AdditiveBlending,
        depthWrite: false,
      })
    );
    moltenSurface.rotation.x = -Math.PI / 2.0;
    moltenSurface.position.y = -1.92;
    reactorGroup.add(moltenSurface);

    const plumeCore = new three.Mesh(
      new three.ConeGeometry(0.26, 3.8, 32, 1, true),
      new three.MeshBasicMaterial({
        color: 0xffe0a1,
        transparent: true,
        opacity: 0.30,
        blending: three.AdditiveBlending,
        depthWrite: false,
      })
    );
    plumeCore.position.y = 0.46;
    reactorGroup.add(plumeCore);

    const sootLayer = new three.Mesh(
      new three.CylinderGeometry(0.84, 0.80, 6.6, 68, 1, true),
      new three.MeshStandardMaterial({
        color: 0x201812,
        metalness: 0.16,
        roughness: 0.93,
        transparent: true,
        opacity: 0.12,
        side: three.DoubleSide,
      })
    );
    reactorGroup.add(sootLayer);

    const carbonPocket = new three.Mesh(
      new three.SphereGeometry(0.34, 26, 18),
      new three.MeshStandardMaterial({
        color: 0x231912,
        roughness: 0.95,
        metalness: 0.06,
        transparent: true,
        opacity: 0.16,
      })
    );
    carbonPocket.position.set(0.18, 2.55, 0.15);
    reactorGroup.add(carbonPocket);

    const restrictionRing = new three.Mesh(
      new three.TorusGeometry(0.58, 0.05, 16, 64),
      new three.MeshStandardMaterial({
        color: 0xb86332,
        emissive: 0x612c10,
        emissiveIntensity: 0.0,
        metalness: 0.43,
        roughness: 0.56,
        transparent: true,
        opacity: 0.0,
      })
    );
    restrictionRing.position.y = -0.02;
    restrictionRing.rotation.x = Math.PI / 2.0;
    reactorGroup.add(restrictionRing);

    function createPointCloud(count, color, size, opacity, additive) {
      const geometry = new three.BufferGeometry();
      const positions = new Float32Array(count * 3);
      const seeds = new Float32Array(count * 3);
      for (let i = 0; i < count; i += 1) {
        seeds[i * 3] = seeded(i, 0.13);
        seeds[(i * 3) + 1] = seeded(i, 0.71);
        seeds[(i * 3) + 2] = seeded(i, 1.91);
        positions[i * 3] = 2.2;
        positions[(i * 3) + 1] = Y_SHIFT + 0.8;
        positions[(i * 3) + 2] = 0.0;
      }
      geometry.setAttribute("position", new three.BufferAttribute(positions, 3));
      const material = new three.PointsMaterial({
        color,
        size,
        transparent: true,
        opacity,
        depthWrite: false,
        blending: additive ? three.AdditiveBlending : three.NormalBlending,
      });
      const points = new three.Points(geometry, material);
      root.add(points);
      return { geometry, positions, seeds, material, count };
    }

    const ch4Particles = createPointCloud(110, 0x45d9ff, 0.204, 0.86, true);
    const h2Particles = createPointCloud(100, 0xbef7ff, 0.196, 0.92, true);
    const plumeParticles = createPointCloud(280, 0xffe1aa, 0.264, 0.58, true);
    const carbonParticles = createPointCloud(150, 0x251912, 0.112, 0.70, false);
    const reactionZoneParticles = [
      createPointCloud(110, 0x3f93ff, 0.224, 0.42, true),
      createPointCloud(130, 0xffdc66, 0.248, 0.48, true),
      createPointCloud(140, 0xff5d26, 0.272, 0.56, true),
    ];

    function createOverlay(styleBuilder) {
      const element = document.createElement("div");
      element.style.position = "absolute";
      element.style.pointerEvents = "none";
      styleBuilder(element.style);
      mountEl.appendChild(element);
      return element;
    }

    const badge = createOverlay((style) => {
      style.top = "10px";
      style.right = "10px";
      style.padding = "6px 10px";
      style.fontFamily = "Inter, sans-serif";
      style.fontSize = "11px";
      style.letterSpacing = "1.2px";
      style.textTransform = "uppercase";
      style.background = "rgba(10, 16, 26, 0.86)";
      style.border = "1px solid rgba(96,126,159,0.70)";
      style.color = "#d3e6ff";
    });

    const trustAnchor = createOverlay((style) => {
      style.top = "62px";
      style.right = "10px";
      style.padding = "5px 8px";
      style.fontFamily = "Inter, sans-serif";
      style.fontSize = "9px";
      style.letterSpacing = "1.1px";
      style.textTransform = "uppercase";
      style.lineHeight = "1.35";
      style.background = "rgba(10, 16, 26, 0.66)";
      style.border = "1px solid rgba(80, 104, 132, 0.45)";
      style.color = "#8ea6c0";
    });

    const legend = createOverlay((style) => {
      style.left = "10px";
      style.bottom = "8px";
      style.padding = "4px 8px";
      style.fontFamily = "Inter, sans-serif";
      style.fontSize = "10px";
      style.letterSpacing = "1.1px";
      style.background = "rgba(9, 13, 21, 0.68)";
      style.border = "1px solid rgba(78,105,136,0.58)";
      style.color = "#93b7d8";
    });

    const bboxError = createOverlay((style) => {
      style.left = "50%";
      style.top = "10px";
      style.transform = "translateX(-50%)";
      style.padding = "5px 9px";
      style.fontFamily = "Inter, sans-serif";
      style.fontSize = "10px";
      style.letterSpacing = "0.8px";
      style.textTransform = "uppercase";
      style.background = "rgba(44, 20, 10, 0.88)";
      style.border = "1px solid rgba(197, 96, 62, 0.78)";
      style.color = "#ffceb8";
      style.display = "none";
    });

    const debugPanel = createOverlay((style) => {
      style.left = "10px";
      style.top = "10px";
      style.padding = "4px 8px";
      style.fontFamily = "Inter, sans-serif";
      style.fontSize = "10px";
      style.letterSpacing = "0.7px";
      style.background = "rgba(8, 14, 22, 0.70)";
      style.border = "1px solid rgba(86, 110, 141, 0.45)";
      style.color = "#98b3ce";
      style.display = "none";
    });

    function makeLabel(text) {
      return createOverlay((style) => {
        style.padding = "2px 6px";
        style.fontFamily = "Inter, sans-serif";
        style.fontSize = "10px";
        style.letterSpacing = "0.9px";
        style.background = "rgba(8, 14, 22, 0.58)";
        style.border = "1px solid rgba(86, 110, 141, 0.45)";
        style.color = "#b4cee8";
        style.display = "none";
      });
    }

    const labels = [
      { point: new three.Vector3(5.30, 6.98 + Y_SHIFT, 0.0), el: makeLabel("Hydrogen outlet") },
      { point: new three.Vector3(5.12, 0.42 + Y_SHIFT, 0.0), el: makeLabel("Natural gas feed") },
      { point: new three.Vector3(-3.35, 3.15 + Y_SHIFT, 0.68), el: makeLabel("Solids handling") },
      { point: new three.Vector3(1.46, 2.52 + Y_SHIFT, 0.38), el: makeLabel("Solid carbon") },
      { point: new three.Vector3(0.92, 1.70 + Y_SHIFT, -0.36), el: makeLabel("High-T multi-phase reactor") },
      { point: new three.Vector3(1.20, -1.85 + Y_SHIFT, 0.22), el: makeLabel("High-temperature liquid") },
    ];
    labels[0].el.textContent = "Hydrogen outlet";
    labels[1].el.textContent = "Natural gas feed";
    labels[2].el.textContent = "Solids handling";
    labels[3].el.textContent = "Solid carbon";
    labels[4].el.textContent = "High-T multi-phase reactor";
    labels[5].el.textContent = "High-temperature liquid";

    root.position.set(0.0, 0.0, 0.0);
    root.rotation.set(0.0, 0.0, 0.0);
    root.scale.set(1.0, 1.0, 1.0);

    const bbox = new three.Box3();
    const bboxSize = new three.Vector3();
    const bboxCenter = new three.Vector3();

    function updateRootBounds() {
      bbox.setFromObject(root);
      if (bbox.isEmpty()) {
        return false;
      }
      bbox.getSize(bboxSize);
      bbox.getCenter(bboxCenter);
      const extent = bboxSize.length();
      if (!Number.isFinite(extent) || extent < 0.5) {
        return false;
      }
      return true;
    }

    function fitCamera(reason) {
      if (!updateRootBounds()) {
        bboxError.style.display = "block";
        bboxError.textContent = "Renderer geometry bounds invalid - skip fit (" + reason + ")";
        return false;
      }
      bboxError.style.display = "none";
      let maxDim = Math.max(bboxSize.x, bboxSize.y, bboxSize.z);
      if (!Number.isFinite(maxDim) || maxDim <= 0.0) {
        maxDim = 10.0;
      }
      if (maxDim > 40.0) {
        bboxError.style.display = "block";
        bboxError.textContent = "Renderer bounds too large - using fallback fit";
        maxDim = 12.0;
        bboxCenter.set(0.45, 4.0, 0.0);
      }
      const fovRad = camera.fov * (Math.PI / 180.0);
      const fitHeightDistance = (maxDim * 0.72) / Math.tan(Math.max(0.1, fovRad * 0.5));
      const fitWidthDistance = fitHeightDistance / Math.max(camera.aspect, 0.65);
      let distance = Math.max(fitHeightDistance, fitWidthDistance) * 1.25;
      if (!Number.isFinite(distance) || distance <= 0.0) {
        distance = 12.0;
      }
      distance = clamp(distance, 8.0, 28.0);
      const direction = new three.Vector3(1.08, 0.62, 1.20).normalize();
      camera.position.copy(bboxCenter).addScaledVector(direction, distance);
      camera.near = Math.max(0.05, distance / 240.0);
      camera.far = Math.max(100.0, distance * 16.0);
      camera.updateProjectionMatrix();
      if (controls) {
        controls.target.copy(bboxCenter);
        controls.minDistance = Math.max(2.0, maxDim * 0.28);
        controls.maxDistance = Math.max(26.0, maxDim * 8.5);
        controls.update();
      } else {
        camera.lookAt(bboxCenter);
      }
      saveCamera();
      return true;
    }

    function saveCamera() {
      if (!controls) return;
      try {
        window.localStorage.setItem(
          CAMERA_STORAGE_KEY,
          JSON.stringify({
            pos: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
            target: { x: controls.target.x, y: controls.target.y, z: controls.target.z },
          })
        );
      } catch (err) {
        // ignore storage errors
      }
    }

    function restoreCameraIfValid() {
      if (!controls) {
        return false;
      }
      if (!updateRootBounds()) {
        return false;
      }
      try {
        const raw = window.localStorage.getItem(CAMERA_STORAGE_KEY);
        if (!raw) {
          return false;
        }
        const parsed = JSON.parse(raw);
        if (!parsed || !parsed.pos || !parsed.target) {
          return false;
        }
        camera.position.set(parsed.pos.x, parsed.pos.y, parsed.pos.z);
        controls.target.set(parsed.target.x, parsed.target.y, parsed.target.z);
        const dist = camera.position.distanceTo(controls.target);
        const span = Math.max(1.0, bboxSize.length());
        if (
          !Number.isFinite(dist) ||
          dist < 2.0 ||
          dist > Math.min(45.0, span * 5.0)
        ) {
          return false;
        }
        controls.update();
        return true;
      } catch (err) {
        return false;
      }
    }

    let saveCameraTimer = null;
    if (controls) {
      controls.addEventListener("change", function () {
        if (saveCameraTimer) {
          window.clearTimeout(saveCameraTimer);
        }
        saveCameraTimer = window.setTimeout(saveCamera, 150);
      });
    }

    const state = {
      frameSeq: "",
      lastRecenterToken: 0,
      debug: false,
      tick: 0,
      conversion: 0.0,
      h2Norm: 0.0,
      carbonNorm: 0.0,
      fouling: 0.0,
      dpNorm: 0.0,
      powerNorm: 0.0,
      health: 1.0,
      confidence: 0.85,
      methaneFraction: 1.0,
      hydrogenFraction: 0.0,
      oodScore: 0.0,
      isOOD: false,
      zoneTemps: [920.0, 980.0, 1040.0],
      validationRmse: null,
      validationMae: null,
      thermalView: false,
      isPlaying: false,
      flowTimeSec: 0.0,
      lastAnimTimeSec: null,
    };
    const target = { ...state };

    function setFrame(nextFrame) {
      if (!nextFrame || typeof nextFrame !== "object") {
        return;
      }
      target.frameSeq = String(nextFrame.frame_seq || "");
      target.tick = Number(nextFrame.tick || 0.0);
      const temps = Array.isArray(nextFrame.zone_temps_c) ? nextFrame.zone_temps_c : [920.0, 980.0, 1040.0];
      target.zoneTemps = [Number(temps[0] || 920.0), Number(temps[1] || temps[0] || 980.0), Number(temps[2] || temps[1] || temps[0] || 1040.0)];
      target.conversion = clamp(Number(nextFrame.conversion || 0.0), 0.0, 1.0);
      target.h2Norm = clamp(Number(nextFrame.h2_rate || 0.0) / 13.0, 0.0, 1.4);
      target.carbonNorm = clamp(Number(nextFrame.carbon_rate || 0.0) / 35.0, 0.0, 1.4);
      target.fouling = clamp(Number(nextFrame.fouling_index || 0.0), 0.0, 1.8);
      target.dpNorm = clamp(Number(nextFrame.deltaP_kpa || 0.0) / 14.0, 0.0, 1.25);
      target.powerNorm = clamp(Number(nextFrame.power_kw || nextFrame.q_required_kw || 0.0) / 500.0, 0.0, 1.8);
      target.health = clamp(Number(nextFrame.health || (1.0 - target.fouling)), 0.0, 1.0);
      target.confidence = clamp(Number(nextFrame.confidence || 0.82), 0.0, 1.0);
      target.methaneFraction = clamp(Number(nextFrame.methane_fraction || (1.0 - target.conversion)), 0.0, 1.0);
      target.hydrogenFraction = clamp(Number(nextFrame.hydrogen_fraction || target.conversion), 0.0, 1.0);
      target.oodScore = Math.max(0.0, Number(nextFrame.ood_score || 0.0));
      target.isOOD = !!nextFrame.is_out_of_distribution;
      target.validationRmse = hasFiniteNumber(nextFrame.validation_rmse) ? Number(nextFrame.validation_rmse) : null;
      target.validationMae = hasFiniteNumber(nextFrame.validation_mae) ? Number(nextFrame.validation_mae) : null;
      target.thermalView = !!nextFrame.thermal_view;
      target.isPlaying = !!nextFrame.is_playing;
      target.debug = !!nextFrame.debug_mode;

      const recenterToken = Math.max(0, Number(nextFrame.recenter_token || 0));
      if (recenterToken > state.lastRecenterToken) {
        state.lastRecenterToken = recenterToken;
        fitCamera("recenter");
      }
    }

    function updateVisualState(timeSec) {
      state.frameSeq = target.frameSeq;
      state.tick = target.tick;
      state.conversion = lerp(state.conversion, target.conversion, 0.08);
      state.h2Norm = lerp(state.h2Norm, target.h2Norm, 0.08);
      state.carbonNorm = lerp(state.carbonNorm, target.carbonNorm, 0.08);
      state.fouling = lerp(state.fouling, target.fouling, 0.08);
      state.dpNorm = lerp(state.dpNorm, target.dpNorm, 0.08);
      state.powerNorm = lerp(state.powerNorm, target.powerNorm, 0.08);
      state.health = lerp(state.health, target.health, 0.08);
      state.confidence = lerp(state.confidence, target.confidence, 0.08);
      state.methaneFraction = lerp(state.methaneFraction, target.methaneFraction, 0.08);
      state.hydrogenFraction = lerp(state.hydrogenFraction, target.hydrogenFraction, 0.08);
      state.oodScore = lerp(state.oodScore, target.oodScore, 0.08);
      state.isOOD = target.isOOD;
      state.zoneTemps = target.zoneTemps.slice();
      state.validationRmse = target.validationRmse;
      state.validationMae = target.validationMae;
      state.thermalView = target.thermalView;
      state.isPlaying = target.isPlaying;
      state.debug = target.debug;

      if (!state.isPlaying) {
        state.flowTimeSec = state.tick * 0.028;
      }

      const avgTemp = (state.zoneTemps[0] + state.zoneTemps[1] + state.zoneTemps[2]) / 3.0;
      const tempNorm = clamp((avgTemp - 820.0) / 420.0, 0.0, 1.0);
      const thermalViewBlend = state.thermalView ? 1.0 : 0.0;

      const moltenScaleY = clamp(0.72 + (state.conversion * 0.95), 0.72, 1.55);
      moltenPool.scale.set(1.0, moltenScaleY, 1.0);
      moltenPool.position.y = -2.60 + ((moltenScaleY - 1.0) * 0.56);
      moltenSurface.position.y = -2.00 + ((moltenScaleY - 1.0) * 0.58);
      moltenPool.material.opacity = clamp(
        0.26 + state.powerNorm * 0.32 + state.conversion * 0.20 + (thermalViewBlend * 0.12),
        0.20,
        0.92
      );
      moltenPool.material.emissiveIntensity = clamp(
        (0.62 + state.powerNorm * 1.00) * clamp(0.45 + (0.55 * state.health), 0.35, 1.0),
        0.45,
        2.20
      );
      moltenPool.material.color.setHSL(0.09 - tempNorm * 0.03, 0.90, 0.46 + tempNorm * 0.10);
      moltenSurface.material.opacity = clamp(0.18 + state.powerNorm * 0.34 + (thermalViewBlend * 0.12), 0.16, 0.88);

      plumeCore.material.opacity = clamp(0.10 + state.conversion * 0.48 + state.powerNorm * 0.14, 0.10, 0.76);
      plumeCore.material.color.setHSL(0.11 - tempNorm * 0.05, 0.90, 0.58);
      plumeCore.scale.y = 0.66 + state.conversion * 0.46 + state.powerNorm * 0.22;

      sootLayer.material.opacity = clamp(
        0.05 + Math.min(1.0, state.fouling) * 0.22 + state.carbonNorm * 0.08,
        0.05,
        0.34
      );
      sootLayer.scale.set(1.0 + Math.min(1.0, state.fouling) * 0.12, 1.0, 1.0 + Math.min(1.0, state.fouling) * 0.12);
      carbonPocket.material.opacity = clamp(
        0.08 + Math.min(1.0, state.fouling) * 0.22 + state.carbonNorm * 0.08,
        0.08,
        0.48
      );
      carbonPocket.scale.setScalar(0.72 + Math.min(1.0, state.fouling) * 0.78 + state.carbonNorm * 0.18);

      restrictionRing.material.opacity = clamp((state.dpNorm - 0.10) * 0.80, 0.0, 0.70);
      restrictionRing.material.emissiveIntensity = clamp(state.dpNorm * 0.85, 0.0, 0.95);
      restrictionRing.scale.setScalar(1.0 + state.dpNorm * 0.09);

      const compression = clamp(state.dpNorm * 0.06, 0.0, 0.08);
      reactorGroup.scale.set(1.0 - compression, 1.0, 1.0 - compression * 0.78);
      shellMat.opacity = 0.23;
      thermalUniforms.uZoneTemps.value.set(state.zoneTemps[0], state.zoneTemps[1], state.zoneTemps[2]);
      thermalUniforms.uPower.value = clamp(state.powerNorm, 0.0, 1.4);
      thermalUniforms.uHealth.value = clamp(state.health, 0.0, 1.0);
      thermalUniforms.uThermalView.value = thermalViewBlend;
      thermalUniforms.uTime.value = timeSec;
      thermalUniforms.uOpacity.value = state.thermalView ? 1.05 : 0.90;
      thermalMat.needsUpdate = false;

      if (state.isOOD) {
        badge.style.borderColor = "rgba(255,121,72,0.92)";
        badge.style.color = "#ffd2bb";
        badge.style.boxShadow = "0 0 14px rgba(255,103,45,0.38)";
        badge.innerHTML = "OOD " + state.oodScore.toFixed(2) + "<br/><span style='font-size:10px;color:#ffcfb4'>CONF " + Math.round(state.confidence * 100) + "%</span>";
      } else {
        badge.style.borderColor = "rgba(96,126,159,0.70)";
        badge.style.color = "#d3e6ff";
        badge.style.boxShadow = "0 0 10px rgba(96,150,228,0.20)";
        badge.innerHTML = "CONF " + Math.round(state.confidence * 100) + "%<br/><span style='font-size:10px;color:#9ac0e5'>OOD LOW</span>";
      }

      if (state.validationRmse !== null && state.validationMae !== null) {
        trustAnchor.innerHTML =
          "PHYSICS-CALIBRATED SURROGATE<br/>" +
          "EXP. RMSE: " + state.validationRmse.toFixed(2) + " | MAE: " + state.validationMae.toFixed(2);
      } else {
        trustAnchor.innerHTML = "PHYSICS-CALIBRATED SURROGATE<br/>VALIDATION: NOT LOADED";
      }
      legend.textContent = "CH4 " + Math.round(state.methaneFraction * 100) + "%  |  H2 " + Math.round(state.hydrogenFraction * 100) + "%";
      debugPanel.style.display = state.debug ? "block" : "none";
    }

    function curvePoint(curve, t) {
      return curve.getPoint(clamp(t, 0.0, 1.0));
    }

    function updateParticles(timeSec) {
      const throughput = clamp(
        (0.46 + state.conversion * 0.66 + state.h2Norm * 0.18) * (1.0 - state.dpNorm * 0.40),
        0.10,
        1.45
      );
      const ch4Speed = (0.14 + state.methaneFraction * 0.76) * throughput;
      const h2Speed = (0.18 + state.h2Norm * 0.84) * throughput;
      const plumeSpeed = (0.22 + state.conversion * 1.05 + state.powerNorm * 0.18) * throughput;
      const carbonDensity = clamp(
        0.08 + Math.min(1.0, state.fouling) * 0.66 + state.carbonNorm * 0.50,
        0.08,
        1.0
      );
      const viewGlowBoost = state.thermalView ? 1.0 : 1.28;

      const ch4Active = Math.floor(ch4Particles.count * clamp((1.0 - state.conversion) * 0.98, 0.15, 1.0));
      ch4Particles.geometry.setDrawRange(0, ch4Active);
      for (let i = 0; i < ch4Active; i += 1) {
        const idx = i * 3;
        const phase = fract(ch4Particles.seeds[idx] + timeSec * 0.09 * ch4Speed);
        const p = curvePoint(feedPipe.curve, phase);
        ch4Particles.positions[idx] = p.x;
        ch4Particles.positions[idx + 1] = p.y + Math.sin((timeSec * 4.0) + (ch4Particles.seeds[idx + 1] * 12.0)) * 0.03;
        ch4Particles.positions[idx + 2] = p.z + (ch4Particles.seeds[idx + 2] - 0.5) * 0.18;
      }
      ch4Particles.geometry.attributes.position.needsUpdate = true;
      ch4Particles.material.opacity = clamp((0.20 + state.methaneFraction * 0.74) * viewGlowBoost, 0.18, 0.98);

      const h2Active = Math.floor(h2Particles.count * clamp(0.10 + state.h2Norm * 0.95, 0.08, 1.0));
      h2Particles.geometry.setDrawRange(0, h2Active);
      for (let i = 0; i < h2Active; i += 1) {
        const idx = i * 3;
        const phase = fract(h2Particles.seeds[idx] + timeSec * 0.10 * h2Speed);
        const p = curvePoint(topOutlet.curve, phase);
        h2Particles.positions[idx] = p.x;
        h2Particles.positions[idx + 1] = p.y + Math.sin((timeSec * 3.4) + (h2Particles.seeds[idx + 1] * 10.0)) * 0.03;
        h2Particles.positions[idx + 2] = p.z + (h2Particles.seeds[idx + 2] - 0.5) * 0.14;
      }
      h2Particles.geometry.attributes.position.needsUpdate = true;
      h2Particles.material.opacity = clamp((0.22 + state.hydrogenFraction * 0.80) * viewGlowBoost, 0.20, 0.98);

      for (let i = 0; i < plumeParticles.count; i += 1) {
        const idx = i * 3;
        const phase = fract(plumeParticles.seeds[idx] + timeSec * 0.07 * plumeSpeed);
        const swirl = (phase * Math.PI * 11.0) + (timeSec * 0.85);
        const radius = (0.03 + plumeParticles.seeds[idx + 1] * 0.36) * (1.0 - phase * 0.42);
        plumeParticles.positions[idx] = 2.20 + Math.cos(swirl) * radius;
        plumeParticles.positions[idx + 1] = (-1.60 + Y_SHIFT) + phase * (2.7 + state.powerNorm * 1.2 + state.conversion * 1.3);
        plumeParticles.positions[idx + 2] = Math.sin(swirl) * radius;
      }
      plumeParticles.geometry.attributes.position.needsUpdate = true;
      plumeParticles.material.opacity = clamp(
        (0.13 + state.conversion * 0.46 + state.powerNorm * 0.16) * viewGlowBoost,
        0.12,
        0.92
      );

      const tempNorms = [
        clamp((state.zoneTemps[0] - 820.0) / 420.0, 0.0, 1.0),
        clamp((state.zoneTemps[1] - 820.0) / 420.0, 0.0, 1.0),
        clamp((state.zoneTemps[2] - 820.0) / 420.0, 0.0, 1.0),
      ];
      const zoneBands = [
        [-2.75 + Y_SHIFT, -0.85 + Y_SHIFT],
        [-0.85 + Y_SHIFT, 1.05 + Y_SHIFT],
        [1.05 + Y_SHIFT, 3.15 + Y_SHIFT],
      ];
      for (let zoneIdx = 0; zoneIdx < 3; zoneIdx += 1) {
        const zoneCloud = reactionZoneParticles[zoneIdx];
        const tempNorm = tempNorms[zoneIdx];
        const active = Math.floor(
          zoneCloud.count * clamp(0.20 + (tempNorm * 0.72) + (state.thermalView ? 0.14 : 0.0), 0.15, 1.0)
        );
        zoneCloud.geometry.setDrawRange(0, active);
        const yLow = zoneBands[zoneIdx][0];
        const yHigh = zoneBands[zoneIdx][1];
          const zoneRise = 0.045 + (tempNorm * 0.10) + (state.powerNorm * 0.035);
        for (let i = 0; i < active; i += 1) {
          const idx = i * 3;
          const phase = fract(zoneCloud.seeds[idx] + timeSec * zoneRise);
          const theta = (zoneCloud.seeds[idx + 1] * Math.PI * 2.0) + (timeSec * (0.55 + (tempNorm * 0.85)));
          const radial = (0.08 + (zoneCloud.seeds[idx + 2] * 0.52)) * (1.0 - (phase * 0.18));
          zoneCloud.positions[idx] = 2.20 + Math.cos(theta) * radial;
          zoneCloud.positions[idx + 1] = yLow + (phase * (yHigh - yLow));
          zoneCloud.positions[idx + 2] = Math.sin(theta) * radial;
        }
        zoneCloud.geometry.attributes.position.needsUpdate = true;
        zoneCloud.material.opacity = clamp(
          (0.16 + (tempNorm * 0.52) + (state.thermalView ? 0.18 : 0.0)) * viewGlowBoost,
          0.12,
          0.94
        );
      }

      const carbonActive = Math.floor(carbonParticles.count * carbonDensity);
      carbonParticles.geometry.setDrawRange(0, carbonActive);
      for (let i = 0; i < carbonActive; i += 1) {
        const idx = i * 3;
        const phase = fract(carbonParticles.seeds[idx] + timeSec * (0.022 + state.dpNorm * 0.012));
        const theta = (carbonParticles.seeds[idx + 1] * Math.PI * 2.0) + (timeSec * 0.26);
        const radius = 0.84 + Math.min(1.0, state.fouling) * 0.08;
        carbonParticles.positions[idx] = 2.20 + Math.cos(theta) * radius;
        carbonParticles.positions[idx + 1] = (1.00 + Y_SHIFT) + phase * 3.4;
        carbonParticles.positions[idx + 2] = Math.sin(theta) * radius;
      }
      carbonParticles.geometry.attributes.position.needsUpdate = true;
      carbonParticles.material.opacity = clamp(0.16 + Math.min(1.0, state.fouling) * 0.60, 0.14, 0.88);
    }

    function updateLabelPositions() {
      const w = Math.max(1, mountEl.clientWidth || width);
      const h = Math.max(1, mountEl.clientHeight || height);
      for (let i = 0; i < labels.length; i += 1) {
        const projected = labels[i].point.clone().project(camera);
        const visible = projected.z > -1.0 && projected.z < 1.0;
        labels[i].el.style.display = visible ? "block" : "none";
        if (visible) {
          labels[i].el.style.left = (((projected.x * 0.5) + 0.5) * w).toFixed(1) + "px";
          labels[i].el.style.top = (((-projected.y * 0.5) + 0.5) * h).toFixed(1) + "px";
          labels[i].el.style.transform = "translate(-50%, -50%)";
        }
      }
    }

    function updateDebugOverlay() {
      if (!state.debug) {
        return;
      }
      updateRootBounds();
      const targetPoint = controls ? controls.target : bboxCenter;
      debugPanel.textContent =
        "CTRLS " + (controls ? "ON" : "OFF") + " | " +
        "CAM " +
        camera.position.x.toFixed(2) + "," +
        camera.position.y.toFixed(2) + "," +
        camera.position.z.toFixed(2) +
        " | TARGET " +
        targetPoint.x.toFixed(2) + "," +
        targetPoint.y.toFixed(2) + "," +
        targetPoint.z.toFixed(2) +
        " | BB " +
        bboxSize.x.toFixed(2) + "x" +
        bboxSize.y.toFixed(2) + "x" +
        bboxSize.z.toFixed(2);
    }

    function resizeRenderer(refit) {
      const w = Math.max(380, mountEl.clientWidth || width);
      const h = Math.max(320, mountEl.clientHeight || height);
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      if (refit) {
        fitCamera("resize");
      } else {
        updateRootBounds();
      }
      updateLabelPositions();
      updateDebugOverlay();
    }

    function onKeyDown(event) {
      if (event.key === "r" || event.key === "R") {
        fitCamera("hotkey");
      }
    }

    let resizeObserver = null;
    if (typeof window.ResizeObserver !== "undefined") {
      resizeObserver = new window.ResizeObserver(function () {
        resizeRenderer(true);
      });
      resizeObserver.observe(mountEl);
    }
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("resize", function () {
      resizeRenderer(true);
    });

    const restoredCamera = restoreCameraIfValid();
    if (!restoredCamera) {
      fitCamera("init");
    } else {
      updateRootBounds();
    }

    let frameReader = function () {
      return {};
    };
    let pollTimer = window.setInterval(function () {
      const polled = frameReader();
      if (polled && String(polled.frame_seq || "") !== state.frameSeq) {
        setFrame(polled);
      }
    }, 90);

    function animate(nowMs) {
      const nowSec = nowMs * 0.001;
      if (state.lastAnimTimeSec === null) {
        state.lastAnimTimeSec = nowSec;
      }
      const dt = clamp(nowSec - state.lastAnimTimeSec, 0.0, 0.12);
      state.lastAnimTimeSec = nowSec;
      if (state.isPlaying) {
        state.flowTimeSec += dt;
      }
      updateVisualState(state.flowTimeSec);
      updateParticles(state.flowTimeSec);
      if (controls) {
        controls.update();
      }
      updateLabelPositions();
      updateDebugOverlay();
      renderer.render(scene, camera);
      state.rafHandle = requestAnimationFrame(animate);
    }
    state.rafHandle = requestAnimationFrame(animate);

    return {
      container: mountEl,
      setFrame: setFrame,
      setFrameReader: function (readerFn) {
        frameReader = readerFn;
      },
      dispose: function () {
        try {
          if (state.rafHandle) {
            cancelAnimationFrame(state.rafHandle);
          }
          if (pollTimer) {
            window.clearInterval(pollTimer);
            pollTimer = null;
          }
          if (resizeObserver) {
            resizeObserver.disconnect();
          }
          window.removeEventListener("keydown", onKeyDown);
          renderer.dispose();
        } catch (err) {
          // ignore disposal errors
        }
      },
    };
  }

  function ensureThreeAndRender() {
    if (!window.THREE) {
      requestAnimationFrame(ensureThreeAndRender);
      return;
    }
    const incomingFrame = readLatestFrame();
    const existing = window[ENGINE_KEY];
    if (existing && existing.container === container) {
      existing.setFrameReader(readLatestFrame);
      existing.setFrame(incomingFrame);
      return;
    }
    if (existing && typeof existing.dispose === "function") {
      existing.dispose();
    }
    container.innerHTML = "";
    const engine = createEngine(window.THREE, container);
    engine.setFrameReader(readLatestFrame);
    engine.setFrame(incomingFrame);
    window[ENGINE_KEY] = engine;
  }

  ensureThreeAndRender();
})();
</script>
"""

    html = html_template.replace("__HEIGHT__", str(height))
    components.html(html, height=height + 2, scrolling=False)


__all__ = ["render_reactor_3d"]
