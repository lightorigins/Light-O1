import * as THREE from "three";
import { MannequinAvatar, REST_PELVIS_Y, loadMannequinTemplate } from "./mannequin_model.js";

const DEFAULT_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19];
const DEFAULT_CONNECTIONS = DEFAULT_PARENTS.flatMap((parent, joint) => (parent >= 0 ? [[parent, joint]] : []));
// Match the renderer's per-frame camera override; scene XML defaults are ignored.
const SONIC_CAMERA_AZIMUTH_DEGREES = 135;
const SONIC_CAMERA_ELEVATION_DEGREES = -15;
const SONIC_CAMERA_DISTANCE = 3.0;
const SONIC_CAMERA_LOOK_AT_HEIGHT = 0.8;
const SONIC_ORBIT_YAW = THREE.MathUtils.degToRad(SONIC_CAMERA_AZIMUTH_DEGREES + 180);
const SONIC_ORBIT_PITCH = THREE.MathUtils.degToRad(-SONIC_CAMERA_ELEVATION_DEGREES);

const STANDBY_POSE = [
  [0, 0.9, 0],
  [0.09, 0.82, 0],
  [-0.09, 0.82, 0],
  [0, 1.01, 0],
  [0.09, 0.4, 0],
  [-0.09, 0.4, 0],
  [0, 1.13, 0],
  [0.09, -0.01, 0],
  [-0.09, -0.01, 0],
  [0, 1.27, 0],
  [0.09, -0.07, 0.13],
  [-0.09, -0.07, 0.13],
  [0, 1.43, 0],
  [0.08, 1.33, 0],
  [-0.08, 1.33, 0],
  [0, 1.61, 0],
  [0.22, 1.33, 0],
  [-0.22, 1.33, 0],
  [0.47, 1.33, 0],
  [-0.47, 1.33, 0],
  [0.71, 1.33, 0],
  [-0.71, 1.33, 0]
];

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function normalizeAction(payload) {
  const positions = payload?.positions ?? payload?.joints ?? payload?.frames ?? payload?.data?.joints;
  if (!Array.isArray(positions) || !positions.length || !Array.isArray(positions[0])) {
    return { frames: [], rotations: [], connections: DEFAULT_CONNECTIONS };
  }
  const parents = payload?.parents;
  const connections = Array.isArray(parents)
    ? parents.flatMap((parent, joint) => (Number.isInteger(parent) && parent >= 0 ? [[parent, joint]] : []))
    : DEFAULT_CONNECTIONS;
  return {
    frames: positions,
    rotations: Array.isArray(payload?.rotations) ? payload.rotations : [],
    connections: connections.length ? connections : DEFAULT_CONNECTIONS,
  };
}

export class PreviewSurface {
  constructor(canvas, { onFrame } = {}) {
    this.canvas = canvas;
    this.onFrame = onFrame;
    this.action = normalizeAction(null);
    this.frameCount = 0;
    this.frame = 0;
    this.fps = 20;
    this.playing = false;
    this.lastTick = 0;
    this.reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x080b12);
    this.scene.fog = new THREE.FogExp2(0x080b12, 0.065);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.04, 80);
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.canvas.dataset.renderer = "three";
    this.canvas.dataset.cameraView = "sonic-mujoco";

    this.scene.add(new THREE.HemisphereLight(0xc9dcff, 0x121728, 2.6));
    const keyLight = new THREE.DirectionalLight(0xffffff, 3.3);
    keyLight.position.set(3.5, 5.5, 4.5);
    this.scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0x477cff, 2.1);
    rimLight.position.set(-4, 2.5, -3);
    this.scene.add(rimLight);

    this.grid = new THREE.GridHelper(20, 40, 0x4269a8, 0x1b2a43);
    this.grid.position.y = -0.075;
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.62;
    this.scene.add(this.grid);

    this.skeleton = new THREE.Group();
    this.scene.add(this.skeleton);
    this.linePositions = new Float32Array(DEFAULT_CONNECTIONS.length * 6);
    this.lineGeometry = new THREE.BufferGeometry();
    this.lineGeometry.setAttribute("position", new THREE.BufferAttribute(this.linePositions, 3));
    this.lineMaterial = new THREE.LineBasicMaterial({ color: 0x73a7ff, transparent: true, opacity: 0.98 });
    this.lines = new THREE.LineSegments(this.lineGeometry, this.lineMaterial);
    this.skeleton.add(this.lines);

    this.jointGeometry = new THREE.SphereGeometry(0.034, 18, 12);
    this.jointMaterial = new THREE.MeshStandardMaterial({
      color: 0xeaf2ff,
      emissive: 0x14294a,
      emissiveIntensity: 0.65,
      metalness: 0.08,
      roughness: 0.28,
    });
    this.rootMaterial = new THREE.MeshStandardMaterial({
      color: 0x73a7ff,
      emissive: 0x1d4d9b,
      emissiveIntensity: 1.15,
      metalness: 0.1,
      roughness: 0.24,
    });
    this.joints = Array.from({ length: STANDBY_POSE.length }, (_, index) => {
      const joint = new THREE.Mesh(this.jointGeometry, index === 0 ? this.rootMaterial : this.jointMaterial);
      this.skeleton.add(joint);
      return joint;
    });
    this.rootAxes = new THREE.AxesHelper(0.28);
    this.skeleton.add(this.rootAxes);
    this.contactGeometry = new THREE.CircleGeometry(0.42, 48);
    this.contactMaterial = new THREE.MeshBasicMaterial({
      color: 0x2e64ee,
      transparent: true,
      opacity: 0.14,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    this.contact = new THREE.Mesh(this.contactGeometry, this.contactMaterial);
    this.contact.rotation.x = -Math.PI / 2;
    this.contact.position.y = -0.07;
    this.scene.add(this.contact);

    this.orbit = {
      yaw: SONIC_ORBIT_YAW,
      pitch: SONIC_ORBIT_PITCH,
      distance: SONIC_CAMERA_DISTANCE,
      target: new THREE.Vector3(0, SONIC_CAMERA_LOOK_AT_HEIGHT, 0),
      pan: new THREE.Vector3(),
      dragging: false,
      pointerId: null,
      mode: "rotate",
      x: 0,
      y: 0,
    };
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.attachControls();
    this.resize();
    this.loop = this.loop.bind(this);
    this.animationId = window.requestAnimationFrame(this.loop);
    this.destroyed = false;
    this.canvas.dataset.avatar = "loading";
    loadMannequinTemplate().then((template) => {
      if (this.destroyed) return;
      this.avatar = new MannequinAvatar(template);
      this.scene.add(this.avatar.mesh);
      this.render();
    }).catch(() => {
      if (!this.destroyed) this.canvas.dataset.avatar = "fallback";
    });
  }

  attachControls() {
    this.onPointerDown = (event) => {
      this.orbit.dragging = true;
      this.orbit.pointerId = event.pointerId;
      this.orbit.mode = event.shiftKey || event.button !== 0 ? "pan" : "rotate";
      this.orbit.x = event.clientX;
      this.orbit.y = event.clientY;
      this.canvas.setPointerCapture(event.pointerId);
      event.preventDefault();
    };
    this.onPointerMove = (event) => {
      if (!this.orbit.dragging || event.pointerId !== this.orbit.pointerId) return;
      const deltaX = event.clientX - this.orbit.x;
      const deltaY = event.clientY - this.orbit.y;
      this.orbit.x = event.clientX;
      this.orbit.y = event.clientY;
      if (this.orbit.mode === "pan") {
        const scale = this.orbit.distance * 0.0018;
        const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0).normalize();
        const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 1).normalize();
        this.orbit.pan.addScaledVector(right, -deltaX * scale).addScaledVector(up, deltaY * scale);
      } else {
        this.orbit.yaw -= deltaX * 0.006;
        this.orbit.pitch = clamp(this.orbit.pitch - deltaY * 0.006, -0.58, 1.08);
      }
      this.render();
    };
    this.onPointerEnd = (event) => {
      if (event.pointerId === this.orbit.pointerId) {
        this.orbit.dragging = false;
        this.orbit.pointerId = null;
      }
    };
    this.onWheel = (event) => {
      this.orbit.distance = clamp(this.orbit.distance * Math.exp(event.deltaY * 0.001), 1.3, 12);
      this.render();
      event.preventDefault();
    };
    this.onContextMenu = (event) => event.preventDefault();
    this.canvas.addEventListener("pointerdown", this.onPointerDown);
    this.canvas.addEventListener("pointermove", this.onPointerMove);
    this.canvas.addEventListener("pointerup", this.onPointerEnd);
    this.canvas.addEventListener("pointercancel", this.onPointerEnd);
    this.canvas.addEventListener("wheel", this.onWheel, { passive: false });
    this.canvas.addEventListener("contextmenu", this.onContextMenu);
  }

  resize() {
    const width = Math.max(2, this.canvas.clientWidth);
    const height = Math.max(2, this.canvas.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  setActions(payloads, fps = 20) {
    const actions = (Array.isArray(payloads) ? payloads : []).slice(0, 2).map(normalizeAction);
    this.action = actions[0] || normalizeAction(null);
    this.frameCount = this.action.frames.length;
    this.frame = 0;
    this.fps = Number(fps) > 0 ? Number(fps) : 20;
    this.playing = this.frameCount > 1 && !this.reducedMotion;
    this.onFrame?.(this.frame, this.frameCount);
    this.render();
    return actions.map((action) => action.frames.length);
  }

  setPlaying(playing) {
    this.playing = Boolean(playing) && this.frameCount > 1;
    return this.playing;
  }

  toggle() {
    return this.setPlaying(!this.playing);
  }

  seek(frame) {
    this.frame = Math.max(0, Math.min(Number(frame) || 0, Math.max(0, this.frameCount - 1)));
    this.onFrame?.(this.frame, this.frameCount);
    this.render();
  }

  resetView() {
    this.orbit.yaw = SONIC_ORBIT_YAW;
    this.orbit.pitch = SONIC_ORBIT_PITCH;
    this.orbit.distance = SONIC_CAMERA_DISTANCE;
    this.orbit.pan.set(0, 0, 0);
    this.seek(0);
  }

  render() {
    const frameIndex = Math.min(this.frame, Math.max(0, this.action.frames.length - 1));
    const pose = this.action.frames[frameIndex] || STANDBY_POSE;
    const root = new THREE.Vector3().fromArray(pose[0] || STANDBY_POSE[0]);
    const rotations = this.action.rotations[frameIndex];
    const validRotations = Array.isArray(rotations) && rotations.length === 22
      && rotations.every((q) => Array.isArray(q) && q.length === 4
        && q.every(Number.isFinite) && Math.hypot(...q) > 1e-8);
    const standby = !this.action.frames.length;
    const showMesh = Boolean(this.avatar && (standby || validRotations));
    this.skeleton.visible = !showMesh;
    if (this.avatar) {
      if (showMesh) this.avatar.setPose(
        standby ? [0, REST_PELVIS_Y, 0] : pose[0],
        standby ? Array.from({ length: 22 }, () => [1, 0, 0, 0]) : rotations,
      );
      this.avatar.setVisible(showMesh);
      this.canvas.dataset.avatar = showMesh ? "mannequin" : "fallback";
    }
    let cursor = 0;
    for (const [from, to] of this.action.connections) {
      for (const jointIndex of [from, to]) {
        const point = pose[jointIndex] || STANDBY_POSE[jointIndex];
        this.linePositions[cursor] = Number(point[0]);
        this.linePositions[cursor + 1] = Number(point[1]);
        this.linePositions[cursor + 2] = Number(point[2]);
        cursor += 3;
      }
    }
    this.lineGeometry.setDrawRange(0, cursor / 3);
    this.lineGeometry.attributes.position.needsUpdate = true;
    this.joints.forEach((joint, index) => joint.position.fromArray(pose[index] || STANDBY_POSE[index]));
    this.rootAxes.position.copy(root);
    const rootRotation = this.action.rotations[frameIndex]?.[0];
    if (Array.isArray(rootRotation) && rootRotation.length === 4) {
      this.rootAxes.quaternion.set(rootRotation[1], rootRotation[2], rootRotation[3], rootRotation[0]);
    } else {
      this.rootAxes.quaternion.identity();
    }
    this.contact.position.x = root.x;
    this.contact.position.z = root.z;
    this.grid.position.x = Math.round(root.x / 2) * 2;
    this.grid.position.z = Math.round(root.z / 2) * 2;
    this.orbit.target.set(root.x, SONIC_CAMERA_LOOK_AT_HEIGHT, root.z).add(this.orbit.pan);
    const cosine = Math.cos(this.orbit.pitch);
    this.camera.position.set(
      this.orbit.target.x + Math.sin(this.orbit.yaw) * cosine * this.orbit.distance,
      this.orbit.target.y + Math.sin(this.orbit.pitch) * this.orbit.distance,
      this.orbit.target.z + Math.cos(this.orbit.yaw) * cosine * this.orbit.distance,
    );
    this.camera.lookAt(this.orbit.target);
    this.renderer.render(this.scene, this.camera);
  }

  loop(time) {
    if (this.playing && time - this.lastTick >= 1000 / this.fps) {
      this.frame = (this.frame + 1) % this.frameCount;
      this.lastTick = time;
      this.onFrame?.(this.frame, this.frameCount);
      this.render();
    }
    this.animationId = window.requestAnimationFrame(this.loop);
  }

  destroy() {
    this.destroyed = true;
    this.avatar?.dispose();
    this.resizeObserver.disconnect();
    window.cancelAnimationFrame(this.animationId);
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    this.canvas.removeEventListener("pointermove", this.onPointerMove);
    this.canvas.removeEventListener("pointerup", this.onPointerEnd);
    this.canvas.removeEventListener("pointercancel", this.onPointerEnd);
    this.canvas.removeEventListener("wheel", this.onWheel);
    this.canvas.removeEventListener("contextmenu", this.onContextMenu);
    this.lineGeometry.dispose();
    this.lineMaterial.dispose();
    this.jointGeometry.dispose();
    this.jointMaterial.dispose();
    this.rootMaterial.dispose();
    this.contactGeometry.dispose();
    this.contactMaterial.dispose();
    this.grid.geometry.dispose();
    this.grid.material.dispose();
    this.renderer.dispose();
  }
}
