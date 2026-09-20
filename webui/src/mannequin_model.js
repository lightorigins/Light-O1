import * as THREE from "three";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { clone } from "three/addons/utils/SkeletonUtils.js";

// Bone names match JOINT_NAMES in light_deploy.action_tokenizer.representation.
export const BONE_NAMES = ["pelvis", "left_hip", "right_hip", "spine1",
  "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
  "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist"];

let templatePromise;
export function loadMannequinTemplate() {
  return templatePromise ??= loadTemplate().catch((error) => {
    templatePromise = null;
    throw error;
  });
}

async function loadTemplate() {
  // Ignore stale texture references embedded in the FBX; use our shipped map.
  const manager = new THREE.LoadingManager();
  const placeholder = new THREE.Texture();
  manager.addHandler(/.*/, { load: () => placeholder, setPath() { return this; } });
  const [model, map] = await Promise.all([
    new FBXLoader(manager).loadAsync("/models/mannequin/mannequin.fbx"),
    new THREE.TextureLoader().loadAsync("/models/mannequin/robot_base_color.jpg"),
  ]);
  map.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.MeshStandardMaterial({ map, roughness: 0.8, metalness: 0 });
  model.traverse((node) => {
    if (!node.isMesh) return;
    for (const old of Array.isArray(node.material) ? node.material : [node.material]) old.dispose();
    node.material = material;
    node.frustumCulled = false;
  });
  placeholder.dispose();
  return model;
}

// Rest pelvis height of HUMANOID22_V1, the rig the generated rotations and root
// translation are expressed in: light_deploy.action_tokenizer.fk.pelvis_rest_height().
export const REST_PELVIS_Y = 0.97;

/** The mannequin mesh and its 22-joint rig, in centimetres. */
export class MannequinAvatar {
  constructor(template) {
    this.mesh = new THREE.Group();
    const model = clone(template);
    model.scale.multiplyScalar(0.01);
    // Both the FBX and human_action_138_v1 preview use Y-up.
    this.mesh.add(model);
    const byName = {};
    model.traverse((node) => { if (node.isBone) byName[node.name] = node; });
    this.bones = BONE_NAMES.map((name) => byName[name]);
    if (this.bones.some((bone) => !bone)) throw new Error("Mannequin is missing required body joints");
    // The asset carries its own proportions, so normalize it the way the hosted
    // viewer does: rest pelvis height above the lowest bone becomes REST_PELVIS_Y,
    // which puts the soles on the ground at the root height the model generates.
    this.mesh.updateMatrixWorld(true);
    const probe = new THREE.Vector3();
    let lowest = Infinity;
    for (const bone of Object.values(byName)) {
      bone.getWorldPosition(probe);
      lowest = Math.min(lowest, probe.y);
    }
    this.bones[0].getWorldPosition(probe);
    model.scale.multiplyScalar(REST_PELVIS_Y / Math.max(probe.y - lowest, 1e-6));
    this.parentRotation = new THREE.Quaternion();
    this.pelvisPosition = new THREE.Vector3();
  }

  setPose(rootPosition, rotations) {
    this.setVisible(true);
    this.mesh.position.set(0, 0, 0);
    this.mesh.updateMatrixWorld(true);
    // The compact FK endpoint supplies GLOBAL wxyz rotations, not local xyzw.
    // Resolve each parent first, then convert the target into the bone's frame.
    for (let i = 0; i < this.bones.length; i++) {
      const bone = this.bones[i];
      const q = rotations[i];
      bone.parent.getWorldQuaternion(this.parentRotation);
      bone.quaternion.set(q[1], q[2], q[3], q[0]).normalize()
        .premultiply(this.parentRotation.invert());
      bone.updateMatrixWorld(true);
    }
    this.mesh.updateMatrixWorld(true);
    this.bones[0].getWorldPosition(this.pelvisPosition);
    this.mesh.position.fromArray(rootPosition).sub(this.pelvisPosition);
    this.mesh.updateMatrixWorld(true);
  }

  setVisible(value) { this.mesh.visible = value; }
  dispose() {
    // Geometry/material/texture are shared by the cached template and candidates.
    const skeletons = new Set();
    this.mesh.traverse((node) => { if (node.skeleton) skeletons.add(node.skeleton); });
    for (const skeleton of skeletons) skeleton.dispose();
  }
}
