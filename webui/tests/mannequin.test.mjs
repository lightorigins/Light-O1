import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import * as THREE from "three";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { MannequinAvatar, REST_PELVIS_Y } from "../src/mannequin_model.js";

test("mannequin follows global wxyz rotations in Y-up without sharing candidate poses", () => {
  const manager = new THREE.LoadingManager();
  manager.addHandler(/.*/, { load: () => new THREE.Texture(), setPath() { return this; } });
  const buffer = fs.readFileSync(new URL("../public/models/mannequin/mannequin.fbx", import.meta.url));
  assert.ok(buffer.length > 100000, "FBX must be fetched with Git LFS");
  assert.ok(!buffer.includes(Buffer.from("/Users/")), "no workstation paths in asset");
  assert.ok(!buffer.includes(Buffer.from("/home/")), "no workstation paths in asset");
  // No third-party body-model naming may creep back into the asset.
  for (const marker of ["m_avg", "SMPL", "smpl"]) {
    assert.ok(!buffer.includes(Buffer.from(marker)), `no ${marker} naming in asset`);
  }
  const template = new FBXLoader(manager).parse(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength), "");
  const avatar = new MannequinAvatar(template), other = new MannequinAvatar(template);
  const rotations = Array.from({ length: 22 }, () => [1, 0, 0, 0]);
  avatar.setPose([1, REST_PELVIS_Y, 2], rotations);
  const box = new THREE.Box3().setFromObject(avatar.mesh);
  assert.ok(box.max.y - box.min.y > 1.75 && box.max.y - box.min.y < 1.9);
  assert.ok(Math.abs(box.min.y) < 0.02);
  const targets = rotations.map((_, index) => new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.03 * index));
  avatar.setPose([1, 1.1, 2], targets.map((q) => [q.w, q.x, q.y, q.z]));
  assert.ok(avatar.bones[0].getWorldPosition(new THREE.Vector3()).distanceTo(new THREE.Vector3(1, 1.1, 2)) < 1e-6);
  avatar.bones.forEach((bone, i) => assert.ok(bone.getWorldQuaternion(new THREE.Quaternion()).angleTo(targets[i]) < 1e-6));
  assert.ok(avatar.bones[18].quaternion.angleTo(other.bones[18].quaternion) > 0.01);
  assert.deepEqual(avatar.bones.map((bone) => bone.name).slice(0, 3),
    ["pelvis", "left_hip", "right_hip"]);
  avatar.dispose(); other.dispose();
});
