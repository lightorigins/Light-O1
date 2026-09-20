# Unitree G1 model assets

This directory contains the minimum 29-DoF G1 model subset used by the
Sonic MuJoCo simulation example. It is vendored from Unitree's official
[`unitree_mujoco`](https://github.com/unitreerobotics/unitree_mujoco)
repository at commit:

```text
1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d
```

Included files:

- `g1_29dof.xml`: the official MJCF model definition.
- `meshes/*.STL`: only the mesh files referenced by that definition.
- `LICENSE`: the upstream BSD-3-Clause license.

The upstream terrain scenes, simulation controller, and models for other
robots are not included. The example adds its own procedural floor and lighting.
