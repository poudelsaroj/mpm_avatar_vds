"""
Generate per-frame smplx_fitted OBJ files from params_*.npz + split_idx.npz.
The SMPLX body vertices are extracted from the full mesh using split_idx.

Run on Lightning:
    python gen_smplx_fitted.py
"""
import numpy as np
import os
import glob

PRETRAINED   = "/teamspace/studios/this_studio/pretrained_models"
TRACKING_DIR = f"{PRETRAINED}/output/tracking/a1_s1_460_200"
SPLIT_IDX    = "/teamspace/studios/this_studio/mpm_avatar_vds/data/a1_s1/split_idx.npz"
OUT_DIR      = f"{TRACKING_DIR}/a1_s1/smplx_fitted"

# Load split indices
split = np.load(SPLIT_IDX, allow_pickle=True)
human_v_idx = split["reordered_human_v_idx"]  # indices into full mesh for body vertices
human_f_idx = split["reordered_human_f_idx"]  # face indices for body

print(f"Human verts: {len(human_v_idx)}  Human faces: {len(human_f_idx)}")

# Find all params files
params_files = sorted(glob.glob(f"{TRACKING_DIR}/params_*.npz"))
print(f"Found {len(params_files)} params files")

for pf in params_files:
    frame = int(os.path.basename(pf).replace("params_", "").replace(".npz", ""))
    data  = np.load(pf)
    verts = data["vertices"]          # [N_verts, 3]
    faces = data["faces"]             # [N_faces, 3]  full mesh faces

    # Extract body vertices
    body_verts = verts[human_v_idx]   # [N_body_verts, 3]

    # Extract body faces (reindex to body-local vertex indices)
    # Build a mapping from global vertex index → local body vertex index
    global_to_local = np.full(len(verts), -1, dtype=np.int64)
    global_to_local[human_v_idx] = np.arange(len(human_v_idx))

    body_faces_global = faces[human_f_idx]   # [N_body_faces, 3] global indices
    body_faces_local  = global_to_local[body_faces_global]  # remap to local

    # Write OBJ
    out_path = os.path.join(OUT_DIR, f"{frame:06d}", "smplx_icp.obj")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        for v in body_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in body_faces_local:
            if np.all(tri >= 0):
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    print(f"  frame {frame:06d} → {out_path}  ({len(body_verts)} verts)")

print("Done.")
