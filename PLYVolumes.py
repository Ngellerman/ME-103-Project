#!/usr/bin/env python3
"""
Batch-mesh .ply point clouds and report enclosed volume.

For every .ply in a folder:
  1. Load the point cloud (convert units to mm if needed)
  2. Remove stray outlier points
  3. Estimate + consistently orient normals (pointing outward)
  4. Screened Poisson reconstruction, with octree depth chosen so the
     finest cell size is <= the target resolution (default 0.2 mm)
  5. Compute volume of the closed mesh (divergence theorem via trimesh)
  6. Write a CSV: file name, volume, and quality checks

Usage:
    pip install open3d trimesh
    python ply_volumes.py /path/to/scans                       # scans already in mm
    python ply_volumes.py /path/to/scans --scale 1000          # scans in metres
    python ply_volumes.py /path/to/scans --save-meshes meshes  # also keep meshes to inspect
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def orient_normals_outward(pcd: o3d.geometry.PointCloud, k: int) -> None:
    """Make normals consistent across the surface, then flip them all if most point inward."""
    try:
        pcd.orient_normals_consistent_tangent_plane(k)
    except RuntimeError:
        pcd.orient_normals_towards_camera_location(pcd.get_center())
        pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))
        return
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    outward = np.einsum("ij,ij->i", nrm, pts - pts.mean(axis=0))
    if np.mean(outward > 0) < 0.5:
        pcd.normals = o3d.utility.Vector3dVector(-nrm)


def process(path: Path, res: float, scale: float, save_dir: Path | None) -> dict:
    pcd = o3d.io.read_point_cloud(str(path))
    n_raw = len(pcd.points)
    if n_raw == 0:
        raise ValueError("no points found")
    if scale != 1.0:
        pcd.scale(scale, center=(0, 0, 0))

    # 1) light cleanup
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)

    # 2) normals (always re-estimate orientation; scanner normals are often view-dependent)
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
    orient_normals_outward(pcd, k=15)

    # 3) Poisson depth so finest octree cell <= res
    #    Open3D's octree spans max_extent * poisson_scale
    poisson_scale = 1.1
    extent = float(np.max(pcd.get_axis_aligned_bounding_box().get_extent()))
    depth = max(6, math.ceil(math.log2(extent * poisson_scale / res)))
    cell = extent * poisson_scale / 2**depth

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=poisson_scale, linear_fit=True
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=True)
    # keep the largest closed component (drops small floating blobs)
    parts = tm.split(only_watertight=False)
    if len(parts) > 1:
        tm = max(parts, key=lambda p: abs(p.volume) if p.is_watertight else 0)
    if not tm.is_watertight:
        trimesh.repair.fill_holes(tm)
    trimesh.repair.fix_normals(tm)

    hull_vol = trimesh.Trimesh(np.asarray(pcd.points)).convex_hull.volume

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        tm.export(save_dir / f"{path.stem}_mesh.ply")

    vol = abs(tm.volume)
    return {
        "file": path.name,
        "volume_mm3": round(vol, 3),
        "volume_cm3": round(vol / 1000, 5),
        "watertight": tm.is_watertight,
        "mesh_to_hull_ratio": round(vol / hull_vol, 4) if hull_vol > 0 else "",
        "points": n_raw,
        "triangles": len(tm.faces),
        "poisson_depth": depth,
        "cell_size_mm": round(cell, 4),
        "error": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder containing .ply scans")
    ap.add_argument("--resolution", type=float, default=0.2, help="target mesh resolution in mm (default 0.2)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply coordinates by this to get mm (metres -> 1000)")
    ap.add_argument("--out", type=Path, default=Path("volumes.csv"), help="output CSV path")
    ap.add_argument("--save-meshes", type=Path, default=None, help="folder to write reconstructed meshes")
    args = ap.parse_args()

    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() == ".ply")
    if not files:
        sys.exit(f"No .ply files in {args.folder}")

    rows = []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name} ... ", end="", flush=True)
        try:
            row = process(f, args.resolution, args.scale, args.save_meshes)
            flag = "" if row["watertight"] else "  (NOT watertight - check)"
            print(f"{row['volume_mm3']:.2f} mm^3{flag}")
        except Exception as e:  # keep going on bad files
            row = {"file": f.name, "error": str(e)}
            print(f"FAILED: {e}")
        rows.append(row)

    fields = ["file", "volume_mm3", "volume_cm3", "watertight", "mesh_to_hull_ratio",
              "points", "triangles", "poisson_depth", "cell_size_mm", "error"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()