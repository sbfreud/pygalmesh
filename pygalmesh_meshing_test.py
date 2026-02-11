#(Author: Victor (kinda))
#!/usr/bin/env python3
"""
TIFF (segmented volume) -> CGAL (pygalmesh) conformal tetra mesh -> Nastran .bdf

Run inside WSL in your pygalmesh virtualenv (or in linux ir ur a nerd), for example:

    source ~/cgalenv/bin/activate
    python tiff_to_cgal_bdf.py \
        --tiff path/to/volume.tif \
        --dx 1 --dy 1 --dz 1 \
        --max-facet-distance 1 \
        --max-cell-circumradius 2 \
        --max-circumradius-edge-ratio 2 \
        --min-facet-angle 30 \
        --max-edge-size-at-feature-edges 0.25 \
        --max-radius-surface-delaunay-ball 0.5 \
        --exude-time 1000 \
        --exude-sliver-bound 10 \
        --out-prefix binary_multiphase

Notes
-----
- pygalmesh.generate_from_array():
    * Takes an int 3D array of labels.
    * Subdomains with key 0 are NOT meshed (Pygalmesh does this automatically idk why).
    * Nonzero labels become subdomain indices, interfaces are conformal.

- This script:
    * Optionally shifts labels so that background=0 and phases=1..N.
    * Reorders axes so array.shape = (Nx, Ny, Nz) matches voxel_size=(dx, dy, dz).
"""

import argparse
import os

#Requirements
import numpy as np
import tifffile as tiff
import pygalmesh
import meshio
from skimage.morphology import ball, cube, disk, square
import scipy.ndimage as spim
import porespy as ps


def is_percolating(im, axis=None, inlets=None, outlets=None, conn='min'):
    
    r"""
    Determines if a percolating path exists across the domain (in the specified
    direction) or between given inlets and outlets.

    Parameters
    ----------
    im : ndarray
        Image of the void space with `True` indicating void space.
    axis : int
        The axis along which percolation is checked. If `None` (default) then
        percolation is checked in all dimensions.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    percolating : bool or list of bools
        A boolean value indicating if the domain percolates in the given direction.
        If `axis=None` then all directions are checked and the result is returned
        as a list like `[True, False, True]` indicating that the domain percolates
        in the `x` and `z` directions, but not `y`.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/is_percolating.html>`_
    to view online example.

    """
    strel = {2: {"min": disk(1), "max": square(3)}, 3: {"min": ball(1), "max": cube(3)}}
    if (inlets is not None) and (outlets is not None):
        pass
    elif axis is not None:
        im = np.swapaxes(im, 0, axis) == 1
        inlets = np.zeros_like(im)
        inlets[0, ...] = True
        inlets *= im
        outlets = np.zeros_like(im)
        outlets[-1, ...] = True
        outlets *= im
    else:
        ans = []
        for ax in range(im.ndim):
            ans.append(is_percolating(im, axis=ax, conn=conn))
        return ans

    labels, N = spim.label(im, structure=strel[im.ndim][conn])
    a = np.unique(labels[inlets])
    a = a[a > 0]
    b = np.unique(labels[outlets])
    b = b[b > 0]
    hits = np.isin(a, b)
    return np.any(hits)                    
                        

def crop_image(image, crop_factor, type_range):
    """
    Crops the input 3D image by the specified crop factor.

    Parameters:
    image (ndarray): The input 3D image to be cropped.
    crop_factor (float): The factor by which to crop the image (0 < crop_factor < 1).

    Returns:
    ndarray: The cropped 3D image.
    """
    if not (0 < crop_factor < 100):
        raise ValueError("crop_factor must be between 0 and 100")
    crop_factor = crop_factor / 100.0



    percolating_flag = False
    cropped_subimage = image
    count = 0
    while not percolating_flag:
        print(f"Trying crop factor of {crop_factor}")
        z, y, x = image.shape
        new_z = int(z * crop_factor)
        new_y = int(y * crop_factor)
        new_x = int(x * crop_factor)

        if count < 20:
            start_z = np.random.randint(0, z - new_z + 1)
            start_y = np.random.randint(0, y - new_y + 1)
            start_x = np.random.randint(0, x - new_x + 1)
            phase_flags = []
            cropped_subimage = image[start_z:start_z + new_z,
                                    start_y:start_y + new_y,
                                    start_x:start_x + new_x]
            for value in type_range:
                binary_image = (cropped_subimage == value)
                flag_array = is_percolating(binary_image)
                phase_flags.append(np.all(flag_array))
            percolating_flag = np.all(phase_flags)
            count += 1
        else:
            crop_factor += 0.05
            count = 0
    print("Found a percolating crop at a crop factor of ", crop_factor*100)
    
    #Removes disconnected domain regions
    for i in range(1, len(type_range)):
        binary_mask = cropped_subimage==type_range[i]
        im = ps.filters.find_disconnected_voxels(binary_mask, conn = "min")
        # deactivation_mask = im ^ binary_mask
        print(np.sum(im))
        cropped_subimage[im] = type_range[0]
    return cropped_subimage

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate CGAL conformal tet mesh from segmented TIFF and export as Nastran .bdf"
    )
    p.add_argument("--tiff", help="Path to segmented 3D TIFF (integer labels)", default="volume.tif")
    p.add_argument("--cropfactor", help="Crop factor of 3D TIFF image", default=int(10))
    # p.add_argument("--phases", help="Comma separated list of integer labels (eg. 100,155,255)", default="100,177,255")
    p.add_argument("--dx", type=float, help="Voxel dimension in X", default=1)
    p.add_argument("--dy", type=float, help="Voxel dimension in Y", default=1)
    p.add_argument("--dz", type=float, help="Voxel dimension in Z", default=1)

    p.add_argument(
        "--max-facet-distance",
        type=float,
        default=1,
        help="Surface approximation tolerance (CGAL Mesh_3 'max_facet_distance'). "
             "If None, defaults to 1",
    )
    p.add_argument(
        "--max-cell-circumradius",
        type=float,
        default=2,
        help="Target tetra size (CGAL Mesh_3 'max_cell_circumradius'). "
             "If None, defaults to 2 (slightly larger than facet_length)",
    )

    p.add_argument(
        "--max-circumradius-edge-ratio",
        type=float,
        default=2,
        help="This parameter controls the shape of mesh cells . " \
        "The Delaunay refinement process is guaranteed to terminate for values of max-circumradius-edge-ratio bigger than 2. " \
        "Default 2",
    )
    

    p.add_argument(
        "--min-facet-angle",
        type=float,
        default=30,
        help="This parameter controls the shape of surface facets. " \
        "Actually, it is a lower bound for the angle (in degree) of surface facets." \
        " Default 30",
    )
    
    p.add_argument(
        "--max-edge-size-at-feature-edges",
        type=float,
        default=0.25,
        help="This constant or variable scalar field is used as an upper bound for the distance " \
        "between two protecting ball centers that are consecutive on a 1-feature. " 
             "If None, defaults to 0.25",
    )

    p.add_argument(
        "--max-radius-surface-delaunay-ball",
        type=float,
        default=0.5,
        help=" This parameter controls the size of surface facets. " 
             "If None, defaults to 0.5",
    )

    p.add_argument(
        "--exude-time-limit",
        type=float,
        default=1000,
        help="time limit of the exude process. "
             "If None, defaults to 1000s.",
    )

    p.add_argument(
        "--exude-sliver-bound",
        type=float,
        default=10,
        help="Targeted sliver dihedral angle in degrees. "
             "If None, defaults to 10.",
    )        

    p.add_argument(
        "--no-shift-labels",
        action="store_true",
        help="Do NOT shift labels so min label is 1. "
             "By default, shift so that 0 is background and all phases >0 are meshed.",
    )
    p.add_argument(
        "--out-prefix",
        default="cgal_multiphase_test1",
        help="Prefix for output files (.bdf, .vtk)",
    )
    p.add_argument(
        "--transpose-zyx-to-xyz",
        action="store_true",
        help=(
            "If TIFF is stored as (Z, Y, X) but want voxel_size = (dx,dy,dz) "
            "to align with (X,Y,Z), enable this to transpose array from (Z,Y,X) to (X,Y,Z)."
        ),
    )
    return p



def write_phases_exodus(mesh, filename, domain_key="medit:ref"):
    """
    EXPORTS AN EXODUS .e MESH with phase Ids per element
    with Phase ID encoded into CTETRA as the property field.
    Example output:
        CTETRA, 123456, 2, n1, n2, n3, n4

    COMSOL interprets field #3 as domain.
    """

    if domain_key not in mesh.cell_data:
        raise RuntimeError("No domain/subdomain data found for BDF export!")

    cells_info = []
    point_set = {}
    cell_set = {}
    for block_idx, cell_block in enumerate(mesh.cells):
        ctype = cell_block.type
        cells = cell_block.data

            # Pull matching domain label array
        doms = mesh.cell_data[domain_key][block_idx]
        unique_doms = np.unique(doms)
        for value in unique_doms:
            if ctype == "tetra":
                cells_info.append((ctype, cells[doms==value]))
            else:
                # point_set[f'{value}'] = cells[doms==value].flatten()
                cell_set[f'{value}'] = cells[doms==value].flatten()
    new_mesh = meshio.Mesh(points=mesh.points, cells = cells_info)#, point_sets = point_set)
    new_mesh.write(filename)

    print(f"[SUCCESS] Exported Exodus → {filename}")

def main():
    parser = build_parser()
    args = parser.parse_args()

    
    
    tiff_path = args.tiff
    crop_factor = args.cropfactor
    dx, dy, dz = args.dx, args.dy, args.dz
    type_range = np.unique(tiff.imread(tiff_path))
    if not os.path.isfile(tiff_path):
        raise FileNotFoundError(f"TIFF not found: {tiff_path}")

    print(f"[INFO] Reading TIFF: {tiff_path}")
    vol = crop_image(tiff.imread(tiff_path), float(crop_factor), type_range)
    # vol = tiff.imread(tiff_path)
    


    print(f"[INFO] Original volume shape: {vol.shape}, dtype={vol.dtype}")

    # Optionally transpose (Z,Y,X) -> (X,Y,Z)
    if args.transpose_zyx_to_xyz:
        if vol.ndim != 3:
            raise ValueError("Expected 3D volume for transpose_zyx_to_xyz")
        vol = np.transpose(vol, (2, 1, 0))
        print(f"[INFO] Transposed volume to (X,Y,Z). New shape: {vol.shape}")

    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got ndim={vol.ndim}")

    # Ensure integer labels
    if not np.issubdtype(vol.dtype, np.integer):
        print(f"[WARN] Volume dtype {vol.dtype} is not integer; casting to uint16.")
        vol = vol.astype(np.uint16)
    else:
        # Make a copy in a small int type if possible
        max_label = int(vol.max())
        if max_label <= 255 and vol.dtype != np.uint8:
            vol = vol.astype(np.uint8)
        elif max_label <= 65535 and vol.dtype not in (np.uint16, np.uint8):
            vol = vol.astype(np.uint16)

    unique_labels = np.unique(vol)
    print(f"[INFO] Unique labels before shift: {unique_labels}")
    count = 1
    
    for value in unique_labels:
        while count in unique_labels:
            count += 1
        vol[vol == value] = count
        count += 1
    # unique_labels =(np.unique(vol))
    # Optionally shift labels so that min label is 1; 0 becomes background (non-meshed in standard pygalmesh flow)
    shift_applied = False
    if not args.no_shift_labels:
        min_label = int(unique_labels.min())
        if min_label <= 0:
            shift = 1 - min_label
            vol = vol + shift
            shift_applied = True
            unique_labels = np.unique(vol)
            print(
                f"[INFO] Shifted labels by +{shift} so that background=0 is reserved "
                f"and phases are >0. New unique labels: {unique_labels}"
            )
        else:
            print("[INFO] Labels already >0, no shift needed.")
    else:
        print(
            "[INFO] --no-shift-labels enabled. "
            "Remember: subdomain 0 (if present) will NOT be meshed by pygalmesh."
        )
    # At this point, label 0 (if present) is background; labels>0 are meshed
    nonzero_labels = [int(v) for v in unique_labels if v != 0]
    if not nonzero_labels:
        raise ValueError("No non-zero labels present; nothing to mesh.")
    
    print(f"[INFO] Will mesh subdomains with labels: {nonzero_labels}")
    voxel_size = (dx, dy, dz)
    pygalmesh.save_inr(vol, voxel_size, 'temp.inr')
    print(f"[INFO] Using voxel_size (dx,dy,dz) = {voxel_size}")

    # Choose default meshing parameters if not provided
    # vmin = min(dx, dy, dz)
    max_facet_distance = args.max_facet_distance
    max_cell_circumradius = args.max_cell_circumradius
    max_circumradius_edge_ratio = args.max_circumradius_edge_ratio
    min_facet_angle = args.min_facet_angle
    max_edge_size_at_feature_edges = args.max_edge_size_at_feature_edges
    max_radius_surface_delaunay_ball = args.max_radius_surface_delaunay_ball
    exude_time_limit = args.exude_time_limit
    exude_sliver_bound = args.exude_sliver_bound


    print(f"[INFO] max_facet_distance      = {max_facet_distance}")
    print(f"[INFO] max_cell_circumradius   = {max_cell_circumradius}")

    # --- CGAL meshing via pygalmesh ---
    print("[INFO] Calling pygalmesh.generate_from_array()...")
    mesh = pygalmesh.generate_from_array_with_features(
        vol,
        voxel_size,
        lloyd=True,
        odt=True,
        perturb=True,
        exude=True,
        max_facet_distance=max_facet_distance,
        max_cell_circumradius=max_cell_circumradius,
        max_circumradius_edge_ratio=max_circumradius_edge_ratio,
        min_facet_angle=min_facet_angle,
        max_edge_size_at_feature_edges=max_edge_size_at_feature_edges,
        max_radius_surface_delaunay_ball=max_radius_surface_delaunay_ball,
        exude_time_limit=exude_time_limit,
        exude_sliver_bound=exude_sliver_bound,
        verbose=True,
    )
    mesh.write('testmesh.vtk')
    # mesh is a meshio.Mesh-like object; ensure it's meshio.Mesh
    if not isinstance(mesh, meshio.Mesh):
        # Older/newer versions might still be compatible; wrap if needed
        mesh = meshio.Mesh(points=mesh.points, cells=mesh.cells)

    print("[INFO] Meshing done.")
    print(f"       Points: {mesh.points.shape}")
    for cb in mesh.cells:
        print(f"       Cell block: {cb.type}, count={cb.data.shape[0]}")

    # (depends on pygalmesh version; often "tetra" block has cell_data["subdomain"] or similar)
    if mesh.cell_data:
        print("[INFO] Available cell_data keys:")
        for key, data_list in mesh.cell_data.items():
            print(f"  - {key}: {[d.shape for d in data_list]}")
    else:
        print("[WARN] No cell_data present; subdomain labels may not be exported.")

    base = args.out_prefix
    out_exodus = base + ".e"
    # out_vtk = base + ".vtk"


    print(f"[INFO] Writing exodus mesh: {out_exodus}")
    
    write_phases_exodus(mesh, out_exodus, domain_key="medit:ref") # replaces Meshio's garbage BDF writer

    # print(f"[INFO] Writing VTK mesh for inspection: {out_vtk}")
    # mesh.write(out_vtk)

    # print("[DONE] Mesh generation completed.")
    # print(f"       -> Nastran: {out_bdf}")
    # print(f"       -> VTK:     {out_vtk}")
    # if shift_applied:
    #     print("       NOTE: Labels were shifted so that original phases are now >0.")
    #     print("             You may want to keep a mapping of original_label -> CGAL_subdomain_index.")


if __name__ == "__main__":
    main()
