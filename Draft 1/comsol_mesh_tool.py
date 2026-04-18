"""
COMSOL Mesh Tool
================
Parses a COMSOL .mphtxt mesh export, reconstructs the model,
and lets you extract points on any named boundary / side.

Supported element types (from COMSOL text export):
  - vtx   : 1-node vertex
  - edg   : 2-node line edge
  - tri   : 3-node triangle
  - quad  : 4-node quad
  - tet   : 4-node tetrahedron
  - hex   : 8-node hexahedron
  - prism : 6-node prism

Usage
-----
    mesh = COMSOLMesh("my_model.mphtxt")
    mesh.summary()

    # List available boundaries
    mesh.list_boundaries()

    # Extract all node coordinates on boundary tag 3
    pts = mesh.get_boundary_points(tag=3)

    # Or select by domain index (0-based index in the .mphtxt domain list)
    pts = mesh.get_boundary_points(domain_index=0)

    # Quick 3-D scatter plot of the extracted points
    mesh.plot_boundary_points(tag=3)

    # Export extracted points to a CSV
    mesh.export_boundary_points(tag=3, filename="side_points.csv")
"""

import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 – registers 3-D projection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MeshSection:
    """One element-type section inside a COMSOL mesh text file."""
    elem_type: str              # e.g. "tri", "tet", "edg"
    n_nodes_per_elem: int       # nodes per element
    connectivity: np.ndarray    # shape (n_elems, n_nodes_per_elem), 0-based
    domains: np.ndarray         # shape (n_elems,)  geometric domain tag


@dataclass
class COMSOLMesh:
    """
    Parses a COMSOL .mphtxt mesh file and exposes helpers
    for model reconstruction and boundary-point extraction.
    """

    filepath: str
    nodes: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    sections: List[MeshSection] = field(default_factory=list)
    spacedim: int = 3
    _domain_names: Dict[int, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructor: parse on init
    # ------------------------------------------------------------------

    def __post_init__(self):
        self._parse(self.filepath)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, filepath: str):
        """Read a COMSOL text mesh (.mphtxt) file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Mesh file not found: {filepath}")

        with open(filepath, "r") as fh:
            lines = [ln.rstrip("\n") for ln in fh]

        i = 0
        n_lines = len(lines)

        def skip_blanks_and_comments():
            nonlocal i
            while i < n_lines and (lines[i].strip() == "" or lines[i].strip().startswith("#")):
                i += 1

        def next_tokens():
            """Return stripped tokens of the current line and advance."""
            nonlocal i
            skip_blanks_and_comments()
            if i >= n_lines:
                return []
            toks = lines[i].split()
            i += 1
            return toks

        # ---- header -------------------------------------------------------
        skip_blanks_and_comments()
        while i < n_lines:
            line = lines[i].strip()
            if line.startswith("# Space dimensions"):
                self.spacedim = int(lines[i + 1].strip())
            if line.startswith("# Mesh vertex coordinates"):
                i += 1
                break
            i += 1

        # ---- node coordinates ---------------------------------------------
        nodes_list: List[List[float]] = []
        while i < n_lines:
            line = lines[i].strip()
            if line == "" or line.startswith("#"):
                i += 1
                # stop at section boundary
                if i < n_lines and lines[i].strip().startswith("# "):
                    break
                continue
            try:
                coords = list(map(float, line.split()))
            except ValueError:
                i += 1
                break
            # Pad to 3-D if the mesh is 2-D
            while len(coords) < 3:
                coords.append(0.0)
            nodes_list.append(coords[:3])
            i += 1

        self.nodes = np.array(nodes_list, dtype=float)
        print(f"[COMSOLMesh] Loaded {len(self.nodes)} nodes "
              f"(space dim = {self.spacedim})")

        # ---- element sections --------------------------------------------
        ELEM_INFO = {
            "vtx":   1,
            "edg":   2,
            "edg2":  3,
            "tri":   3,
            "tri2":  6,
            "quad":  4,
            "quad2": 8,
            "tet":   4,
            "tet2": 10,
            "hex":   8,
            "pyr":   5,
            "prism": 6,
        }

        while i < n_lines:
            line = lines[i].strip()
            # Detect element-type header  e.g.  "4 # number of element types"
            m = re.match(r"(\d+)\s*#\s*number of element types", line, re.I)
            if m:
                n_types = int(m.group(1))
                i += 1
                for _ in range(n_types):
                    # skip to element-type name line
                    while i < n_lines and not lines[i].strip().startswith("#"):
                        i += 1
                    # e.g. "# Mesh element: tri"
                    m2 = re.search(r"Mesh element[s]?:\s*(\w+)", lines[i], re.I)
                    elem_type = m2.group(1).lower() if m2 else "unknown"
                    i += 1

                    toks = next_tokens()
                    n_elems = int(toks[0]) if toks else 0
                    n_npe   = ELEM_INFO.get(elem_type, 0)

                    # Read connectivity (0-based in COMSOL .mphtxt)
                    conn_rows = []
                    for _ in range(n_elems):
                        toks = next_tokens()
                        conn_rows.append(list(map(int, toks)))
                    connectivity = np.array(conn_rows, dtype=int) if conn_rows else np.empty((0, max(n_npe, 1)), dtype=int)

                    # Skip to domain line
                    while i < n_lines and not re.search(r"\d+\s*#\s*number of (geometric )?domains", lines[i], re.I):
                        i += 1
                    i += 1  # consume the domain-count line

                    dom_rows = []
                    for _ in range(n_elems):
                        toks = next_tokens()
                        dom_rows.append(int(toks[0]) if toks else -1)
                    domains = np.array(dom_rows, dtype=int) if dom_rows else np.empty(0, dtype=int)

                    sec = MeshSection(
                        elem_type=elem_type,
                        n_nodes_per_elem=n_npe,
                        connectivity=connectivity,
                        domains=domains,
                    )
                    self.sections.append(sec)
                    print(f"[COMSOLMesh]   section '{elem_type}': "
                          f"{n_elems} elements, domains {np.unique(domains).tolist()}")
                break
            i += 1

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def summary(self):
        """Print a short summary of the loaded mesh."""
        print("=" * 55)
        print(f"  Mesh file : {self.filepath}")
        print(f"  Nodes     : {len(self.nodes)}")
        print(f"  Sections  : {len(self.sections)}")
        for sec in self.sections:
            doms = np.unique(sec.domains).tolist()
            print(f"    [{sec.elem_type:6s}] {len(sec.connectivity):6d} elems  "
                  f"domains: {doms}")
        print("=" * 55)

    def list_boundaries(self):
        """List every (section, domain tag) combination found in the mesh."""
        print("\nAvailable boundary tags:")
        print(f"  {'Section':<10} {'Domain tag':<12} {'# elements'}")
        print("  " + "-" * 38)
        for sec in self.sections:
            for tag in np.unique(sec.domains):
                n = int(np.sum(sec.domains == tag))
                print(f"  {sec.elem_type:<10} {tag:<12d} {n}")
        print()

    # ------------------------------------------------------------------
    # Point extraction
    # ------------------------------------------------------------------

    def get_boundary_points(
        self,
        tag: Optional[int] = None,
        domain_index: Optional[int] = None,
        elem_types: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Return unique node coordinates that belong to a selected side.

        Parameters
        ----------
        tag          : int  – domain tag in the COMSOL mesh (1-based as in GUI).
        domain_index : int  – 0-based index into the sorted unique domain list
                              (alternative to `tag`).
        elem_types   : list[str]  – restrict to these element types, e.g.
                                    ["tri"] for surface faces in a 3-D model.
                                    Defaults to the lowest-dimensional type.

        Returns
        -------
        pts : np.ndarray, shape (N, 3)
        """
        if tag is None and domain_index is None:
            raise ValueError("Provide either `tag` or `domain_index`.")

        # Resolve domain_index → tag
        if domain_index is not None:
            all_tags = sorted({int(d) for sec in self.sections
                               for d in np.unique(sec.domains)})
            if domain_index >= len(all_tags):
                raise IndexError(
                    f"domain_index={domain_index} out of range "
                    f"(0 .. {len(all_tags)-1})"
                )
            tag = all_tags[domain_index]
            print(f"[get_boundary_points] domain_index {domain_index} → tag {tag}")

        # Auto-select lowest-dimensional element type if not specified
        if elem_types is None:
            dim_order = ["vtx", "edg", "edg2", "tri", "tri2", "quad", "quad2",
                         "tet", "tet2", "hex", "pyr", "prism"]
            present = {sec.elem_type for sec in self.sections}
            # surface elements for 3-D: tri / quad; edge elements for 2-D: edg
            surface_types = [t for t in dim_order if t in present]
            if not surface_types:
                surface_types = list(present)
            # For 3-D meshes keep only the surface-boundary types
            boundary_types = []
            for t in surface_types:
                if self.spacedim == 3 and t in ("tri", "tri2", "quad", "quad2",
                                                 "edg", "edg2", "vtx"):
                    boundary_types.append(t)
                elif self.spacedim == 2 and t in ("edg", "edg2", "vtx"):
                    boundary_types.append(t)
            if not boundary_types:
                boundary_types = surface_types[:1]
            elem_types = boundary_types

        node_indices: set = set()
        for sec in self.sections:
            if sec.elem_type not in elem_types:
                continue
            mask = sec.domains == tag
            if not np.any(mask):
                continue
            node_indices.update(sec.connectivity[mask].ravel().tolist())

        if not node_indices:
            print(f"[Warning] No nodes found for tag={tag}, "
                  f"elem_types={elem_types}.")
            return np.empty((0, 3))

        idx_arr = np.array(sorted(node_indices), dtype=int)
        pts = self.nodes[idx_arr]
        print(f"[get_boundary_points] tag={tag}: {len(pts)} unique nodes extracted.")
        return pts

    def get_boundary_elements(
        self,
        tag: int,
        elem_types: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (connectivity, node_coords) for elements on the given boundary.

        Returns
        -------
        conn   : np.ndarray  – element connectivity (0-based, global indices)
        coords : np.ndarray  – all node coordinates (same as self.nodes)
        """
        if elem_types is None:
            elem_types = [sec.elem_type for sec in self.sections]

        conn_all = []
        for sec in self.sections:
            if sec.elem_type not in elem_types:
                continue
            mask = sec.domains == tag
            if np.any(mask):
                conn_all.append(sec.connectivity[mask])

        if not conn_all:
            return np.empty((0,), dtype=int), self.nodes

        return np.vstack(conn_all), self.nodes

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_boundary_points(
        self,
        tag: Optional[int] = None,
        domain_index: Optional[int] = None,
        title: str = "",
        color: str = "#D85A30",
    ):
        """
        3-D scatter plot of boundary points.
        All other nodes are shown as light background dots for context.
        """
        pts = self.get_boundary_points(tag=tag, domain_index=domain_index)
        if len(pts) == 0:
            print("No points to plot.")
            return

        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection="3d")

        # Background mesh nodes
        ax.scatter(
            self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2],
            c="#B4B2A9", s=2, alpha=0.2, linewidths=0, label="All nodes"
        )
        # Selected boundary
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=color, s=18, alpha=0.85, linewidths=0,
            label=f"Boundary tag={tag}"
        )

        label = title or f"Boundary points  (tag={tag}, n={len(pts)})"
        ax.set_title(label, fontsize=13, pad=12)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(markerscale=2, fontsize=10)
        plt.tight_layout()
        plt.show()

    def plot_all_boundaries(self, show_labels: bool = True):
        """
        Color-coded 3-D scatter of all domain boundaries found in
        the lowest-dimensional element sections.
        """
        fig = plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(111, projection="3d")

        colors = plt.cm.tab20.colors
        color_idx = 0

        for sec in self.sections:
            # Only surface / boundary sections
            if self.spacedim == 3 and sec.elem_type not in (
                "tri", "tri2", "quad", "quad2"
            ):
                continue
            if self.spacedim == 2 and sec.elem_type not in ("edg", "edg2"):
                continue

            for tag in np.unique(sec.domains):
                mask = sec.domains == tag
                idxs = np.unique(sec.connectivity[mask].ravel())
                pts  = self.nodes[idxs]
                c    = colors[color_idx % len(colors)]
                color_idx += 1
                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    c=[c], s=12, alpha=0.8, linewidths=0,
                    label=f"{sec.elem_type} tag={tag}"
                )

        ax.set_title("All boundaries (color-coded by domain tag)", fontsize=13)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        if show_labels:
            ax.legend(loc="upper left", fontsize=8, markerscale=1.5,
                      ncol=2, framealpha=0.6)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_boundary_points(
        self,
        filename: str = "boundary_points.csv",
        tag: Optional[int] = None,
        domain_index: Optional[int] = None,
    ):
        """Write extracted boundary points to a CSV file."""
        pts = self.get_boundary_points(tag=tag, domain_index=domain_index)
        if len(pts) == 0:
            print("Nothing to export.")
            return

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["x", "y", "z"])
            writer.writerows(pts.tolist())

        print(f"[Export] {len(pts)} points written to '{filename}'")


# ---------------------------------------------------------------------------
# Utility: create a simple synthetic COMSOL-style .mphtxt for testing
# ---------------------------------------------------------------------------

def create_sample_mphtxt(filepath: str = "sample_mesh.mphtxt"):
    """
    Write a minimal synthetic COMSOL text mesh representing a unit cube
    with 8 vertices, 6 quad faces (one per side), and 6 surface quads.
    Use this to test the parser when no real .mphtxt file is available.
    """
    content = """\
# Created by COMSOL Multiphysics (synthetic sample)

# Space dimensions
3

# Mesh vertex coordinates  (8 corners of a unit cube)
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1

# ---- Element sections -----------------------------------------------

1 # number of element types

# Mesh element: quad
6 # number of elements
# connectivity (0-based)
0 1 2 3
4 5 6 7
0 1 5 4
2 3 7 6
0 3 7 4
1 2 6 5
# 1 # number of geometric domains
1 # number of domains per element (column)
1
2
3
4
5
6
"""
    with open(filepath, "w") as fh:
        fh.write(content)
    print(f"[Sample] Wrote synthetic mesh to '{filepath}'")
    return filepath


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # -----------------------------------------------------------------
    # 1. Use a real file if passed on the command line, else create sample
    # -----------------------------------------------------------------
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    else:
        mesh_file = create_sample_mphtxt("sample_mesh.mphtxt")

    # -----------------------------------------------------------------
    # 2. Load mesh
    # -----------------------------------------------------------------
    mesh = COMSOLMesh(mesh_file)
    mesh.summary()

    # -----------------------------------------------------------------
    # 3. Inspect available boundaries
    # -----------------------------------------------------------------
    mesh.list_boundaries()

    # -----------------------------------------------------------------
    # 4. Extract points on the first boundary  (domain tag = 1)
    # -----------------------------------------------------------------
    pts = mesh.get_boundary_points(tag=1)
    print(f"\nPoints on tag=1:\n{pts}")

    # -----------------------------------------------------------------
    # 5. Extract by domain_index (e.g. second boundary in the sorted list)
    # -----------------------------------------------------------------
    pts2 = mesh.get_boundary_points(domain_index=1)
    print(f"\nPoints on domain_index=1:\n{pts2}")

    # -----------------------------------------------------------------
    # 6. Export to CSV
    # -----------------------------------------------------------------
    mesh.export_boundary_points(tag=1, filename="side_1_points.csv")

    # -----------------------------------------------------------------
    # 7. Plot  (comment out in headless environments)
    # -----------------------------------------------------------------
    try:
        mesh.plot_boundary_points(tag=1, title="Side 1 – extracted points")
        mesh.plot_all_boundaries()
    except Exception as e:
        print(f"[Plot skipped] {e}")
