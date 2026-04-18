# COMSOL Extract Simulation

A Python-based toolkit for extracting, analyzing, and simulating particle flow through COMSOL mesh models. This project focuses on mesh processing, boundary point extraction, and particle residence time calculations.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Python Scripts](#python-scripts)
- [Files & Data](#files--data)
- [How to Run](#how-to-run)
- [Requirements](#requirements)

---

## Project Overview

This repository contains tools for:
1. **Mesh Visualization**: Parse and visualize COMSOL mesh files (.mphtxt format)
2. **Boundary Extraction**: Extract inlet, outlet, and side boundary points from meshes
3. **Particle Simulation**: Trace particle flow through the mesh to calculate residence times

---

## Python Scripts

### 1. **3D_structure_builder.py**
**Purpose**: Visualizes COMSOL mesh in 3D

**What it does**:
- Reads a COMSOL `.mphtxt` mesh file
- Extracts all vertex coordinates
- Creates a 3D scatter plot visualization using matplotlib
- Subsamples points for performance optimization

**Input**: COMSOL mesh file (`.mphtxt`)  
**Output**: 3D visualization plot

---

### 2. **Inlets_outlets_points_extractor.py**
**Purpose**: Extracts inlet and outlet boundary points from mesh

**What it does**:
- Parses COMSOL mesh file and locates all vertices
- Filters vertices at inlet boundary (Y ≈ -45)
- Filters vertices at outlet boundary (Z ≈ 60)
- Splits inlets/outlets by X coordinate (left/right)
- Exports to `.txt` and `.csv` files

**Input**: COMSOL mesh file (`.mphtxt`)  
**Output**:
- `inlet_1.txt` - Left inlet points
- `inlet_2.txt` - Right inlet points
- `outlet_1.txt` - Left outlet points
- `outlet_2.txt` - Right outlet points

---

### 3. **Point extracter.py**
**Purpose**: Extracts inlet and outlet points with flexible boundary filtering

**What it does**:
- Reads COMSOL mesh and extracts all vertices
- Identifies inlet points at a target Y coordinate
- Identifies outlet points at a target Z coordinate
- Exports to CSV files with X,Y,Z coordinates
- Useful for fluid flow simulations

**Input**: COMSOL mesh file (`.mphtxt`)  
**Output**:
- `inlets_y_minus_45.csv` - All inlet points
- `outlets_z_60.csv` - All outlet points

---

### 4. **Mesh_side_points_finder.py** (also in Draft 1 as `comsol_mesh_tool.py`)
**Purpose**: Comprehensive COMSOL mesh parser for extracting boundary points

**What it does**:
- Parses COMSOL `.mphtxt` mesh files
- Reconstructs mesh geometry from connectivity data
- Supports multiple element types: vertices, edges, triangles, quads, tetrahedra, hexahedra, prisms
- Extracts points from named boundaries or specific domains
- Provides visualization and export functionality
- Enables interactive boundary selection

**Usage Examples**:
```python
from comsol_mesh_tool import COMSOLMesh

# Load mesh
mesh = COMSOLMesh("model.mphtxt")
mesh.summary()

# List available boundaries
mesh.list_boundaries()

# Extract points from specific boundary
pts = mesh.get_boundary_points(tag=3)

# Export to CSV
mesh.export_boundary_points(tag=3, filename="side_points.csv")

# Visualize
mesh.plot_boundary_points(tag=3)
```

**Input**: COMSOL mesh file (`.mphtxt`)  
**Output**: Extracted points, CSV exports, 3D plots

---

### 5. **particle_tracing_module.py**
**Purpose**: Particle residence time simulation and tracing

**What it does**:
- Performs Lagrangian particle tracing through velocity fields
- Reads velocity data from two files (e.g., `Velo_1.txt`, `Velo_2.txt`)
- Uses inlet points as starting positions for particles
- Simulates particle trajectories using ODE solver
- Tracks when particles exit through outlets
- Calculates residence time (time particle spends in domain)
- Generates residence time histogram
- Uses multi-core parallel processing for performance
- Includes PyVista-based interactive 3D visualization

**Key Features**:
- Parallel particle tracing (uses all CPU cores)
- Spatial interpolation for velocity field
- KD-Tree acceleration for nearest-neighbor searches
- Interactive 3D visualization dashboard
- Residence time statistical analysis

**Input Files** (configurable in script):
- `Velo_1.txt` - Velocity field data (first component)
- `Velo_2.txt` - Velocity field data (second component)
- `inlets_y_minus_45.csv` - Inlet point coordinates
- `outlets_z_60.csv` - Outlet point coordinates

**Output**:
- `blood_residence_times.csv` - Residence time data for each particle
- `tracing_debug.log` - Simulation log and debug information
- Residence time histogram plot

---

## Files & Data

### What to Upload to Run the Simulation

To use this toolkit with your own COMSOL model, you need to provide the following input files:

**Required Input Files** (from your COMSOL model):
1. **Mesh File** (`.mphtxt` format):
   - Export your COMSOL model as text: **File → Export → Mesh as text**
   - Save as a `.mphtxt` file in the project directory

2. **Velocity Field Data** (`.txt` files):
   - `Velo_1.txt` - Velocity component 1 (X or radial direction)
   - `Velo_2.txt` - Velocity component 2 (Y or axial direction)
   - Format: Space or comma-separated coordinate + velocity values
   - These are generated from COMSOL simulations

3. **Optional**: Original COMSOL model file (`.mph`)
   - Useful for reference or further modifications
   - Not required to run the Python scripts

**Note**: The Python scripts are already in this repository. Users only need to provide their own COMSOL data files.

---

## How to Run

### Prerequisites

Install required Python packages:
```bash
pip install numpy matplotlib scipy pyvista scikit-learn joblib
```

### Step 1: Prepare Mesh Files

Export your COMSOL model as text format:
1. In COMSOL, go to **File → Export → Mesh as text**
2. Save as `.mphtxt` file in the project directory

### Step 2: Extract Boundary Points

Run one of the extraction scripts:

```bash
# Option A: Quick inlet/outlet extraction
python "Point_extracter.py"

# Option B: Comprehensive boundary extraction
python "Mesh_side_points_finder.py"
```

This generates CSV files with boundary point coordinates.

### Step 3: Visualize Mesh (Optional)

```bash
python "3D_structure_builder.py"
```

This opens an interactive 3D plot of the mesh.

### Step 4: Run Particle Simulation

```bash
python "particle_tracing_module.py"
```

**What happens**:
1. Script loads velocity field data and inlet/outlet points
2. Traces 2500 particles through the domain
3. Calculates residence time for each particle
4. Generates histogram and CSV output
5. Shows summary statistics in console

**Output files**:
- `blood_residence_times.csv` - Particle residence times
- `tracing_debug.log` - Detailed simulation log

---

## Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
- **numpy** - Array operations and matrix math
- **matplotlib** - 2D/3D plotting
- **scipy** - Scientific computing (ODE solver, interpolation, spatial algorithms)
- **pyvista** - 3D visualization and rendering
- **joblib** - Parallel processing

### System Requirements
- Multi-core CPU recommended (for parallel particle tracing)
- 8GB+ RAM (for large meshes and simulations)
- COMSOL installation (for generating `.mph` and `.mphtxt` files)

---

## Quick Reference

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| `3D_structure_builder.py` | .mphtxt | 3D Plot | Mesh visualization |
| `Inlets_outlets_points_extractor.py` | .mphtxt | .txt files | Extract boundary points (split left/right) |
| `Point_extracter.py` | .mphtxt | .csv files | Extract inlet/outlet points |
| `Mesh_side_points_finder.py` | .mphtxt | .csv, plots | Comprehensive boundary extraction |
| `particle_tracing_module.py` | Velocity + points files | .csv, histogram | Particle residence time simulation |

---

## Troubleshooting

**Q: "ModuleNotFoundError" when running scripts?**  
A: Install missing package with `pip install <package_name>`

**Q: Mesh visualization is slow/laggy?**  
**A: The subsample_rate in `3D_structure_builder.py` reduces plotted points for performance

**Q: Particle simulation taking too long?**  
**A: Reduce `NUM_PARTICLES` in `particle_tracing_module.py` or run on a machine with more CPU cores

**Q: No inlet/outlet points extracted?**  
A: Verify coordinate values (Y=-45, Z=60) match your COMSOL model geometry

---

## Contact & Support

For questions or issues, refer to the COMSOL documentation or review the script docstrings and comments.
