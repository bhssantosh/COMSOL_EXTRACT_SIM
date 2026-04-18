import numpy as np

def export_fpt_boundaries(filename):
    print(f"Reading '{filename}'...")
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    start_index = 0
    num_vertices = 0
    
    # Locate the vertex data
    for i, line in enumerate(lines):
        if "number of mesh vertices" in line:
            num_vertices = int(line.split()[0])
        if "Mesh vertex coordinates" in line:
            start_index = i + 1
            break
            
    # Extract all vertices
    vertices = []
    for i in range(start_index, start_index + num_vertices):
        coords = list(map(float, lines[i].split()))[:3]
        vertices.append(coords)
        
    vertices = np.array(vertices)
    
    # ---------------------------------------------------------
    # 1. Extract and Split Inlets (Y ≈ -45)
    # ---------------------------------------------------------
    inlets = vertices[np.abs(vertices[:, 1] - (-45.0)) < 1e-3]
    
    # Split into left and right inlets based on X coordinate
    inlet_1 = inlets[inlets[:, 0] < 0]
    inlet_2 = inlets[inlets[:, 0] >= 0]
    
    # Save as plain text format (comma separated)
    np.savetxt('inlet_1.txt', inlet_1, delimiter=',', fmt='%.6f', header='X,Y,Z', comments='')
    np.savetxt('inlet_2.txt', inlet_2, delimiter=',', fmt='%.6f', header='X,Y,Z', comments='')
    print(f"✅ Saved Inlet 1 ({len(inlet_1)} points) to 'inlet_1.txt'")
    print(f"✅ Saved Inlet 2 ({len(inlet_2)} points) to 'inlet_2.txt'")

    # ---------------------------------------------------------
    # 2. Extract and Split Outlets (Z ≈ 60)
    # ---------------------------------------------------------
    outlets = vertices[np.abs(vertices[:, 2] - 60.0) < 1e-3]
    
    # Split into left and right outlets based on X coordinate
    outlet_1 = outlets[outlets[:, 0] < 0]
    outlet_2 = outlets[outlets[:, 0] >= 0]
    
    np.savetxt('outlet_1.txt', outlet_1, delimiter=',', fmt='%.6f', header='X,Y,Z', comments='')
    np.savetxt('outlet_2.txt', outlet_2, delimiter=',', fmt='%.6f', header='X,Y,Z', comments='')
    print(f"✅ Saved Outlet 1 ({len(outlet_1)} points) to 'outlet_1.txt'")
    print(f"✅ Saved Outlet 2 ({len(outlet_2)} points) to 'outlet_2.txt'")

# Run the extraction
export_fpt_boundaries('mesh_1.mphtxt')