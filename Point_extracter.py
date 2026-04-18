import numpy as np
import matplotlib.pyplot as plt

def extract_inlets_and_outlets(filename):
    print(f"Reading '{filename}'...")
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    start_index = 0
    num_vertices = 0
    
    # Locate the vertex data in the COMSOL mphtxt file
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
    print(f"Total vertices loaded: {len(vertices)}\n")
    
    # ---------------------------------------------------------
    # 1. Extract Inlets (Negative Y direction ends)
    # ---------------------------------------------------------
    target_y = -45.0
    inlet_mask = np.abs(vertices[:, 1] - target_y) < 1e-3
    inlet_points = vertices[inlet_mask]
    
    np.savetxt('inlets_y_minus_45.csv', inlet_points, delimiter=',', header='X,Y,Z', comments='')
    print(f"✅ Extracted {len(inlet_points)} Inlet points to 'inlets_y_minus_45.csv'")

    # ---------------------------------------------------------
    # 2. Extract Outlets (Z direction ends)
    # ---------------------------------------------------------
    target_z = 60.0
    outlet_mask = np.abs(vertices[:, 2] - target_z) < 1e-3
    outlet_points = vertices[outlet_mask]
    
    np.savetxt('outlets_z_60.csv', outlet_points, delimiter=',', header='X,Y,Z', comments='')
    print(f"✅ Extracted {len(outlet_points)} Outlet points to 'outlets_z_60.csv'\n")

    # ---------------------------------------------------------
    # 3. 3D Visualization
    # ---------------------------------------------------------
    print("Opening 3D viewer...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot a faint outline of the entire mesh (subsampled so it doesn't lag)
    sub_mesh = vertices[::50] 
    ax.scatter(sub_mesh[:, 0], sub_mesh[:, 1], sub_mesh[:, 2], 
               c='lightgray', marker='.', s=1, alpha=0.1)

    # Plot the Inlets (Blue)
    if len(inlet_points) > 0:
        ax.scatter(inlet_points[:, 0], inlet_points[:, 1], inlet_points[:, 2], 
                   c='blue', marker='.', s=10, label='Inlets (Y ≈ -45)')
        
    # Plot the Outlets (Red)
    if len(outlet_points) > 0:
        ax.scatter(outlet_points[:, 0], outlet_points[:, 1], outlet_points[:, 2], 
                   c='red', marker='.', s=10, label='Outlets (Z ≈ 60)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('COMSOL Mesh Boundaries Extracted')
    ax.legend()
    
    # Clean up the background grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.show()

# Run the extraction on your uploaded file
extract_inlets_and_outlets('mesh_1.mphtxt')