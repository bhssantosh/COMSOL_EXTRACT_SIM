import numpy as np
import matplotlib.pyplot as plt

def visualize_full_mesh(filename, subsample_rate=10):
    print(f"Reading '{filename}'...")
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    start_index = 0
    num_vertices = 0
    
    # Locate the vertex data in the file
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
    print(f"Total vertices loaded: {len(vertices)}")
    
    # Subsample the points to prevent the 3D viewer from lagging
    # A rate of 10 means it takes every 10th point
    plot_points = vertices[::subsample_rate]
    print(f"Plotting {len(plot_points)} points for better performance...")

    # --- 3D Visualization ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of the full mesh
    ax.scatter(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2], 
               c='gray', marker='.', s=1, alpha=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Full COMSOL Mesh Visualization')
    
    # Set background to white
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.show()

# Run the visualization 
# (You can change subsample_rate=1 if you have a powerful GPU and want ALL points)
visualize_full_mesh('mesh_1.mphtxt', subsample_rate=10)