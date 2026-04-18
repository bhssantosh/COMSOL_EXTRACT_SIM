import numpy as np                      # For heavy matrix math and array manipulation
import pyvista as pv                    # For high-performance 3D rendering and the UI dashboard
from scipy.interpolate import LinearNDInterpolator # For spatial velocity interpolation
from scipy.integrate import solve_ivp   # The core Ordinary Differential Equation (ODE) solver
from scipy.spatial import cKDTree, Delaunay # For fast nearest-neighbor searches and mesh building
import bisect                           # For fast binary searching through time arrays
import time                             # To track how long calculations take
import gc                               # Garbage Collector to manually free up RAM
from joblib import Parallel, delayed    # For utilizing all CPU cores simultaneously
import vtk                              # The core C++ engine behind PyVista (used for the zoom tool)
import tkinter as tk                    # For native Windows pop-up dialog boxes
from tkinter import simpledialog, filedialog        # Specifically for the text-entry pop-up
import matplotlib.pyplot as plt         # For plotting the Residence Time histogram
import logging                          # For debug logging to console and file

# ==========================================
# 0. Initialize Debug Logger
# ==========================================
# Sets up a system that prints updates to the console AND saves them to a text file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("tracing_debug.log", mode='w'),
        logging.StreamHandler()
    ]
)

logging.info("--- STARTING PARTICLE TRACING SIMULATION ---")

# ==========================================
# 1. User Parameters & File Paths
# ==========================================
VELOCITY_FILE_1 = "Velo_1.txt"  
VELOCITY_FILE_2 = "Velo_2.txt"  
INLET_FILE = "inlets_y_minus_45.csv"
OUTLET_FILE = "outlets_z_60.csv"

NUM_PARTICLES = 2500 
PARTICLE_RADIUS = 0.5 

# How close a particle must get to the outlet plane to be considered "exited"
HIT_TOLERANCE = 2.0          

# Creates an array of time steps [0.0, 0.01, 0.02 ... 3.00]
TIME_STEPS_ARRAY = np.arange(0.0, 3.01, 0.01)  
SOLVER_END_TIME = 3.0 

# Velocities below 10 mm/s will trigger the magenta stagnation warning
STAGNATION_THRESHOLD = 10.0  

# ==========================================
# 2. Load Data and Apply Scale Corrections
# ==========================================
# Load boundary coordinates
inlet_nodes = np.loadtxt(INLET_FILE, delimiter=',', skiprows=1)
outlet_nodes = np.loadtxt(OUTLET_FILE, delimiter=',', skiprows=1)

# Dynamically find the absolute top and bottom of the medical model
OUTLET_Z_MAX = np.max(outlet_nodes[:, 2])
INLET_Z_MIN = np.min(inlet_nodes[:, 2])

logging.info(f"Z-Bounds detected: Min={INLET_Z_MIN:.2f}, Max={OUTLET_Z_MAX:.2f}")

# Load the massive velocity text files in chunks to prevent RAM overflow
logging.info("Loading velocity data...")
data1 = np.loadtxt(VELOCITY_FILE_1, comments='%')
points = data1[:, 0:3]          # Extract X, Y, Z coordinates
vels1 = data1[:, 3:]            # Extract velocity vectors
del data1; gc.collect()         # Delete raw data and force RAM cleanup

data2 = np.loadtxt(VELOCITY_FILE_2, comments='%')
vels2 = data2[:, 3:]
del data2; gc.collect()

# Combine chunks and convert COMSOL's (m/s) to our physical mesh scale (mm/s)
all_velocities = np.column_stack((vels1, vels2)) * 1000.0 
del vels1, vels2; gc.collect()

# Create the visual 3D shell for PyVista from our raw data points
pv_mesh = pv.PolyData(points)

# ==========================================
# 3. Interpolation and Failsafes
# ==========================================
# cKDTree organizes points spatially so we can instantly find the "nearest neighbor" later
point_tree = cKDTree(points) 

# Delaunay connects all our raw points into a 3D web of triangles (tetrahedrons)
triangulation = Delaunay(points)

logging.info("Building 4D Spatio-Temporal Interpolators...")
# This loop pre-builds a spatial interpolator for every single time step
interpolators = [LinearNDInterpolator(triangulation, np.column_stack((all_velocities[:, i*3], all_velocities[:, i*3+1], all_velocities[:, i*3+2])), fill_value=np.nan) for i in range(len(TIME_STEPS_ARRAY))]

def get_velocity_4d(t, pos):
    """The core function called thousands of times per second by the ODE solver."""
    
    # HARD CEILING: Instantly stop the particle if it reaches the outlet height
    if pos[2] >= OUTLET_Z_MAX: 
        return [0.0, 0.0, 0.0]
    
    # Find exactly which two time steps the current time 't' falls between
    idx = min(bisect.bisect_right(TIME_STEPS_ARRAY, t) - 1, len(TIME_STEPS_ARRAY)-2)
    idx = max(0, idx)
    
    # Ask the spatial interpolator for the velocity at this exact X,Y,Z for both time steps
    v0 = interpolators[idx](pos[0], pos[1], pos[2])
    v1 = interpolators[idx + 1](pos[0], pos[1], pos[2])

    # FAILSAFE: If the particle stepped outside the mesh boundary (returns NaN)
    if np.isnan(v0).any() or np.isnan(v1).any():
        # Find the nearest valid physical node in our data
        dist, n_idx = point_tree.query(pos)
        # If it flew too far away (2.0mm), kill it.
        if dist > 2.0: return [0.0, 0.0, 0.0]
        # Otherwise, rescue it by assigning it the velocity of that nearest node
        v0 = all_velocities[n_idx, idx*3 : idx*3+3]
        v1 = all_velocities[n_idx, (idx+1)*3 : (idx+1)*3+3]
        # If the nearest node is also corrupt, stop the particle
        if np.isnan(v0).any(): return [0.0, 0.0, 0.0]
    
    # Linear Temporal Interpolation: Blend v0 and v1 based on exactly where 't' is between them
    t0, t1 = TIME_STEPS_ARRAY[idx], TIME_STEPS_ARRAY[idx+1]
    weight = (t - t0) / (t1 - t0)
    return v0 + weight * (v1 - v0)

# ==========================================
# 4. PARALLEL Solver Execution
# ==========================================
def hit_outlet_plane(t, pos): 
    # An event function for the solver: returns 0 when particle hits the Z-ceiling
    return OUTLET_Z_MAX - pos[2] 
hit_outlet_plane.terminal = True # Tell solver to stop tracing when event hits 0
hit_outlet_plane.direction = -1  # Only trigger if particle is moving upwards into the plane

# Randomly pick NUM_PARTICLES starting locations from our list of inlet nodes
spawn_points = inlet_nodes[np.random.choice(len(inlet_nodes), NUM_PARTICLES, replace=(NUM_PARTICLES > len(inlet_nodes)))]
t_eval = np.linspace(0, SOLVER_END_TIME, 500) # Output 500 coordinate points per track

def trace_worker(i, p0):
    """The isolated job sent to individual CPU cores."""
    sol = solve_ivp(get_velocity_4d, (0, SOLVER_END_TIME), p0, t_eval=t_eval, events=[hit_outlet_plane], method='RK45')
    return i, sol.y.T, sol.status, sol.t[-1]

logging.info(f"Tracing {NUM_PARTICLES} particles in parallel...")
# Execute the worker function across all available CPU cores (-1)
results = Parallel(n_jobs=-1, backend="threading")(delayed(trace_worker)(i, p0) for i, p0 in enumerate(spawn_points))

# Unpack the parallel results and sort them back into numerical order
trajectories = [r[1] for r in sorted(results)]
residence_times = [r[3] for r in sorted(results)]
reached_outlet_flags = [r[2] == 1 for r in sorted(results)]

# ==========================================
# 5. Data Export & Plotting
# ==========================================
# Save the Residence Time Histogram as an image (prevents UI freezing)
plt.figure(figsize=(8, 4))
plt.hist(residence_times, bins=30, color='skyblue', edgecolor='black')
plt.title("Blood Residence Time (BRT)")
plt.xlabel("Time (s)")
plt.ylabel("Particle Count")
plt.savefig("Residence_Time_Histogram.png", dpi=300, bbox_inches='tight')
plt.close()

# Export Residence Time Data to CSV
rt_export = np.column_stack((np.arange(1, NUM_PARTICLES+1), reached_outlet_flags, residence_times))
np.savetxt("blood_residence_times.csv", rt_export, delimiter=",", header="ID,Reached_Outlet,Time_s", comments='')

# Find and export exact X,Y,Z coordinates of Stagnant Blood
stagnant_list = []
dt = SOLVER_END_TIME / 500.0
for traj in trajectories:
    if len(traj) < 2: continue
    # Recalculate speed (distance / time)
    v = np.linalg.norm(np.diff(traj, axis=0), axis=1) / dt
    v = np.append(v, v[-1])
    # Mask out coordinates that are stagnant AND inside the physical boundaries
    mask = (v < STAGNATION_THRESHOLD) & (traj[:,2] < OUTLET_Z_MAX - 2.0) & (traj[:,2] > INLET_Z_MIN + 2.0)
    if any(mask): stagnant_list.append(traj[mask])
    
if stagnant_list: 
    np.savetxt("stagnation_points.csv", np.vstack(stagnant_list), delimiter=",", header="X,Y,Z", comments='')

# ==========================================
# 6. Interactive 3D Dashboard
# ==========================================
plotter = pv.Plotter()
plotter.set_background('white')

# Render the faint background shell
mesh_actor = plotter.add_mesh(pv_mesh, color='white', opacity=0.05, render_points_as_spheres=True, point_size=2)

line_actors, stag_actors = [], []
for i, traj in enumerate(trajectories):
    line = pv.lines_from_points(traj)
    vels = np.linalg.norm(np.diff(traj, axis=0), axis=1) / dt
    line["Speed"] = np.append(vels, vels[-1])
    l_act = plotter.add_mesh(line, scalars="Speed", cmap="jet", line_width=4, show_scalar_bar=False)
    
    mask = (line["Speed"] < STAGNATION_THRESHOLD) & (traj[:,2] < OUTLET_Z_MAX - 2) & (traj[:,2] > INLET_Z_MIN + 2)
    s_act = plotter.add_mesh(pv.PolyData(traj[mask]), color='magenta', point_size=15, render_points_as_spheres=True) if any(mask) else None
    line_actors.append(l_act); stag_actors.append(s_act)

plotter.add_scalar_bar(title="Speed (mm/s)", color='black', vertical=True, position_x=0.88, position_y=0.1)

# Initialize UI text actors
status_act = plotter.add_text("Status: READY", position=(20, 20), font_size=11, color='green', font='arial')
id_act = plotter.add_text("Viewing Particle: 1", position=(20, 200), font_size=14, color='blue', font='arial')

ui = {'single': False, 'id': 0, 'mesh': True, 'stag': True, 'iso': False}

def update():
    status_act.SetInput("Status: UPDATING RENDER...")
    status_act.GetTextProperty().SetColor((1.0, 0.5, 0.0)) 
    plotter.render() 
    
    for i in range(NUM_PARTICLES):
        vis = (not ui['single']) or (i == ui['id'])
        if ui['iso']: vis = False 
        line_actors[i].SetVisibility(vis)
        if stag_actors[i]: stag_actors[i].SetVisibility((vis or ui['iso']) and ui['stag'])
    
    mesh_actor.SetVisibility(ui['mesh'])
    id_act.SetInput(f"Viewing Particle: {ui['id'] + 1} / {NUM_PARTICLES}")
    
    status_act.SetInput("Status: READY")
    status_act.GetTextProperty().SetColor((0.0, 0.6, 0.0))
    plotter.render()

# --- Core Action Functions ---
def action_next(): ui['id'] = (ui['id'] + 1) % NUM_PARTICLES; ui['single'] = True; update()
def action_prev(): ui['id'] = (ui['id'] - 1) % NUM_PARTICLES; ui['single'] = True; update()

def action_search():
    root = tk.Tk(); root.attributes('-topmost', True); root.withdraw() 
    val = simpledialog.askinteger("Search", f"Enter Particle ID (1 - {NUM_PARTICLES}):", minvalue=1, maxvalue=NUM_PARTICLES, parent=root)
    root.destroy()
    if val: ui['id'] = val - 1; ui['single'] = True; update()

# NEW: Action to capture a high-res image
def action_snapshot():
    root = tk.Tk(); root.attributes('-topmost', True); root.withdraw()
    filepath = filedialog.asksaveasfilename(title="Save Snapshot", defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")], parent=root)
    root.destroy()
    if filepath:
        plotter.screenshot(filepath)
        logging.info(f"Saved snapshot to: {filepath}")

# NEW: Action to record an animated 360 orbit
def action_record_orbit():
    root = tk.Tk(); root.attributes('-topmost', True); root.withdraw()
    filepath = filedialog.asksaveasfilename(title="Save 360 Video", defaultextension=".gif", filetypes=[("GIF Animation", "*.gif")], parent=root)
    root.destroy()
    if filepath:
        status_act.SetInput("Status: RECORDING ORBIT VIDEO...")
        status_act.GetTextProperty().SetColor((1.0, 0.0, 0.0)) # Turn status Red
        plotter.render()
        
        logging.info("Recording 360-degree orbit...")
        plotter.open_gif(filepath)
        plotter.orbit_on_path(step=0.05, write_frames=True, progress_bar=False)
        plotter.close()
        
        status_act.SetInput("Status: READY")
        status_act.GetTextProperty().SetColor((0.0, 0.6, 0.0))
        plotter.render()
        logging.info(f"Saved orbit video to: {filepath}")

# --- Checkbox Callbacks ---
def cb_mesh(state): ui['mesh'] = state; update()
def cb_stag(state): ui['stag'] = state; update()
def cb_single(state): ui['single'] = state; update()
def cb_iso(state): ui['iso'] = state; update()
def cb_cam(state): plotter.view_isometric(); plotter.reset_camera()
def cb_zoom(state): plotter.iren.interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandZoom()) if state else plotter.enable_trackball_style()

def cb_next(state): action_next()
def cb_prev(state): action_prev()
def cb_search(state): action_search()
def cb_snap(state): action_snapshot()     # NEW
def cb_orbit(state): action_record_orbit() # NEW

plotter.add_key_event('Right', action_next)
plotter.add_key_event('Left', action_prev)

# --- Render Clean UI Elements ---
f_args = dict(font='arial', color='black', font_size=10)

# Shifted base toggles up to fit the media controls
plotter.add_checkbox_button_widget(cb_mesh, value=True, position=(20, 480))
plotter.add_text("Show Manifold Shell", position=(60, 480), **f_args)

plotter.add_checkbox_button_widget(cb_stag, value=True, position=(20, 440), color_on='magenta')
plotter.add_text("Show Stagnant Zones", position=(60, 440), **f_args)

plotter.add_checkbox_button_widget(cb_single, value=False, position=(20, 400), color_on='blue')
plotter.add_text("Isolate Single Particle", position=(60, 400), **f_args)

plotter.add_checkbox_button_widget(cb_iso, value=False, position=(20, 360), color_on='red')
plotter.add_text("ISOLATE ONLY STAGNANT ZONES", position=(60, 360), font='arial', color='red', font_size=10)

plotter.add_checkbox_button_widget(cb_zoom, value=False, position=(20, 320), color_on='orange')
plotter.add_text("Area Zoom Tool (Drag box)", position=(60, 320), font='arial', color='orange', font_size=10)

# THE NEW MEDIA EXPORT CONTROLS
plotter.add_checkbox_button_widget(cb_snap, value=False, position=(20, 280), color_on='cyan')
plotter.add_text("Take Camera Snapshot (PNG)", position=(60, 280), font='arial', color='teal', font_size=10)

plotter.add_checkbox_button_widget(cb_orbit, value=False, position=(20, 240), color_on='darkred')
plotter.add_text("Record 360 Orbit Video (GIF)", position=(60, 240), font='arial', color='maroon', font_size=10)

# Standard Navigation Controls
plotter.add_checkbox_button_widget(cb_prev, value=False, position=(20, 150))
plotter.add_text("< Prev", position=(60, 150), **f_args)

plotter.add_checkbox_button_widget(cb_next, value=False, position=(120, 150))
plotter.add_text("Next >", position=(160, 150), **f_args)

plotter.add_checkbox_button_widget(cb_search, value=False, position=(20, 110), color_on='blue')
plotter.add_text("Search ID (Type Number)...", position=(60, 110), font='arial', color='darkblue', font_size=10)

plotter.add_checkbox_button_widget(cb_cam, value=False, position=(20, 70), color_on='green')
plotter.add_text("Reset Camera View", position=(60, 70), font='arial', color='darkgreen', font_size=10)

plotter.add_camera_orientation_widget()
plotter.show_grid(color='black', font_size=10)

logging.info("Opening viewer window. Script will exit when the window is closed.")
plotter.show()