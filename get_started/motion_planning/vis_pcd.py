import open3d as o3d
import open3d.visualization.rendering as rendering

pcd = o3d.io.read_point_cloud("get_started/output/motion_planning.ply")  # Replace with your point cloud file

# vis = o3d.visualization.Visualizer()
# vis.create_window(visible=False)  # Offscreen
# vis.add_geometry(pcd)
# vis.poll_events()
# vis.update_renderer()
# vis.capture_screen_image("LiberoPickAlphabetSoup_PCD.png")
# vis.destroy_window()

# Set up renderer
width, height = 640, 480
renderer = rendering.OffscreenRenderer(width, height)
mat = rendering.MaterialRecord()
mat.shader = "defaultUnlit"

# Add the geometry
scene = renderer.scene
scene.set_background([1, 1, 1, 1])  # white background
scene.add_geometry("pcd", pcd, mat)

# Setup camera
bounds = scene.bounding_box
center = bounds.get_center()
scene.camera.look_at(center, center + [0, 0, -1], [0, -1, 0])

# Render and save
img = renderer.render_to_image()
o3d.io.write_image("get_started/output/motion_planning/LiberoPickAlphabetSoup_PCD.png", img)
