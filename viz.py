import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv

import plotly.io as pio
def save_viz(bboxes,image_3d, save_path="volume.png"):
    """
    Renders a 3D volume with bounding boxes, preserving original intensities.
    """
    D,H,W = image_3d.shape
    
    # --- create PyVista uniform grid ---
    grid = pv.UniformGrid()
    grid.dimensions = np.array(image_3d.shape) + 1
    grid.cell_data["values"] = image_3d.flatten(order="F")
    
    # --- Plotter with offscreen rendering ---
    p = pv.Plotter(off_screen=True)
    p.add_volume(grid, cmap="gray", opacity="sigmoid", opacity_unit_distance=1)
    
    # --- Add bounding boxes ---
    for box in bboxes:
        x0,y0,z0,x1,y1,z1 = np.array(box) * np.array([W,H,D,W,H,D])
        cube = pv.Box(bounds=(x0,x1,y0,y1,z0,z1))
        p.add_mesh(cube, color="red", style="wireframe", line_width=3)
    
    # --- Render & save ---
    p.show(screenshot=save_path)
    print(f"✅ Saved volume render with bounding boxes to {save_path}")
if __name__ == "__main__":  
    from src.dataset.dataloader import load_data
    from src.data_process.util import group_and_merge_3d_bboxes_v2
    train_val_dir= "pseudo_3d/32_overlap_slices/4f6cab46-2746-48c2-a8fc-cff454b3e15a/train/data"
    test_dir="pseudo_3d/32_overlap_slices/4f6cab46-2746-48c2-a8fc-cff454b3e15a/test/data"
    image_train_path="clean_data/train/image"
    image_test_path= "clean_data/test/image"
    train_set, val_set, test_set = load_data(
        train_val_dir=train_val_dir,
        test_dir=test_dir,
        image_train_path=image_train_path,
        image_test_path=image_test_path,
        dataset="trac_white",
    )   
    sample=test_set[0]
    image,Q1,A1,Q2,A2,Q3,A3,Q4,A4=sample["image"],sample["Q1"],sample["A1"],sample["Q2"],sample["A2"],sample["Q3"],sample["A3"],sample["Q4"],sample["A4"]
    image=image.numpy().squeeze()
    bboxes=group_and_merge_3d_bboxes_v2(Q1, num_concat=32)
    print("bboxes",bboxes)  
    print("image shape", image.shape)
    print("a1", A1)
    save_viz(bboxes, image, save_path="bbox_viz_A1.png")