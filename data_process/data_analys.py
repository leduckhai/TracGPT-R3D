import csv
import sys
csv_path="/home/ducnguyen/sync_local/repo/TracGPT/clean_data_3d/data_1374d6bc-82ac-4f4c-9838-3626eb480192/train/data.csv"
import ast
csv.field_size_limit(sys.maxsize)
# with open(csv_path, 'r') as file:
#     reader = csv.DictReader(file)  
#     first_row = next(reader)      

# for key, value in first_row.items():
#     if (type(value) == list):
#         print("value is a list",len(value))
#     if key in ["Original","__index__level_0__","No.", "Column 9","Deliverable","Doctor","Start date","Google Drive Link","rotated_link","vn","fr","de","mandarin","korean","japanese","vi"]:
#         continue
#     if key=="A1":
#         value=parsed = ast.literal_eval(value)
#         # print("len A1",len(value), value[:10])
#         print("A1",value[0], value[5])
#     value_str = str(value)  
#     # print(f"key: {key}, value: {value_str[:400]}")

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
import json 
import ast 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
from PIL import Image
import os
df = load_dataset('tungvu3196/vlm-project-with-images-with-bbox-images-v3')['train']

sample=df[2]
image=sample["image"]
image_with_bboxes=sample["image_with_bboxes"]
bboxes=sample["A1"]
bboxes=ast.literal_eval(bboxes)


def visualize_compare(
    img, 
    gt_img=None,  # Ground truth image with boxes already drawn
    pred_bboxes=None, 
    output_path=None, 
    figsize=(18, 6),
    pred_color='r', 
    pred_linewidth=2,
    pred_alpha=1.0,
    show_pred_labels=False,
    pred_labels=None,
    label_fontsize=10
):
    """
    Compare original image, pre-rendered ground truth image, and predicted boxes
    
    Args:
        img (ndarray/str): Original image (array or path)
        gt_img (ndarray/str): Ground truth image with boxes already drawn
        pred_bboxes (list): Predicted boxes in format [x, y, w, h]
        output_path (str): Path to save visualization
        figsize (tuple): Figure size
        pred_color (str): Color for predicted boxes
        pred_linewidth (int): Box line width
        pred_alpha (float): Box transparency
        show_pred_labels (bool): Show labels on predicted boxes
        pred_labels (list): Labels for predicted boxes
        label_fontsize (int): Label font size
    """
    # Load images
    if isinstance(img, str):
        img = np.array(Image.open(img))
    else:
        img = np.array(img)
    
    if isinstance(gt_img, str):
        gt_img = np.array(Image.open(gt_img))
    elif gt_img is not None:
        gt_img = np.array(gt_img)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3 if gt_img is not None else 2, figsize=figsize)
    axes = axes.flatten()
    
    # Original Image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground Truth Image (pre-rendered with boxes)
    if gt_img is not None:
        axes[1].imshow(gt_img)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        pred_ax = 2  # Predictions will be on third panel
    else:
        pred_ax = 1  # Predictions will be on second panel
    
    # Image with Predicted Bounding Boxes
    axes[pred_ax].imshow(img)  # Use original image as base
    axes[pred_ax].imshow(gt_img, alpha=0.5)
    if pred_bboxes:
        axes[pred_ax].set_title('Predicted Boxes')
        img_h, img_w = img.shape[:2]
        
        for i, bbox in enumerate(pred_bboxes):
            x, y, w, h = bbox[:4]
            x_px, y_px = x * img_w, y * img_h
            w_px, h_px = (w-x) * img_w, (h-y) * img_h
            # w_px, h_px = w * img_w, h * img_h

            # Draw predicted box
            rect = patches.Rectangle(
                (x_px, y_px), w_px, h_px,
                linewidth=pred_linewidth,
                edgecolor=pred_color,
                facecolor='none',
                alpha=pred_alpha
            )
            axes[pred_ax].add_patch(rect)
            
            # Add label if requested
            if show_pred_labels:
                label = pred_labels[i] if (pred_labels and i < len(pred_labels)) else str(i+1)
                axes[pred_ax].text(
                    x_px, y_px - 5,
                    label,
                    color='white',
                    bbox=dict(facecolor=pred_color, alpha=0.8, edgecolor='none'),
                    fontsize=label_fontsize
                )
    
    axes[pred_ax].axis('off')
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
def visualize_bboxes_comparison(
    img, 
    pred_bboxes=None, 
    gt_bboxes=None, 
    output_path=None, 
    figsize=(18, 6),
    pred_color='r', 
    gt_color='g',
    pred_linewidth=2,
    gt_linewidth=2,
    pred_alpha=1.0,
    gt_alpha=1.0,
    show_labels=False,
    pred_labels=None,
    gt_labels=None,
    label_fontsize=10
):
    """
    Visualize image with predicted and ground truth bounding boxes side by side
    
    Args:
        img (ndarray or str): Image array or path to the image file
        pred_bboxes (list): List of predicted bounding boxes in format [x, y, w, h]
        gt_bboxes (list): List of ground truth bounding boxes in format [x, y, w, h]
        output_path (str, optional): Path to save the visualization
        figsize (tuple): Figure size (width, height) in inches
        pred_color (str): Color for predicted boxes (default: 'r' [red])
        gt_color (str): Color for ground truth boxes (default: 'g' [green])
        pred_linewidth (int): Line width for predicted boxes
        gt_linewidth (int): Line width for ground truth boxes
        pred_alpha (float): Transparency for predicted boxes (0-1)
        gt_alpha (float): Transparency for ground truth boxes (0-1)
        show_labels (bool): Whether to show labels on boxes
        pred_labels (list): Labels for predicted boxes
        gt_labels (list): Labels for ground truth boxes
        label_fontsize (int): Font size for labels
    """
    # Load the image if path is provided
    if isinstance(img, str):
        img = plt.imread(img)
    img = np.array(img)
    
    # Get image dimensions
    img_h, img_w = img.shape[:2]
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Original Image
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Image with Predicted Bounding Boxes
    ax2.imshow(img)
    if pred_bboxes:
        ax2.set_title('Predicted Bounding Boxes')
        for i, bbox in enumerate(pred_bboxes):
            x, y, w, h = bbox[:4]
            x_px, y_px = x * img_w, y * img_h
            w_px, h_px = w * img_w, h * img_h
            
            rect = patches.Rectangle(
                (x_px, y_px), w_px, h_px,
                linewidth=pred_linewidth,
                edgecolor=pred_color,
                facecolor='none',
                alpha=pred_alpha
            )
            ax2.add_patch(rect)
            
            if show_labels and pred_labels:
                label = pred_labels[i] if i < len(pred_labels) else str(i+1)
                ax2.text(
                    x_px, y_px - 5,
                    label,
                    color='white',
                    bbox=dict(facecolor=pred_color, alpha=0.8, edgecolor='none'),
                    fontsize=label_fontsize
                )
    ax2.axis('off')
    
    # Image with Ground Truth Bounding Boxes
    ax3.imshow(img)
    if gt_bboxes:
        ax3.set_title('Ground Truth Bounding Boxes')
        for i, bbox in enumerate(gt_bboxes):
            x, y, w, h = bbox[:4]
            x_px, y_px = x * img_w, y * img_h
            w_px, h_px = w * img_w, h * img_h
            
            rect = patches.Rectangle(
                (x_px, y_px), w_px, h_px,
                linewidth=gt_linewidth,
                edgecolor=gt_color,
                facecolor='none',
                alpha=gt_alpha
            )
            ax3.add_patch(rect)
            
            if show_labels and gt_labels:
                label = gt_labels[i] if i < len(gt_labels) else str(i+1)
                ax3.text(
                    x_px, y_px - 5,
                    label,
                    color='white',
                    bbox=dict(facecolor=gt_color, alpha=0.8, edgecolor='none'),
                    fontsize=label_fontsize
                )
    ax3.axis('off')
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

viz_dir="viz"
os.makedirs(viz_dir, exist_ok=True)
path_1=os.path.join(viz_dir,"image.png")
path_2=os.path.join(viz_dir,"image_with_bboxes.png")
path_3=os.path.join(viz_dir,"image_with_bboxes_2.png")
# visualize_bboxes(image, bboxes, output_path=path_2, figsize=(12, 6))
# visualize_bboxes(image, bboxes, output_path=path_3, figsize=(12, 6))
print("type before",type(image),type(image_with_bboxes),type(bboxes))
# print("image",image,image_with_bboxes,bboxes)
visualize_compare(img=image,gt_img=image_with_bboxes,pred_bboxes=bboxes ,output_path=path_1, figsize=(12, 6))
# result = df.query(
#     '`Patient ID` == "OAS1_0004" and Slide == "mpr-3_114"'
# )['A1'].tolist()

