import os
import json
from tqdm import tqdm
import shutil
from collections import defaultdict
# from src.dataset.dataloader import load_data
# from src.data_process.util import bboxes_to_filled_volume, bboxes_to_wireframe_volume,draw_3d_bbox_wireframe_v2, draw_3d_bbox_filled,draw_3d_bbox_wireframe,convert_list_slice_paths_to_3d,save_nifti,group_and_merge_3d_bboxes_v2
import pandas as pd

significant_slice_path= "significant_slice.csv"
df=pd.read_csv(significant_slice_path)
print(df.columns.tolist())
# print(df.iloc[0])
p_id="OAS1_0001"
slices=df[df["Patient ID"] == p_id][["Slide"]]
print(f"Significant slices for patient {p_id}: {slices}")