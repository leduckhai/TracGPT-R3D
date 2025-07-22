from collator import BboxAwareCollator
from data.dataloader import load_data
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from model.bbox3d.helper import corners_to_center,center_to_corners
from model.bbox3d.bbox_head import AnchorBBox3DHeadV2

model_max_length = 512
image_token_name = "<im_patch>"
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
collator = BboxAwareCollator(
    tokenizer=tokenizer,
    max_length=model_max_length,
    max_bbox_length=9,
    num_vision_token=256,
    token_name=image_token_name,
)
train_set, val_set, test_set = load_data(bbox_only=True)
print("len trainset", len(train_set))
dl = DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=collator)

with torch.no_grad():
    for i, batch in enumerate(dl):
        if i == 5:
            break
        (
            images,
            input_ids,
            attention_mask,
            positive_centers,
            labels,
            center_bbox_gts,
            bbox_mask,
            position_ids,
            answer_types,
            questions,
            answers,
            corner_bbox_gts,
        ) = batch.values()
        
    
        print("positive center",positive_centers)
        # print("all positive centers",positive_centers.sum())
        anchorbox=AnchorBBox3DHeadV2(config=None)
        test_image=torch.rand(2,64,768)
        binary_output,delta_xyz,log_dwh,conf=anchorbox(
            test_image
        )
        break
        # print("center bbox gt", center_bbox_gts.tolist())
        # print("corner bbox gt", corner_bbox_gts.tolist())
        # print("positive centers", positive_centers.tolist())
        # # center=get_center(center_bbox_gt, patch_grid=[4,4,4])
        # # print("center", center)
        # print("center sample",center_bbox_gt, "mask",bbox_mask[0])
        # sample=center_bbox_gt[0][bbox_mask[0]]
        
        # print("sample",sample)
        # pos_center=get_center(sample)
        # print("pos center",pos_center)
        # print("corner bbox gt", corner_bbox_gts)
        # inverted_bbox_gt = center_to_corners(bbox_gt)
        # print("inverted bbox gt", inverted_bbox_gt)
        # print("invert bbox gt", center_to_corners(bbox_gt).tolist())