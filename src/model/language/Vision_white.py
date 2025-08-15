import sys 

sys.path.append(".")

from transformers import AutoConfig, AutoModel
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from model.bbox3d.builder import BBox3DPredictor
import yaml
from model.Encoder.encoder import build_vision_tower
from model.Projector.projector import build_mm_projector
from utils.type import dict_to_namespace
from model.bbox3d.helper import compute_ious
from model.bbox3d.builder import BBox3DPredictor
from model.bbox3d.bbox_decoder import BBox3DDecoder
from src.model.LanguageModel.model_output import TracVisionModelOutput
class VisionEncoder(nn.Module):
    """Handles vision encoding and projection"""

    def __init__(self, config):
        super().__init__()
        self.vision_tower_config = config.vision_tower_config
        self.mm_projector_config = config.projector
        self.vision_tower = None
        self.mm_projector = None
        if self.vision_tower_config:
            self.vision_tower = build_vision_tower(
                self.vision_tower_config,
            )
            print("Vision tower built successfully.")
            # self.mm_projector = build_mm_projector(self.mm_projector_config)

    def encode_images(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode images to features"""
        if images is None:
            return None

        image_features = self.vision_tower(images)
        # print("after vision tower stats", image_features.shape,image_features.min(), image_features.max(),image_features.mean(), image_features.std())
        # image_features = self.mm_projector(image_features)
        # print("after mm projector stats",  image_features.shape,image_features.min(), image_features.max(),image_features.mean(), image_features.std())
        return image_features


class TracVisionConfig(AutoConfig):
    model_type = "TracVision"
    
    def __init__(
        self,
        vision_config=None,
        num_labels=1000,
        hidden_size=768,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_config = vision_config
        self.num_labels = num_labels
        self.hidden_size = hidden_size

class TracVisionModel(nn.Module):
    
    def __init__(self, ):
        super().__init__()
        module_config_path = "config/llama_yolo.yaml"
        with open(module_config_path, "r") as f:
            module_config = yaml.safe_load(f)
        self.module_config = dict_to_namespace(module_config)
        self.module_config = self.module_config.tiny_llama
        self.vision_encoder = VisionEncoder(self.module_config.vision_encoder)
        self.bbox3d_predictor = BBox3DPredictor(self.module_config.bbox_predictor)
        self.bbox_decoder=BBox3DDecoder(conf_threshold=0.5, nms_threshold=0.4)
        # self.init_weights()
  
    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             m.weight.data.normal_(0.0, self.config.initializer_range)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        images=None,
        labels=None,
        bbox_gts=None,
        bbox_masks=None,
        return_dict=None,
        **kwargs
    ):
        
        features = self.vision_encoder.encode_images(images)
        # pooled = self.dropout(features)
        # logits = self.classifier(pooled)
    
        loss = None
        aux_loss = {}
        predicts=None
        logits=None
        predictor =  self.bbox3d_predictor.predict_bboxes
        compute_bbox_loss = self.bbox3d_predictor.compute_bbox_loss

        bbox_samples = bbox_masks.any(dim=1)
        if  bbox_samples.any():

            targets = bbox_gts[bbox_samples]
            masks = bbox_masks[bbox_samples]

            vision_features = features[bbox_samples]
            # center_pred,delta_xyz,log_dwh,conf= predictor(vision_features)
            center_pred,delta_xyz,log_dwh= predictor(vision_features)
            bbox_aux_loss = compute_bbox_loss(
                delta_xyz=delta_xyz,
                log_dwh=log_dwh,
                # conf_pred=conf,
                center_pred=center_pred,
                targets=targets,
                masks=masks,
            )
            bbox_prediction_decoder = self.bbox_decoder.decode_predictions_v3(center_pred,delta_xyz,log_dwh)
            # print("gt boxes",targets)
            bbox_pred=[d["boxes"] for d in bbox_prediction_decoder]
            
            for b in range(len(bbox_pred)):
                print("n bbox pred samples",len(bbox_pred[b]))
                pairs=compute_ious(bbox_pred[b].detach().cpu(),targets[b].detach().cpu(),masks[b].detach().cpu(),mode="center")
                # print("pairs",pairs)
                print("Pair iou",pairs["iou"])
            loss= bbox_aux_loss["total_loss"]
            predicts = bbox_pred
            aux_loss={
                "bbox_3d_loss":bbox_aux_loss["total_loss"].item(),
                "pos_loss":bbox_aux_loss["pos_loss"].item(),
                "size_loss":bbox_aux_loss["size_loss"].item(),
                "center_loss":bbox_aux_loss["center_loss"].item(),
                # "conf_loss":bbox_aux_loss["conf_loss"].item(),
            }
            print("bbox aux loss",aux_loss)
                
        else:
            print("No bbox samples bbox mask",bbox_masks)
        return TracVisionModelOutput(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss if aux_loss else None,
            predicts=predicts
        )

if __name__=="__main__":
    from collator import BboxAwareCollator
    from torch.utils.data import DataLoader
    from eval import evaluate
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from transformers import AutoConfig, AutoModelForCausalLM
    from data.dataloader import load_data

    model_max_length = 512
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    special_tokens = [
        "<im_patch>",
        "<end>",
    ]

    image_token_name = "<im_patch>"
    end_token = "<end>"
    num_added = tokenizer.add_tokens(special_tokens)
    print(f"Added {num_added} special tokens", len(tokenizer))

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
    img_token_id = tokenizer.convert_tokens_to_ids(image_token_name)

    print("vocab size", len(tokenizer))
    model = TracVisionModel()
    model.to("cuda")
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
            bbox_masks,
            position_ids,
            answer_types,
            questions,
            answers,
            corner_bbox_gts,
            ) = batch.values()
            images = images.to("cuda")
            print("img shape", images.shape)
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
            labels = labels.to("cuda")
            center_bbox_gts = center_bbox_gts.to("cuda")
            
            bbox_mask = bbox_masks.to("cuda")
            position_ids = position_ids.to("cuda")
            # print("forward pass")
            # print("center_bbox_gts shape",center_bbox_gts.shape)
            # print("center_bbox_gts",center_bbox_gts.unique())
            assert positive_centers[positive_centers>0.0].float().sum().item() == bbox_masks.sum().item(), (
            f"Expected center_bbox_gts {positive_centers.sum().item()} "
            f"and bbox_masks to be equal {bbox_masks.sum().item()}"
        )
            outputs = model(
                images=images,
                bbox_gts=center_bbox_gts,
                bbox_masks=bbox_masks,
                labels=labels,
            )
            # print("outputs",outputs)