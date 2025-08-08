import sys 
sys.path.append(".")
from src.model.LanguageModel.model_output import TracVisionModelOutput
import torch
import torch.nn as nn
from src.model.Encoder.resnet import ResNet18_3D
from src.model.Encoder.densenet import DenseNet3D
import torch.nn.functional as F
class TracVisionModel(nn.Module):
    
    def __init__(self, vision_encoder="resnet",num_bbox_classes=4):
        
        super().__init__()
        self.num_bbox_classes=num_bbox_classes
        self.loss= nn.CrossEntropyLoss()
        if vision_encoder == "resnet":
            self.vision_encoder = ResNet18_3D()
        elif vision_encoder == "densenet":
            self.vision_encoder = DenseNet3D()
            
       

    def forward(
        self,
        images=None,
    #    bbox_GCA=None,
    #    bbox_Koedam=None,
    #    bbox_MTA=None,
    #    status_criteria=None,
        **kwargs
    #    bbox_criteria=None,
    ):
        
        subtask_out, finaltask_out  = self.vision_encoder(images)
        # print("subtask out",subtask_out.shape,"finaltask out",finaltask_out.shape)
        # print("sample shape",subtask_out[:,0,:].shape,bbox_criteria[metric].shape)
        # metric_loss={}
        # metric_loss["GCA"] = self.loss(subtask_out[:,0,:],bbox_GCA.long().to("cuda"))
        # metric_loss["Koedam"] = self.loss(subtask_out[:,1,:],bbox_Koedam.long().to("cuda"))
        # metric_loss["MTA"] = self.loss(subtask_out[:,2,:],bbox_MTA.long().to("cuda"))
        # # for i, (metric, val) in enumerate(bbox_criteria.items()):
           
        # metric_loss["status"] = self.loss(finaltask_out.squeeze(1), status_criteria.long().to("cuda"))
        
        # loss = sum(metric_loss.values())
        # return TracVisionModelOutput(
        #     loss=loss,
        #     logits=None,
        #     aux_loss=metric_loss,
        #     predicts=None
            
        # )
        return subtask_out,finaltask_out
if __name__=="__main__":
    from src.data.dataloader import load_data
    from src.collator import WhiteCollator
    from torch.utils.data import DataLoader
    model=TracVisionModel()
    model.to("cuda")
    
    collator=WhiteCollator()
    train_set, val_set, test_set = load_data(train_val_dir="/root/TracGPT-R3D/pseudo_3d/32_overlap_slices/26f67cb9-1efd-4a39-9eda-4fe15eb5127f/train/data",dataset="trac_white")
    train_ld=DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=collator)
    # status_criteria, bbox_GCA,bbox_Koedam,bbox_MTA
    for i, sample in enumerate(train_ld):
        with torch.no_grad():
            if i==3:
                break
            # print("sample",sample)
            image=sample["images"].to("cuda")
            print("image stats",image.shape,image.min(), image.max(),image.mean(), image.std())
            # print("sample",sample)
            # print("---------Sample Criteria",sample["status_criteria"])
            # print("---bbox criteria",sample["bbox_criteria"])
    
            out=model(sample["image"].to("cuda"),bbox_criteria=sample["bbox_criteria"],status_criteria=sample["status_criteria"].to("cuda"))
        