from train import pytorch_model_run
import torch
from predict import eval_gpt_open_ended
# from models import VQAmedModel, VQAmedModel_abl
from models.med_vqa import VQAmedModel, VQAmedModel_abl
from data_loaders.dataloader import medvqaDataset
from torch.utils.data import DataLoader
import argparse
import numpy as np
import random
import os


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt2-xl", choices=("gpt2-xl", "microsoft/biogpt","stanford-crfm/BioMedLM"))
    parser.add_argument("--setting", type=str, default="frozen", choices=("lora", "frozen",'prefixtuning',"p_tuning","prompttuning", "unfrozen"))
    parser.add_argument("--ablation", type=str, default="none", choices=("remove_question", "remove_visual",'replace_visual',"swap"))
    parser.add_argument("--mapping_type", type=str, default="MLP")
    parser.add_argument("--prefix_length", type=int, default=8)
    parser.add_argument(
        "--dataset_path", type=str, default="../vqa_datasets/"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--dataset", type=str, default='pathvqa', choices=('pathvqa', 'ovqa', 'slake'))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters_to_accumulate", type=int, default=4)
    parser.add_argument("--validation_step", type=int, default=1000)
    parser.add_argument("--out_dir", default="./checkpoints")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--eval", dest="eval", action="store_true")
    parser.add_argument("--data_partition", default="part")
    parser.add_argument("--verbose", dest="verbose", action="store_true")

    args = parser.parse_args()
    


    set_random_seeds(args.seed)
    return args


if __name__ == "__main__":
    args = parse_argument() 
    
    train_path="/home/ubuntu/repo/SLAKE/train.pkl"
    val_path="/home/ubuntu/repo/SLAKE/val.pkl"
    test_path="/home/ubuntu/repo/SLAKE/test.pkl"
    
    
    train_dataset = medvqaDataset(train_path,split="train",prefix_length=args.prefix_length,model_type=args.model_type)#,abl=args.ablation)
    val_dataset = medvqaDataset(val_path,split="val",prefix_length=args.prefix_length,model_type=args.model_type)#, abl=args.ablation)
    test_dataset = medvqaDataset(test_path,split="test",prefix_length=args.prefix_length,model_type=args.model_type,like_test=True)

    if args.ablation != "none":
        model = VQAmedModel_abl(
            prefix_length=args.prefix_length,
            clip_length=4,
            setting=args.setting,
            mapping_type=args.mapping_type,
            args=args,
        )
    else:
        model = VQAmedModel(
            prefix_length=args.prefix_length,
            clip_length=4,
            setting=args.setting,
            mapping_type=args.mapping_type,
            args=args,
        )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=True)
    
    print("len train_dataset",len(train_dataset),"len val_dataset",len(val_dataset),"len test_dataset",len(test_dataset))
    if not args.eval:
        model = pytorch_model_run(train_dataloader, val_dataloader, test_dataset, model, args)
    else:
        checkpoint = os.path.join(args.out_dir, f"open_ended_latest.pt")
        if args.verbose:
            print(f">> Loading pre-trained model {checkpoint}!")
        if os.path.exists(checkpoint):
            model.load_state_dict(
                torch.load(checkpoint, map_location=torch.device("cpu")), strict=False
            )
        else:
            raise ValueError("Please provide valid path for loading checkpoint")
        eval_gpt_open_ended(model, test_dataset,args)
