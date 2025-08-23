
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import sys 
# import os 
# sys.path.append("/root/TracGPT-R3D")
# from src.model.language.llama import TracLlamaForCausalLM, TracLlamaConfig
# # def load_pretrain( model_dir:str,config_path:str=None):
# #     with 
# #     tokenizer = AutoTokenizer.from_pretrained(backbone_language_model)
# #     if model_name == "vit_llama":
# #         config = TracLlamaConfig.from_pretrained("meta-llama/Llama-2-7b")
# #         config.custom_config = {
# #         "vision_encoder": "vit-base-patch16-224",
# #         "projector": {"hidden_size": 4096},
# #         "language_model": {"name": "meta-llama/Llama-2-7b"}
# #     }

# # Initialize and load weights
# # model = TracLlamaForCausalLM.from_pretrained(
# #     "./trac_llama_model",  # Path to saved weights
# #     config=config
# # )
# # model.eval()

#     return model, tokenizer
# if __name__ == "__main__":
#     import os
#     model_dir="output/t3bmi4n1/checkpoint-2"
#     assert os.path.exists(model_dir), f"Model directory {model_dir} does not exist."
#     model, tokenizer = load_pretrain(model_dir)
#     print("model", model)
#     # """
#     # Load pretrained model from the specified path.
    
#     # Args:
#     #     config: Configuration object containing model parameters.
#     #     pretrained_path: Path to the pretrained model.
        
#     # Returns:
#     #     model: Loaded pretrained model.
#     # """

#     # # Initialize the model with the configuration
#     # model = LlamaForCausalLM(config)

#     # # Load the state dictionary from the specified path
#     # state_dict = torch.load(pretrain_path, map_location="cpu")

#     # # Load the state dictionary into the model
#     # model.load_state_dict(state_dict, strict=False)

#     # return model