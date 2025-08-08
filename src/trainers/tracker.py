import wandb
from datetime import datetime
now = datetime.now()
class WandbTracker:
    def __init__(self,project="TracGPT",tags=[]):
        print("Init Wandb Tracker")
        date_time_string = now.strftime("%d-%m-%Y--%H-%M-%S")
        wandb.init(
        project=project,
        name=f"Trac_llama-{date_time_string}",
        tags=tags
    )
        
    def log(self, log_dict, step):
        print(log_dict)
        wandb.log(log_dict, step=step)
    def on_train_end(self):
        wandb.finish()