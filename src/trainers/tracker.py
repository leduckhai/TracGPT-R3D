import wandb
from datetime import datetime
from uuid import uuid4
now = datetime.now()

class DummyTracker:
    def __init__(self, project="TracGPT", tags=[]):
        print("Init Dummy Tracker")
        now = datetime.now()
        human_time = (
            now.strftime("%I:%M:%S%p").lower().replace(":", "")
        )  # e.g., "43204pm"
        date_str = now.strftime("%d-%m-%Y")  # e.g., "17-08-2025"
        self.id=uuid4().hex  # Generate a unique ID for the run
    def get_id(self):
        return self.id

    def log(self, log_dict, step):
        print("Dummy Tracker Log",step)
        print(log_dict)

    def on_train_end(self):
        pass

class WandbTracker:
    def __init__(self, project="TracGPT", tags=[]):
        print("Init Wandb Tracker")
        # date_time_string = now.strftime("%d-%m-%Y--%H-%M-%S")
        # wandb.init(
        # project=project,
        # name=f"Trac_llama-{date_time_string}",
        # tags=tags
        now = datetime.now()
        human_time = (
            now.strftime("%I:%M:%S%p").lower().replace(":", "")
        )  # e.g., "43204pm"
        date_str = now.strftime("%d-%m-%Y")  # e.g., "17-08-2025"

        wandb.init(
            project=project,
            name=f"Training_{now.strftime('%d-%m-%Y')}_{now.strftime('%I-%M-%S%p').lower()}",
            tags=tags,
        )

    def get_id(self):
        return wandb.run.id

    def log(self, log_dict, step):
        print(log_dict)
        wandb.log(log_dict, step=step)

    def on_train_end(self):
        wandb.finish()
