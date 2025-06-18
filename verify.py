from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("tungvu3196/vlm-project-with-images-with-bbox-images-v3")
train_ds=ds["train"]

A2_vals=train_ds.iloc[ "Patient ID"]