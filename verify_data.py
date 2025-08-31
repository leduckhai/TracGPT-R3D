import os
import json
train_dirs="pseudo_3d/32_overlap_slices/820ac9e9-3f29-498d-b717-466d44081411/train/data"

test_dir= "pseudo_3d/32_overlap_slices/820ac9e9-3f29-498d-b717-466d44081411/test/data"

path=[os.path.join(train_dirs, f) for f in os.listdir(train_dirs) if f.endswith('.json')]
test_path=[os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.json')]

unique_classes = set()
for f in path:
    with open(f, "r") as file:
        data = json.load(file)
        for d in data:
            unique_classes.add(d['A4'])
print("unique_classes",unique_classes)

unique_classes = set()
for f in test_path:
    with open(f, "r") as file:
        data = json.load(file)
        for d in data:
            unique_classes.add(d['A4'])
print("unique_classes",unique_classes)
