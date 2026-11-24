import json
from pathlib import Path

split_path = Path("data/raw/data2017_splits.json")

with open(split_path, "r") as f:
    splits = json.load(f)

# Count the number of field IDs in each split
for split_name, field_ids in splits.items():
    print(f"{split_name}: {len(field_ids)} fields")

# Check for overlap between splits
train = set(splits["train"])
val = set(splits["val"])
test = set(splits["test"])

print("\nOverlap checks:")
print("Train & Val:", len(train & val))
print("Train & Test:", len(train & test))
print("Val & Test:", len(val & test))

print("\nTotal unique fields:", len(train | val | test))