import mpose
import json

data_dir='./data/mpose'

# Load the dataset (PoseNet version, split doesn't matter for names)
dataset = mpose.MPOSE(pose_extractor='posenet', split=1)

# get_labels() returns a dictionary: class_name -> label_id
label_dict = dataset.get_labels()
# Invert to list indexed by label_id
class_names = [None] * (max(label_dict.values()) + 1)
for name, idx in label_dict.items():
    class_names[idx] = name

# Save to JSON
with open('class_info.json', 'w') as f:
    json.dump({'class_names': class_names}, f, indent=2)

print(f"Saved class_info.json with {len(class_names)} classes.")