# extract_mpii_global_embeddings.py
'''python src/pipeline/embedding_from_checkpoint.py
    --mpii_mat data/mpii/annotations/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat
    --mpii_images data/mpii/images
    --output_dir outputs/ssl_mpii_pretrained
    --indices_dir .'''
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


# Add project root
script_path = Path(__file__).resolve()
current = script_path.parent
while current.name != 'src' and current != current.parent:
    current = current.parent
project_root = current.parent
sys.path.insert(0, str(project_root))
# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.encoder.skeleton_encoder import JointFeatureExtractor
from data.mpii.mpii_dataclass import MPIIPoseDataset

# --- Config ---
mat_file = "data/mpii/annotations/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat"
image_dir = "data/mpii/images"
output_dir = "outputs/ssl_mpii_pretrained"
batch_size = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = os.path.join(output_dir, "best_ssl_model.pth")

def collate_skip_none(batch):
    """Custom collate that filters out any sample that is None or has None in dict."""
    # Filter out None samples
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    # Filter out dict items that are None
    batch = [{k: v for k, v in item.items() if v is not None} for item in batch]
    return default_collate(batch)

def extract_embeddings(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the full MPII dataset
    dataset = MPIIPoseDataset(
        mat_file=args.mpii_mat,
        image_dir=args.mpii_images,
        split='train',
        pose_classes=None,
        min_activity_samples=1
    )
    print(f"Loaded {len(dataset)} raw samples")

    # Load person‑based split indices
    train_idx = np.load(os.path.join(args.indices_dir, 'mpii_train_indices.npy'))
    val_idx   = np.load(os.path.join(args.indices_dir, 'mpii_val_indices.npy'))

    # Sanity check: ensure indices are within range
    if max(train_idx) >= len(dataset):
        raise ValueError("Train indices out of range")
    if max(val_idx) >= len(dataset):
        raise ValueError("Val indices out of range")

    # Create subsets
    train_subset = Subset(dataset, train_idx)
    val_subset   = Subset(dataset, val_idx)

    print(f"Train subset size: {len(train_subset)}")
    print(f"Val subset size:   {len(val_subset)}")

    # Create extractor with same config as SSL training
    extractor = JointFeatureExtractor(
        num_joints=17,
        joint_type_embedding_dim=16,
        max_angles=3,
        use_angles=True,
        use_velocities=False,
        use_relative_pos=True,
        use_confidence=True,
        use_temporal=False,
        output_dim=64,
        normalize='torso',
        augmentations={},
        return_raw=False,
        min_torso_height=10.0,
        max_norm_range=5.0,
        fallback_to_bbox=True,
    )

    # Load checkpoint
    checkpoint_path = os.path.join(args.output_dir, 'best_ssl_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    extractor.load_state_dict(checkpoint['model_state_dict'])
    extractor.to(device)
    extractor.eval()
    print("Extractor loaded.")

    def extract_global(subset, desc):
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == 'cuda'),
                            collate_fn=collate_skip_none)
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                if batch is None:
                    continue
                x = batch['keypoints'].to(device)
                out = extractor(x, apply_augmentations=False)
                global_emb = out['joint_embeddings'].mean(dim=1).cpu().numpy()
                embeddings.append(global_emb)
        if len(embeddings) == 0:
            return np.array([])
        return np.concatenate(embeddings, axis=0)

    # Extract embeddings
    train_emb = extract_global(train_subset, "Extracting train embeddings")
    val_emb   = extract_global(val_subset,   "Extracting val embeddings")

    # Get labels (they are available in the original dataset)
    train_labels = np.array([dataset[i]['label'].item() for i in train_idx])
    val_labels   = np.array([dataset[i]['label'].item() for i in val_idx])

    # Save
    np.save(os.path.join(args.output_dir, 'train_emb.npy'), train_emb)
    np.save(os.path.join(args.output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(args.output_dir, 'val_emb.npy'), val_emb)
    np.save(os.path.join(args.output_dir, 'val_labels.npy'), val_labels)

    print(f"Saved embeddings to {args.output_dir}")
    print(f"Train embeddings shape: {train_emb.shape}")
    print(f"Val embeddings shape:   {val_emb.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mpii_mat', required=True,
                        help='Path to MPII .mat annotation file')
    parser.add_argument('--mpii_images', required=True,
                        help='Path to MPII image directory')
    parser.add_argument('--output_dir', default='outputs/ssl_mpii_pretrained',
                        help='Directory containing best_ssl_model.pth and where to save embeddings')
    parser.add_argument('--indices_dir', default='.',
                        help='Directory containing mpii_train_indices.npy and mpii_val_indices.npy')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    extract_embeddings(args)