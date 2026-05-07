import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import load_metadata, split_dataset, HAM10000Dataset, CLASS_TO_IDX
from tta_ensemble import load_efficientnetv2, load_dinov2_registers, get_tta_transforms


def generate_mega_probs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_metadata(r"D:\ML\skin-lesion-vit\data\raw")
    _, _, test_df = split_dataset(df)

    output_dir = "experiments/mega_tta_sync"
    os.makedirs(output_dir, exist_ok=True)

    # FIX 1 — One shared get_tta_transforms(image_size=224) was applied to
    # BOTH models. EfficientNetV2-M was trained at 384px (engine.py phase 2:
    # "input_size": 384). Running TTA at 224px silently degrades its output.
    # Each model now carries its own image_size so the input matches training.
    models_to_run = {
        "eff":  (load_efficientnetv2,   "experiments/phase2_efficientnetv2/best_model.pth",             384),
        "dino": (load_dinov2_registers, "experiments/phase10_dinov2_registers_finetune/best_model.pth", 224),
    }

    for name, (loader_fn, ckpt, img_size) in models_to_run.items():
        print(f"Running TTA for {name} at {img_size}px...")
        model = loader_fn(ckpt).to(device).eval()

        tta_transforms = get_tta_transforms(image_size=img_size)  # FIX 1

        all_view_probs = []
        for i, tf in enumerate(tta_transforms):
            ds = HAM10000Dataset(test_df, transform=tf)
            loader = DataLoader(ds, batch_size=16, shuffle=False)

            view_probs = []
            for imgs, labels in tqdm(loader, desc=f"View {i + 1}"):
                with torch.no_grad():
                    logits = model(imgs.to(device))
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    view_probs.append(probs)
            all_view_probs.append(np.concatenate(view_probs))

        avg_probs = np.mean(all_view_probs, axis=0)
        np.save(f"{output_dir}/{name}_probs.npy", avg_probs)

    # FIX 2 — 'label_idx' column does not exist in the DataFrame.
    # The dataset only stores a 'dx' column with string class names (e.g.
    # 'mel', 'nv'). CLASS_TO_IDX maps those to integers inside __getitem__
    # at runtime but never writes back into the df, so test_df['label_idx']
    # raises a KeyError that crashes the entire script at the very last line.
    # Map 'dx' through CLASS_TO_IDX explicitly to build the integer array.
    labels = test_df["dx"].map(CLASS_TO_IDX).values
    np.save(f"{output_dir}/labels.npy", labels)
    print(f"Sync complete. Files saved to {output_dir}")


if __name__ == "__main__":
    generate_mega_probs()