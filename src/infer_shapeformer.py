import argparse
import torch
import numpy as np
import json
from Models.shapeformer import model_factory
from Shapelet.mul_shapelet_discovery import ShapeletDiscover

# ============================================================

def load_ts(ts_path):
    X_list = []
    y_list = []
    in_data = False
    
    with open(ts_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("@"):
                if line.lower().startswith("@data"):
                    in_data = True
                continue

            if not in_data:
                continue

            parts = line.split(":")
            *dims_str, label = parts

            dims = [
                list(map(float, d.split(",")))
                for d in dims_str
            ]

            X_list.append(np.array(dims, dtype=np.float32))
            y_list.append(label)

    return X_list, y_list


def load_model_exact(config_path, ckpt_path, shapelet_pkl, device, X_tensor=None):
    print("\n=== Reconstructing EXACT ShapeFormer ===")

    # 1) Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # 2) Reconstruct input shape from the test tensor itself
    #    X_tensor comes with shape (N, C, T)
    if X_tensor is None:
        raise ValueError("You must pass X_tensor to load_model_exact()")

    _, ts_dim, len_ts = X_tensor.shape
    print(f"Reconstructed: ts_dim={ts_dim}, len_ts={len_ts}")

    # 3) Important parameters
    window_size = config.get("window_size", 100)
    num_pip = config.get("num_pip", 0.2)
    processes = config.get("processes", 64)
    num_shapelet = config.get("num_shapelet", 3)

    # 4) ShapeletDiscover only to load the PKL
    sd = ShapeletDiscover(
        window_size=window_size,
        num_pip=num_pip,
        processes=processes,
        len_of_ts=len_ts,
        dim=ts_dim
    )
    sd.load_shapelet_candidates(path=shapelet_pkl)
    shapelets_info = sd.get_shapelet_info(number_of_shapelet=num_shapelet)

    # Rescale IG weights as in main.py
    sw = torch.tensor(shapelets_info[:,3])
    sw = torch.softmax(sw*20, dim=0) * sw.shape[0]
    shapelets_info[:,3] = sw.numpy()

    # Dummy shapelets according to their lengths
    shapelets = []
    for si in shapelets_info:
        start = int(si[1])
        end = int(si[2])
        length = max(end - start, 1)
        shapelets.append(np.zeros(length, dtype=np.float32))

    # Insert into config
    config["shapelets_info"] = shapelets_info
    config["shapelets"] = shapelets
    config["len_ts"] = len_ts
    config["ts_dim"] = ts_dim

    # Actual number of classes
    config["num_labels"] = 4     # BasicMotions has 4 classes
    config["Data_shape"] = (1, ts_dim, len_ts)

    # 5) Create empty model
    model = model_factory(config).to(device)

    # 6) Load weights
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("Missing weights:", missing)
    print("Unexpected weights:", unexpected)

    return model


def run_inference(ts_path, ckpt_path, shapelet_pkl, config_path, device="cpu"):
    device = torch.device(device)

    X_list, y_list = load_ts(ts_path)

    X = np.stack([x.T for x in X_list], axis=0)  # (N, T, C)
    X_tensor = torch.from_numpy(X).float().to(device)
    X_tensor = X_tensor.permute(0, 2, 1)         # (N, C, T)

    print("Input to model =", X_tensor.shape)

    model = load_model_exact(
    config_path, ckpt_path, shapelet_pkl,
    device,
    X_tensor=X_tensor
    )

    with torch.no_grad():
        logits = model(X_tensor, ep=0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # Map real labels
    unique_labels = sorted(set(y_list))
    label2idx = {lab: i for i, lab in enumerate(unique_labels)}
    y_true = np.array([label2idx[y] for y in y_list])

    acc = (preds == y_true).mean() * 100
    print(f"\nTEST Accuracy: {acc:.2f}%")

    for i in range(len(preds)):
        print(f"{i}: true={y_list[i]} pred={unique_labels[preds[i]]}")


# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shapelet_pkl", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_inference(
        ts_path=args.ts_path,
        ckpt_path=args.checkpoint,
        shapelet_pkl=args.shapelet_pkl,
        config_path=args.config,
        device=args.device
    )

if __name__ == "__main__":
    main()
