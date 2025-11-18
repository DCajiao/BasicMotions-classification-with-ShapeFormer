import streamlit as st
import torch
import numpy as np
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

from Models.shapeformer import model_factory
from Shapelet.mul_shapelet_discovery import ShapeletDiscover

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
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
    logger.info("Rebuilding ShapeFormer")

    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Rebuild input shape from the test tensor itself
    # X_tensor arrives with shape (N, C, T)
    if X_tensor is None:
        msg = "You must pass X_tensor to load_model_exact()"
        logger.error(msg)
        raise ValueError(msg)

    _, ts_dim, len_ts = X_tensor.shape
    logger.info(f"Rebuilt input shape: ts_dim={ts_dim}, len_ts={len_ts}")

    # Important parameters
    window_size = config.get("window_size", 100)
    num_pip = config.get("num_pip", 0.2)
    processes = config.get("processes", 64)
    num_shapelet = config.get("num_shapelet", 3)

    # ShapeletDiscover just to load the PKL
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
    sw = torch.tensor(shapelets_info[:, 3])
    sw = torch.softmax(sw*20, dim=0) * sw.shape[0]
    shapelets_info[:, 3] = sw.numpy()

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

    # Real number of classes
    config["num_labels"] = 4     # BasicMotions has 4 classes
    config["Data_shape"] = (1, ts_dim, len_ts)

    # Create empty model
    model = model_factory(config).to(device)

    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"]
    model_state = model.state_dict()

    filtered_state = {}

    for k, v in state.items():
        if k in model_state:
            # Verify that the size matches
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                logger.warning(
                    f"[IGNORED] Shape mismatch for {k}: ckpt={tuple(v.shape)}, model={tuple(model_state[k].shape)}")
        else:
            logger.warning(f"[IGNORED] Not found in model: {k}")

    # Load ONLY matching keys
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    logger.info("\n=== FILTERED LOAD ===")
    logger.info("Successfully loaded parameters: %d", len(filtered_state))
    logger.info("Missing weights: %s", missing)
    logger.info("Unexpected weights: %s", unexpected)

    return model

# ============================================================


# Streamlit Interface
st.title("Basic Motions Classification with ShapeFormer")
st.write("Upload dataset (`.ts` file) to run inference.")

# Sidebar: upload files and parameters
st.sidebar.title("⚙ Model Configuration")

ts_file = st.sidebar.file_uploader("TS File", type=["ts"])
device = st.sidebar.selectbox("Device", ["cpu"])

# Preloaded paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = os.path.join(
    BASE_DIR, "Results/Dataset/UEA/2025-11-16_12-02/configuration.json")
DEFAULT_CKPT = os.path.join(
    BASE_DIR, "Results/Dataset/UEA/2025-11-16_12-02/checkpoints/BasicMotionsmodel_last.pth")
DEFAULT_SHAP = os.path.join(BASE_DIR, "store/BasicMotions_80.pkl")
st.sidebar.write("---")

if ts_file:
    if st.button("-> Run Inference"):
        st.write("Loading data...")

        # Save temporary files
        ts_path = "temp_input.ts"
        with open(ts_path, "wb") as f:
            f.write(ts_file.getvalue())

        config_path = DEFAULT_CONFIG
        ckpt_path = DEFAULT_CKPT
        shapelet_pkl = DEFAULT_SHAP

        # Load TS
        X_list, y_list = load_ts(ts_path)
        X = np.stack([x.T for x in X_list], axis=0)  # (N, T, C)
        X_tensor = torch.from_numpy(X).float().to(device)
        X_tensor = X_tensor.permute(0, 2, 1)         # (N, C, T)

        model = load_model_exact(
            config_path,
            ckpt_path,
            shapelet_pkl,
            device,
            X_tensor=X_tensor
        )

        # Inference
        with torch.no_grad():
            logits = model(X_tensor, ep=0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        # Map labels
        unique_labels = sorted(set(y_list))
        label2idx = {lab: i for i, lab in enumerate(unique_labels)}
        y_true = np.array([label2idx[y] for y in y_list])

        acc = (preds == y_true).mean() * 100

        st.subheader(f"Accuracy: **{acc:.2f}%**")

        # Results table
        st.subheader("Results")
        import pandas as pd
        df = pd.DataFrame({
            "True Label": y_list,
            "Pred Label": [unique_labels[p] for p in preds]
        })
        st.dataframe(df)

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, preds)
        cm_norm = confusion_matrix(y_true, preds, normalize="true")

        # Plot 1
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                    xticklabels=unique_labels,
                    yticklabels=unique_labels, ax=ax1)
        ax1.set_title("Confusion Matrix (Raw counts)")
        st.pyplot(fig1)

        # Plot 2
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.heatmap(cm_norm, annot=True, cmap="Greens", fmt=".2f",
                    xticklabels=unique_labels,
                    yticklabels=unique_labels, ax=ax2)
        ax2.set_title("Confusion Matrix (Normalized)")
        st.pyplot(fig2)

else:
    st.info("Upload the files in the sidebar to enable inference.")
