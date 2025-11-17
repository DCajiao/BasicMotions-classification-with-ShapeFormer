import argparse
import torch
import numpy as np
import json
import pickle
from Models.shapeformer import model_factory
from Shapelet.mul_shapelet_discovery import ShapeletDiscover

# ============================================================
# 1. LEER ARCHIVO .TS
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


# ============================================================
# 2. CARGAR MODELO COMPLETO
# ============================================================

def load_model_exact(config_path, ckpt_path, shapelet_pkl, device, X_tensor=None):
    print("\n=== Reconstruyendo ShapeFormer EXACTO ===")

    # 1) Cargar config
    with open(config_path, "r") as f:
        config = json.load(f)

    # 2) Reconstruir forma de entrada desde el propio tensor test
    #    X_tensor llega con shape (N, C, T)
    if X_tensor is None:
        raise ValueError("Debes pasar X_tensor a load_model_exact()")

    _, ts_dim, len_ts = X_tensor.shape
    print(f"Reconstruido: ts_dim={ts_dim}, len_ts={len_ts}")

    # 3) Parámetros importantes
    window_size = config.get("window_size", 100)
    num_pip = config.get("num_pip", 0.2)
    processes = config.get("processes", 64)
    num_shapelet = config.get("num_shapelet", 3)

    # 4) ShapeletDiscover solo para cargar el PKL
    sd = ShapeletDiscover(
        window_size=window_size,
        num_pip=num_pip,
        processes=processes,
        len_of_ts=len_ts,
        dim=ts_dim
    )
    sd.load_shapelet_candidates(path=shapelet_pkl)
    shapelets_info = sd.get_shapelet_info(number_of_shapelet=num_shapelet)

    # Reescalar pesos IG como en main.py
    sw = torch.tensor(shapelets_info[:,3])
    sw = torch.softmax(sw*20, dim=0) * sw.shape[0]
    shapelets_info[:,3] = sw.numpy()

    # Shapelets dummy según sus longitudes
    shapelets = []
    for si in shapelets_info:
        start = int(si[1])
        end = int(si[2])
        length = max(end - start, 1)
        shapelets.append(np.zeros(length, dtype=np.float32))

    # Insertamos en config
    config["shapelets_info"] = shapelets_info
    config["shapelets"] = shapelets
    config["len_ts"] = len_ts
    config["ts_dim"] = ts_dim

    # Número de clases real
    config["num_labels"] = 4     # BasicMotions tiene 4 clases
    config["Data_shape"] = (1, ts_dim, len_ts)

    # 5) Crear modelo vacío
    model = model_factory(config).to(device)

    # 6) Cargar pesos
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("Pesos faltantes:", missing)
    print("Pesos inesperados:", unexpected)

    return model



# ============================================================
# 3. INFERENCIA
# ============================================================

def run_inference(ts_path, ckpt_path, shapelet_pkl, config_path, device="cpu"):
    device = torch.device(device)

    X_list, y_list = load_ts(ts_path)

    X = np.stack([x.T for x in X_list], axis=0)  # (N, T, C)
    X_tensor = torch.from_numpy(X).float().to(device)
    X_tensor = X_tensor.permute(0, 2, 1)         # (N, C, T)

    print("Entrada a modelo =", X_tensor.shape)

    model = load_model_exact(
    config_path, ckpt_path, shapelet_pkl,
    device,
    X_tensor=X_tensor
    )

    with torch.no_grad():
        logits = model(X_tensor, ep=0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # Mapear etiquetas reales
    unique_labels = sorted(set(y_list))
    label2idx = {lab: i for i, lab in enumerate(unique_labels)}
    y_true = np.array([label2idx[y] for y in y_list])

    acc = (preds == y_true).mean() * 100
    print(f"\nAccuracy TEST: {acc:.2f}%")

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
