### Commands 

1. Training
```python
python main.py --dataset_pos=1 --num_shapelet=10 --window_size=80 --pre_shapelet_discovery=0 --processes=12 --epochs=100 --lr=0.001 --weight_decay=5e-4 --dropout=0.2 --shape_embed_dim=128 --pos_embed_dim=128 --emb_size=64 --local_embed_dim=48 --local_pos_dim=48 --dim_ff=256 --num_heads=4 --local_num_heads=4 --num_pip=0.1
```


2. Inference
```python
python infer_shapeformer.py --ts_path "Dataset/UEA/Multivariate_ts/BasicMotions/BasicMotions_TEST.ts" --checkpoint "Results/Dataset/UEA/2025-11-16_12-02/checkpoints/BasicMotionsmodel_last.pth" --shapelet_pkl "store/BasicMotions_80.pkl" --config "Results/Dataset/UEA/2025-11-16_12-02/configuration.json" --device cpu
```