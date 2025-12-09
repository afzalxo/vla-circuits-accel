# Dataset Collection

1. Install dependencies:

```bash
pip install minigrid gymnasium
```

2. Collect 200k samples from the MiniGrid environment:

```bash
python3 collect_data.py
```

The samples and dataset will be saved in a directory named `babyai_vla_dataset`.

3. The training and val can be started by running:

```bash
python3 train_dense_moe_fast.py
```

I will upload the trained model pth file to a file hosting service and share the link here soon.
