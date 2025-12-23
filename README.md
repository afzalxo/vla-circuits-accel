
### Dataset Collection
1. Install dependencies:

```bash
pip install minigrid gymnasium
```

2. Collect 200k samples from the MiniGrid environment:

```bash
python3 collect_data.py
```

The samples and dataset will be saved in a directory named `babyai_vla_dataset`.

### Training and Validation of Task-Sparse MoE with top-32 neurons in fusion layer top-1 expert.
3. The training and val can be started by running:

```bash
python3 train_dense_moe_fast.py
```

The trained model checkpoint is at: [wandb link](https://wandb.ai/afzalxo-hong-kong-university-of-science-and-technology/sparse-vla-accel/runs/s71fp8yl/files?nw=nwuserafzalxo) under the filename `best_vla_moe_fast.pth` and the `vocab.pth` file is also located there.

Train/val log: [wandb link](https://wandb.ai/afzalxo-hong-kong-university-of-science-and-technology/sparse-vla-accel/runs/s71fp8yl/logs?nw=nwuserafzalxo)

### Evaluation

4. The evaluation can be started by running:

```bash
python3 eval_moe.py
```

Please modify the `MODEL_PATH` variable at the top of the file as needed.


## 16 Dec (Update on HW Accelerator):

Please see directory `hw_accel`.

## 21 Dec (Further update on HW Accelerator):

Added Memory Subsystem Integration and Hardware Emulation. Please see directory `hw_accel`.

## 23 Dec (HW Accel)

Convolution of full 128x128x64 feature map with 3x3x64x64 kernel implemented and verified. Takes around 3ms for this size. Perf can be improved using certain optimization techniques like double buffer (WIP). 
