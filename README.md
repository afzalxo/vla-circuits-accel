
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

## 26 Dec (HW Accel)

Added multi-layer support and a basic ISA that allows `CONV` and `HALT` instructions for host-configured multi-layer support. The host writes the instruction packet to the HBM[0] heap space (first 64KB). The accelerator reads the instructions, executes each layer sequentially until it encounters `HALT`. Tested and verified using 1-3 conv layers.


## 30 Dec (HW Accel)

Added strided convolutions.

## 2 Jan (HW Accel)

Real data verification

## 12 Jan (Hw Accel)

Added support for GEMM for dense layers. Moving towards end-to-end verification.

## 12 Jan (Model + Dataset)

Added dataset collection script for carla, and model+training script for carla dense lasso model.

## 18 Jan (Hw Accel)

End-to-end hardware verification of new VGG-style model on Carla dataset with 128 x 128 image inputs. Tested and functional verification done for input images obtained from the dataset, and weights obtained from the saved model pth. So far, only the vision modality is being processed on the hardware. 


## 27 Jan (Model + Dataset)

Collected a large dataset (376k samples, 256x256 images) from Carla with various per-frame concept latbels such as `near_pedestrian`, `env_night`, `env_rain`, `near_vehicle`, `near_junction`.

Expanded model to larger image inputs. Currently in the process of training the model and trying to get its performance right. Currently, there seems to a dataset imbalance issue: most of the data is that of the vehicle driving straight, so the model is biased towards regressing low steer values, hence often performs abysmally (e.g., isnt able to steer well when the road curves). Working on solving this and many other issues. 
