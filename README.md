
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


## 22 Feb

Many updates to dataset, model, training, evaluation, and hardware accelerator.

- Dataset: Collected dataset from Carla for driving in various conditions: `base`, `weather`, and `dynamic`. The `base` dataset is collected in clear weather with no traffic/pedestrians. The `weather` dataset is collected in heavy rain with fog and wetness, while the `dynamic` dataset is collected in clear weather but with 40 traffic actors and 40 pedestrian actors in the scene. Each dataset has around 400k samples, and the images are of size 256x256. I used perturbations to ensure a more robust set of data since an expert driver doesnt make for a good dataset: using an expert driver to collect the dataset, there are no samples where the ego vehicle is recovering from its mistakes. I applied perturbations intermittently to the steer to simulate mistakes in driving, and I captured the experts recovery from those mistakes. Collecting this recovery dataset makes our model more robust to mistakes it might make.

- Model: I am now using separate MLP heads for different driving scenarios: i.e., 'turn left' invokes a different head from 'follow lane' command. This is because the distribution of the data is very different for different commands, and using separate heads allows the model to learn better. I am also using a weighted loss function to address the dataset imbalance issue. The weights are calculated based on the frequency of each command in the dataset, and they are used to give more importance to the less frequent commands during training. The model is still VGG-like, but is deeper and wider. The instruction modality only selects the different MLP heads; it no longer generates features, but is instead used as a switch to select the appropriate head.

- Training: I am using a weighted loss function with branches. For a samples instruction, we calculate the loss using only the results from the head corresponding to that instruction, and we weight of the steer and throttle losses based on their magnitudes. This way, we can address the dataset imbalance issue and ensure that the model learns to steer well even in scenarios where the steer values are low.

- Evaluation: I am now performing evaluation on live carla runs by making the model traverse the vehicle in the carla environment on a predetermined route. The evaluation environment instantiates the same assets (ego vehicle, camera, weather, traffic signals, instructions) as the dataset collection environment. The frames from the camera are fed to the model, which predicts the steer and throttle, which are fed back to the vehicle control, followed by the simulator tick. This way, we can evaluate the model in a closed loop setting and see how well it performs in real-time. I am using the following two metrics for evaluation: 1) Completion percentage: the percentage of the predetermined route that the model is able to complete without crashing, stalling, or going off-road. 2) Cross-track error (CTE): the distance between the center of the vehicle and the center of the lane, averaged over the entire route. This metric gives us an idea of how well the model is able to stay in its lane, and how well it handles curves and turns. I am attaching a couple videos of the model driving the same route in clear and heavy rain environments:




- Hardware Accelerator: I have finished closed-loop tests with carla+hardware accelerator. I have two machines: an A100 GPU server running the carla simulator, and an Alveo U50 server running the hardware accelerator on the FPGA. I used TCP to collect the two machines. The carla machine generates the camera frames, previous action (steer, throttle, brake, ego vehicle speed), and the instruction (based on a preconfigured route). This data is sent to the FPGA machine over TCP. The FPGA machine loads the hardware accelerator bit-stream onto the FPGA, loads the model weights, and enters a listen state waiting for frames to arrive over the TCP connection. When a frame arrives, the host code on the FPGA machine transfers the data to the FPGA's HBM memory, triggers the accelerator to perform inference, waits for the results (steer and acceleration), dequantizes them and sends them back to the machine running carla. This machine then feeds the steer and throttle to the vehicles control, followed by the simulator tick. This way, we have a closed loop system where the model is running on the hardware accelerator and controlling the vehicle in real-time. I will later record and attach a video of the closed loop system in action. Currently, I noticed that the model's performance is bad when `bias=False` in the model's dense layers (I had set `bias=False` to make it hardware friendly since I have not yet added support for bias in the hardware accelerator). I am currently working on adding bias support to the hardware accelerator, and I have tested models performance with `bias=True`, and it achieves `100%` completion and low CTE as shown in the videos above. 
