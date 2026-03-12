
## N-ISA: An Instruction Set Architecture for Compositional Execution of Sparse Vision-Language Agents

### 22 Feb

Many updates to dataset, model, training, evaluation, and hardware accelerator.

- Dataset: Collected dataset from [Carla](https://github.com/carla-simulator/carla) for driving in various conditions: `base`, `weather`, and `dynamic`. The `base` dataset is collected in clear weather with no traffic/pedestrians. The `weather` dataset is collected in heavy rain with fog and wetness, while the `dynamic` dataset is collected in clear weather but with 40 traffic actors and 40 pedestrian actors in the scene. Each dataset has around 400k samples, and the images are of size $256\times256$. I used perturbations to ensure a more robust set of data since an expert driver doesnt make for a good dataset: using an expert driver to collect the dataset, there are no samples where the ego vehicle is recovering from its mistakes. I applied perturbations intermittently to the steer to simulate mistakes in driving, and I captured the experts recovery from those mistakes. Collecting this recovery dataset makes our model more robust to mistakes it might make. A sample driving scene showing the expert driving, perturbations and recovery, is shown below (I recorded this in 3rd person to show perturbations, the actual dataset is recorded in first person with a front-mounted camera. Also, we dont record the perturbations, only the recovery and normal frames). 

[dataset-collection-carla.webm](https://github.com/user-attachments/assets/b6086389-6b46-4916-9fb4-4befe1e42524)

A couple more scenes, but in first person camera, and with perturbation frames removed are shown below:

| collection-weather-carla.webm | collection-dynamic-carla.webm |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/2fb8bdfa-2c02-470c-8e08-60e1664a1ed4" width="512" controls></video> | <video src="https://github.com/user-attachments/assets/63d2a430-33a5-4648-83fd-648169f9b571" width="512" controls></video> |

- Model: I am now using separate MLP heads for different driving scenarios: i.e., 'turn left' invokes a different head from 'follow lane' command. This is because the distribution of the data is very different for different commands, and using separate heads allows the model to learn better. I am also using a weighted loss function to address the dataset imbalance issue. The weights are calculated based on the frequency of each command in the dataset, and they are used to give more importance to the less frequent commands during training. The model is still VGG-like, but is deeper and wider. The instruction modality only selects the different MLP heads; it no longer generates features, but is instead used as a switch to select the appropriate head.

- Training: I am using a weighted loss function with branches. For a sample instruction, we calculate the loss using only the results from the MLP head corresponding to that instruction, and we weight the steer and throttle losses based on their magnitudes; if the model gets wrong a steer value which is high in the ground truth labels, it is penalized more. This way, we can address the dataset imbalance issue and ensure that the model learns to steer well even in rarer scenarios (e.g., high steer values are relatively rare as turns are rarer than going straight).

- Evaluation: I am now performing evaluation on live carla runs by making the model traverse the vehicle in the carla environment on a predetermined route. I perform a live evaluation run on each training epoch. The evaluation environment instantiates the same assets (ego vehicle, camera, weather, traffic signals, instructions) as the dataset collection environment. The frames from the camera are fed to the model, which predicts the steer and throttle, which are fed back to the vehicle control, followed by the simulator tick. This way, we can evaluate the model in a closed loop setting and see how well it performs in real-time. I am using the following two metrics for evaluation: 1) Completion percentage: the percentage of the predetermined route that the model is able to complete without crashing, stalling, or going off-road. 2) Cross-track error (CTE): the distance in meters between the center of the vehicle and the center of the lane, averaged over the entire route. This metric gives us an idea of how well the model is able to stay in its lane, and how well it handles curves and turns. I am attaching a couple videos of the best trained model driving the same route in clear and heavy rain environments:

| ClearNoon | HeavyRain |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/387b22a4-ba63-4ab0-a5ab-5839febfc130" width="512" controls></video> | <video src="https://github.com/user-attachments/assets/58505fe1-babb-4aa7-8f0c-52e7cc6e3eb6" width="512" controls></video> |

The evaluation route is shown in the image below, with spawn point in green and destination in red. Different colors along the route show different commands (e.g., yellow for 'turn right').

<p align="center">
  <img src="https://github.com/user-attachments/assets/abfd5b25-bec7-43f9-b03a-5a16c01d158c" alt="BEV Route Carla Eval" width="512"/>
</p>

Some evaluation results (completion percentage and average CTE) on this route during the model training are shown below:

<p align="center">
  <img src="https://github.com/user-attachments/assets/22bdc4f9-1bd5-4c0d-8bc6-4f66bf416b46" alt="BEV Route Carla Eval" width="512"/>
</p>

Perhaps a pitfall here: the evaluation is performed on a single hardcoded route in `base` envrionment. Perhaps a more rigid evaluation should cover multiple routes and all dataset conditions (`base`, `weather`, and `dynamic`).

- Hardware Accelerator: I have finished closed-loop tests with carla+hardware accelerator. I have two machines: an A100 GPU server running the carla simulator, and an [Alveo U50](https://www.amd.com/en/products/accelerators/alveo/u50/a-u50-p00g-pq-g.html) server running the hardware accelerator on the FPGA. I used TCP to connect the two machines. The carla machine generates the camera frames, previous action (steer, throttle, brake, ego vehicle speed), and the instruction (based on a preconfigured route). This data is sent to the FPGA machine over TCP. The FPGA machine loads the hardware accelerator bit-stream onto the FPGA, loads the model weights, and enters a listen state waiting for frames to arrive over the TCP connection. When a frame arrives, the host code on the FPGA machine transfers the data to the FPGA's HBM memory, triggers the accelerator to perform inference, waits for the results (steer and acceleration), dequantizes them and sends them back to the machine running carla. This machine then feeds the steer and throttle to the vehicles control, followed by the simulator tick. This way, we have a closed loop system where the model is running on the hardware accelerator and controlling the vehicle in real-time. I will later record and attach a video of the closed loop system in action. Currently, I noticed that the model's performance is bad when `bias=False` in the model's dense layers (I had set `bias=False` to make it hardware friendly since I have not yet added support for bias in the hardware accelerator). I am currently working on adding bias support to the hardware accelerator, and I have tested models performance with `bias=True`, and it achieves `100%` completion and low CTE as shown in the videos above. After finishing `bias=True` support, I will evaluate FPGA vs GPU runs in terms of our evaluation metrics (there will be a slight disparity between GPU and FPGA results due to 8-bit quantization on the FPGA, although the disparity isn't that significant).

The following image shows the device view of the AMD/Xilinx's Alveo U50 chip implementing the hardware accelerator. Some components of the design have been highlighed with different colors. For example, the massive red area is the actual CONV/GEMM accelerator, the green area is the logic used for instruction scheduler, the logic in magenta near the HBM stacks are the HBM interfaces used to exchange data (features, weights, biases) between the chip and the HBM. The chip has two vertically stacked dies, the top die is almost entirely unutilized as can be seen in the top half of the chip. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/4b9ee38e-7da2-47e7-ac44-bd5dd1e87b68" alt="Device View Accelerator" width="480"/>
</p>

The design is currently running at only around 148 MHz as I have not optimized certain critical paths as of yet. I am expecting the final design frequency to be in the range 250-300 MHz. Per frame latency is currently around 15ms, but again, this will likely be reduced to sub 10ms when things are finalized. 

The next steps are to train the model with instruction-specific grouping of tiles, similar to what we did for the BabyAI dataset. The hardware accelerator already supports tile masking using `is_sparse`, `ic_tile_mask`, and `oc_tile_mask` flags which allow skipping portions of the model not needed for the active task. Once this is finished, I will work on task composition: Circuit('follow lane' in 'weather') U Circuit('follow lane' in 'dynamic') = Circuit('follow lane' in 'weather' and 'dynamic').


### Updates on Model and Hardware

I have been working on improving the hardware accelerators performance. 

1. I improved the end-to-end latency from the previous result of around 19.4 ms to around 15 ms by improving memory-bandwidth utilization. Previously, I was performing external memory access for feature maps in bursts such that we wait for an entire 4KB burst of data to arrive to the FPGA before the next burst transaction is issued. In the new read master, I am using outstanding reads to issue bursts requests in a pipeline, so we can avoid ~100 cycle overhead of having to wait for a new burst request to arrive at the HBM memory controller. The improvement is achieved without making any sacrifices to resource utilization or design frequency. 

2. I also improved the design frequency from 133 MHz to 181 MHz by tuning critical paths in the design. There is still room for improvement into the 250-300 MHz range.

3. I also added double buffering at the input features, weights and biases level. So now while we are fetching features and weights for tile N+1, we have the accelerator performing computations for tile N. This way, we can hide the latency of memory accesses behind the computation latency. This double buffering works across IC/OC tiles. So if all IC tiles are finished being fetched, we proceed to the next OC's first IC tile for fetch while performing compute on the last IC tile of the previous OC.

On the model side, I added a task-gated mechanism for training the model. I added a `TileGatingNetwork` which predicts which of the hardware tiles to turn on or off for the currently active task. The training process involves training the model parameters and the tile gating network jointly, so the tiles chosen for a certain task are optimized for that task. I trained the gated model on the carla dataset. The trained model achieves around 35.3% tile sparsity (i.e., this proportion of the total tiles in the model are inactive). Here is the breakdown of the MAC and parameter activation for different tasks:

## Hardware-Aligned Computation & Parameter Sparsity (VLA Gated Conv Layers)

| Task               | Active MACs | Total MACs | Active % (MACs) | Active Params | Total Params | Active % (Params) |
|--------------------|-------------|------------|-----------------|---------------|--------------|-------------------|
| Follow Lane        | 240.95M     | 429.39M    | 56.1%           | 6.92M         | 11.01M       | 62.9%             |
| Turn Left          | 273.53M     | 429.39M    | 63.7%           | 5.90M         | 11.01M       | 53.6%             |
| Turn Right         | 301.99M     | 429.39M    | 70.3%           | 6.49M         | 11.01M       | 58.9%             |
| Change Lane Left   | 298.09M     | 429.39M    | 69.4%           | 6.89M         | 11.01M       | 62.6%             |
| Change Lane Right  | 237.95M     | 429.39M    | 55.4%           | 5.49M         | 11.01M       | 49.9%             |
| Stop               | 83.53M      | 429.39M    | 19.5%           | 1.55M         | 11.01M       | 14.1%             | 

Perhaps unsurprisingly, simple tasks such as stop consume the least amount of compute and parameters. The sparsity can be increased by exploring the `LAMBDA_SPARSE` vs driving capability tradeoff. I was conservative in choosing `LAMBDA_SPARSE` because I am strict on driving metrics (I evaluate on multiple driving routes in multiple driving scenarios, and I dont want the model to fail at a route due to the sparsity).

I also measured runtimes of different tasks on the FPGA to ensure that the tile-switching is working as intended and we are indeed getting linear speedups as expected from the sparsity, whereas GPU results show no speedup from the sparsity due to the lack of support for dynamic switching of tiles on the GPU. However, the A100 GPU achieves a much lower latency compared to the FPGA (around 2-3x faster) in the dense case. This is mainly due to incredibly high clock speed and memory bandwidth on the GPU. This can be addressed if we send our design to a fab.  

The task specific activation of tiles is shown below:




Right now, I am working on further improving the design's latency somehow. Three avenues remain:

1. Improving the design frequency by further tuning the critical path (I hope I can get to 250-300 MHz range, thats a 65% improvment in the best case. 

2. Scaling up the design: I am thinking of increasing `IC_PAR`/`OC_PAR` from 16 to 32. This will double the speed and resource utilization. I might have to move to a buffer FPGA for this.

3. Output double buffering: Currently, when we are writing the output for a tile, the compute is sitting idle until the write transaction is finished. This can be remedied by double buffering the output. So while writing the output for tile N, we can start computing tile N+1's output. 

Beyond this, the main experiments will be on the latency and inference per kilo joule improvements as a result of our sparsity. Energy consumption experiments are yet to be performed although the GPU seems to be doing suprisingly well for batch 1 power (around 70-80W).  
