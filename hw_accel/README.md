## Accelerator code for N-ISA for Language-Conditioned Dynamic Sparsity

This is a work in progress. The following gadgets have been implemented so far:

- At the highest level, the accelerator allows performing convolutions on input features of size H x W x IC x OC with kernels of size 3 x 3 x IC x OC.
    - I have tested different configurations of H, W, IC, and OC to ensure correctness. W needs to be a multiple of $PP\_{PAR}$ (described later).
    - Scaling to larger input channels (IC) and output channels (OC) is a WIP, it would require forumulating three nested loops described later.

- The gadgets needed to make this operational are:
    - A line buffer (`line_buffer.sv`) to store input feature maps and provide sliding windows for convolution.
    - A window sequencer (`window_sequencer.sv`) to extract 3x3 windows from the line buffer.
    - A vector compute unit (`vector_compute_unit.sv`) consisting of $PP\_{PAR}$ processing elements (PEs) operating in parallel. This corresponds to pixel parallelism.
    - A processing element (PE) (`processing_element.sv`) consisting of $OC\_{PAR}$ multiply-accumulate (MAC) lanes. This is parallelism in kernels output channel dim.
    - A MAC lane (`mac_lane.sv`) that performs MAC between a feature map of IC channels and a kernel of IC channels (1x1 convolutions). Parallelism in input channel dim.
    - A conv controller (`conv_controller.sv`) module that orchestrates control signals and data movement between different modules.

- The accelerator is designed to be configurable with the following parameters:
    - $PP\_{PAR}$: Number of pixels processed in parallel (I think 8 is a good choice).
    - $IC\_{PAR}$: Number of feature map channels processed in parallel (I think 16 is a good choice).
    - $OC\_{PAR}$: Number of output channels processed in parallel (I think 16 is a good choice).
    - The width of the features needs to be a multiple of $PP\_{PAR}$ for the design to work correctly. (For now, we can apply zero-padding later to allow functionality over arbitrary widths).
    - I have tested heights greater than equal to 3. Not sure if there are any constraints on height for now (I think 3 or more should be fine).
    - IC and OC need to be multiples of $IC\_{PAR}$ and $OC\_{PAR}$ respectively to fully utilize the parallelism. (Scaling to arbitrary IC and OC is a WIP).
    - I have provided test benches for most of the modules. The top-level test bench is `tb_full_image.sv` which tests the entire accelerator for a full image convolution (under the above constraints).
    - For testing, use the following flow:
        - Generate random input features and weights using the provided Python script `generate_test_vectors.py`.
        - `input_image.hex` and `weights.hex` are used by the test bench to load data into the accelerator.
        - Use `xvlog -sv <path to tb.sv> <path to all srcs.sv>` to compile the design. (This is using AMD/Xilinx's tools, you could use something like `iverilog` as well).
        - Use `xelab -debug typical tb_full_image -s tb_snapshot` to elaborate the design.
        - Use `xsim tb_snapshot -R` to run the simulation within the terminal. It will generate `hw_output.csv` containing the output feature map.
        - Or you could use `xsim tb_snapshot -gui` to launch the GUI and inspect waveforms.

### Next Steps:

- Implement nested loops to scale to larger IC and OC. This is going to be a bit tricky as it would require buffering partial sums and accumulating them across multiple passes. I am thinking the following nested loop structure (pseudocode below):

```
for h in range(0, H, H_{tile}):
    Load Feature Tile of size H_{tile} x W x C
    for oc in range(0, OC, OC_{PAR}):
        accum = 0
        for ic in range(0, C, IC_{PAR}):
            // The contents of this loop are what is currently implemented and are described above
            Load Weight Tile of size 3 x 3 x IC_{PAR} x OC_{PAR}
            Stream feature data from URAM -> Line Buffer -> Window Sequencer -> CU
            Compute and accumulate partial sums in CU
        We now have the final result for this Output Tile.
        Apply ReLU -> DMA Write Output Strip to HBM
```


---

## Update 21st December: Hardware Emulation and Memory Subsystem Integration

Focus: System-Level Integration, Memory Tiling, and C++ Host Emulation

### Summary
We have transitioned from block-level Verilog simulation to a full Hardware Emulation Workflow. This update introduces the top-level memory management logic (`tile_manager`), the Direct Memory Access (DMA) engines, and a C++ Host application that drives the simulation. The accelerator now supports loading tiled feature maps and weights from external memory (simulated HBM), enabling the processing of arbitrary image sizes via tiling.

### Architectural Changes

#### 1. Top-Level Hierarchy (`vla_accel_top`)
The design has been wrapped in an AXI-compliant top-level module suitable for AMD/Xilinx Vitis/Vivado integration.
*   **Control Interface:** AXI4-Lite (`s_axi_control`) for register configuration (start, done, pointers).
*   **Data Interface:** AXI4-Master (`m_axi_gmem`) for high-bandwidth access to external memory.
*   **Core Logic:** Instantiates the `tile_manager` which governs the accelerator core.

#### 2. The Tile Manager (`tile_manager.sv`)
A new control unit that abstracts memory management from the compute core.
*   **Function:** Orchestrates the movement of data between Global Memory (HBM), On-Chip Memory (URAM/BRAM), and the Compute Core.
*   **Tiling Strategy:** Implements **Height-Tiled, Input-Stationary** dataflow.
    1.  Loads a horizontal strip of the Input Feature Map (e.g., 4 rows) into **URAM**.
    2.  Iterates through Input Channel (IC) tiles.
    3.  Loads corresponding Weights into **BRAM**.
    4.  Streams data through the `conv_accelerator`.
    5.  Accumulates Partial Sums in the Output URAM.

#### 3. DMA Engines
*   **`tiled_dma.sv`:** Handles 3D strided access to fetch specific spatial tiles ($H_{tile} \times W \times IC_{tile}$) from the flat memory address space in HBM to on-chip URAM.
*   **`weights_dma.sv`:** Fetches weight blocks ($OC_{tile} \times IC_{tile} \times 3 \times 3$) into on-chip BRAM.

### Host-Side Emulation (C++)

We have replaced the SystemVerilog testbench (`tb_full_image.sv`) with a C++ Host Application (`host_main.cpp`) that mimics the runtime driver.

*   **Data Packing:** Implemented custom packing functions to transform standard NCHW/NHWC tensors into the hardware-optimized blocked layout:
*   **Bit-Accurate Verification:** The host code generates a "Golden Model" ground truth using the exact same integer arithmetic and padding logic as the hardware, allowing for automated bit-perfect verification of the accelerator output.

### Accelerator Improvements 

Several critical bugs were resolved in the core logic to support this integration:
*   **Line Buffer:** Fixed Read-After-Write hazards by implementing explicit temporary registers for memory shifts.
*   **Controller:** Simplified the state machine to a "Spatial-First" processing model. The controller now processes a full spatial strip before requesting new IC tiles, reducing control overhead.
*   **Signed Arithmetic:** Aligned the C++ Host and Verilog Hardware to strictly use `int8` (2's complement) arithmetic, resolving negative value discrepancies in the MAC units.

### Current Status

*   **Interface:** AXI-Stream handshaking between DMA and Accelerator is stable.
*   **Memory:** URAM/BRAM read/write logic with partial sum accumulation is verified.
*   **Computation:** First strip of the first tile matches Ground Truth bit-for-bit.
*   **Accumulation:** SIMD-style accumulation of partial sums in the Tile Manager is functional.

### Next Steps
- [x]   **Halo Loading:** Implement automatic loading of "Halo" rows (Top/Bottom padding) in the DMA to support seamless vertical tiling without artifacts at tile boundaries (currently, the conv output at bottom boundaries of feature maps horizontal tiles is incorrect since we only load the current tile, whereas the output depends on the halo). **Update (21st December): Completed and verified.**
- [ ]   **Looping over OC Tiles:** Extend the Tile Manager to loop over Output Channel tiles, enabling processing of arbitrary OC sizes (currently only processing the first OC tile).
- [ ]   **Looping over H Tiles:** Extend the Tile Manager to loop over Height tiles, enabling processing of arbitrary image heights (currently only processing the first OC tile of the first H tile).
- [x]   **Storing Output of Current H Tile/OC Tile:** Implement logic to write back the completed output tile to HBM after processing all IC tiles for the current H/OC tile. **Update (23 Dec): (`output_dma.sv`) Added logic to write only the first H/OC tile (since looping over H/OC hasnt been implemented yet)**
- [ ]   **Double Buffering:** Implement double buffering so the output storing of the current H/OC tile can occur in parallel with loading the next tile.
- [ ]   **Full Image Verification:** Run the emulation on a complete $128 \times 128$ image to verify tile switching logic and currect output computation for one conv layer.
- [ ]   **ReLU Activation:** Integrate ReLU activation as a post-processing after computing current H/OC tile.
- [x]   **Re-Quantization:** Implement re-quantization logic after ReLU to map outputs of accumulator bit-width (28-bits) back to int8 range. **Update (23rd December): Re-quantization/scaling of fmap outputs has been added (need to adjust the `quant_shift` variable according to data scale**)
- [ ]   **Multi-Layer Support:** Extend accelerator to support multiple conv layers in sequence.
- [ ]   **CARLA Integration:** Begin collecting the autonomous driving dataset to train the sparse VLA model for deployment.

---
