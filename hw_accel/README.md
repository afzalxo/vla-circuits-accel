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


