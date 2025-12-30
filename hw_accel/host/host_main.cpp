#include <iostream>
#include <vector>
#include <cstdint> // For uint32_t, int16_t etc.
#include <cmath>
#include <string>
#include <inttypes.h>
#include <thread>
#include <chrono>

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl2.hpp>
#include "xcl2.hpp"

#include "ap_int.h"

constexpr uint16_t NUM_LAYERS = 2;
constexpr uint16_t STRIDE = 2;

constexpr uint16_t FM_WIDTH = 64;    // Feature map width
constexpr uint16_t FM_HEIGHT = 64;   // Feature map height
 
constexpr uint16_t FM_CHANNELS = 32;
constexpr uint16_t OUT_CHANNELS = 32;

constexpr uint16_t IC_PAR = 16;
constexpr uint16_t OC_PAR = 16;
constexpr uint16_t PP_PAR = 8;

constexpr uint16_t TILE_HEIGHT = 4;

constexpr uint16_t FP_WIDTH = 16;
constexpr uint16_t FP_FRAC_BITS = 8;
typedef ap_uint<FP_WIDTH> fixed_t;


template <typename T>
void dump_to_hex_file(const T* data, int bytes_per_line, int length_of_data, const std::string& filename) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    int total_bytes = length_of_data * sizeof(T);
    const uint8_t* raw_ptr = reinterpret_cast<const uint8_t*>(data);

    for (int i = 0; i < total_bytes; i += bytes_per_line) {
        // Print bytes in REVERSE order for this chunk (MSB -> LSB)
        for (int j = bytes_per_line - 1; j >= 0; --j) {
            if (i + j < total_bytes) {
                f << std::hex << std::setw(2) << std::setfill('0') 
                  << (int)raw_ptr[i + j];
            } else {
                // Padding if data isn't perfectly aligned (shouldn't happen with correct dims)
                f << "00";
            }
        }
        f << "\n";
    }
    f.close();
    std::cout << "Dumped " << filename << " (" << bytes_per_line << " bytes per line)" << std::endl;
}

constexpr size_t INSTR_SECTION_SIZE = 64 * 1024; // 64 KB reserved for instructions

// Opcode Definitions
constexpr uint8_t OP_CONV = 1;
constexpr uint8_t OP_HALT = 255;

// --- N-ISA INSTRUCTION STRUCT ---
// Must match the bit slicing in instruction_scheduler.sv
struct alignas(64) NISA_Instruction {
    uint64_t input_offset;   // [63:0]   - Byte offset from Heap Base (HBM0)
    uint64_t output_offset;  // [127:64] - Byte offset from Output Base (HBM2)
    uint64_t weight_offset;  // [191:128]- Byte offset from Weight Base (HBM1)
    uint16_t width;          // [207:192]
    uint16_t height;         // [223:208]
    uint16_t in_channels;    // [239:224]
    uint16_t out_channels;   // [255:240]
    uint8_t  opcode;         // [263:256]
    uint8_t  quant_shift;    // [271:264]
    uint8_t  bank_sel;       // [279:272] select HBM input/output bank for this layer
    uint8_t  relu_en;        // [287:280] ReLU enable flag
    uint8_t  stride;         // [295:288] stride (1 or 2)
    uint8_t  log2_mem_tile_height; // [303:296]
    uint8_t  padding[26];    // Pad to 64 bytes total
};

/**
 * Packs feature map data from standard layout to HBM-optimized tiled layout.
 * 
 * Input Assumption: 
 *   - "Width innermost, then height, then channels" implies a Planar layout [C][H][W].
 *   - If your input is Interleaved [H][W][C], swap the input index calculation (commented below).
 * 
 * @tparam T        Data type (e.g., int8_t, float)
 * @param input     Pointer to source data
 * @param output    Pointer to destination buffer (must be pre-allocated)
 * @param H         Total Height of feature map
 * @param W         Total Width of feature map
 * @param C         Total Channels
 * @param H_tile    Height of the tile (e.g., 32)
 * @param IC_PAR    Parallel Input Channels (e.g., 16)
 */
template <typename T>
void pack_feature_map(const T* input, T* output, 
                      int H, int W, int C, 
                      int H_tile, int IC_PAR) {
    
    // 1. Validation
    if (H % H_tile != 0) {
        std::cerr << "Error: Height (" << H << ") must be divisible by H_tile (" << H_tile << ")" << std::endl;
        return;
    }
    if (C % IC_PAR != 0) {
        std::cerr << "Error: Channels (" << C << ") must be divisible by IC_PAR (" << IC_PAR << ")" << std::endl;
        return;
    }

    int num_h_tiles = H / H_tile;
    int num_c_slots = C / IC_PAR;
    
    int out_idx = 0;

    // 2. Iterate according to the target HBM hierarchy
    
    // Loop 1: Tile Height Dimension (Which vertical tile are we in?)
    for (int ht = 0; ht < num_h_tiles; ++ht) {
        
        // Loop 2: Channel Slot Dimension (Which block of IC_PAR channels?)
        for (int cs = 0; cs < num_c_slots; ++cs) {
            
            // Loop 3: Height Dimension (Inside the current tile)
            for (int h_loc = 0; h_loc < H_tile; ++h_loc) {
                
                // Calculate absolute height in the original image
                int h_abs = (ht * H_tile) + h_loc;

                // Loop 4: Width Dimension (Iterate all W)
                for (int w = 0; w < W; ++w) {
                    
                    // Loop 5: Innermost Channel Dimension (Iterate IC_PAR)
                    for (int c_loc = 0; c_loc < IC_PAR; ++c_loc) {
                        
                        // Calculate absolute channel in the original image
                        int c_abs = (cs * IC_PAR) + c_loc;

                        // --- INPUT INDEX CALCULATION ---
                        
                        // OPTION A: If Input is [C][H][W] (Planar / Width Innermost)
                        // This matches "Width innermost, then height, then channels"
                        int in_idx = (c_abs * H * W) + (h_abs * W) + w;

                        // OPTION B: If Input is [H][W][C] (Interleaved / Channels Innermost)
                        // Use this if your input comes directly from OpenCV/TensorFlow standard format
                        // int in_idx = (h_abs * W * C) + (w * C) + c_abs;

                        // Perform the copy
                        output[out_idx++] = input[in_idx];
                    }
                }
            }
        }
    }
}

/**
 * Packs weights from standard format to Tiled Hardware format.
 * 
 * Source Layout: [OC][IC][3][3] (Row-major linear memory)
 *   - Iterates OC, then IC, then 3x3 spatial.
 * 
 * Destination Layout: [IC_Tiles][OC_Tiles][3][3][OC_Par][IC_Par]
 *   - Optimized for Input Stationary hardware dataflow.
 * 
 * @tparam T        Data type (e.g., int8_t, float)
 * @param src       Pointer to source weight buffer
 * @param dst       Pointer to destination buffer
 * @param OC        Total Output Channels
 * @param IC        Total Input Channels
 * @param OC_PAR    Output Parallelism (Tile size, e.g., 16)
 * @param IC_PAR    Input Parallelism (Tile size, e.g., 16)
 */
template <typename T>
void pack_weights(const T* src, T* dst, 
                  int OC, int IC, 
                  int OC_PAR, int IC_PAR) {
    
    // 1. Validation
    if (OC % OC_PAR != 0 || IC % IC_PAR != 0) {
        std::cerr << "Error: Dimensions must be divisible by parallelism factors." << std::endl;
        return;
    }

    int num_ic_tiles = IC / IC_PAR;
    int num_oc_tiles = OC / OC_PAR;
    int K = 3; // Kernel size is fixed at 3x3

    int dst_idx = 0;

    // 2. Iterate in Destination Order (Hardware Order)
    // Loop 2: Output Channel Tiles
    for (int t_oc = 0; t_oc < num_oc_tiles; ++t_oc) {
    
        // Loop 1: Input Channel Tiles (Outer - Input Stationary)
        for (int t_ic = 0; t_ic < num_ic_tiles; ++t_ic) {
            
            // Loop 3 & 4: Kernel Spatial Dimensions (3x3)
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    
                    // Loop 5: Output Parallelism (Hardware Array Rows)
                    for (int p_oc = 0; p_oc < OC_PAR; ++p_oc) {
                        
                        // Loop 6: Input Parallelism (Hardware Array Cols - Contiguous)
                        for (int p_ic = 0; p_ic < IC_PAR; ++p_ic) {
                            
                            // --- Calculate Absolute Indices ---
                            int global_oc = (t_oc * OC_PAR) + p_oc;
                            int global_ic = (t_ic * IC_PAR) + p_ic;

                            // --- Calculate Source Index ---
                            // Source Layout: OC -> IC -> 3x3 (9 elements)
                            // Offset = (OC_idx * Stride_OC) + (IC_idx * Stride_IC) + Spatial_Offset
                            int stride_ic = 9;          // 3x3
                            int stride_oc = IC * 9;     // IC * 3x3
                            
                            int spatial_offset = (ky * 3) + kx;
                            
                            int src_idx = (global_oc * stride_oc) + 
                                          (global_ic * stride_ic) + 
                                          spatial_offset;

                            // --- Copy ---
                            dst[dst_idx++] = src[src_idx];
                        }
                    }
                }
            }
        }
    }
}

// Helper: Perform software convolution (Reference Model)
// Input/Output Layout: Planar [Channels][Height][Width]
// Weight Layout: [OutCh][InCh][3][3]
std::vector<int8_t> compute_gold_conv_layer(
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& weights,
    int H, int W, int IC, int OC,
    int quant_shift,
    int relu_en,
    int stride
) {

    int H_out = H / stride;
    int W_out = W / stride;

    std::vector<int8_t> output(OC * H_out * W_out);
    // Loop over Output Channels
    for (int oc = 0; oc < OC; ++oc) {
        // Loop over Spatial Dimensions
        for (int y = 0; y < H_out; ++y) {
            for (int x = 0; x < W_out; ++x) {
                
		int y_in_center = y * stride;
                int x_in_center = x * stride;
                int32_t accumulator = 0;

                // Convolution (3x3)
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        int in_y = y_in_center + ky - 1; // Padding handling (3x3 specific)
                        int in_x = x_in_center + kx - 1;

                        bool is_pad = (in_y < 0 || in_y >= H || in_x < 0 || in_x >= W);

                        for (int ic = 0; ic < IC; ++ic) {
                            int16_t pixel_val = 0;
                            if (!is_pad) {
                                // Input Index: [IC][H][W]
                                int in_idx = (ic * H * W) + (in_y * W) + in_x;
                                pixel_val = input[in_idx];
                            }

                            // Weight Index: [OC][IC][3][3]
                            int w_idx = (oc * IC * 9) + (ic * 9) + (ky * 3 + kx);
                            int16_t weight_val = weights[w_idx];

                            accumulator += pixel_val * weight_val;
                        }
                    }
                }

		if (relu_en == 1) {
		    if (accumulator < 0) accumulator = 0;
		}
                // Quantization (Scale & Clamp)
                int32_t shifted = accumulator >> quant_shift;
                int8_t result;
                if (shifted > 127) result = 127;
                else if (shifted < -128) result = -128;
                else result = (int8_t)shifted;

                // Output Index: [OC][H][W]
                int out_idx = (oc * H_out * W_out) + (y * W_out) + x;
                output[out_idx] = result;
            }
        }
    }
    return output;
}

// Helper to convert float to our fixed-point representation
int16_t float_to_fixed(double val) {
    double scaled = val * (1 << FP_FRAC_BITS);
    // Basic saturation (optional, depends on how you want to handle overflow)
    double max_val = (double)((1 << (FP_WIDTH - 1)) - 1); // Max positive signed
    double min_val = (double)(-(1 << (FP_WIDTH - 1)));   // Min negative signed
    if (scaled > max_val) scaled = max_val;
    if (scaled < min_val) scaled = min_val;
    return static_cast<int16_t>(round(scaled)); // Rounding is often better
}

// Helper to convert fixed-point back to float for display
double fixed_to_float(int16_t val) {
    return static_cast<double>(val) / (1 << FP_FRAC_BITS);
}

typedef uint64_t output_buffer_t; // Or uint16_t if unsigned

int main(int argc, char **argv) {
  if (argc < 1) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }

  std::string xclbin_path = argv[1];

  cl_int err;
  cl::Context context;
  cl::CommandQueue q;

  cl::Kernel vlaAccelKernel;

  // --- Boilerplate OpenCL Setup ---
  std::cout << "INFO: Loading XCLBIN: " << xclbin_path << std::endl;
  auto devices = xcl::get_xil_devices();
  auto device = devices[0]; // Assuming one device

  // Create Context and Command Queue
  OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
  cl::Program::Binaries bins = xcl::import_binary_file(xclbin_path);
  cl::Program program(context, {device}, bins);
  std::cout << "INFO: Program created" << std::endl;

  OCL_CHECK(err, q = cl::CommandQueue(
				     context, device,
				     CL_QUEUE_PROFILING_ENABLE |
				     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, // Streams benefit from OOO
				     &err));

  OCL_CHECK(err, vlaAccelKernel = cl::Kernel(program, "vla_accel_top:{vla_accel_top_0}", &err));
  std::cout << "INFO: Kernel created" << std::endl;

  size_t input_buffer_size_bytes =  FM_WIDTH * FM_HEIGHT * FM_CHANNELS;
  size_t output_buffer_size_bytes = FM_WIDTH * FM_HEIGHT * OUT_CHANNELS;
  size_t fmap_size = FM_WIDTH * FM_HEIGHT * FM_CHANNELS;

  std::vector<uint8_t> raw_input(fmap_size);
  for (size_t i = 0; i < FM_WIDTH*FM_HEIGHT*FM_CHANNELS; i++) {
      raw_input[i] = uint8_t(i);
  }
  size_t heap_buffer_size = INSTR_SECTION_SIZE + input_buffer_size_bytes;
  std::vector<uint8_t, aligned_allocator<uint8_t>> host_heap(heap_buffer_size);
  pack_feature_map<uint8_t>(raw_input.data(), host_heap.data() + INSTR_SECTION_SIZE, 
			    FM_HEIGHT, FM_WIDTH, FM_CHANNELS, TILE_HEIGHT, IC_PAR);

  size_t weight_buffer_size_bytes = FM_CHANNELS * OUT_CHANNELS * 3 * 3;
  uint8_t weight_arr[FM_CHANNELS * OUT_CHANNELS * 3 * 3];
  uint8_t packed_weight_arr[FM_CHANNELS * OUT_CHANNELS * 3 * 3];
  for (size_t i = 0; i < FM_CHANNELS * OUT_CHANNELS * 3 * 3; i++) {
      weight_arr[i] = uint8_t(i); // Example weight initialization
  }
  pack_weights<uint8_t>(weight_arr, packed_weight_arr, OUT_CHANNELS, FM_CHANNELS, OC_PAR, IC_PAR);

  std::vector<uint8_t, aligned_allocator<uint8_t>> host_weight_buffer(packed_weight_arr, packed_weight_arr + sizeof(packed_weight_arr) / sizeof(packed_weight_arr[0]));

  NISA_Instruction instr_list[3];
  // Instruction 0: Conv Layer 0
  instr_list[0].input_offset  = INSTR_SECTION_SIZE; // Data starts after 64KB
  instr_list[0].output_offset = 0;                  // Output starts at 0 of HBM[2]
  instr_list[0].weight_offset = 0;                  // Weights start at 0 of HBM[1]
  instr_list[0].width         = FM_WIDTH;
  instr_list[0].height        = FM_HEIGHT;
  instr_list[0].in_channels   = FM_CHANNELS;
  instr_list[0].out_channels  = OUT_CHANNELS;
  instr_list[0].opcode        = OP_CONV;
  instr_list[0].bank_sel      = 0x04;  // 0b0100: HBM1 output, HBM0 input
  instr_list[0].relu_en       = 1;
  instr_list[0].stride        = STRIDE;
  instr_list[0].log2_mem_tile_height = 2; // log2(4) = 2
  instr_list[0].quant_shift   = 10;
  // Instruction 1: Conv Layer 1
  instr_list[1].input_offset  = 0;
  instr_list[1].output_offset = 0;
  instr_list[1].weight_offset = 0;
  instr_list[1].width         = FM_WIDTH / STRIDE;
  instr_list[1].height        = FM_HEIGHT / STRIDE;
  instr_list[1].in_channels   = OUT_CHANNELS;
  instr_list[1].out_channels  = OUT_CHANNELS;
  instr_list[1].opcode        = OP_CONV;
  instr_list[1].bank_sel      = 0x09;  // 0b1001: HBM2 output, HBM1 input
  instr_list[1].relu_en       = 1;
  instr_list[1].stride        = STRIDE;
  instr_list[1].log2_mem_tile_height = 1; // log2(2) = 1
  instr_list[1].quant_shift   = 10;
  /*
  // Instruction 2: Conv Layer 2
  instr_list[2].input_offset  = 0;
  instr_list[2].output_offset = 0;
  instr_list[2].weight_offset = 0;
  instr_list[2].width         = FM_WIDTH;
  instr_list[2].height        = FM_HEIGHT;
  instr_list[2].in_channels   = FM_CHANNELS;
  instr_list[2].out_channels  = OUT_CHANNELS;
  instr_list[2].opcode        = OP_CONV;
  instr_list[2].bank_sel      = 0x06;  // 0b0110: HBM1 output, HBM2 input
  instr_list[2].relu_en       = 1;
  instr_list[2].quant_shift   = 10;
  */

  // HALT Instruction
  instr_list[2].opcode        = OP_HALT;

  std::cout << "Size of instr_list: %zu bytes" << sizeof(instr_list) << std::endl;
  memcpy(host_heap.data(), instr_list, sizeof(instr_list));

  OCL_CHECK(err, cl::Buffer dev_heap(
	      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      heap_buffer_size, host_heap.data(), &err));

  OCL_CHECK(err, cl::Buffer dev_buf_a(
	      context, CL_MEM_READ_WRITE,
	      output_buffer_size_bytes, NULL, &err));

  OCL_CHECK(err, cl::Buffer dev_buf_b(
	      context, CL_MEM_READ_WRITE,
	      output_buffer_size_bytes, NULL, &err));

  OCL_CHECK(err, cl::Buffer dev_buf_weights(
	      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      weight_buffer_size_bytes, host_weight_buffer.data(), &err));

  // dump_to_hex_file(packed_weight_arr, OC_PAR * IC_PAR, 3*3*IC_PAR*OC_PAR, "golden_weights.hex");
  // dump_to_hex_file(packed_arr, IC_PAR * PP_PAR, 8 * PP_PAR * IC_PAR, "golden_features.hex");
 
  // --- Set Kernel Arguments ---
  std::cout << "INFO: Setting kernel arguments..." << std::endl;
  OCL_CHECK(err, err = vlaAccelKernel.setArg(0, dev_heap));
  OCL_CHECK(err, err = vlaAccelKernel.setArg(1, dev_buf_weights));
  OCL_CHECK(err, err = vlaAccelKernel.setArg(2, dev_buf_a));
  OCL_CHECK(err, err = vlaAccelKernel.setArg(3, dev_buf_b));
  // --- Execute Kernels ---
  // Enqueue tasks. Streams handle synchronization.
  // Using Out-of-Order queue allows runtime to potentially overlap execution.
  std::cout << "INFO: Enqueuing tasks..." << std::endl;
  OCL_CHECK(err, err = q.enqueueTask(vlaAccelKernel));
  OCL_CHECK(err, err = q.finish());
  std::cout << "INFO: Kernels finished execution." << std::endl;

  std::cout << "INFO: Reading output data from device..." << std::endl;
  
  // Calculate total output size
  // Note: Hardware writes TILE_HEIGHT rows per tile. 
  // If FM_HEIGHT is a multiple of TILE_HEIGHT, this matches the full image size.
  size_t output_size_bytes = (FM_HEIGHT / (STRIDE * NUM_LAYERS)) * (FM_WIDTH / (STRIDE * NUM_LAYERS)) * OUT_CHANNELS * sizeof(int8_t);
  
  // Allocate Host Buffer (Aligned for OpenCL)
  std::vector<int8_t, aligned_allocator<int8_t>> host_output_buffer(output_size_bytes);

  cl::Buffer* final_buffer_ptr;
    
  if (NUM_LAYERS % 2 != 0) {
      std::cout << "INFO: Odd layers. Reading output from Buffer A (HBM[1])." << std::endl;
      final_buffer_ptr = &dev_buf_a;
  } else {
      std::cout << "INFO: Even layers. Reading output from Buffer B (HBM[2])." << std::endl;
      final_buffer_ptr = &dev_buf_b;
  }

  // Enqueue Read Command
  OCL_CHECK(err, err = q.enqueueReadBuffer(
      *final_buffer_ptr,            // Buffer object
      CL_TRUE,              // Blocking read
      0, // INSTR_SECTION_SIZE,   // Offset
      output_size_bytes,    // Size
      host_output_buffer.data(),
      nullptr, nullptr)
  );

  // =========================================================================
  // 6. VERIFY AGAINST GROUND TRUTH
  // =========================================================================
  std::cout << "INFO: Verifying results..." << std::endl;

  // True verification starts here:
  // 1. Prepare Data Containers
  // Convert raw_input (uint8) to signed int8 for calculation
  std::vector<int8_t> input_L1(raw_input.begin(), raw_input.end());
    
    // Split weights for Layer 1 and Layer 2
    // Assuming both layers have same dims: 32x32x3x3
  size_t layer_weight_size = OUT_CHANNELS * FM_CHANNELS * 9;
    
  std::vector<int8_t> weights_L1(layer_weight_size);
  std::vector<int8_t> weights_L2(layer_weight_size);
    
    // Copy from your main raw_weights buffer
    // (Assuming you packed L1 then L2 sequentially in raw_weights)
  for(size_t i=0; i<layer_weight_size; ++i) {
      weights_L1[i] = (int8_t)weight_arr[i];
      weights_L2[i] = (int8_t)weight_arr[i];
  }

    // 2. Compute Golden Output Sequentially
    std::cout << "INFO: Computing Golden Model for Layer 1..." << std::endl;
    std::vector<int8_t> golden_L1 = compute_gold_conv_layer(
        input_L1, weights_L1, 
        FM_HEIGHT, FM_WIDTH, FM_CHANNELS, OUT_CHANNELS, 
        10, // quant_shift
	1, // relu_en
	STRIDE // stride
    );

    std::cout << "INFO: Computing Golden Model for Layer 2..." << std::endl;
    int l2_in_height = FM_HEIGHT / STRIDE;
    int l2_in_width  = FM_WIDTH / STRIDE;
    std::vector<int8_t> golden_L2 = compute_gold_conv_layer(
        golden_L1, weights_L2, // Input is Output of L1
        l2_in_height, l2_in_width, OUT_CHANNELS, OUT_CHANNELS, 
        10, // quant_shift
	1, // relu_en
	STRIDE
    );

    /*
    std::vector<int8_t> golden_L3 = compute_gold_conv_layer(
	golden_L2, weights_L2, // Input is Output of L2
	FM_HEIGHT, FM_WIDTH, OUT_CHANNELS, OUT_CHANNELS, 
	10, // quant_shift
	1
    );
    */

    // 3. Verify Final Result
    // The hardware output is Tiled. The golden_L3 is Planar.
    // We iterate in Hardware Order and look up the Planar index.
    //
    //
    std::cout << "DEBUG: Verifying Layer 1 Output..." << std::endl;
  
  // 1. Read the Intermediate Buffer
  // If NUM_LAYERS=2, Layer 1 output is in the "other" buffer.
  // If Final is Buf_B, then Intermediate is Buf_A.
  cl::Buffer* intermediate_buffer = (NUM_LAYERS % 2 != 0) ? &dev_buf_b : &dev_buf_a;
  
  size_t l1_size = (FM_WIDTH/2) * (FM_HEIGHT/2) * OUT_CHANNELS;
  std::vector<int8_t> host_l1_debug(l1_size);
  
  q.enqueueReadBuffer(*intermediate_buffer, CL_TRUE, 0, l1_size, host_l1_debug.data());

  // 2. Compare against golden_L1
  int l1_errors = 0;
  // Note: golden_L1 is Planar. Hardware output is Tiled.
  // We need to iterate in Hardware Order for the L1 dimensions (32x32).
  
  int l1_width = FM_WIDTH / 2;
  int l1_height = FM_HEIGHT / 2;
  int l1_tile_h = TILE_HEIGHT / 2; // Stride 2 output tile height
  int l1_pp_par = PP_PAR / 2;      // Stride 2 output packing
  
  int l1_linear_idx = 0;

  for (int ht = 0; ht < FM_HEIGHT / TILE_HEIGHT; ++ht) { // Input Tiles
      for (int ot = 0; ot < OUT_CHANNELS / OC_PAR; ++ot) {
          for (int r = 0; r < l1_tile_h; ++r) {
              for (int w_strip = 0; w_strip < FM_WIDTH / PP_PAR; ++w_strip) {
                  for (int p = 0; p < l1_pp_par; ++p) {
                      for (int o = 0; o < OC_PAR; ++o) {
                          
                          // Coords in 32x32 image
                          int h = (ht * l1_tile_h) + r;
                          int w = (w_strip * l1_pp_par) + p;
                          int oc = (ot * OC_PAR) + o;
                          
                          int gold_idx = (oc * l1_height * l1_width) + (h * l1_width) + w;
                          int8_t exp = golden_L1[gold_idx];
                          int8_t got = host_l1_debug[l1_linear_idx++];
                          
                          if (got != exp) {
                              if (l1_errors < 5) 
                                  std::cout << "L1 Error [H=" << h << " W=" << w << "]: Exp " << (int)exp << " Got " << (int)got << std::endl;
                              l1_errors++;
                          }
                      }
                  }
              }
          }
      }
  }
  
  if (l1_errors > 0) {
      std::cout << "STOP: Layer 1 Output is incorrect! Fix this first." << std::endl;
      return EXIT_FAILURE;
  }
  std::cout << "DEBUG: Layer 1 Output is CORRECT." << std::endl;

  // =========================================================================
  // 6. VERIFY RESULTS (With Stride Support)
  // =========================================================================
  std::cout << "INFO: Verifying results..." << std::endl;
  
  // Configuration (Must match Instruction)
  int current_stride = STRIDE;
    // Dimensions of the Final Output (16x16)
  int final_out_height = l2_in_height / current_stride;
  int final_out_width  = l2_in_width / current_stride;
  
  // Hardware Processing Parameters for Layer 2
  // The hardware processes the *Input of Layer 2* (32x32)
  int hw_input_height = l2_in_height;
  int hw_input_width  = l2_in_width;
  
  // Loop Bounds based on Layer 2 Input
  int num_height_tiles = hw_input_height / TILE_HEIGHT;
  int num_width_strips = hw_input_width / PP_PAR;
  
  // Output Packing for Layer 2
  int out_tile_height  = TILE_HEIGHT / current_stride;
  int effective_pp_par = PP_PAR / current_stride; 

  int errors = 0;
  int linear_idx = 0; 

  // 1. Height Tiles (Iterate over Layer 2 Input Tiles)
  for (int ht = 0; ht < num_height_tiles; ++ht) {
      
      // 2. Output Channel Tiles
      for (int ot = 0; ot < OUT_CHANNELS / OC_PAR; ++ot) {
          
          // 3. Output Rows inside the Tile
          for (int r = 0; r < out_tile_height; ++r) {
              
              // 4. Width Strips (Iterate over Layer 2 Input Strips)
              for (int w_strip = 0; w_strip < num_width_strips; ++w_strip) {
                  
                  // 5. Output Pixels inside the Packed Chunk
                  for (int p = 0; p < effective_pp_par; ++p) {
                      
                      // 6. Output Channels
                      for (int o = 0; o < OC_PAR; ++o) {
                          
                          // --- Calculate Global Output Coordinates ---
                          // These are coordinates in the 16x16 output image
                          int global_h_out = (ht * out_tile_height) + r;
                          int global_w_out = (w_strip * effective_pp_par) + p;
                          int global_oc    = (ot * OC_PAR) + o;

                          // --- Look up Golden Value ---
                          int gold_idx = (global_oc * final_out_height * final_out_width) + 
                                         (global_h_out * final_out_width) + global_w_out;
                          
                          int8_t expected = golden_L2[gold_idx];
                          int8_t hw_val = host_output_buffer[linear_idx];
                          
                          if (hw_val != expected) {
                              if (errors < 20) {
                                  std::cout << "Error at [H=" << global_h_out << " W=" << global_w_out 
                                            << " OC=" << global_oc << "] "
                                            << "Exp: " << (int)expected << " Got: " << (int)hw_val 
                                            << " (HBM Index " << linear_idx << ")" << std::endl;
                              }
                              errors++;
                          }
                          
                          linear_idx++;
                      }
                  }
              }
          }
      }
  }

  if (errors == 0) {
      std::cout << "\n*** TEST PASSED! Hardware output matches Ground Truth. ***" << std::endl;
  } else {
      std::cout << "\n*** TEST FAILED! Total Errors: " << errors << " ***" << std::endl;
      // return EXIT_FAILURE;
  }
    
  return EXIT_SUCCESS;
}
