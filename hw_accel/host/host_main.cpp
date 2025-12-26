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

constexpr uint16_t FM_WIDTH = 64;    // Feature map width
constexpr uint16_t FM_HEIGHT = 64;   // Feature map height
 
constexpr uint16_t FM_CHANNELS = 64;
constexpr uint16_t OUT_CHANNELS = 64;

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
    uint8_t  padding[21];    // Pad to 64 bytes total
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
    int quant_shift
) {
    std::vector<int8_t> output(OC * H * W);

    // Loop over Output Channels
    for (int oc = 0; oc < OC; ++oc) {
        // Loop over Spatial Dimensions
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                
                int32_t accumulator = 0;

                // Convolution (3x3)
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        int in_y = y + ky - 1; // Padding handling (3x3 specific)
                        int in_x = x + kx - 1;

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

                // Quantization (Scale & Clamp)
                int32_t shifted = accumulator >> quant_shift;
                int8_t result;
                if (shifted > 127) result = 127;
                else if (shifted < -128) result = -128;
                else result = (int8_t)shifted;

                // Output Index: [OC][H][W]
                int out_idx = (oc * H * W) + (y * W) + x;
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

  NISA_Instruction instr_list[4];
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
  instr_list[0].quant_shift   = 10;
  // Instruction 1: Conv Layer 1
  instr_list[1].input_offset  = 0;
  instr_list[1].output_offset = 0;
  instr_list[1].weight_offset = 0;
  instr_list[1].width         = FM_WIDTH;
  instr_list[1].height        = FM_HEIGHT;
  instr_list[1].in_channels   = FM_CHANNELS;
  instr_list[1].out_channels  = OUT_CHANNELS;
  instr_list[1].opcode        = OP_CONV;
  instr_list[1].bank_sel      = 0x09;  // 0b1001: HBM2 output, HBM1 input
  instr_list[1].quant_shift   = 10;
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
  instr_list[2].quant_shift   = 10;

  // HALT Instruction
  instr_list[3].opcode        = OP_HALT;

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
  size_t output_size_bytes = FM_HEIGHT * FM_WIDTH * OUT_CHANNELS * sizeof(int8_t);
  
  // Allocate Host Buffer (Aligned for OpenCL)
  std::vector<int8_t, aligned_allocator<int8_t>> host_output_buffer(output_size_bytes);

  // Enqueue Read Command
  OCL_CHECK(err, err = q.enqueueReadBuffer(
      dev_buf_a,            // Buffer object
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

  /*
  int errors = 0;
  int max_errors_to_print = 20;
  int linear_idx = 0; // Tracks position in the raw HBM buffer

  // Quantization Shift (Must match Hardware)
  int quant_shift = 10; 

  // --- ITERATE IN HARDWARE WRITE ORDER ---
  // 1. Height Tiles
  for (int ht = 0; ht < FM_HEIGHT / TILE_HEIGHT; ++ht) {
      
      // 2. Output Channel Tiles
      for (int ot = 0; ot < OUT_CHANNELS / OC_PAR; ++ot) {
          
          // 3. Rows inside the Tile
          for (int r = 0; r < TILE_HEIGHT; ++r) {
              
              // 4. Width Strips (PP_PAR pixels per strip)
              for (int w_strip = 0; w_strip < FM_WIDTH / PP_PAR; ++w_strip) {
                  
                  // 5. Pixels inside the Strip
                  for (int p = 0; p < PP_PAR; ++p) {
                      
                      // 6. Output Channels inside the Block
                      for (int o = 0; o < OC_PAR; ++o) {
                          
                          // --- A. Calculate Global Coordinates ---
                          int global_h = (ht * TILE_HEIGHT) + r;
                          int global_w = (w_strip * PP_PAR) + p;
                          int global_oc = (ot * OC_PAR) + o;

                          // --- B. Calculate Ground Truth (Software Conv) ---
                          int32_t accumulator = 0;

                          // Loop over Kernel (3x3)
                          for (int ky = 0; ky < 3; ++ky) {
                              for (int kx = 0; kx < 3; ++kx) {
                                  int in_y = global_h + ky - 1; // Handle Padding
                                  int in_x = global_w + kx - 1;
                                  
                                  bool is_padding = (in_y < 0) || (in_y >= FM_HEIGHT) || 
                                                    (in_x < 0) || (in_x >= FM_WIDTH);

                                  // Loop over ALL Input Channels
                                  for (int ic = 0; ic < FM_CHANNELS; ++ic) {
                                      int16_t pixel_val = 0;
                                      if (!is_padding) {
                                          // Input is Planar [C][H][W] (Matches your pack function)
                                          int in_idx = (ic * FM_HEIGHT * FM_WIDTH) + 
                                                       (in_y * FM_WIDTH) + in_x;
                                          pixel_val = (int16_t)(int8_t)raw_input[in_idx];
                                      }

                                      // Weights are [OC][IC][3][3] (Matches your pack function)
                                      int w_idx = (global_oc * FM_CHANNELS * 9) + 
                                                  (ic * 9) + 
                                                  (ky * 3 + kx);
                                      int16_t weight_val = (int16_t)(int8_t)weight_arr[w_idx];

                                      accumulator += (pixel_val * weight_val);
                                  }
                              }
                          }

                          // --- C. Quantize Ground Truth ---
                          int32_t shifted = accumulator >> quant_shift;
                          int8_t expected_val;
                          if (shifted > 127) expected_val = 127;
                          else if (shifted < -128) expected_val = -128;
                          else expected_val = (int8_t)shifted;

                          // --- D. Compare with Hardware Output ---
                          // The linear_idx walks through the buffer in the exact order HW wrote it
                          int8_t hw_val = host_output_buffer[linear_idx];
                          
                          if (hw_val != expected_val) {
                              if (errors < max_errors_to_print) {
                                  std::cout << "ERROR at [H=" << global_h << ", W=" << global_w 
                                            << ", OC=" << global_oc << "]: "
                                            << "Expected " << (int)expected_val 
                                            << ", Got " << (int)hw_val 
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
*/

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
        10 // quant_shift
    );

    std::cout << "INFO: Computing Golden Model for Layer 2..." << std::endl;
    std::vector<int8_t> golden_L2 = compute_gold_conv_layer(
        golden_L1, weights_L2, // Input is Output of L1
        FM_HEIGHT, FM_WIDTH, OUT_CHANNELS, OUT_CHANNELS, 
        10 // quant_shift
    );

    std::vector<int8_t> golden_L3 = compute_gold_conv_layer(
	golden_L2, weights_L2, // Input is Output of L2
	FM_HEIGHT, FM_WIDTH, OUT_CHANNELS, OUT_CHANNELS, 
	10 // quant_shift
    );

    // 3. Verify Final Result
    // The hardware output is Tiled. The golden_L3 is Planar.
    // We iterate in Hardware Order and look up the Planar index.
    
    std::cout << "INFO: Verifying Layer 3 Results..." << std::endl;
    int errors = 0;
    int linear_idx = 0;

    for (int ht = 0; ht < FM_HEIGHT / TILE_HEIGHT; ++ht) {
        for (int ot = 0; ot < OUT_CHANNELS / OC_PAR; ++ot) {
            for (int r = 0; r < TILE_HEIGHT; ++r) {
                for (int w_strip = 0; w_strip < FM_WIDTH / PP_PAR; ++w_strip) {
                    for (int p = 0; p < PP_PAR; ++p) {
                        for (int o = 0; o < OC_PAR; ++o) {
                            
                            // Calculate Global Coordinates
                            int global_h = (ht * TILE_HEIGHT) + r;
                            int global_w = (w_strip * PP_PAR) + p;
                            int global_oc = (ot * OC_PAR) + o;

                            // Look up Golden Value (Planar Indexing)
                            int gold_idx = (global_oc * FM_HEIGHT * FM_WIDTH) + 
                                           (global_h * FM_WIDTH) + global_w;
                            
                            int8_t expected = golden_L3[gold_idx];
                            int8_t hw_val = host_output_buffer[linear_idx];

                            if (hw_val != expected) {
                                if (errors < 10) {
                                    std::cout << "Error at [H=" << global_h << " W=" << global_w 
                                              << " OC=" << global_oc << "] "
                                              << "Exp: " << (int)expected << " Got: " << (int)hw_val 
                                              << std::endl;
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

  return EXIT_SUCCESS;
}
