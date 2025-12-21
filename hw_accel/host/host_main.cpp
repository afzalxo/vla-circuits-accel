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

constexpr uint16_t FM_WIDTH = 8;    // Feature map width
constexpr uint16_t FM_HEIGHT = 8;   // Feature map height
constexpr uint16_t FM_CHANNELS = 32;
constexpr uint16_t OUT_CHANNELS = 32;

constexpr uint16_t IC_PAR = 16;
constexpr uint16_t OC_PAR = 16;
constexpr uint16_t PP_PAR = 4;

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
    
    // Loop 1: Input Channel Tiles (Outer - Input Stationary)
    for (int t_ic = 0; t_ic < num_ic_tiles; ++t_ic) {
        
        // Loop 2: Output Channel Tiles
        for (int t_oc = 0; t_oc < num_oc_tiles; ++t_oc) {
            
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

  uint8_t arr[FM_WIDTH*FM_HEIGHT*FM_CHANNELS]; 
  uint8_t packed_arr[FM_WIDTH * FM_HEIGHT * FM_CHANNELS];
  
  for (size_t i = 0; i < FM_WIDTH*FM_HEIGHT*FM_CHANNELS; i++) {
      arr[i] = uint8_t(i); // Example data initialization
  }

  pack_feature_map<uint8_t>(arr, packed_arr, FM_HEIGHT, FM_WIDTH, FM_CHANNELS, TILE_HEIGHT, IC_PAR);

  std::cout << "\nFirst 16 output bytes (Tile 0, Slot 0, Row 0, All Widths):" << std::endl;
  for (int i = 0; i < 16; ++i) {
      std::cout << (int)packed_arr[i] << " ";
  }
  std::cout << std::endl;

  std::vector<uint8_t, aligned_allocator<uint8_t>> host_input_buffer(packed_arr, packed_arr + sizeof(packed_arr) / sizeof(packed_arr[0]));

  OCL_CHECK(err, cl::Buffer device_input_buffer(
	      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      input_buffer_size_bytes, host_input_buffer.data(), &err));

  // --- Prepare Weight Buffer ---
  size_t weight_buffer_size_bytes = FM_CHANNELS * OUT_CHANNELS * 3 * 3;
  uint8_t weight_arr[FM_CHANNELS * OUT_CHANNELS * 3 * 3];
  uint8_t packed_weight_arr[FM_CHANNELS * OUT_CHANNELS * 3 * 3];
  for (size_t i = 0; i < FM_CHANNELS * OUT_CHANNELS * 3 * 3; i++) {
      weight_arr[i] = uint8_t(i); // Example weight initialization
  }
  pack_weights<uint8_t>(weight_arr, packed_weight_arr, OUT_CHANNELS, FM_CHANNELS, OC_PAR, IC_PAR);


  std::vector<uint8_t, aligned_allocator<uint8_t>> host_weight_buffer(packed_weight_arr, packed_weight_arr + sizeof(packed_weight_arr) / sizeof(packed_weight_arr[0]));

  dump_to_hex_file(packed_weight_arr, OC_PAR * IC_PAR, 3*3*IC_PAR*OC_PAR, "golden_weights.hex");
  dump_to_hex_file(packed_arr, IC_PAR * PP_PAR, 8 * PP_PAR * IC_PAR, "golden_features.hex");

  OCL_CHECK(err, cl::Buffer device_weight_buffer(
	      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	      weight_buffer_size_bytes, host_weight_buffer.data(), &err));
 
  // --- Set Kernel Arguments ---
  std::cout << "INFO: Setting kernel arguments..." << std::endl;
  OCL_CHECK(err, err = vlaAccelKernel.setArg(0, device_input_buffer));
  OCL_CHECK(err, err = vlaAccelKernel.setArg(1, device_weight_buffer));
  // --- Execute Kernels ---
  // Enqueue tasks. Streams handle synchronization.
  // Using Out-of-Order queue allows runtime to potentially overlap execution.
  std::cout << "INFO: Enqueuing tasks..." << std::endl;
  OCL_CHECK(err, err = q.enqueueTask(vlaAccelKernel));
  OCL_CHECK(err, err = q.finish());
  std::cout << "INFO: Kernels finished execution." << std::endl;

  std::cout << "INFO: Host execution finished successfully." << std::endl;

  // =========================================================================
  // GROUND TRUTH GENERATION (Full First Tile: Rows 0-3, IC 0-15)
  // =========================================================================
  std::cout << "\n--- Generating Ground Truth for Tile 0 (Rows 0-" << TILE_HEIGHT-1 
            << ", IC 0-" << IC_PAR-1 << ") ---" << std::endl;

  // Configuration
  int check_ic_start = 0;      // First IC Tile starts at 0
  int check_oc_start = 0;      // First OC Tile starts at 0
  int strips_per_row = FM_WIDTH / PP_PAR;
  int total_strips   = TILE_HEIGHT * strips_per_row;

  // Iterate through the strips in the order hardware produces them (Row-Major)
  for (int strip_idx = 0; strip_idx < total_strips; ++strip_idx) {
      
      // Map linear strip index to 2D coordinates
      int row_idx = strip_idx / strips_per_row;
      int col_strip_idx = strip_idx % strips_per_row;
      int col_start = col_strip_idx * PP_PAR;

      // Storage for this strip's results: [PP_PAR][OC_PAR]
      int32_t strip_results[PP_PAR][OC_PAR]; 

      // 1. Loop over Pixels in this Strip
      for (int p = 0; p < PP_PAR; ++p) {
          int out_x = col_start + p;
          int out_y = row_idx; // Relative to Tile 0 (Global Row 0..3)

          // 2. Loop over Output Channels
          for (int local_oc = 0; local_oc < OC_PAR; ++local_oc) {
              int global_oc = check_oc_start + local_oc;
              int32_t accumulator = 0;

              // 3. Convolution (3x3)
              for (int ky = 0; ky < 3; ++ky) {
                  for (int kx = 0; kx < 3; ++kx) {
                      
                      // Calculate Input Coordinates (Handle Padding)
                      // Note: out_y is relative to the tile. 
                      // Since this is Tile 0, out_y=0 is the top of the image.
                      int in_y = out_y + ky - 1; 
                      int in_x = out_x + kx - 1;

                      bool is_padding = (in_y < 0) || (in_y >= FM_HEIGHT) || 
                                        (in_x < 0) || (in_x >= FM_WIDTH);

                      // 4. Loop over Input Channels (Partial Sum: 0 to IC_PAR-1)
                      for (int local_ic = 0; local_ic < IC_PAR; ++local_ic) {
                          int global_ic = check_ic_start + local_ic;

                          int16_t pixel_val = 0;
                          if (!is_padding) {
                              // Planar Layout Indexing [C][H][W]
                              int in_idx = (global_ic * FM_HEIGHT * FM_WIDTH) + 
                                           (in_y * FM_WIDTH) + in_x;
                              // CRITICAL: Cast to signed 8-bit first
                              pixel_val = (int16_t)(int8_t)arr[in_idx];
                          }

                          // Weight Layout Indexing [OC][IC][3][3]
                          int w_idx = (global_oc * (FM_CHANNELS * 9)) + 
                                      (global_ic * 9) + 
                                      (ky * 3 + kx);
                          // CRITICAL: Cast to signed 8-bit first
                          int16_t weight_val = (int16_t)(int8_t)weight_arr[w_idx];

                          accumulator += (pixel_val * weight_val);
                      }
                  }
              }
              strip_results[p][local_oc] = accumulator;
          }
      }

      // --- PRINT HEX STRING FOR THIS STRIP ---
      // This matches one line in the hardware dump file
      std::cout << "Strip " << strip_idx << " (Row " << row_idx << ", Col " << col_start << "): 0x";
      
      // Iterate backwards (MSB -> LSB)
      // Hardware packing: P=0,OC=0 is LSB. P=3,OC=15 is MSB.
      for (int p = PP_PAR - 1; p >= 0; --p) {
          for (int o = OC_PAR - 1; o >= 0; --o) {
              uint32_t val = strip_results[p][o] & 0x0FFFFFFF; // Mask to 28 bits
              std::cout << std::hex << std::setw(7) << std::setfill('0') << val;
          }
      }
      std::cout << std::dec << std::endl;
  }

  return EXIT_SUCCESS;
}
