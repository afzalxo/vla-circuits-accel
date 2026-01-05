#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <string>
#include <fstream>
#include <iomanip>
#include <cstring>

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl2.hpp>
#include "xcl2.hpp"

constexpr uint16_t IC_PAR = 16;
constexpr uint16_t OC_PAR = 16;
constexpr uint16_t PP_PAR = 8;
constexpr uint16_t TILE_HEIGHT = 4;
constexpr size_t INSTR_SECTION_SIZE = 64 * 1024;

// opcodes
constexpr uint8_t OP_CONV = 1;
constexpr uint8_t OP_HALT = 255;

// User-defined Layer Configuration
struct LayerConfig {
    std::string name;
    int in_w, in_h, in_c;
    int out_c;
    int stride;
    int quant_shift;
    bool relu;
    std::string weight_file;
    std::string golden_file; // For verification
};

// N-ISA Instruction Format
struct alignas(64) NISA_Instruction {
    uint64_t input_offset;     // [63:0]
    uint64_t output_offset;    // [127:64]
    uint64_t weight_offset;    // [191:128]
    uint16_t width;            // [207:192]
    uint16_t height;           // [223:208]
    uint16_t in_channels;      // [239:224]
    uint16_t out_channels;     // [255:240]
    uint8_t  opcode;           // [263:256]
    uint8_t  quant_shift;      // [271:264]
    uint8_t  bank_sel;         // [279:272]
    uint8_t  relu_en;          // [287:280]
    uint8_t  stride;           // [295:288]
    uint8_t  log2_mem_tile_height;  // [303:296]
    uint8_t  is_sparse;             // [311:304]
    // Compiler inserts implicit padding 8-bit here     // [319:312]
    uint32_t ic_tile_mask;          // [351:320]
    uint32_t oc_tile_mask;          // [383:352]
    uint8_t  padding[16];           // [511:384] - Padding to make 64 bytes
};

// --- HELPER FUNCTIONS ---

void load_bin(const std::string& filename, std::vector<uint8_t>& buffer) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        exit(1);
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    buffer.resize(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
}

// Pack Feature Map (Host -> Hardware Layout)
template <typename T>
void pack_feature_map(const T* input, T* output, int H, int W, int C, int H_tile, int IC_PAR_HW) {
    int num_h_tiles = H / H_tile;
    int num_c_slots = (C + IC_PAR_HW - 1) / IC_PAR_HW;
    int out_idx = 0;

    for (int ht = 0; ht < num_h_tiles; ++ht) {
        for (int cs = 0; cs < num_c_slots; ++cs) {
            for (int h_loc = 0; h_loc < H_tile; ++h_loc) {
                int h_abs = (ht * H_tile) + h_loc;
                for (int w = 0; w < W; ++w) {
                    for (int c_loc = 0; c_loc < IC_PAR_HW; ++c_loc) {
                        int c_abs = (cs * IC_PAR_HW) + c_loc;
                        if (c_abs >= C) {
                            output[out_idx++] = 0; // Padding
                        } else {
                            // Planar Input [C][H][W]
                            int in_idx = (c_abs * H * W) + (h_abs * W) + w;
                            output[out_idx++] = input[in_idx];
                        }
                    }
                }
            }
        }
    }
}

// Pack Weights (Host -> Hardware Layout)
template <typename T>
void pack_weights(const T* src, T* dst, int OC, int IC, int OC_PAR_HW, int IC_PAR_HW) {
    int num_ic_tiles = (IC + IC_PAR_HW - 1) / IC_PAR_HW;
    int num_oc_tiles = (OC + OC_PAR_HW - 1) / OC_PAR_HW;
    int K = 3;
    int dst_idx = 0;

    for (int t_oc = 0; t_oc < num_oc_tiles; ++t_oc) {
        for (int t_ic = 0; t_ic < num_ic_tiles; ++t_ic) {
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    for (int p_oc = 0; p_oc < OC_PAR_HW; ++p_oc) {
                        for (int p_ic = 0; p_ic < IC_PAR_HW; ++p_ic) {
                            int global_oc = (t_oc * OC_PAR_HW) + p_oc;
                            int global_ic = (t_ic * IC_PAR_HW) + p_ic;
                            
                            if (global_oc >= OC || global_ic >= IC) {
                                dst[dst_idx++] = 0;
                            } else {
                                // Source: [OC][IC][3][3]
                                int src_idx = (global_oc * IC * 9) + (global_ic * 9) + (ky * 3 + kx);
                                dst[dst_idx++] = src[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

// Verification Function
int verify_output(const std::vector<int8_t>& hw_output, const std::string& golden_file, 
                  int H, int W, int C, int stride) {
    
    std::vector<uint8_t> file_golden_raw;
    load_bin(golden_file, file_golden_raw);
    
    // Cast to signed for comparison
    std::vector<int8_t> golden(file_golden_raw.begin(), file_golden_raw.end());

    int out_tile_h = TILE_HEIGHT / stride;
    int eff_pp_par = PP_PAR / stride;
    int out_h = H / stride;
    int out_w = W / stride;
    
    int max_diff = 0;
    int errors = 0;
    int linear_idx = 0;

    // Iterate in Hardware Tiled Order
    for (int ht = 0; ht < H / TILE_HEIGHT; ++ht) {
        for (int ot = 0; ot < C / OC_PAR; ++ot) {
            for (int r = 0; r < out_tile_h; ++r) {
                for (int w_strip = 0; w_strip < W / PP_PAR; ++w_strip) {
                    for (int p = 0; p < eff_pp_par; ++p) {
                        for (int o = 0; o < OC_PAR; ++o) {
                            
                            int global_h = (ht * out_tile_h) + r;
                            int global_w = (w_strip * eff_pp_par) + p;
                            int global_oc = (ot * OC_PAR) + o;

                            // Planar Index for Golden
                            int gold_idx = (global_oc * out_h * out_w) + (global_h * out_w) + global_w;
                            
                            int8_t exp = golden[gold_idx];
                            int8_t got = hw_output[linear_idx];
                            
                            int diff = std::abs((int)got - (int)exp);
                            if (diff > 1) {
                                if (errors < 10) {
                                    std::cout << "Mismatch [H=" << global_h << " W=" << global_w << " C=" << global_oc 
                                              << "] Exp: " << (int)exp << " Got: " << (int)got 
                                              << " (Diff: " << diff << ")" << std::endl;
                                }
                                errors++;
				if (diff > max_diff) {
				    max_diff = diff;
				}
                            }
                            linear_idx++;
                        }
                    }
                }
            }
        }
    }
    if (errors > 0) {
	std::cout << "Max Difference: " << max_diff << std::endl;
    }
    return errors;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    // Network topology here:
    std::vector<LayerConfig> layers;

    // Layer 0
    layers.push_back({
        "Layer 0",
        64, 64, 12, 32,   // InW, InH, InC, OutC
        2, 8, true,       // Stride, Shift, ReLU
        "fpga_data/weights_layer0.bin",
        "fpga_data/golden_output_layer0.bin"
    });

    // Layer 1
    layers.push_back({
        "Layer 1",
        32, 32, 32, 64,   // Input dims match L0 output
        2, 8, true,
        "fpga_data/weights_layer1.bin",
        "fpga_data/golden_output_layer1.bin"
    });

    layers.push_back({
	"Layer 2",
	16, 16, 64, 128,
	2, 7, true,
	"fpga_data/weights_layer2.bin",
	"fpga_data/golden_output_layer2.bin"
    });

    // A. Input Feature Map
    std::vector<uint8_t> raw_input;
    load_bin("fpga_data/input_layer0.bin", raw_input);
    
    // Calculate Heap Size (Instr + Input)
    // Input needs to be padded to IC_PAR
    size_t l0_input_padded_size = layers[0].in_w * layers[0].in_h * 
                                  ((layers[0].in_c + IC_PAR - 1) / IC_PAR) * IC_PAR;
    size_t heap_size = INSTR_SECTION_SIZE + l0_input_padded_size;
    std::vector<uint8_t, aligned_allocator<uint8_t>> host_heap(heap_size);

    // Pack Input into Heap
    pack_feature_map<uint8_t>(raw_input.data(), host_heap.data() + INSTR_SECTION_SIZE, 
                              layers[0].in_h, layers[0].in_w, layers[0].in_c, 
                              TILE_HEIGHT, IC_PAR);

    // B. Weights (Stack all layers)
    size_t total_weight_size = 0;
    for (const auto& layer : layers) {
        // Align to IC_PAR/OC_PAR
        int ic_pad = (layer.in_c + IC_PAR - 1) / IC_PAR * IC_PAR;
        int oc_pad = (layer.out_c + OC_PAR - 1) / OC_PAR * OC_PAR;
        total_weight_size += oc_pad * ic_pad * 9;
    }
    std::vector<uint8_t, aligned_allocator<uint8_t>> host_weights(total_weight_size);
    
    // Pack Weights
    size_t current_weight_offset = 0;
    std::vector<size_t> layer_weight_offsets; // Store for instruction gen

    for (const auto& layer : layers) {
        std::vector<uint8_t> w_raw;
        load_bin(layer.weight_file, w_raw);
        
        layer_weight_offsets.push_back(current_weight_offset);
        
        pack_weights<uint8_t>(w_raw.data(), host_weights.data() + current_weight_offset, 
                              layer.out_c, layer.in_c, OC_PAR, IC_PAR);
        
        int ic_pad = (layer.in_c + IC_PAR - 1) / IC_PAR * IC_PAR;
        int oc_pad = (layer.out_c + OC_PAR - 1) / OC_PAR * OC_PAR;
        current_weight_offset += oc_pad * ic_pad * 9;
    }

    // C. Output Buffers (Max size needed)
    // Just allocate enough for the largest feature map
    size_t max_fmap_size = 64 * 64 * 64; 
    std::vector<int8_t, aligned_allocator<int8_t>> host_buf_a(max_fmap_size);
    std::vector<int8_t, aligned_allocator<int8_t>> host_buf_b(max_fmap_size);

    // Gen instructions
    std::vector<NISA_Instruction> instr_list(layers.size() + 1); // +1 for HALT

    // Bank Mapping: 0=Heap, 1=BufA, 2=BufB
    int prev_bank = 0; // Input starts at Heap
    int prev_stride = 1; // Virtual stride of "input"

    for (size_t i = 0; i < layers.size(); ++i) {
        auto& layer = layers[i];
        auto& instr = instr_list[i];

        // Determine Banks (Ping-Pong)
        // Input is prev_bank. Output toggles between 1 and 2.
        int in_bank = prev_bank;
        int out_bank = (in_bank == 1) ? 2 : 1; 
        
        // Bank Sel Byte: [Output(3:2) | Input(1:0)]
        instr.bank_sel = (out_bank << 2) | in_bank;

        // Offsets
        instr.input_offset = (i == 0) ? INSTR_SECTION_SIZE : 0; // L0 reads from Heap offset
        instr.output_offset = 0;
        instr.weight_offset = layer_weight_offsets[i];

        // Dimensions
        instr.width = layer.in_w;
        instr.height = layer.in_h;
        // Pad Input Channels for Hardware Loop
        instr.in_channels = (layer.in_c + IC_PAR - 1) / IC_PAR * IC_PAR;
        instr.out_channels = layer.out_c;

        // Config
        instr.opcode = OP_CONV;
        instr.quant_shift = layer.quant_shift;
        instr.relu_en = layer.relu ? 1 : 0;
        instr.stride = layer.stride;

        // Memory Layout Logic
        // If previous layer had stride 2, current input is packed in blocks of TILE_HEIGHT/2
        // L0 Input is packed by Host at TILE_HEIGHT (log2=2)
        // L1 Input is L0 Output. If L0 stride=2, L1 Input is packed at TILE_HEIGHT/2 (log2=1)
        int mem_tile_h = (i == 0) ? TILE_HEIGHT : (TILE_HEIGHT / prev_stride);
        instr.log2_mem_tile_height = (int)std::log2(mem_tile_h);
	// instr.is_sparse = 0;
	// instr.ic_tile_mask = 0xFFFFFFFF;
	// instr.oc_tile_mask = 0xFFFFFFFF;
	if (i == 0) {
	   instr.is_sparse = 1;
	   instr.ic_tile_mask = 0x00000001;
	   instr.oc_tile_mask = 0x00000001;
	} else if (i == 1) {
	   instr.is_sparse = 1;
	   instr.ic_tile_mask = 0x00000001;
	   instr.oc_tile_mask = 0x00000001;
	} else if (i == 2) {
	   instr.is_sparse = 1;
	   instr.ic_tile_mask = 0x00000001;
	   instr.oc_tile_mask = 0x0000000F;
	}
        // Update state for next layer
        prev_bank = out_bank;
        prev_stride = layer.stride;
    }

    // HALT Instruction
    instr_list[layers.size()].opcode = OP_HALT;

    std::cout << "Size of NISA instruction struct (in bytes): " << sizeof(NISA_Instruction) << std::endl;
    std::cout << "Length of instruction list: " << instr_list.size() << std::endl;
    std::cout << "Length of layers list: " << layers.size() << std::endl;
    // Copy to Heap
    memcpy(host_heap.data(), instr_list.data(), sizeof(NISA_Instruction) * instr_list.size());

    // OpenCL routines
    cl_int err;
    auto devices = xcl::get_xil_devices();
    auto device = devices[0];
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    cl::Program::Binaries bins = xcl::import_binary_file(argv[1]);
    OCL_CHECK(err, cl::Program program(context, {device}, bins, NULL, &err));
    OCL_CHECK(err, cl::Kernel kernel(program, "vla_accel_top:{vla_accel_top_0}", &err));

    OCL_CHECK(err, cl::Buffer d_heap(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, heap_size, host_heap.data(), &err));
    OCL_CHECK(err, cl::Buffer d_buf_a(context, CL_MEM_READ_WRITE, max_fmap_size, NULL, &err));
    OCL_CHECK(err, cl::Buffer d_buf_b(context, CL_MEM_READ_WRITE, max_fmap_size, NULL, &err));
    OCL_CHECK(err, cl::Buffer d_wgt(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, total_weight_size, host_weights.data(), &err));

    kernel.setArg(0, d_heap);
    kernel.setArg(1, d_wgt);
    kernel.setArg(2, d_buf_a);
    kernel.setArg(3, d_buf_b);

    std::cout << "INFO: Running Kernel..." << std::endl;
    q.enqueueTask(kernel);
    q.finish();

    std::cout << "INFO: Verifying Last Layer Output..." << std::endl;
    
    // Determine where the final output is
    // Logic matches the loop: L0->BufA(1), L1->BufB(2), L2->BufA(1)...
    // Odd layers (1, 3) -> BufA. Even layers (2, 4) -> BufB.
    // Note: layers.size() is 1-based count.
    cl::Buffer* final_buf = (layers.size() % 2 != 0) ? &d_buf_a : &d_buf_b;
    // cl::Buffer* final_buf = &d_buf_b;
    
    auto& last_layer = layers.back();
    size_t out_bytes = (last_layer.in_w / last_layer.stride) * 
                       (last_layer.in_h / last_layer.stride) * 
                       last_layer.out_c;
    
    std::vector<int8_t> hw_results(out_bytes);
    q.enqueueReadBuffer(*final_buf, CL_TRUE, 0, out_bytes, hw_results.data());

    int errors = verify_output(hw_results, last_layer.golden_file, 
                               last_layer.in_h, last_layer.in_w, last_layer.out_c, 
                               last_layer.stride);

    if (errors == 0) std::cout << "TEST PASSED!" << std::endl;
    else std::cout << "TEST FAILED with " << errors << " errors." << std::endl;

    return (errors == 0) ? 0 : 1;
}
