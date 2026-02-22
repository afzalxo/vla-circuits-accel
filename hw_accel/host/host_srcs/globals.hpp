#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include <cstdint>
#include <string>

constexpr uint16_t COMM_PORT = 12345;

constexpr uint16_t IMG_WIDTH  = 256;
constexpr uint16_t IMG_HEIGHT = 256;
constexpr uint16_t IMG_CHANNELS = 3;
constexpr uint32_t IMG_SIZE_BYTES = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS;
constexpr uint32_t EXTRA_FEATURES_BYTES = 128;
constexpr float OUTPUT_SCALE = 87.6368f;

constexpr uint16_t IC_PAR = 16;
constexpr uint16_t OC_PAR = 16;
constexpr uint16_t PP_PAR = 8;
constexpr uint16_t TILE_HEIGHT = 4;
constexpr size_t INSTR_SECTION_SIZE = 64 * 1024;

// opcodes
constexpr uint8_t OP_CONV   = 1;
constexpr uint8_t OP_GEMM   = 2;
constexpr uint8_t OP_MEMCPY = 3;
constexpr uint8_t OP_GAP    = 4;
constexpr uint8_t OP_HALT   = 255;

// User-defined Layer Configuration
struct LayerConfig {
    std::string name;
    int in_w, in_h, in_c;
    int out_c;
    int stride;
    int quant_shift;
    bool relu;
    int opcode;
    std::string weight_file;
    std::string golden_file; // For verification
};

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
    uint8_t  flatten;               // [311:304]
    uint8_t  is_sparse;             // [319:312]
    uint32_t ic_tile_mask;          // [351:320]
    uint32_t oc_tile_mask;          // [383:352]
    uint8_t  padding[16];           // [511:384] - Padding to make 64 bytes
};


#pragma pack(push, 1)
struct InferenceRequest {
    int32_t command_id;
    int32_t terminate;
};

struct InferenceResponse {
    float steer;
    float accel;
};
#pragma pack(pop)


#endif
