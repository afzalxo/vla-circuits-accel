`timescale 1ns / 1ps

module vector_compute_unit #(
    parameter IC_PAR = 16,
    parameter OC_PAR = 16,
    parameter PP_PAR = 8,   // 8 Pixels per clock
    parameter DATA_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,
    input wire en,
    input wire clear_acc,
    input wire valid_in,
    
    // Input: 8 Pixels x 16 Channels - Packed 3D
    // [Pixel_Index][Channel_Index][Bit_Index]
    input wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] img_vector_in,
    
    // Weights: 16 Filters x 16 Channels - Packed 3D
    // Shared across all pixels
    input wire signed [OC_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] weights_in,
    
    // Output: 8 Pixels x 16 Channels - Packed 3D
    output wire signed [PP_PAR-1:0][OC_PAR-1:0][27:0] img_vector_out,
    output wire valid_out
);

    wire [PP_PAR-1:0] valid_outs;
    assign valid_out = valid_outs[0];

    genvar p;
    generate
        for (p = 0; p < PP_PAR; p = p + 1) begin : GEN_PIXELS
            processing_element #(
                .IC_PAR(IC_PAR),
                .OC_PAR(OC_PAR),
                .DATA_WIDTH(DATA_WIDTH)
            ) pe_inst (
                .clk(clk),
                .rst_n(rst_n),
                .en(en),
                .clear_acc(clear_acc),
		.valid_in(valid_in),
                .pixel_in(img_vector_in[p]), // Slice the p-th pixel
                .weights_in(weights_in),     // Pass full weight matrix (Shared)
                .pixel_out(img_vector_out[p]),
		.valid_out(valid_outs[p])
            );
        end
    endgenerate

endmodule
