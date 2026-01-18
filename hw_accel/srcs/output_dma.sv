`timescale 1ns / 1ps

module output_dma #(
    parameter OC_PAR = 16,
    parameter PP_PAR = 8,
    parameter DATA_WIDTH = 8,
    parameter ACCUM_WIDTH = 28,
    parameter TILE_HEIGHT = 4,
    parameter HBM_DATA_WIDTH = 512,
    parameter ADDR_WIDTH = 64,
    
    // URAM Width: PP_PAR * OC_PAR * 28-bit = 3584 bits
    parameter URAM_WIDTH = PP_PAR * OC_PAR * ACCUM_WIDTH
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // Configuration
    input wire [15:0] img_width,      // Full Image Width
    input wire [15:0] oc_channels,   // Total Output Channels
    input wire [15:0] tile_y_index,   // Current Vertical Tile
    input wire [15:0] tile_oc_index,  // Current Output Channel Tile

    input wire [15:0] active_height,

    input wire [4:0] quant_shift,
    input wire relu_en,
    input wire [1:0] stride,
    input wire flatten,
    // URAM Interface (Read Port)
    output reg [15:0] uram_addr,
    input wire [URAM_WIDTH-1:0] uram_rdata,
    output reg uram_ren,
    
    // Interface to burst_axi_write_master
    output reg start_write,           // Pulse to start the Master
    output reg [ADDR_WIDTH-1:0] write_base_addr,
    output reg [15:0] write_count,    // Total 512-bit words to write
    
    output reg [HBM_DATA_WIDTH-1:0] write_data,
    output reg write_valid,
    input wire write_ready,           // From Master
    input wire write_done,            // From Master
    
    output reg done
);
    // Calculate Packing Density
    localparam BITS_PER_PIXEL_BLOCK = OC_PAR * DATA_WIDTH; 
    // Serialization Constants
    // TODO: What happens when QUANTIZED_BLOCK_WIDTH < HBM_DATA_WIDTH?
    localparam QUANTIZED_BLOCK_WIDTH = PP_PAR * OC_PAR * DATA_WIDTH;
    localparam CHUNKS_PER_URAM_WORD = (QUANTIZED_BLOCK_WIDTH + HBM_DATA_WIDTH - 1) / HBM_DATA_WIDTH;
    
    wire [15:0] out_width = img_width / stride;
    // Stride Calculations (in Bytes)
    wire [63:0] bits_per_row = out_width * BITS_PER_PIXEL_BLOCK;
    wire [63:0] words_per_row_packed = (out_width * BITS_PER_PIXEL_BLOCK) / HBM_DATA_WIDTH;
    wire [63:0] words_per_row_flat = out_width * CHUNKS_PER_URAM_WORD; 
    wire [63:0] words_per_row = flatten ? words_per_row_flat : words_per_row_packed;

    wire [63:0] stride_oc_tile_words = words_per_row * (active_height / stride);
    wire [63:0] stride_height_tile_words = stride_oc_tile_words * (oc_channels / OC_PAR);

    // Base Address for this Tile
    wire [63:0] tile_base_addr = (tile_y_index * stride_height_tile_words) + 
                                 (tile_oc_index * stride_oc_tile_words);

    reg [3:0] flat_pixel_idx; 
    reg [3:0] flat_chunk_idx;
    // Flattened write constants
    wire [15:0] num_w_strips = (img_width + PP_PAR - 1) / PP_PAR;
    wire [15:0] total_uram_words = num_w_strips * (active_height / stride);

    // TODO: Support non-divisible cases?
    wire [7:0] chunks_per_word = CHUNKS_PER_URAM_WORD / stride;
    
    // Total 512-bit words to write to HBM
    wire [15:0] total_hbm_words = flatten ? (total_uram_words * CHUNKS_PER_URAM_WORD * PP_PAR) : total_uram_words * chunks_per_word;

    reg [QUANTIZED_BLOCK_WIDTH-1:0] quantized_data;
    reg [HBM_DATA_WIDTH-1:0] tmp;
    integer p, o;
    always @(*) begin
	quantized_data = 0;
        for (p = 0; p < PP_PAR; p = p + 1) begin
            for (o = 0; o < OC_PAR; o = o + 1) begin
                // 1. Extract 28-bit Accumulator
                reg signed [ACCUM_WIDTH-1:0] acc_val;
                reg signed [ACCUM_WIDTH-1:0] shifted_val;
                reg signed [DATA_WIDTH-1:0] final_val;
                
                acc_val = uram_rdata[((p*OC_PAR + o)*ACCUM_WIDTH) +: ACCUM_WIDTH];

		if (relu_en && acc_val[ACCUM_WIDTH-1]) begin // Check Sign Bit
                    acc_val = 0;
                end
                
                // 2. Arithmetic Shift (Preserves Sign)
                shifted_val = acc_val >>> quant_shift;
                
                // 3. Saturate / Clamp to 8-bit range (-128 to 127)
                if (shifted_val > 127) 
                    final_val = 8'd127;
                else if (shifted_val < -128)
                    final_val = -8'd128;
                else
                    final_val = shifted_val[DATA_WIDTH-1:0];
                
                // 4. Pack into intermediate buffer
                quantized_data[((p*OC_PAR + o)*DATA_WIDTH) +: DATA_WIDTH] = final_val;
            end
        end
    end
 

    localparam S_IDLE        = 0;
    localparam S_START_AXI   = 1;
    localparam S_READ_URAM   = 2;
    localparam S_WAIT_URAM   = 3;
    localparam S_SERIALIZE   = 4;
    localparam S_WAIT_DONE   = 5;
    localparam S_DONE        = 6;
    
    reg [2:0] output_dma_state;
    reg [15:0] word_cnt;      // URAM words processed
    reg [7:0] chunk_cnt;      // 512-bit chunks processed for current URAM word
    reg [7:0] pixel_cnt;
    reg [7:0] input_pixel_cnt;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_dma_state <= S_IDLE;
            done <= 0;
            start_write <= 0;
            write_valid <= 0;
            uram_addr <= 0;
	    uram_ren <= 0;
            write_base_addr <= 0;
            write_count <= 0;
            word_cnt <= 0;
            chunk_cnt <= 0;
	    pixel_cnt <= 0;
	    input_pixel_cnt <= 0;
	    flat_chunk_idx <= 0;
            write_data <= 0;
        end else begin
            start_write <= 0;
            done <= 0;
            
            case (output_dma_state)
                S_IDLE: begin
                    write_valid <= 0;
                    if (start) begin
                        // Setup AXI Master Parameters
                        write_base_addr <= tile_base_addr;
                        write_count <= total_hbm_words;
                        
                        // Reset Counters
                        uram_addr <= 0;
                        word_cnt <= 0;
                        chunk_cnt <= 0;
			flat_chunk_idx <= 0;
                        
                        output_dma_state <= S_START_AXI;
                    end
                end
                
                S_START_AXI: begin
                    // Pulse start for the AXI Master
                    start_write <= 1;
		    uram_ren <= 1;
                    output_dma_state <= S_READ_URAM;
                end
                
                S_READ_URAM: begin
                    // Address is already set (starts at 0, increments in S_SERIALIZE)
                    // Just wait for RAM latency
		    uram_ren <= 0;
                    output_dma_state <= S_WAIT_URAM;
                end
                
                S_WAIT_URAM: begin
		    if (flatten) begin
			pixel_cnt <= 0;
			input_pixel_cnt <= 0;
			flat_chunk_idx <= 0;
		    end else begin
		        write_data <= quantized_data[0 +: HBM_DATA_WIDTH];
		        chunk_cnt <= 0;
		        write_valid <= 1;
		    end

                    output_dma_state <= S_SERIALIZE;
                end
                
                S_SERIALIZE: begin
                    // Wait for Master to accept data
                    if (write_ready) begin
			if (flatten) begin
                            // --- FLATTEN MODE ---
		            // We are writing 16 units of valid data + PP_PAR-1 x 16
		            // units of zero padding. A single HBM word holds 16 units
		            // of valid data and 48 units of zero padding. The first
		            // HBM word we will write will have the valid data + 48
		            // zeros and the second one will have all zeros. This is
		            // for one input pixel location. We will repeat this for
		            // all PP_PAR input pixel locations.
			    if (input_pixel_cnt < PP_PAR) begin
			        if (flat_chunk_idx == 0) begin
			            write_data <= {{(HBM_DATA_WIDTH - BITS_PER_PIXEL_BLOCK){1'b0}}, quantized_data[(input_pixel_cnt) * BITS_PER_PIXEL_BLOCK +: BITS_PER_PIXEL_BLOCK]};
			            write_valid <= 1;
			            flat_chunk_idx <= flat_chunk_idx + 1;
		                end else if (flat_chunk_idx == CHUNKS_PER_URAM_WORD - 1) begin
			  	    // This will not work when
				    // CHUNKS_PER_URAM_WORD is not 2 FIXME
			            write_data <= 0;
			            write_valid <= 1;
			            input_pixel_cnt <= input_pixel_cnt + 1;
			            flat_chunk_idx <= 0;
			        end
		            end else begin
			        // Finished all input pixels
			        // Move to next URAM word or done
			        input_pixel_cnt <= 0;
			        flat_chunk_idx <= 0;
			        if (word_cnt == total_uram_words - 1) begin
			    	    write_valid <= 0;
			    	    output_dma_state <= S_WAIT_DONE;
			        end else begin
			    	    write_valid <= 0;
			    	    word_cnt <= word_cnt + 1;
			    	    uram_addr <= uram_addr + 1;
			    	    uram_ren <= 1;
			    	    output_dma_state <= S_READ_URAM;
			        end
			    end
                        end else begin
                            // Chunk accepted
                            if (chunk_cnt == chunks_per_word - 1) begin
                                // Finished one URAM word
                                chunk_cnt <= 0;
                                
                                if (word_cnt == total_uram_words - 1) begin
                                    // All data sent
                                    write_valid <= 0;
                                    output_dma_state <= S_WAIT_DONE;
                                end else begin
                                    // Move to next URAM word
                                    word_cnt <= word_cnt + 1;
                                    uram_addr <= uram_addr + 1;
			    	    uram_ren <= 1;
                                    write_valid <= 0; // Pause valid while reading new data
                                    output_dma_state <= S_READ_URAM;
                                end
                            end else begin
                                chunk_cnt <= chunk_cnt + 1;
			        write_data <= quantized_data[(chunk_cnt + 1) * HBM_DATA_WIDTH +: HBM_DATA_WIDTH];
			        write_valid <= 1;
                            end
			end
		    end
                end
                
                S_WAIT_DONE: begin
                    // Wait for AXI Master to finish the B-Channel response
                    if (write_done) begin
                        output_dma_state <= S_DONE;
                    end
                end
                
                S_DONE: begin
                    done <= 1;
                    output_dma_state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
