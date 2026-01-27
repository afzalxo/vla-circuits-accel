`timescale 1ns / 1ps

module output_dma #(
    parameter OC_PAR = 16,
    parameter PP_PAR = 8,
    parameter DATA_WIDTH = 8,
    parameter ACCUM_WIDTH = 28,
    parameter TILE_HEIGHT = 4,
    parameter HBM_DATA_WIDTH = 512,
    parameter ADDR_WIDTH = 64,
    parameter URAM_WIDTH = PP_PAR * OC_PAR * ACCUM_WIDTH
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // Configuration
    input wire [15:0] img_width,
    input wire [15:0] oc_channels,
    input wire [15:0] tile_y_index,
    input wire [15:0] tile_oc_index,
    input wire [15:0] active_height,
    input wire [4:0] quant_shift,
    input wire relu_en,
    input wire [1:0] stride,
    input wire flatten,
    
    // URAM Interface
    output reg [15:0] uram_addr,
    input wire [URAM_WIDTH-1:0] uram_rdata,
    output reg uram_ren,
    
    // Interface to burst_axi_write_master
    output reg start_write,
    output reg [ADDR_WIDTH-1:0] write_base_addr,
    output reg [15:0] write_count,
    output reg [HBM_DATA_WIDTH-1:0] write_data,
    output reg write_valid,
    input wire write_ready,
    input wire write_done,
    output reg done
);

    // 1. Basic Block Sizes
    localparam BITS_PER_PIXEL_BLOCK = OC_PAR * DATA_WIDTH; 
    localparam QUANTIZED_BLOCK_WIDTH = PP_PAR * OC_PAR * DATA_WIDTH;
    localparam CHUNKS_PER_URAM_WORD = (QUANTIZED_BLOCK_WIDTH + HBM_DATA_WIDTH - 1) / HBM_DATA_WIDTH;

    // --- PIPELINE REGISTERS ---
    reg [15:0] r_out_width;
    reg [15:0] r_num_w_strips;
    reg [15:0] r_valid_pixels_per_strip;
    reg [15:0] r_active_bits_per_strip;
    reg [31:0] r_global_tile_index;
    
    reg [15:0] r_num_oc_tiles;
    reg [31:0] r_tile_y_stride;
    reg [31:0] r_words_per_row;
    reg [15:0] r_total_uram_words;
    
    reg [31:0] r_stride_oc_tile_words;
    
    // --- QUANTIZATION ---
    reg [URAM_WIDTH-1:0] r_raw_uram_data;
    reg [QUANTIZED_BLOCK_WIDTH-1:0] quantized_data;
    reg [QUANTIZED_BLOCK_WIDTH-1:0] r_quantized_data;

    integer p, o;
    always @(*) begin
        quantized_data = 0;
        for (p = 0; p < PP_PAR; p = p + 1) begin
            for (o = 0; o < OC_PAR; o = o + 1) begin
                reg signed [ACCUM_WIDTH-1:0] acc_val;
                reg signed [ACCUM_WIDTH-1:0] shifted_val;
                reg signed [DATA_WIDTH-1:0] final_val;
                
                acc_val = r_raw_uram_data[((p*OC_PAR + o)*ACCUM_WIDTH) +: ACCUM_WIDTH];
                if (relu_en && acc_val[ACCUM_WIDTH-1]) acc_val = 0;
                shifted_val = acc_val >>> quant_shift;
                
                if (shifted_val > 127) final_val = 8'd127;
                else if (shifted_val < -128) final_val = -8'd128;
                else final_val = shifted_val[DATA_WIDTH-1:0];
                
                quantized_data[((p*OC_PAR + o)*DATA_WIDTH) +: DATA_WIDTH] = final_val;
            end
        end
    end

    // --- GEARBOX ---
    reg [2047:0] gearbox_buffer;
    reg [15:0] gearbox_fill_level;

    // --- FSM ---
    localparam S_IDLE        = 0;
    localparam S_CALC_1      = 1;
    localparam S_CALC_2      = 2;
    localparam S_CALC_3      = 3;
    localparam S_CALC_4      = 4;
    localparam S_START_AXI   = 5;
    localparam S_READ_URAM   = 6;
    localparam S_WAIT_URAM   = 7;
    localparam S_PROCESS     = 8;
    localparam S_FLUSH       = 9;
    localparam S_WAIT_DONE   = 10;
    localparam S_DONE        = 11;
    localparam S_QUANTIZE    = 12;
    localparam S_LATCH_RAW   = 13;
    
    reg [3:0] output_dma_state;
    reg [15:0] word_cnt;      
    reg [7:0] input_pixel_cnt;
    reg [3:0] flat_chunk_idx;

    (* max_fanout = 50 *) reg raw_data_en;
    
    always @(posedge clk) begin
        if (!rst_n) raw_data_en <= 0;
        else begin
            // Pre-calculate enable for next cycle (S_READ_URAM -> S_LATCH_RAW)
            // Or just base it on current state if timing allows.
            // Ideally, register it to break the path from FSM logic.
            raw_data_en <= (output_dma_state == S_READ_URAM);
        end
    end
    
    always @(posedge clk) begin
        // Use the replicated enable signal
        if (raw_data_en) begin
            r_raw_uram_data <= uram_rdata;
        end
    end

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
            input_pixel_cnt <= 0;
            flat_chunk_idx <= 0;
            write_data <= 0;
            gearbox_buffer <= 0;
            gearbox_fill_level <= 0;
            
            // Reset Pipeline Regs
            r_out_width <= 0;
            r_num_w_strips <= 0;
            r_valid_pixels_per_strip <= 0;
            r_active_bits_per_strip <= 0;
            r_global_tile_index <= 0;
            r_words_per_row <= 0;
            r_total_uram_words <= 0;
	    r_num_oc_tiles <= 0;
            r_stride_oc_tile_words <= 0;
        end else begin
            start_write <= 0;
            done <= 0;
            
            case (output_dma_state)
                S_IDLE: begin
                    write_valid <= 0;
                    if (start) begin
                        output_dma_state <= S_CALC_1;
			r_out_width <= 0;
			r_num_w_strips <= 0;
			r_valid_pixels_per_strip <= 0;
			r_active_bits_per_strip <= 0;
			r_words_per_row <= 0;
			r_total_uram_words <= 0;
			r_num_oc_tiles <= 0;
			r_stride_oc_tile_words <= 0;
                    end
                end

		// --- PIPELINE STAGE 1: Basic Dimensions ---
                S_CALC_1: begin
                    r_out_width <= (stride == 2) ? (img_width >> 1) : img_width;
                    r_num_w_strips <= (img_width + PP_PAR - 1) / PP_PAR;
                    r_valid_pixels_per_strip <= (stride == 2) ? (PP_PAR >> 1) : PP_PAR;
                    output_dma_state <= S_CALC_2;
                end
                
                // --- PIPELINE STAGE 2: Derived Counts ---
                S_CALC_2: begin
                    r_active_bits_per_strip <= r_valid_pixels_per_strip * BITS_PER_PIXEL_BLOCK;
                    if (flatten) begin
                        r_words_per_row <= r_out_width * CHUNKS_PER_URAM_WORD;
                    end else begin
                        r_words_per_row <= (r_out_width * BITS_PER_PIXEL_BLOCK + HBM_DATA_WIDTH - 1) / HBM_DATA_WIDTH;
                    end
                    r_total_uram_words <= (stride == 2) ? r_num_w_strips * (active_height >> 1) : r_num_w_strips * active_height;
		    r_num_oc_tiles <= oc_channels / OC_PAR;
                    output_dma_state <= S_CALC_3;
                end
                
                // --- PIPELINE STAGE 3: Strides & Beats ---
                S_CALC_3: begin
                    // 2. Stride OC Tile (Words)
		    if (stride == 2) begin
                        r_stride_oc_tile_words <= r_words_per_row * (active_height >> 1);
		    end
		    else begin
                        r_stride_oc_tile_words <= r_words_per_row * active_height;
		    end
                    
                    // 3. Write Count
                    if (flatten) begin
                        write_count <= r_total_uram_words * r_valid_pixels_per_strip * CHUNKS_PER_URAM_WORD;
                    end else begin
                        write_count <= (r_total_uram_words * r_active_bits_per_strip + HBM_DATA_WIDTH - 1) / HBM_DATA_WIDTH;
                    end
		    r_tile_y_stride <= tile_y_index * r_num_oc_tiles;

                    output_dma_state <= S_CALC_4;
                end

		S_CALC_4: begin
		    // write_base_addr <= tile_y_index * (r_stride_oc_tile_words * (oc_channels / OC_PAR)) +
		    //	    	       tile_oc_index * r_stride_oc_tile_words;
		    write_base_addr <= (r_tile_y_stride + tile_oc_index) * r_stride_oc_tile_words;
                    
		    uram_addr <= 0;
		    word_cnt <= 0;
		    flat_chunk_idx <= 0;
		    gearbox_fill_level <= 0;
		    output_dma_state <= S_START_AXI;
		end 

                S_START_AXI: begin
                    start_write <= 1;
                    uram_ren <= 1;
                    output_dma_state <= S_READ_URAM;
                end
                
                S_READ_URAM: begin
                    uram_ren <= 0;
                    output_dma_state <= S_LATCH_RAW;
                end

		S_LATCH_RAW: begin
		    // r_raw_uram_data <= uram_rdata;
		    output_dma_state <= S_QUANTIZE;
		end

		S_QUANTIZE: begin
		    r_quantized_data <= quantized_data;
		    output_dma_state <= S_WAIT_URAM;
		end
                
                S_WAIT_URAM: begin
                    if (flatten) begin
                        input_pixel_cnt <= 0;
                        flat_chunk_idx <= 0;
                        output_dma_state <= S_PROCESS;
                    end else begin
                        // STANDARD MODE: Load Gearbox
                        gearbox_buffer[gearbox_fill_level +: QUANTIZED_BLOCK_WIDTH] <= r_quantized_data;
                        gearbox_fill_level <= gearbox_fill_level + r_active_bits_per_strip;
                        output_dma_state <= S_PROCESS;
                    end
                end
                
                S_PROCESS: begin
                    if (flatten) begin
                        // --- FLATTEN MODE ---
                        if (write_ready || !write_valid) begin
                            if (input_pixel_cnt < r_valid_pixels_per_strip) begin
                                if (flat_chunk_idx == 0) begin
                                    write_data <= {{(HBM_DATA_WIDTH - BITS_PER_PIXEL_BLOCK){1'b0}}, 
                                                   r_quantized_data[(input_pixel_cnt) * BITS_PER_PIXEL_BLOCK +: BITS_PER_PIXEL_BLOCK]};
                                end else begin
                                    write_data <= 0;
                                end
                                write_valid <= 1;

                                if (flat_chunk_idx == CHUNKS_PER_URAM_WORD - 1) begin
                                    flat_chunk_idx <= 0;
                                    input_pixel_cnt <= input_pixel_cnt + 1;
                                end else begin
                                    flat_chunk_idx <= flat_chunk_idx + 1;
                                end
                            end else begin
                                input_pixel_cnt <= 0;
                                flat_chunk_idx <= 0;
                                if (word_cnt == r_total_uram_words - 1) begin
                                    write_valid <= 0;
                                    output_dma_state <= S_WAIT_DONE;
                                end else begin
                                    word_cnt <= word_cnt + 1;
                                    uram_addr <= uram_addr + 1;
                                    uram_ren <= 1;
                                    write_valid <= 0; 
                                    output_dma_state <= S_READ_URAM;
                                end
                            end
                        end
                    end else begin
                        // --- STANDARD MODE (Gearbox) ---
                        if (gearbox_fill_level >= HBM_DATA_WIDTH) begin
                            if (write_ready || !write_valid) begin
                                write_data <= gearbox_buffer[0 +: HBM_DATA_WIDTH];
                                write_valid <= 1;
                                gearbox_buffer <= (gearbox_buffer >> HBM_DATA_WIDTH);
                                gearbox_fill_level <= gearbox_fill_level - HBM_DATA_WIDTH;
                            end
                        end else begin
                            if (word_cnt == r_total_uram_words - 1) begin
                                if (gearbox_fill_level > 0) begin
                                    output_dma_state <= S_FLUSH;
                                end else begin
                                    write_valid <= 0;
                                    output_dma_state <= S_WAIT_DONE;
                                end
                            end else begin
                                word_cnt <= word_cnt + 1;
                                uram_addr <= uram_addr + 1;
                                uram_ren <= 1;
                                write_valid <= 0;
                                output_dma_state <= S_READ_URAM;
                            end
                        end
                    end
                end
                
                S_FLUSH: begin
                    if (write_ready || !write_valid) begin
                        write_data <= gearbox_buffer[0 +: HBM_DATA_WIDTH];
                        write_valid <= 1;
                        gearbox_fill_level <= 0;
                        output_dma_state <= S_WAIT_DONE;
                    end
                end
                
                S_WAIT_DONE: begin
                    if (write_ready) write_valid <= 0;
                    if (write_done) output_dma_state <= S_DONE;
                end
                
                S_DONE: begin
                    done <= 1;
                    output_dma_state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
