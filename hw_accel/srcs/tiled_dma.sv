`timescale 1ns / 1ps

module tiled_dma #(
    parameter IC_PAR = 16,
    parameter PP_PAR = 8,
    parameter DATA_WIDTH = 8,
    parameter TILE_HEIGHT = 4,
    parameter HBM_DATA_WIDTH = 512,  // Ensure matches bus width
    parameter BEATS_PER_URAM_IDX = (IC_PAR * PP_PAR) / (HBM_DATA_WIDTH / DATA_WIDTH),
    parameter ADDR_WIDTH = 64,  // HBM Address width
    parameter URAM_DEPTH = 4096
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // Configuration
    input wire [15:0] img_width,      // Full Image Width
    input wire [15:0] img_height,     // Full Image Height
    input wire [15:0] img_channels,   // Number of Channels
    input wire [15:0] tile_y_index,   // Current vertical tile index
    input wire [15:0] tile_ic_index,  // Current IC tile index

    output wire [15:0] feature_map_words,
    
    // HBM Interface (Simulated as a large array port)
    input wire [HBM_DATA_WIDTH-1:0] hbm_data_in,
    output reg [ADDR_WIDTH-1:0] hbm_addr,
    output reg hbm_ren,
    input wire hbm_rvalid,
    
    // URAM Interface (Write Port)
    output reg [15:0] uram_addr,
    output reg [PP_PAR*IC_PAR*DATA_WIDTH-1:0] uram_wdata,
    output reg uram_wen,
    
    output wire done
);

    // State Machine
    localparam S_IDLE = 0;
    localparam S_READ = 1;
    localparam S_WAIT = 2;
    localparam S_DONE = 3;
    
    reg [1:0] state;
    reg [15:0] curr_y; // Relative to tile start
    reg [7:0] beat_count;
    reg done_reg;
    assign done = done_reg;
    
    wire [15:0] tile_start_index = tile_ic_index * (img_width * TILE_HEIGHT) / PP_PAR + tile_y_index * (img_width * img_channels * TILE_HEIGHT) / (PP_PAR * IC_PAR);
    wire [15:0] words_per_tile = (img_width * TILE_HEIGHT * IC_PAR) / (HBM_DATA_WIDTH / DATA_WIDTH);

    assign feature_map_words = words_per_tile;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done_reg <= 0;
            hbm_ren <= 0;
            uram_wen <= 0;
	    beat_count <= 0;
        end else begin
            hbm_ren <= 0;
            uram_wen <= 0;
            done_reg <= 0;
            
            case (state)
                S_IDLE: begin
                    if (start) begin
                        curr_y <= 0;
			beat_count <= 0;
                        uram_addr <= 0;
                        state <= S_READ;
                    end
                end
                
                S_READ: begin
		    hbm_addr <= tile_start_index;
		    hbm_ren <= 1; 
		    state <= S_WAIT;
		end

		S_WAIT: begin
		    if (hbm_rvalid) begin
		        uram_wdata[(beat_count % BEATS_PER_URAM_IDX) * HBM_DATA_WIDTH +: HBM_DATA_WIDTH] <= hbm_data_in; 
			if ((beat_count + 1) % BEATS_PER_URAM_IDX == 0) begin
		            uram_wen <= 1;
                            uram_addr <= uram_addr + 1;
			end

			if (beat_count < words_per_tile - 1) begin
			    beat_count <= beat_count + 1;
			    state <= S_WAIT;
			end else begin
			    beat_count <= 0;
			    hbm_ren <= 0;
			    state <= S_DONE;
			end
	            end
                end
                
                S_DONE: begin
                    done_reg <= 1;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
