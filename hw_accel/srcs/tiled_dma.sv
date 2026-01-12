`timescale 1ns / 1ps

module tiled_dma #(
    parameter IC_PAR = 16,
    parameter PP_PAR = 8,
    parameter DATA_WIDTH = 8,
    parameter TILE_HEIGHT = 4,
    parameter INPUT_TILE_HEIGHT = TILE_HEIGHT + 2,  // Top and bottom halo
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
    input wire [15:0] active_height,
    input wire [2:0] log2_mem_tile_height, // (0=1, 1=2, 2=4)

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

    localparam BITS_PER_VECTOR = IC_PAR * DATA_WIDTH;
    localparam VECTORS_PER_HBM_WORD = HBM_DATA_WIDTH / BITS_PER_VECTOR;  // 4

    wire [31:0] stride_row_words = img_width / VECTORS_PER_HBM_WORD;  // 8 / 4 = 2
    
    // Size of one IC Tile Block (contains TILE_HEIGHT rows)
    wire [31:0] stride_ic_tile_words = stride_row_words << log2_mem_tile_height;
    
    // Size of one Height Tile Block (contains all IC tiles for this height)
    // This is the jump required to go from Height Tile N to Height Tile N+1
    wire [31:0] stride_height_tile_words = stride_ic_tile_words * (img_channels / IC_PAR);

    // B. Row Iterator Logic
    // -------------------------------------
    // We need to load: [Start_Row, End_Row]
    // Start = Top Halo (-1 relative to tile)
    // End   = Bottom Halo (+TILE_HEIGHT relative to tile)
    wire signed [15:0] abs_start_row = (tile_y_index * TILE_HEIGHT) - 1;
    wire signed [15:0] abs_end_row   = (tile_y_index * TILE_HEIGHT) + active_height;
    
    // Current Row Iterator
    reg signed [15:0] curr_row;
    
    // C. Dynamic Address Calculation for 'curr_row'
    // -------------------------------------
    // Determine which physical Height Tile block this row belongs to.
    // This handles the Halo case where we read a row from the Next/Prev tile.
    // wire [15:0] target_height_tile = curr_row / TILE_HEIGHT;
    // wire [15:0] row_in_tile        = curr_row % TILE_HEIGHT;
    wire [15:0] target_height_tile = curr_row >>> log2_mem_tile_height;
    wire [15:0] row_mask = (16'd1 << log2_mem_tile_height) - 1;
    wire [15:0] row_in_tile = curr_row & row_mask;
    
    // Final Base Address for the current row
    wire [63:0] curr_row_base_addr = (target_height_tile * stride_height_tile_words) + 
                                     (tile_ic_index      * stride_ic_tile_words) + 
                                     (row_in_tile        * stride_row_words);

    assign feature_map_words = stride_row_words;


    // State Machine
    localparam S_IDLE = 0;
    localparam S_CHECK_BOUNDS = 1;
    localparam S_READ = 3;
    localparam S_WAIT = 4;
    localparam S_DONE = 5;
    
    reg [2:0] tiled_dma_state;
    reg [15:0] beat_count;
    reg [15:0] req_count;
    reg [7:0] uram_beat_count;
    reg done_reg;
    assign done = done_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tiled_dma_state <= S_IDLE;
            done_reg <= 0;
            hbm_ren <= 0;
	    hbm_addr <= 0;
            uram_wen <= 0;
	    uram_addr <= 0;
	    curr_row <= 0;
	    beat_count <= 0;
	    uram_beat_count <= 0;
	    uram_wdata <= 0;
	    req_count <= 0;
        end else begin
            hbm_ren <= 0;
            uram_wen <= 0;
            done_reg <= 0;
            
            case (tiled_dma_state)
                S_IDLE: begin
                    if (start) begin
			curr_row <= abs_start_row;
                        uram_addr <= 0;
			uram_beat_count <= 0;
			tiled_dma_state <= S_CHECK_BOUNDS;
                    end
                end

		S_CHECK_BOUNDS: begin
                    if (curr_row > abs_end_row) begin
                        tiled_dma_state <= S_DONE;
                    end else begin
                        if (curr_row >= 0 && curr_row < img_height) begin
                            hbm_addr <= curr_row_base_addr;
			    hbm_ren <= 1;
                            beat_count <= 0;
                            uram_beat_count <= 0;
                            // Transition to REQ tiled_dma_state to assert REN next cycle
                            tiled_dma_state <= S_READ; 
                        end else begin
                            // PADDING ROW
                            uram_addr <= uram_addr + (img_width / PP_PAR);
                            curr_row <= curr_row + 1;
                            tiled_dma_state <= S_CHECK_BOUNDS;
                        end
                    end
                end

                S_READ: begin
                    if (hbm_rvalid) begin
                        uram_wdata[uram_beat_count * HBM_DATA_WIDTH +: HBM_DATA_WIDTH] <= hbm_data_in;
                        
                        if (uram_beat_count == BEATS_PER_URAM_IDX - 1) begin
                            uram_wen <= 1;
                            uram_addr <= uram_addr + 1;
                            uram_beat_count <= 0;
                        end else begin
                            uram_beat_count <= uram_beat_count + 1;
                        end
                        
                        if (beat_count == stride_row_words - 1) begin
                            curr_row <= curr_row + 1;
                            hbm_ren <= 0;
                            beat_count <= 0;
                            uram_beat_count <= 0;
                            tiled_dma_state <= S_CHECK_BOUNDS;
                        end else begin
                            beat_count <= beat_count + 1;
			    hbm_ren <= 1;
                            hbm_addr <= hbm_addr + 1; 
                        end
                    end
                end

                S_DONE: begin
                    done_reg <= 1;
                    tiled_dma_state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
