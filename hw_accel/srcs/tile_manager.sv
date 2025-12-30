`timescale 1ns / 1ps

module tile_manager #(
    parameter IC_PAR = 16,
    parameter OC_PAR = 16,
    parameter PP_PAR = 8,
    parameter DATA_WIDTH = 8,
    parameter MAX_IMG_WIDTH = 128,
    parameter TILE_HEIGHT = 4, // Height of strip to process per pass
    parameter GMEM_DATA_WIDTH = 512,
    parameter ACC_WIDTH = 28,
    parameter HBM_SIZE = 65536
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    input wire [15:0] full_img_width_strips,
    input wire [15:0] full_img_height,
    input wire [15:0] full_img_channels,

    input wire [15:0] output_channels,
    output wire [15:0] feature_map_words,
    output wire [15:0] weight_words,
    output wire [15:0] fmap_out_words,
    
    input wire relu_en,
    input wire [1:0] stride,
    input wire [2:0] log2_mem_tile_height,
    // Input FMap DMA Interface
    input wire [PP_PAR*IC_PAR*DATA_WIDTH-1:0] hbm_data_in,
    output wire [63:0] hbm_addr,
    output wire hbm_ren,
    input wire hbm_rvalid,
    
    // Weights DMA Interface
    input wire [OC_PAR*IC_PAR*DATA_WIDTH-1:0] hbm_data_in_w,
    output wire [63:0] hbm_addr_w,
    output wire hbm_ren_w,
    input wire hbm_rvalid_w,
    
    // Output FMap DMA Interface
    output wire [PP_PAR*OC_PAR*ACC_WIDTH-1:0] hbm_fmap_out_data,
    output wire [63:0] hbm_fmap_out_addr,
    output wire hbm_fmap_out_wen,
    output wire hbm_fmap_wvalid,
    input wire hbm_fmap_out_ready,
    input wire hbm_fmap_out_done,
    // Status
    output reg done
);

    // !! Used to iterate over tiles !!
    wire [15:0] num_tiles_y = full_img_height / TILE_HEIGHT;
    wire [15:0] num_tiles_ic = full_img_channels / IC_PAR;
    wire [15:0] num_tiles_oc = output_channels / OC_PAR;
    
    // --- INTERNAL URAMs ---
    // Input Buffer: Stores Tile + Halo
    // Size: (TILE_HEIGHT + 2) * MAX_IMG_WIDTH
    // Make sure tools infer URAM here.
    (* ram_style = "uram" *)
    reg [PP_PAR*IC_PAR*DATA_WIDTH-1:0] uram_input [0:(TILE_HEIGHT+2)*MAX_IMG_WIDTH/PP_PAR - 1];

    (* ram_style = "bram" *)
    reg [OC_PAR*IC_PAR*DATA_WIDTH-1:0] weights_bram [0:3*3-1];
    
    // Output Buffer: Stores Result
    // Size: TILE_HEIGHT * MAX_IMG_WIDTH
    // Note: We store the full result here for now.
    // In a real system, we would DMA this back to HBM.
    (* ram_style = "uram" *)
    reg [PP_PAR*OC_PAR*ACC_WIDTH-1:0] uram_output [0:TILE_HEIGHT*MAX_IMG_WIDTH/PP_PAR - 1];

    reg [15:0] current_tile_y;
    reg [15:0] current_tile_ic;

    reg [15:0] current_tile_oc;
    
    // DMA Signals
    reg dma_start;
    reg weights_dma_start;
    wire dma_done;
    wire dma_w_done;
    reg dma_done_r;
    reg dma_w_done_r;
    wire [15:0] dma_uram_addr;
    wire [15:0] dma_w_bram_addr;
    wire [15:0] dma_uram_fmap_out_addr;
    wire [PP_PAR*IC_PAR*DATA_WIDTH-1:0] dma_wdata;
    wire [IC_PAR*OC_PAR*DATA_WIDTH-1:0] dma_w_wdata;
    reg  [PP_PAR*OC_PAR*ACC_WIDTH-1:0] dma_fmap_out_wdata;
    wire dma_wen;
    wire dma_w_wen;
    wire dma_uram_ren;
    
    reg output_dma_start;
    reg output_dma_done;
    // Accelerator Signals
    reg acc_start;
    wire acc_done;
    wire acc_din_ready;
    reg acc_din_valid;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] acc_din_data;
    wire signed [PP_PAR-1:0][OC_PAR-1:0][27:0] acc_dout_data;
    wire acc_dout_valid;
    
    wire [1:0] k_x;
    wire [1:0] k_y;
    wire signed [OC_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] acc_weights_data = weights_bram[k_y*3 + k_x];
    wire acc_weights_req;
    wire acc_weights_ack;
    // Streamer Signals
    reg [15:0] stream_ptr;
    reg [15:0] out_ptr;

    tiled_dma #(
        .IC_PAR(IC_PAR),
	.PP_PAR(PP_PAR),
	.DATA_WIDTH(DATA_WIDTH),
	.TILE_HEIGHT(TILE_HEIGHT),
	.HBM_DATA_WIDTH(GMEM_DATA_WIDTH)
    ) dma (
        .clk(clk), .rst_n(rst_n), .start(dma_start),
        .img_width(full_img_width_strips * PP_PAR),
        .img_height(full_img_height),
	.img_channels(full_img_channels),
	.feature_map_words(feature_map_words),
        .tile_y_index(current_tile_y),
	.tile_ic_index(current_tile_ic),
	.log2_mem_tile_height(log2_mem_tile_height),
        .hbm_data_in(hbm_data_in),
        .hbm_addr(hbm_addr),
        .hbm_ren(hbm_ren),
	.hbm_rvalid(hbm_rvalid),
        .uram_addr(dma_uram_addr),
        .uram_wdata(dma_wdata),
        .uram_wen(dma_wen),
        .done(dma_done)
    );
    
    weights_dma #(
        .IC_PAR(IC_PAR),
	.OC_PAR(OC_PAR),
	.DATA_WIDTH(DATA_WIDTH),
	.HBM_DATA_WIDTH(GMEM_DATA_WIDTH)
    ) weights_dma (
        .clk(clk), .rst_n(rst_n), .start(weights_dma_start),
        .oc(output_channels),
        .ic(full_img_channels),
	.weight_words(weight_words),
	.ic_tile(current_tile_ic),
        .oc_tile(current_tile_oc),
	.hbm_data_in(hbm_data_in_w),
        .hbm_addr(hbm_addr_w),
        .hbm_ren(hbm_ren_w),
	.hbm_rvalid(hbm_rvalid_w),
        .bram_addr(dma_w_bram_addr),
        .bram_wdata(dma_w_wdata),
        .bram_wen(dma_w_wen),
        .done(dma_w_done)
    );

    output_dma #(
	.OC_PAR(OC_PAR),
	.PP_PAR(PP_PAR),
	.DATA_WIDTH(DATA_WIDTH),
	.ACCUM_WIDTH(ACC_WIDTH),
	.TILE_HEIGHT(TILE_HEIGHT),
	.HBM_DATA_WIDTH(GMEM_DATA_WIDTH)
    ) out_dma (
	.clk(clk), .rst_n(rst_n), .start(output_dma_start),
	.img_width(full_img_width_strips * PP_PAR),
	.oc_channels(output_channels),
	.tile_y_index(current_tile_y),
	.tile_oc_index(current_tile_oc),
	.relu_en(relu_en),
	.stride(stride),
	.write_count(fmap_out_words),
	.uram_addr(dma_uram_fmap_out_addr),
	.uram_rdata(dma_fmap_out_wdata),
	.uram_ren(dma_uram_ren),
	.start_write(hbm_fmap_out_wen),
	.write_base_addr(hbm_fmap_out_addr),
	.write_data(hbm_fmap_out_data),
	.write_valid(hbm_fmap_wvalid),
	.write_ready(hbm_fmap_out_ready),
	.write_done(hbm_fmap_out_done),
	.done(output_dma_done)
    );
 
    conv_accelerator #(
        .IC_PAR(IC_PAR), .OC_PAR(OC_PAR), .PP_PAR(PP_PAR), 
        .DATA_WIDTH(DATA_WIDTH), .MAX_IMG_WIDTH(MAX_IMG_WIDTH)
    ) acc (
        .clk(clk), .rst_n(rst_n), .start(acc_start),
        .img_width_strips(full_img_width_strips),
        .img_height(TILE_HEIGHT + 2), // Accelerator sees a small image
	.img_channels(full_img_channels),
        .weights(acc_weights_data),
	.k_x(k_x), .k_y(k_y),
	.stride(stride),
	.weight_req(acc_weights_req),
	.weight_ack(acc_weights_ack),
        .din_data(acc_din_data),
        .din_valid(acc_din_valid),
        .din_ready(acc_din_ready),
        .dout_data(acc_dout_data),
        .dout_valid(acc_dout_valid),
        .done(acc_done)
    );

    // --- URAM WRITE LOGIC (DMA -> URAM) ---
    always @(posedge clk) begin
        if (dma_wen) begin
            uram_input[dma_uram_addr-1] <= dma_wdata;
        end
	if (dma_w_wen) begin
	    weights_bram[dma_w_bram_addr-1] <= dma_w_wdata;
	end
	if (dma_uram_ren) begin
	    dma_fmap_out_wdata <= uram_output[dma_uram_fmap_out_addr];
	end
    end
    
    reg [3:0] state;

    localparam S_IDLE = 0;
    localparam S_DMA_FM  = 1;
    localparam S_DMA_W   = 6;
    localparam S_INIT_URAM = 7;
    localparam S_ACC_START = 2;
    localparam S_ACC_STREAM = 3;
    localparam S_ACC_WAIT = 4;
    localparam S_NEXT_TILE = 5;
    localparam S_WRITE_OUTPUT = 8;
 
    integer p, o;
    reg [15:0] acc_out_row_cnt;
    reg [15:0] acc_out_col_cnt;
    // --- URAM WRITE LOGIC (Accelerator -> Output URAM) ---
    // We simply append results linearly
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_ptr <= 0;
	end else if (state == S_ACC_START) begin
	    out_ptr <= 0;
	    acc_out_row_cnt <= 0;
            acc_out_col_cnt <= 0;
        end else if (acc_dout_valid) begin
	    // if (acc_out_row_cnt >= 1 && acc_out_row_cnt <= TILE_HEIGHT) begin
	        for (p = 0; p < PP_PAR; p = p + 1) begin
	            if (p % stride == 0) begin
	                integer p_out = p / stride;
	                for (o = 0; o < OC_PAR; o = o + 1) begin
	            	integer dst_bit = (p_out * OC_PAR + o) * ACC_WIDTH;
	            	if (current_tile_ic == 0) begin
	            	    // First Pass: Overwrite
	            	    uram_output[out_ptr][dst_bit +: ACC_WIDTH] <= acc_dout_data[p][o];
	            	end else begin
	            	    // Accumulation Pass: Read-Modify-Write
	            	    // Note: We read from the PACKED location (dst_bit)
	            	    uram_output[out_ptr][dst_bit +: ACC_WIDTH] <= 
	            		$signed(uram_output[out_ptr][dst_bit +: ACC_WIDTH]) + 
	            		$signed(acc_dout_data[p][o]);
	            	end
	                end
	            end
	        end
                out_ptr <= out_ptr + 1;
	    // end

	    if (acc_out_col_cnt == full_img_width_strips - 1) begin
                acc_out_col_cnt <= 0;
                acc_out_row_cnt <= acc_out_row_cnt + 1;
            end else begin
                acc_out_col_cnt <= acc_out_col_cnt + 1;
            end
 
        end
    end
   
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            current_tile_y <= 0;
	    current_tile_ic <= 0;
	    current_tile_oc <= 0;
            dma_start <= 0;
            acc_start <= 0;
            acc_din_valid <= 0;
            stream_ptr <= 0;
            done <= 0;
	    dma_done_r <= 0;
	    dma_w_done_r <= 0;
	    output_dma_start <= 0;
        end else begin
            dma_start <= 0;
	    weights_dma_start <= 0;
	    output_dma_start <= 0;
            acc_start <= 0;
            case (state)
                S_IDLE: begin
		    done <= 0;
                    if (start) begin
                        current_tile_y <= 0;
			current_tile_ic <= 0;
			current_tile_oc <= 0;
                        out_ptr <= 0; // Reset output pointer
                        state <= S_INIT_URAM;
                    end
                end

		S_INIT_URAM: begin
		    for (i = 0; i < (TILE_HEIGHT+2)*MAX_IMG_WIDTH/PP_PAR; i = i + 1) begin
			uram_input[i] <= 0;
		    end
		    dma_start <= 1;
                    weights_dma_start <= 1;
                    dma_done_r <= 0;
                    dma_w_done_r <= 0;
		    state <= S_DMA_FM;
		end
                
                // 1. Load Tile from HBM to URAM
                S_DMA_FM: begin
		    dma_start <= 0;
		    weights_dma_start <= 0;
		    if (dma_done) begin
		        dma_done_r <= 1;
		    end
		    if (dma_w_done) begin
			dma_w_done_r <= 1;
		    end
                    if (dma_done_r && dma_w_done_r) begin
                        state <= S_ACC_START;
                    end
                end
                
                // 2. Start Accelerator
                S_ACC_START: begin
                    acc_start <= 1;
                    stream_ptr <= 0;
		    out_ptr <= 0;
                    state <= S_ACC_STREAM;
                end
                
                // 3. Stream Data from URAM to Accelerator
                S_ACC_STREAM: begin
                    // Simple Valid/Ready Handshake
                    // We assume URAM read is 0-latency (Register file behavior for sim)
                    // In real HW, need 1-cycle read latency handling
                    if (stream_ptr < (TILE_HEIGHT + 2) * full_img_width_strips) begin
                        acc_din_valid <= 1;
                        acc_din_data <= uram_input[stream_ptr];
                        
                        if (acc_din_ready) begin
                            stream_ptr <= stream_ptr + 1;
                        end
                    end else begin
                        acc_din_valid <= 0;
                        state <= S_ACC_WAIT;
                    end
                end
                
                // 4. Wait for Accelerator to Finish
                S_ACC_WAIT: begin
                    if (acc_done) begin
                        state <= S_NEXT_TILE;
                    end
                end
                
                // 5. Check if more tiles needed
                S_NEXT_TILE: begin
		    // Next IC tile
		    dma_done_r <= 0;
		    dma_w_done_r <= 0;
		    if (current_tile_ic + 1 >= num_tiles_ic) begin
			state <= S_WRITE_OUTPUT;
			output_dma_start <= 1;      // Trigger Output DMA
			current_tile_ic <= 0;
		    end else begin
			current_tile_ic <= current_tile_ic + 1;
			dma_start <= 1;
			weights_dma_start <= 1;
			state <= S_DMA_FM;
		    end
                end

		S_WRITE_OUTPUT: begin
                    output_dma_start <= 0;
                    if (output_dma_done) begin
                        // Now we can move to the next OC Tile or Height Tile
                        if (current_tile_oc + 1 >= num_tiles_oc) begin
			    // Done with all OC tiles for this Y
			    // Next Y tile
			    if (current_tile_y + 1 >= num_tiles_y) begin
				// All done
                                done <= 1;
			        dma_start <= 0;
			        weights_dma_start <= 0;
                                state <= S_IDLE;
			    end else begin
				current_tile_y <= current_tile_y + 1;
				current_tile_oc <= 0;
				current_tile_ic <= 0;
				state <= S_INIT_URAM;
			    end
                        end else begin
                            current_tile_oc <= current_tile_oc + 1;
			    current_tile_ic <= 0;
                            dma_start <= 1;
			    weights_dma_start <= 1;
                            state <= S_DMA_FM;
                        end
                    end
                end
            endcase
        end
    end

`ifdef XILINX_SIMULATOR
    integer f_dump;
    integer i_dump;
    integer ff_dump;
    integer out_dump;

    reg [PP_PAR*OC_PAR*8-1:0] packed_quant_row;
    reg signed [27:0] raw_acc;
    reg signed [27:0] shifted_acc;
    reg signed [7:0]  clamped_val;
    integer p_idx, o_idx;
    
    initial begin

	wait(state == S_ACC_START);
        #1;
        
        $display("[SIM-DEBUG] Dumping Input URAM Content...");
        f_dump = $fopen("hw_dump_input_uram.txt", "w");
        
        // Dump all rows (TILE_HEIGHT + 2)
        // Each row has (img_width / PP_PAR) words
        for (i_dump = 0; i_dump < (TILE_HEIGHT + 2) * (full_img_width_strips); i_dump = i_dump + 1) begin
            $fdisplay(f_dump, "Addr %0d: %h", i_dump, uram_input[i_dump]);
        end
        $fclose(f_dump);
	
        // Wait until the controller signals that loading is finished
        // You might need to pass 'load_done' into this module or trigger on a state
        wait(state == S_ACC_START && current_tile_y == 0 && current_tile_ic == 0);
        
        // Wait a few cycles for stability
        #2;
        
        $display("[SIM-DEBUG] Dumping Weight Memory to file...");
        f_dump = $fopen("hw_dump_weights.txt", "w");
        $display("[SIM-DEBUG] Dumping Feature Memory to file...");
        ff_dump = $fopen("hw_dump_features.txt", "w");
        
        if (f_dump == 0) begin
            $display("Error: Could not open dump file!");
            $finish;
        end

        // Iterate through the memory depth
        for (i_dump = 0; i_dump < 9; i_dump = i_dump + 1) begin
            // %h prints hex. It prints MSB to LSB.
            $fdisplay(f_dump, "%h", weights_bram[i_dump]);
        end

	for (i_dump = 0; i_dump < TILE_HEIGHT * full_img_width_strips; i_dump = i_dump + 1) begin
	    // %h prints hex. It prints MSB to LSB.
	    $fdisplay(ff_dump, "%h", uram_input[i_dump]);
	end
        
        $fclose(f_dump);
	$fclose(ff_dump);
        $display("[SIM-DEBUG] Dump complete.");

        wait(current_tile_oc == 0 && current_tile_y == 1 && current_tile_ic == 0);

	#2;
	
	$display("[SIM-DEBUG] Dumping Output to file...");
	out_dump = $fopen("hw_dump_output.txt", "w");
	for (i_dump = 0; i_dump < TILE_HEIGHT * full_img_width_strips; i_dump = i_dump + 1) begin
	    $fdisplay(out_dump, "%h", uram_output[i_dump]);
	end
        $fclose(out_dump);
	$display("[SIM-DEBUG] Output Dump complete...");
	f_dump = $fopen("hw_dump_output_quant.txt", "w");
        for (i_dump = 0; i_dump < TILE_HEIGHT * full_img_width_strips; i_dump = i_dump + 1) begin
            // Process one URAM word (contains PP_PAR * OC_PAR accumulators)
            for (p_idx = 0; p_idx < PP_PAR; p_idx = p_idx + 1) begin
                for (o_idx = 0; o_idx < OC_PAR; o_idx = o_idx + 1) begin
                    
                    // 1. Extract the 28-bit Accumulator
                    // Note: 28 is the ACC_WIDTH
                    raw_acc = uram_output[i_dump][((p_idx*OC_PAR + o_idx)*28) +: 28];
                    
                    // 2. Apply Arithmetic Right Shift (Matches C++ ">> 10")
                    shifted_acc = raw_acc >>> 10;
                    
                    // 3. Saturate / Clamp to 8-bit Signed Range
                    if (shifted_acc > 127) 
                        clamped_val = 8'd127;
                    else if (shifted_acc < -128) 
                        clamped_val = -8'd128;
                    else 
                        clamped_val = shifted_acc[7:0];
                    
                    // 4. Pack into the temporary 8-bit bus
                    // Note: 8 is the OUT_WIDTH
                    packed_quant_row[((p_idx*OC_PAR + o_idx)*8) +: 8] = clamped_val;
                end
            end
            
            // Write the packed 8-bit hex string to file
            $fdisplay(f_dump, "%h", packed_quant_row);
        end
        $fclose(f_dump);
    end
`endif



endmodule
