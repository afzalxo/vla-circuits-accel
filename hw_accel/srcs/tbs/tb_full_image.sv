`timescale 1ns / 1ps

module tb_full_image;

    // --- PARAMETERS ---
    localparam IC_PAR = 16;
    localparam OC_PAR = 16;
    localparam PP_PAR = 8;
    localparam DATA_WIDTH = 8;
    localparam MAX_IMG_WIDTH = 128;
    
    // Image Config
    localparam IMG_H = 64;
    localparam IMG_W = 64;
    localparam IMG_W_STRIPS = IMG_W / PP_PAR;
    localparam TOTAL_STRIPS = IMG_H * IMG_W_STRIPS;

    // --- SIGNALS ---
    reg clk, rst_n, start;
    
    // Data Stream In
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] din_data;
    reg din_valid;
    wire din_ready;
    
    // Weights
    reg signed [OC_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] weights;
    
    // Data Stream Out
    wire signed [PP_PAR-1:0][OC_PAR-1:0][27:0] dout_data;
    wire dout_valid;
    wire done;

    // --- MEMORIES ---
    reg [PP_PAR*IC_PAR*DATA_WIDTH-1:0] img_mem [0:TOTAL_STRIPS-1];
    reg [OC_PAR*IC_PAR*DATA_WIDTH-1:0] w_mem [0:0];

    // --- INSTANTIATION ---
    conv_accelerator #(
        .IC_PAR(IC_PAR), 
        .OC_PAR(OC_PAR), 
        .PP_PAR(PP_PAR), 
        .DATA_WIDTH(DATA_WIDTH), 
        .MAX_IMG_WIDTH(MAX_IMG_WIDTH)
    ) dut (
        .clk(clk), 
        .rst_n(rst_n), 
        .start(start),
        
        // Configuration
        .img_width_strips(16'd8), // W=16 -> 2 Strips
        .img_height(16'd64),       // H=3
        
        .weights(weights),
        .din_data(din_data), 
        .din_valid(din_valid), 
        .din_ready(din_ready),
        
        .dout_data(dout_data), 
        .dout_valid(dout_valid), 
        .done(done)
    );

    // --- CLOCK GENERATION ---
    always #5 clk = ~clk;

    // --- FILE OUTPUT HANDLES ---
    integer f_out;
    integer f_lb;       // Line Buffer Debug
    integer f_seq_full; // Sequencer Debug
    
    // --- MAIN PROCESS ---
    integer strip_ptr;
    
    initial begin
        // 1. Initialize Signals
        clk = 0; 
        rst_n = 0; 
        start = 0; 
        din_valid = 0;
        strip_ptr = 0;
        
        // 2. Load Hex Files
        $readmemh("image_input.hex", img_mem);
        $readmemh("weights.hex", w_mem);
        weights = w_mem[0];

        f_out = $fopen("hw_output.csv", "w");
        f_lb = $fopen("debug_line_buffer.txt", "w");
        f_seq_full = $fopen("debug_sequencer_full.txt", "w");

        // 3. Reset Sequence
        #20 rst_n = 1;
        #20 start = 1;
        #10 start = 0;
        
        // Timeout watchdog
        #200000;
        $display("Error: Simulation Timed Out");
        $finish;
    end

    // --- INPUT DRIVER (Fixed Handshake) ---
    always @(posedge clk) begin
        if (!rst_n) begin
            din_valid <= 0;
            strip_ptr <= 0;
        end else begin
            // Case 1: Bus is not valid yet (Start or Bubble)
            if (!din_valid) begin
                if (strip_ptr < TOTAL_STRIPS) begin
                    din_data <= img_mem[strip_ptr];
                    din_valid <= 1;
                end
            end 
            // Case 2: Bus is Valid. Check if Consumer accepted it.
            else if (din_ready) begin
                // Handshake occurred! (Valid=1, Ready=1)
                // Prepare the NEXT data immediately (Pipelined)
                
                if (strip_ptr < TOTAL_STRIPS - 1) begin
                    // Fetch next strip
                    din_data <= img_mem[strip_ptr + 1];
                    din_valid <= 1;
                    strip_ptr <= strip_ptr + 1;
                end else begin
                    // No more data
		    din_data <= 0;
                    din_valid <= 0;
                    strip_ptr <= strip_ptr + 1; // Increment to indicate completion
                end
            end
            // Case 3: Valid=1 but Ready=0. Hold current data.
        end
    end

    // --- OUTPUT MONITOR ---
    integer p, c;
    always @(posedge clk) begin
        if (dout_valid) begin
            for (p = 0; p < PP_PAR; p = p + 1) begin
                for (c = 0; c < OC_PAR; c = c + 1) begin
                    $fwrite(f_out, "%0d", $signed(dout_data[p][c]));
                    if (c < OC_PAR-1) $fwrite(f_out, ",");
                end
                $fwrite(f_out, "\n");
            end
        end
        
        if (done) begin
            $display("Simulation Done. Output written to hw_output.csv");
            $fclose(f_out);
            $fclose(f_lb);
            $fclose(f_seq_full);
            $finish;
        end
    end

    // --- DEBUG: LINE BUFFER MONITOR ---
    // Dumps the state of the Line Buffer every cycle
    integer d_lb;
    always @(posedge clk) begin
        if (rst_n) begin
            $fwrite(f_lb, "Time: %0t | ShiftEn: %b | WrPtr: %0d | RdPtr: %0d\n", 
                    $time, dut.lb.shift_en, dut.lb.wr_ptr, dut.lb.rd_ptr);
            
            // Print Channel 0 of each pixel in the strips for readability
            $fwrite(f_lb, "  Row2 (In):  ");
            for (d_lb=0; d_lb<PP_PAR; d_lb=d_lb+1) $fwrite(f_lb, "%4d ", $signed(dut.lb.row_2[d_lb][0]));
            $fwrite(f_lb, "\n");

            $fwrite(f_lb, "  Row1 (Mid): ");
            for (d_lb=0; d_lb<PP_PAR; d_lb=d_lb+1) $fwrite(f_lb, "%4d ", $signed(dut.lb.row_1[d_lb][0]));
            $fwrite(f_lb, "\n");

            $fwrite(f_lb, "  Row0 (Old): ");
            for (d_lb=0; d_lb<PP_PAR; d_lb=d_lb+1) $fwrite(f_lb, "%4d ", $signed(dut.lb.row_0[d_lb][0]));
            $fwrite(f_lb, "\n");
            $fwrite(f_lb, "--------------------------------------------------\n");
        end
    end

    // --- DEBUG: SEQUENCER MONITOR ---
    // Dumps the state of the Window Sequencer every cycle
    integer d_seq;
    always @(posedge clk) begin
        if (rst_n) begin
            $fwrite(f_seq_full, "Time: %0t | Load: %b | Row: %0d Col: %0d | Ky: %0d Kx: %0d\n",
                    $time, dut.seq.load_new_data, dut.seq.row_idx, dut.seq.col_idx, dut.seq.kernel_y, dut.seq.kernel_x);
            
            $fwrite(f_seq_full, "  SeqOut (Ch0): ");
            for (d_seq=0; d_seq<PP_PAR; d_seq=d_seq+1) $fwrite(f_seq_full, "%4d ", $signed(dut.seq.pixels_out[d_seq][0]));
            $fwrite(f_seq_full, "\n");
            
            // Also dump the internal registers to see what was latched
            $fwrite(f_seq_full, "  Internal Regs (Ch0):\n");
            $fwrite(f_seq_full, "    Curr0: ");
            for (d_seq=0; d_seq<PP_PAR; d_seq=d_seq+1) $fwrite(f_seq_full, "%4d ", $signed(dut.seq.curr_strip_0[d_seq][0]));
            $fwrite(f_seq_full, "\n");
            $fwrite(f_seq_full, "    Curr1: ");
            for (d_seq=0; d_seq<PP_PAR; d_seq=d_seq+1) $fwrite(f_seq_full, "%4d ", $signed(dut.seq.curr_strip_1[d_seq][0]));
            $fwrite(f_seq_full, "\n");
            $fwrite(f_seq_full, "    Curr2: ");
            for (d_seq=0; d_seq<PP_PAR; d_seq=d_seq+1) $fwrite(f_seq_full, "%4d ", $signed(dut.seq.curr_strip_2[d_seq][0]));
            $fwrite(f_seq_full, "\n");
            
            $fwrite(f_seq_full, "--------------------------------------------------\n");
        end
    end

endmodule
