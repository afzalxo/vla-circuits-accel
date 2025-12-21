`timescale 1ns / 1ps

module tb_integration;

    // Params
    localparam IC_PAR = 16;
    localparam OC_PAR = 16;
    localparam PP_PAR = 8;
    localparam DATA_WIDTH = 8;
    localparam MAX_IMG_WIDTH = 128;

    // Signals
    reg clk, rst_n;
    
    // Line Buffer Signals
    reg shift_en;
    reg [15:0] img_width_strips;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] data_in;
    wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] row_0, row_1, row_2;
    
    // Sequencer Signals
    reg [1:0] k_y, k_x;
    reg load_new_data;
    reg [15:0] col_idx;
    reg [15:0] row_idx; 
    reg [15:0] img_height;
    wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] seq_out;
    
    // Compute Unit Signals
    reg cu_en, cu_clear;
    reg signed [OC_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] weights;
    wire signed [PP_PAR-1:0][OC_PAR-1:0][27:0] results;
    wire cu_valid_out;

    // File Handles
    integer f_in, f_seq, f_out;
    integer p, c; // Loop iterators

    // --- INSTANCES ---
    line_buffer #(
        .IC_PAR(IC_PAR), .PP_PAR(PP_PAR), .DATA_WIDTH(DATA_WIDTH), .MAX_IMG_WIDTH(MAX_IMG_WIDTH)
    ) lb (
        .clk(clk), .rst_n(rst_n), .shift_en(shift_en), 
        .img_width_strips(img_width_strips),
        .data_in(data_in),
        .row_0(row_0), .row_1(row_1), .row_2(row_2)
    );

    window_sequencer #(
        .IC_PAR(IC_PAR), .PP_PAR(PP_PAR), .DATA_WIDTH(DATA_WIDTH)
    ) seq (
        .clk(clk), .rst_n(rst_n),
        .kernel_y(k_y), .kernel_x(k_x), .load_new_data(load_new_data),
        .col_idx(col_idx), .img_width_strips(img_width_strips),
        .row_idx(row_idx), .img_height(img_height), 
        .lb_row_0(row_0), .lb_row_1(row_1), .lb_row_2(row_2),
        .pixels_out(seq_out)
    );

    vector_compute_unit #(
        .IC_PAR(IC_PAR), .OC_PAR(OC_PAR), .PP_PAR(PP_PAR), .DATA_WIDTH(DATA_WIDTH)
    ) cu (
        .clk(clk), .rst_n(rst_n), .en(cu_en), .clear_acc(cu_clear),
	.valid_in(cu_en),
        .img_vector_in(seq_out),
        .weights_in(weights),
        .img_vector_out(results),
	.valid_out(cu_valid_out)
    );

    // --- CLOCK ---
    always #5 clk = ~clk;

    // --- HELPER TASK ---
    // Drives inputs on NEGEDGE to ensure setup time is met
    // Also dumps the input data to file
    task feed_strip(input int val);
        integer i, j;
        begin
            @(negedge clk); // Wait for safe time to change inputs
            for(i=0; i<PP_PAR; i=i+1) 
                for(j=0; j<IC_PAR; j=j+1) 
                    data_in[i][j] = val;
            
            // Dump Input to File
            $fwrite(f_in, "Strip Input: ");
            for(i=0; i<PP_PAR; i=i+1) begin
                $fwrite(f_in, "[P%0d:", i);
                for(j=0; j<IC_PAR; j=j+1) begin
                    $fwrite(f_in, "%0d ", data_in[i][j]);
                end
                $fwrite(f_in, "] ");
            end
            $fwrite(f_in, "\n");

            shift_en = 1; 
            
            @(negedge clk); // Hold for one cycle
            shift_en = 0;
        end
    endtask

    // --- TEST ---
    integer i, j;
    initial begin
        // Open Files
        f_in = $fopen("dump_input.txt", "w");
        f_seq = $fopen("dump_sequencer.txt", "w");
        f_out = $fopen("dump_output.txt", "w");

        clk = 0; rst_n = 0;
        shift_en = 0; load_new_data = 0; cu_en = 0; cu_clear = 1;
        k_x = 0; k_y = 0; col_idx = 0; row_idx = 0;
        
        // CONFIG
        img_width_strips = 1; 
        img_height = 16; 

        // Init Weights to 1
        for(i=0; i<OC_PAR; i=i+1) for(j=0; j<IC_PAR; j=j+1) weights[i][j] = 1;
        // Init Data to 0
        for(i=0; i<PP_PAR; i=i+1) for(j=0; j<IC_PAR; j=j+1) data_in[i][j] = 0;

        #20 rst_n = 1;

        $display("--- FILLING LINE BUFFER (Top Edge Case) ---");
        
        // Feed Row 0 (Value 10)
        feed_strip(10); 
	feed_strip(10);
        
        // Feed Row 1 (Value 20)
        feed_strip(20); 
        
        // Load Strip 0 into Sequencer
        @(negedge clk);
        col_idx = 0;
        row_idx = 0; // TOP EDGE
        load_new_data = 1; 
        
        @(negedge clk);
        load_new_data = 0;

        // --- START CONVOLUTION ---
        $display("--- STARTING CONVOLUTION (Row 0, Strip 0) ---");
        
        @(negedge clk);
        cu_clear = 0; 
        cu_en = 1;    

        // 3x3 Loop
        for (int y=0; y<3; y=y+1) begin
            for (int x=0; x<3; x=x+1) begin
                k_y = y;
                k_x = x;
                
                // Wait for posedge to latch inputs
                @(posedge clk); 
                #1; // Small delay for display stability
                
                // Dump Sequencer State
                $fwrite(f_seq, "Ky=%0d Kx=%0d | SeqOut: ", y, x);
                for(p=0; p<PP_PAR; p=p+1) begin
                    $fwrite(f_seq, "[P%0d:", p);
                    for(c=0; c<IC_PAR; c=c+1) begin
                        $fwrite(f_seq, "%0d ", seq_out[p][c]);
                    end
                    $fwrite(f_seq, "] ");
                end
                $fwrite(f_seq, "\n");

                // Debug print for Pixel 0
                $display("Cycle Ky=%d Kx=%d | SeqOut[0]=%d | PartialSum[0]=%d", 
                         y, x, seq_out[0][0], results[0][0]);
            end
        end
        
        // Stop Compute
        @(negedge clk);
        cu_en = 0;
        
        // Wait for pipeline flush (Adder tree latency)
        repeat(10) @(posedge clk);
        
        // Dump Final Results
        $fwrite(f_out, "Final Results (Strip 0):\n");
        for(p=0; p<PP_PAR; p=p+1) begin
            $fwrite(f_out, "Pixel %0d: ", p);
            for(c=0; c<OC_PAR; c=c+1) begin
                $fwrite(f_out, "%0d ", results[p][c]);
            end
            $fwrite(f_out, "\n");
        end

        $display("--- FINAL RESULT ---");
        $display("Pixel 0 Result: %d", results[0][0]);
        $display("Pixel 1 Result: %d", results[1][0]);
	$display("Pixel 2 Result: %d", results[2][0]);
	$display("Pixel 3 Result: %d", results[3][0]);
        $display("Pixel 4 Result: %d", results[4][0]);
	$display("Pixel 5 Result: %d", results[5][0]);
	$display("Pixel 6 Result: %d", results[6][0]);
	$display("Pixel 7 Result: %d", results[7][0]);
        if (results[0][0] == 960) 
            $display("SUCCESS: Result matches expected value (960).");
        else
            $display("FAILURE: Expected 960, got %d", results[0][0]);

        // Close Files
        $fclose(f_in);
        $fclose(f_seq);
        $fclose(f_out);

        $finish;
    end

endmodule
