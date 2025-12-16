`timescale 1ns / 1ps

module tb_line_buffer;

    // Params
    localparam IC_PAR = 16;
    localparam PP_PAR = 8;
    localparam DATA_WIDTH = 8;
    localparam MAX_IMG_WIDTH = 128;

    // Signals
    reg clk, rst_n, shift_en;
    reg [15:0] img_width_strips;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] data_in;
    wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] row_0, row_1, row_2;

    // Instance
    line_buffer #(
        .IC_PAR(IC_PAR), .PP_PAR(PP_PAR), .DATA_WIDTH(DATA_WIDTH), .MAX_IMG_WIDTH(MAX_IMG_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n), .shift_en(shift_en),
        .img_width_strips(img_width_strips),
        .data_in(data_in),
        .row_0(row_0), .row_1(row_1), .row_2(row_2)
    );

    always #5 clk = ~clk;

    // Helper to set data_in to a specific value for all pixels/channels
    task set_input(input int val);
        integer i, j;
        begin
            for(i=0; i<PP_PAR; i=i+1) 
                for(j=0; j<IC_PAR; j=j+1) 
                    data_in[i][j] = val;
        end
    endtask

    // Helper to check outputs
    task check_output(input int exp_r0, input int exp_r1, input int exp_r2, input string msg);
        begin
            // Check just the first pixel/channel for simplicity
            if (row_0[0][0] !== exp_r0 || row_1[0][0] !== exp_r1 || row_2[0][0] !== exp_r2) begin
                $display("FAIL at %t: %s", $time, msg);
                $display("  Expected: R0=%d, R1=%d, R2=%d", exp_r0, exp_r1, exp_r2);
                $display("  Got:      R0=%d, R1=%d, R2=%d", row_0[0][0], row_1[0][0], row_2[0][0]);
                $stop;
            end else begin
                $display("PASS at %t: %s (R0=%d, R1=%d, R2=%d)", $time, msg, row_0[0][0], row_1[0][0], row_2[0][0]);
            end
        end
    endtask

    initial begin
        clk = 0; rst_n = 0; shift_en = 0;
        img_width_strips = 4; // Width = 4 Strips
        set_input(0);

        #20 rst_n = 1;

        // --- PHASE 1: FEED ROW 0 (Values 10, 11, 12, 13) ---
        $display("\n--- FEEDING ROW 0 ---");
        
        // Strip 0
        set_input(10); shift_en = 1; @(posedge clk); #1;
        check_output(0, 0, 10, "Row 0, Strip 0"); // R0, R1 empty. R2 has input.

        // Strip 1
        set_input(11); @(posedge clk); #1;
        check_output(0, 0, 11, "Row 0, Strip 1");

        // Strip 2
        set_input(12); @(posedge clk); #1;
        check_output(0, 0, 12, "Row 0, Strip 2");

        // Strip 3 (End of Row 0)
        set_input(13); @(posedge clk); #1;
        check_output(0, 0, 13, "Row 0, Strip 3");

        // --- PHASE 2: FEED ROW 1 (Values 20, 21, 22, 23) ---
        $display("\n--- FEEDING ROW 1 (Wrapping) ---");
        
        // Strip 0 (Should align with Row 0 Strip 0)
        set_input(20); @(posedge clk); #1;
        // R2 = 20 (Current)
        // R1 = 10 (From Row 0, Strip 0)
        // R0 = 0  (From Row -1)
        check_output(0, 10, 20, "Row 1, Strip 0");

        // Strip 1
        set_input(21); @(posedge clk); #1;
        check_output(0, 11, 21, "Row 1, Strip 1");

        // Strip 2
        set_input(22); @(posedge clk); #1;
        check_output(0, 12, 22, "Row 1, Strip 2");

        // Strip 3
        set_input(23); @(posedge clk); #1;
        check_output(0, 13, 23, "Row 1, Strip 3");

        // --- PHASE 3: FEED ROW 2 (Values 30, 31, 32, 33) ---
        $display("\n--- FEEDING ROW 2 (Full Pipeline) ---");

        // Strip 0
        set_input(30); @(posedge clk); #1;
        // R2 = 30 (Current)
        // R1 = 20 (From Row 1)
        // R0 = 10 (From Row 0)
        check_output(10, 20, 30, "Row 2, Strip 0");

        // --- PHASE 4: HOLD TEST ---
        $display("\n--- HOLD TEST ---");
        shift_en = 0;
        set_input(99); // Change input, but don't shift
        @(posedge clk); #1;
        // Outputs should NOT change (Registered outputs hold state)
        check_output(10, 20, 30, "Hold State");

        $display("\nSUCCESS: Line Buffer Verified.");
        $finish;
    end

endmodule
