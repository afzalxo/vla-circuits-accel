`timescale 1ns / 1ps

module tb_window_sequencer;

    // Params
    localparam IC_PAR = 16;
    localparam PP_PAR = 8;
    localparam DATA_WIDTH = 8;

    // Signals
    reg clk, rst_n;
    reg [1:0] k_y, k_x;
    reg load_new_data;
    reg [15:0] col_idx, row_idx, img_width_strips, img_height;
    
    // Inputs (3 Rows)
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] lb_row_0;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] lb_row_1;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] lb_row_2;
    
    // Output
    wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] pixels_out;

    // Instance
    window_sequencer #(
        .IC_PAR(IC_PAR), .PP_PAR(PP_PAR), .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .kernel_y(k_y), .kernel_x(k_x), .load_new_data(load_new_data),
        .col_idx(col_idx), .img_width_strips(img_width_strips),
        .row_idx(row_idx), .img_height(img_height),
        .lb_row_0(lb_row_0), .lb_row_1(lb_row_1), .lb_row_2(lb_row_2),
        .pixels_out(pixels_out)
    );

    always #5 clk = ~clk;

    // --- HELPER: SET ROWS ---
    task set_rows(input int val0, input int val1, input int val2);
        integer p, c;
        begin
            for(p=0; p<PP_PAR; p=p+1) begin
                for(c=0; c<IC_PAR; c=c+1) begin
                    lb_row_0[p][c] = val0 + p;
                    lb_row_1[p][c] = val1 + p;
                    lb_row_2[p][c] = val2 + p;
                end
            end
        end
    endtask

    // --- HELPER: CHECK OUTPUT ---
    task check(input int pixel_idx, input int expected_val, input string msg);
        begin
            if (pixels_out[pixel_idx][0] !== expected_val) begin
                $display("FAIL: %s | Pixel[%0d] = %0d (Expected %0d)", 
                         msg, pixel_idx, pixels_out[pixel_idx][0], expected_val);
                // Debug internal state
                $display("  DEBUG: curr_strip_1[0][0] = %0d", dut.curr_strip_1[0][0]);
                $display("  DEBUG: lb_row_1[0][0]     = %0d", lb_row_1[0][0]);
                $stop;
            end else begin
                $display("PASS: %s | Pixel[%0d] = %0d", msg, pixel_idx, expected_val);
            end
        end
    endtask

    initial begin
        clk = 0; rst_n = 0; load_new_data = 0;
        k_x = 0; k_y = 0;
        col_idx = 0; row_idx = 0;
        img_width_strips = 2; img_height = 16;
        set_rows(0, 0, 0);
        
        #20 rst_n = 1;

        // ============================================================
        // TEST 1: TOP-LEFT CORNER
        // ============================================================
        $display("\n--- TEST 1: TOP-LEFT CORNER ---");
        row_idx = 0; col_idx = 0;
        
        // 1. Setup Inputs on NEGEDGE to ensure stability for POSEDGE
        @(negedge clk);
        set_rows(10, 20, 30);
        load_new_data = 1; 
        
        // 2. Latch on POSEDGE
        @(posedge clk); 
        #1; // Hold slightly past edge
        load_new_data = 0;

        // A. Check Vertical Padding (Look Up)
        k_y = 0; k_x = 1; #1; 
        check(0, 0, "Top Padding (Row -1)");

        // B. Check Center (Row 0)
        // Note: In the sequencer, Ky=1 maps to curr_strip_1.
        // We loaded '20' into lb_row_1, so curr_strip_1 should be 20.
        k_y = 1; k_x = 1; #1;
        check(0, 20, "Center (Row 0, Px 0)");
        check(1, 21, "Center (Row 0, Px 1)");

        // C. Check Horizontal Padding (Look Left)
        k_y = 1; k_x = 0; #1;
        check(0, 0,  "Left Padding (Px -1)");
        check(1, 20, "Left Shift (Px 0 moved to Px 1 slot)");

        // ============================================================
        // TEST 2: CONTINUITY (Strip 1)
        // ============================================================
        $display("\n--- TEST 2: CONTINUITY (Strip 1) ---");
        col_idx = 1; 
        
        // New Inputs for Strip 1: Row 1=100+
        @(negedge clk);
        set_rows(10, 100, 30);
        load_new_data = 1;
        
        @(posedge clk); 
        #1;
        load_new_data = 0;
        
        // Check Left Neighbor (Look Left)
        k_y = 1; k_x = 0; #1;
        
        // Pixel 0 of Strip 1 should see Pixel 7 of Strip 0.
        // Strip 0 Row 1 was 20..27. So expected is 27.
        check(0, 27, "Stitching (Px 0 sees Prev Strip Px 7)");
        
        // Pixel 1 of Strip 1 should see Pixel 0 of Strip 1 (100)
        check(1, 100, "Internal Shift (Px 1 sees Px 0)");

        $display("\nSUCCESS: Window Sequencer Verified.");
        $finish;
    end

endmodule
