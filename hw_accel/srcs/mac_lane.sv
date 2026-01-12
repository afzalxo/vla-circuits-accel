`timescale 1ns / 1ps

module mac_lane #(
    parameter IC_PAR = 16,      
    parameter DATA_WIDTH = 8,   
    parameter ACC_WIDTH = 28    
)(
    input wire clk,
    input wire rst_n,
    input wire en,          
    input wire clear_acc,   
    input wire valid_in,    
    
    input wire signed [IC_PAR-1:0][DATA_WIDTH-1:0] activations,
    input wire signed [IC_PAR-1:0][DATA_WIDTH-1:0] weights,
    
    output reg signed [ACC_WIDTH-1:0] result,
    output reg valid_out
);

    localparam PIPE_DEPTH = 4;
    
    reg [PIPE_DEPTH-1:0] valid_pipe;
    reg [PIPE_DEPTH-1:0] clear_pipe;

    // --- CONTROL PIPELINE (FIXED) ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= 0;
            clear_pipe <= 0;
        end else begin
            // FIX 1: Valid depends on EN (Data is only valid if enabled)
            valid_pipe <= {valid_pipe[PIPE_DEPTH-2:0], en ? valid_in : 1'b0};
            
            // FIX 2: Clear does NOT depend on EN. It must propagate even if EN=0.
            clear_pipe <= {clear_pipe[PIPE_DEPTH-2:0], clear_acc};
        end
    end

    wire valid_delayed = valid_pipe[PIPE_DEPTH-1];
    wire clear_delayed = clear_pipe[PIPE_DEPTH-1];
    
    // --- DATA PIPELINE ---
    (* use_dsp = "yes" *) reg signed [15:0] products [0:IC_PAR-1];
    reg signed [16:0] sum_l1 [0:7];
    reg signed [17:0] sum_l2 [0:3];
    reg signed [19:0] dot_product;

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i=0; i<IC_PAR; i=i+1) products[i] <= 0;
            dot_product <= 0;
        end else begin
            // Data logic remains the same
            if (en) begin
                for (i=0; i<IC_PAR; i=i+1) products[i] <= $signed(activations[i]) * $signed(weights[i]);
            end else begin
                for (i=0; i<IC_PAR; i=i+1) products[i] <= 0;
            end
            
            // Adder Tree (Always runs, but inputs might be 0)
            for (i=0; i<8; i=i+1) sum_l1[i] <= $signed(products[2*i]) + $signed(products[2*i+1]);
            for (i=0; i<4; i=i+1) sum_l2[i] <= $signed(sum_l1[2*i]) + $signed(sum_l1[2*i+1]);
            dot_product <= ($signed(sum_l2[0]) + $signed(sum_l2[1])) + ($signed(sum_l2[2]) + $signed(sum_l2[3]));
        end
    end
    
    /*
    always @(posedge clk) begin
        if (en) begin
            // Print the first channel of the first pixel
            $display("MAC Input: Act=%d Weight=%d | Product=%d", 
                     activations[0], weights[0], activations[0]*weights[0]);
        end
    end
    */

    // --- ACCUMULATOR ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 0;
        end else begin
	    valid_out <= valid_delayed;
            // Priority: Clear > Accumulate
            if (clear_delayed) begin
                // If clearing, we reset. 
                // If there is ALSO valid data arriving this exact cycle, we load it.
                // Otherwise, we reset to 0.
                result <= (valid_delayed) ? $signed(dot_product) : 0;
            end else if (valid_delayed) begin
                // Normal accumulation
                result <= $signed(result) + $signed(dot_product);
            end
            // Else: Hold value
        end
    end

endmodule
