`timescale 1ns / 1ps

module instruction_scheduler #(
    parameter ADDR_WIDTH = 64,
    parameter DATA_WIDTH = 512
)(
    input wire clk,
    input wire rst_n,
    
    // --- Host Control Interface ---
    input wire start,                 // Pulse to start the instruction stream
    input wire [ADDR_WIDTH-1:0] base_addr, // Address of the first instruction
    output reg done,                  // Asserted when HALT opcode is encountered
    
    // --- AXI4 Read Master Interface (Lite version for Instructions) ---
    // We share the GMEM port, so we need standard AXI signals
    output reg m_axi_arvalid,
    input wire m_axi_arready,
    output reg [ADDR_WIDTH-1:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,    // Always 0 (Single beat)
    output wire [2:0] m_axi_arsize,   // Always 6 (64 bytes)
    output wire [1:0] m_axi_arburst,  // INCR
    
    input wire m_axi_rvalid,
    output reg m_axi_rready,
    input wire [DATA_WIDTH-1:0] m_axi_rdata,
    
    // --- Tile Manager Control Interface ---
    output reg tm_start,
    input wire tm_done,
    
    // --- Configuration Outputs (Driven to Tile Manager) ---
    output reg [ADDR_WIDTH-1:0] cfg_input_addr,
    output reg [ADDR_WIDTH-1:0] cfg_output_addr,
    output reg [ADDR_WIDTH-1:0] cfg_weight_addr,
    output reg [15:0] cfg_img_width,
    output reg [15:0] cfg_img_height,
    output reg [15:0] cfg_in_channels,
    output reg [15:0] cfg_out_channels,
    output reg [4:0]  cfg_quant_shift,
    output reg        cfg_is_conv, // 1=Conv, 0=Dense/Other
    output reg 	      cfg_relu_en,
    output reg [1:0]  cfg_stride,
    output reg [2:0]  cfg_log2_mem_tile_height,
    output reg [1:0]  cfg_input_bank,
    output reg [1:0]  cfg_output_bank
);

    // --- AXI Constants ---
    assign m_axi_arlen = 8'd0;       // Single beat (64 bytes)
    assign m_axi_arsize = 3'b110;    // 64 bytes
    assign m_axi_arburst = 2'b01;    // INCR

    // --- State Machine ---
    localparam S_IDLE        = 0;
    localparam S_FETCH_REQ   = 1; // Send AR
    localparam S_FETCH_WAIT  = 2; // Wait for R
    localparam S_DECODE      = 3; // Latch values
    localparam S_EXECUTE     = 4; // Trigger Tile Manager
    localparam S_WAIT_TM     = 5; // Wait for Tile Manager Done
    localparam S_NEXT_INSTR  = 6; // Increment PC
    localparam S_DONE        = 7;
    localparam S_FETCH_ACK   = 8;

    reg [3:0] state;
    reg [ADDR_WIDTH-1:0] pc; // Program Counter (Current Instruction Address)
    
    // Instruction Register
    reg [DATA_WIDTH-1:0] instr_reg;
    wire [7:0] opcode = instr_reg[263:256];

    // Opcode Definitions
    localparam OP_CONV  = 8'h01;
    localparam OP_DENSE = 8'h02;
    localparam OP_HALT  = 8'hFF;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 0;
            m_axi_arvalid <= 0;
            m_axi_rready <= 0;
            m_axi_araddr <= 0;
            tm_start <= 0;
            pc <= 0;
            // Config defaults
            cfg_input_addr <= 0;
            cfg_output_addr <= 0;
            cfg_weight_addr <= 0;
            cfg_img_width <= 0;
            cfg_img_height <= 0;
            cfg_in_channels <= 0;
            cfg_out_channels <= 0;
            cfg_quant_shift <= 0;
            cfg_is_conv <= 0;
	    cfg_relu_en <= 0;
	    cfg_input_bank <= 0;
	    cfg_output_bank <= 0;
	    cfg_stride <= 2'b01;
        end else begin
            // Default Pulses
            tm_start <= 0;
            done <= 0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        pc <= base_addr;
                        state <= S_FETCH_REQ;
                    end
                end

                // 1. Request Instruction
                S_FETCH_REQ: begin
                    m_axi_araddr <= pc;
                    m_axi_arvalid <= 1;
		    state <= S_FETCH_ACK;
                end

		S_FETCH_ACK: begin
		    if (m_axi_arready) begin
                        m_axi_arvalid <= 0; // Deassert
                        m_axi_rready <= 1;  // Ready to receive data
                        state <= S_FETCH_WAIT;
                    end
		end

                // 2. Receive Instruction
                S_FETCH_WAIT: begin
                    if (m_axi_rvalid) begin
                        instr_reg <= m_axi_rdata;
                        m_axi_rready <= 0;
                        state <= S_DECODE;
                    end
                end

                // 3. Decode Instruction
                S_DECODE: begin
                    // Map bits to configuration registers
                    // Layout:
                    // [63:0] Input Addr
                    // [127:64] Output Addr
                    // [191:128] Weight Addr
                    // [207:192] Width
                    // [223:208] Height
                    // [239:224] In Channels
                    // [255:240] Out Channels
                    // [263:256] Opcode
                    // [271:264] Quant Shift
		    // [279:272] Bank Select (2 LSB bits input, next 2 bits
		    // output)
		    // [280] ReLU Enable
                    
                    cfg_input_addr   <= instr_reg[63:0];
                    cfg_output_addr  <= instr_reg[127:64];
                    cfg_weight_addr  <= instr_reg[191:128];
                    cfg_img_width    <= instr_reg[207:192];
                    cfg_img_height   <= instr_reg[223:208];
                    cfg_in_channels  <= instr_reg[239:224];
                    cfg_out_channels <= instr_reg[255:240];
                    
                    // Opcode Check
                    if (opcode == OP_HALT) begin
			cfg_is_conv <= 0;
			cfg_quant_shift <= 0;
                        state <= S_DONE;
                    end else if (opcode == OP_CONV) begin
                        cfg_is_conv <= (opcode == OP_CONV);
			cfg_quant_shift <= instr_reg[268:264];
                        state <= S_EXECUTE;
		    end else begin
			$display("ERROR: Unknown Opcode %h at PC %h", opcode, pc);
			state <= S_DONE;
		    end

		    cfg_input_bank  <= instr_reg[273:272];
		    cfg_output_bank <= instr_reg[275:274];
		    cfg_relu_en     <= instr_reg[280];
		    cfg_stride      <= instr_reg[289:288];
		    cfg_log2_mem_tile_height <= instr_reg[298:296];
                end

                // 4. Execute Layer
                S_EXECUTE: begin
                    tm_start <= 1; // Pulse start to Tile Manager
                    state <= S_WAIT_TM;
                end

                // 5. Wait for Layer Completion
                S_WAIT_TM: begin
                    if (tm_done) begin
                        state <= S_NEXT_INSTR;
                    end
                end

                // 6. Update Program Counter
                S_NEXT_INSTR: begin
                    pc <= pc + 64; // Move 64 bytes (512 bits) forward
                    state <= S_FETCH_REQ;
                end

                S_DONE: begin
                    done <= 1;
                    // Wait for reset or new start
                    if (start) begin
                        pc <= base_addr;
                        state <= S_FETCH_REQ;
                    end
                end
            endcase
        end
    end

endmodule
