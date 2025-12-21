`default_nettype none
/*
 * burst_axi_read_master
 *
 * Reads a burst of data from a specified AXI address.
 * Designed for loading datasets from HBM into on-chip memory.
 */
module burst_axi_read_master #(
    parameter AXI_ADDR_WIDTH = 64,
    parameter AXI_DATA_WIDTH = 512,
    parameter MAX_BURST_LEN  = 64
) (
    input wire clk,
    input wire rst_n, // Active low reset

    // --- Control Interface ---
    input wire start, // Pulse to initiate the burst read
    input wire [AXI_ADDR_WIDTH-1:0] base_address,

    // --- AXI4 Master Read Interface ---
    output reg                            m_axi_arvalid,
    input wire                            m_axi_arready,
    output wire [AXI_ADDR_WIDTH-1:0]      m_axi_araddr,
    output wire [7:0]                     m_axi_arlen,

    output wire [2:0]                     m_axi_arsize,
    output wire [1:0]                     m_axi_arburst, 

    input wire                            m_axi_rvalid,
    output reg                            m_axi_rready,
    input wire [AXI_DATA_WIDTH-1:0]       m_axi_rdata,
    input wire                            m_axi_rlast,

    // --- Data Output Interface (FIFO-like) ---
    input  wire [15:0] 		 	  total_words_to_read,
    output wire                           data_valid,
    output wire [AXI_DATA_WIDTH-1:0]      data_out,

    // --- Status ---
    output reg done // Pulse high when the entire burst is complete
);

    // --- FSM Definition ---
    localparam IDLE         = 3'd0;
    localparam SEND_AR      = 3'd1;
    localparam WAIT_AR      = 3'd2;
    localparam READ_DATA    = 3'd3;
    localparam DONE_S       = 3'd4;
    localparam CALC_BURST_LEN = 3'd5;

    reg [2:0] state = IDLE;

    // --- Internal Registers ---
    reg [AXI_ADDR_WIDTH-1:0] address_reg;
    reg [7:0]                burst_len_reg;
    reg [31:0] 		     words_remaining;
    reg [15:0] 		     current_transaction_size;

    // --- AXI Signal Assignments ---
    assign m_axi_araddr = address_reg;
    assign m_axi_arlen = burst_len_reg;
    assign m_axi_arsize = $clog2(AXI_DATA_WIDTH/8);
    assign m_axi_arburst = 2'b01;

    // --- Data Output Assignments ---
    assign data_out = m_axi_rdata;
    assign data_valid = m_axi_rvalid && m_axi_rready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            m_axi_arvalid <= 1'b0;
            m_axi_rready <= 1'b0;
            done <= 1'b0;
            address_reg <= 0;
            burst_len_reg <= 0;
	    words_remaining <= 0;
        end else begin
            // Default assignments
            done <= 1'b0; // done is a pulse

            case (state)
                IDLE: begin
                    m_axi_arvalid <= 1'b0;
                    m_axi_rready <= 1'b0;
		    words_remaining <= 0;
                    if (start) begin
                        address_reg <= base_address;
			words_remaining <= total_words_to_read;
                        state <= CALC_BURST_LEN;
                    end
                end

		CALC_BURST_LEN: begin
		    if (words_remaining > MAX_BURST_LEN) begin
                         burst_len_reg <= MAX_BURST_LEN - 1; // ARLEN is N-1
                         current_transaction_size <= MAX_BURST_LEN;
                     end else begin
                         burst_len_reg <= words_remaining[7:0] - 1;
                         current_transaction_size <= words_remaining;
                     end
                     
                     m_axi_arvalid <= 1'b1;
                     state <= SEND_AR;
		end

                SEND_AR: begin
                    // Wait for the address to be accepted
                    if (m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready <= 1'b1; // Signal that we are ready to receive data
                        state <= READ_DATA;
                    end
                end

                // Note: Some AXI interconnects might deassert ARREADY before the first RVALID.
                // A more robust FSM might have a WAIT_AR state, but this is simpler and often works.

                READ_DATA: begin
                    if (m_axi_rvalid) begin // Data is available from the slave
                        if (m_axi_rlast) begin
                            // This is the last beat of the burst
                            m_axi_rready <= 1'b0;
			    words_remaining <= words_remaining - current_transaction_size;
			    address_reg <= address_reg + (current_transaction_size * (AXI_DATA_WIDTH/8));
			    if (words_remaining == current_transaction_size) begin
                                done <= 1'b1;
                                state <= DONE_S;
			    end else begin
				state <= CALC_BURST_LEN;
			    end
                        end
                    end
                end

                DONE_S: begin
                    // Stay here for one cycle to ensure done pulse width
                    state <= IDLE;
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
`default_nettype wire
