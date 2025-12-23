`default_nettype none
/*
 * burst_axi_write_master
 *
 * Writes a burst of data to a specified AXI address.
 * Designed for offloading results from on-chip memory to HBM/DDR.
 */
module burst_axi_write_master #(
    parameter AXI_ADDR_WIDTH = 64,
    parameter AXI_DATA_WIDTH = 512,
    parameter MAX_BURST_LEN  = 64
) (
    input wire clk,
    input wire rst_n, // Active low reset

    // --- Control Interface ---
    input wire start, // Pulse to initiate the burst write
    input wire [AXI_ADDR_WIDTH-1:0] base_address,
    input wire [15:0]               total_words_to_write,

    // --- Data Input Interface (FIFO-like) ---
    input wire [AXI_DATA_WIDTH-1:0] data_in,
    input wire                      data_valid, // User logic has data ready
    output wire                     data_ready, // Master is ready to accept data (WREADY && State)

    // --- AXI4 Master Write Interface ---
    // Write Address Channel (AW)
    output reg                            m_axi_awvalid,
    input wire                            m_axi_awready,
    output wire [AXI_ADDR_WIDTH-1:0]      m_axi_awaddr,
    output wire [7:0]                     m_axi_awlen,
    output wire [2:0]                     m_axi_awsize,
    output wire [1:0]                     m_axi_awburst,

    // Write Data Channel (W)
    output wire                           m_axi_wvalid,
    input wire                            m_axi_wready,
    output wire [AXI_DATA_WIDTH-1:0]      m_axi_wdata,
    output wire [AXI_DATA_WIDTH/8-1:0]    m_axi_wstrb,
    output wire                           m_axi_wlast,

    // Write Response Channel (B)
    input wire                            m_axi_bvalid,
    output reg                            m_axi_bready,
    input wire [1:0]                      m_axi_bresp,

    // --- Status ---
    output reg done // Pulse high when the entire transfer is complete
);

    // --- FSM Definition ---
    localparam IDLE           = 3'd0;
    localparam CALC_BURST_LEN = 3'd1;
    localparam SEND_AW        = 3'd2;
    localparam SEND_W         = 3'd3;
    localparam WAIT_B         = 3'd4;
    localparam DONE_S         = 3'd5;

    reg [2:0] state = IDLE;

    // --- Internal Registers ---
    reg [AXI_ADDR_WIDTH-1:0] address_reg;
    reg [7:0]                burst_len_reg;
    reg [31:0]               words_remaining;
    reg [15:0]               current_transaction_size;
    reg [15:0]               w_beat_count;

    // --- AXI Signal Assignments ---
    assign m_axi_awaddr  = address_reg;
    assign m_axi_awlen   = burst_len_reg;
    assign m_axi_awsize  = $clog2(AXI_DATA_WIDTH/8);
    assign m_axi_awburst = 2'b01; // INCR burst type

    // --- Write Data Channel Assignments ---
    // Pass data directly from input to AXI bus
    assign m_axi_wdata   = data_in;
    // Write Strobe: All ones (assuming full bus width writes)
    assign m_axi_wstrb   = {(AXI_DATA_WIDTH/8){1'b1}};
    
    // Valid logic: We are valid if the user has data AND we are in the SEND_W state
    assign m_axi_wvalid  = (state == SEND_W) && data_valid;
    
    // Last logic: High on the last beat of the current burst
    assign m_axi_wlast   = (w_beat_count == burst_len_reg);

    // --- User Interface Assignments ---
    // We are ready for data if the slave is ready AND we are actively sending
    assign data_ready    = (state == SEND_W) && m_axi_wready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            m_axi_awvalid <= 1'b0;
            m_axi_bready  <= 1'b0;
            done <= 1'b0;
            address_reg <= 0;
            burst_len_reg <= 0;
            words_remaining <= 0;
            current_transaction_size <= 0;
            w_beat_count <= 0;
        end else begin
            // Default assignments
            done <= 1'b0;

            case (state)
                IDLE: begin
                    m_axi_awvalid <= 1'b0;
                    m_axi_bready  <= 1'b0;
                    words_remaining <= 0;
                    if (start) begin
                        address_reg <= base_address;
                        words_remaining <= total_words_to_write;
                        state <= CALC_BURST_LEN;
                    end
                end

                CALC_BURST_LEN: begin
                    // Determine the length of the next burst
                    if (words_remaining > MAX_BURST_LEN) begin
                        burst_len_reg <= MAX_BURST_LEN - 1; // AWLEN is N-1
                        current_transaction_size <= MAX_BURST_LEN;
                    end else begin
                        burst_len_reg <= words_remaining[7:0] - 1;
                        current_transaction_size <= words_remaining;
                    end
                    
                    // Prepare for Address Phase
                    m_axi_awvalid <= 1'b1;
                    w_beat_count <= 0;
                    state <= SEND_AW;
                end

                SEND_AW: begin
                    // Wait for the write address to be accepted
                    if (m_axi_awready) begin
                        m_axi_awvalid <= 1'b0;
                        state <= SEND_W;
                    end
                end

                SEND_W: begin
                    // Handshake: Wait for both Master Valid (data_valid) and Slave Ready
                    if (m_axi_wvalid && m_axi_wready) begin
                        
                        // Check if this is the last beat of the burst
                        if (m_axi_wlast) begin
                            // Burst complete, wait for response
                            m_axi_bready <= 1'b1;
                            state <= WAIT_B;
                        end else begin
                            w_beat_count <= w_beat_count + 1;
                        end
                    end
                end

                WAIT_B: begin
                    // Wait for Write Response (BVALID)
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 1'b0; // Deassert ready
                        
                        // Update tracking registers
                        words_remaining <= words_remaining - current_transaction_size;
                        address_reg <= address_reg + (current_transaction_size * (AXI_DATA_WIDTH/8));

                        // Check if we are done with the total transfer
                        if (words_remaining == current_transaction_size) begin
                            done <= 1'b1;
                            state <= DONE_S;
                        end else begin
                            // More data to write, calculate next burst
                            state <= CALC_BURST_LEN;
                        end
                    end
                end

                DONE_S: begin
                    // Pulse done for one cycle
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
