`default_nettype none

module vla_accel_top #(
    parameter IMG_W = 8,
    parameter IMG_H = 8,
    parameter IMG_C = 32,

    parameter OC    = 32,

    parameter IC_PAR = 16,
    parameter OC_PAR = 16,
    parameter PP_PAR = 4,
    parameter DATA_WIDTH = 8,

    parameter MAX_IMG_WIDTH = 128,
    parameter TILE_HEIGHT = 4,
    
    parameter C_M_AXI_GMEM_ADDR_WIDTH = 64,
    parameter C_M_AXI_GMEM_DATA_WIDTH = 512,

    parameter C_S_AXI_ADDR_WIDTH = 8,
    parameter C_S_AXI_DATA_WIDTH = 32

) (
    input  wire                                         ap_clk,
    input  wire                                         ap_rst_n,
    input  wire  [C_S_AXI_ADDR_WIDTH-1:0]            	s_axi_control_awaddr,
    input  wire                                         s_axi_control_awvalid,
    output wire                                         s_axi_control_awready,
    input  wire  [C_S_AXI_DATA_WIDTH-1:0]             	s_axi_control_wdata,
    input  wire  [C_S_AXI_DATA_WIDTH/8-1:0]           	s_axi_control_wstrb,
    input  wire                                         s_axi_control_wvalid,
    output wire                                         s_axi_control_wready,
    output wire  [1:0]                                	s_axi_control_bresp,
    output wire                                         s_axi_control_bvalid,
    input  wire                                         s_axi_control_bready,
    input  wire  [C_S_AXI_ADDR_WIDTH-1:0]             	s_axi_control_araddr,
    input  wire                                         s_axi_control_arvalid,
    output wire                                         s_axi_control_arready,
    output wire  [C_S_AXI_DATA_WIDTH-1:0]             	s_axi_control_rdata,
    output wire  [1:0]                                	s_axi_control_rresp,
    output wire                                         s_axi_control_rvalid,
    input  wire                                         s_axi_control_rready,

    // AXI Master Interface for Global Memory Reads for Feature Map
    output wire       					m_axi_gmem_arvalid,
    input  wire                                         m_axi_gmem_arready,
    output wire  [C_M_AXI_GMEM_ADDR_WIDTH-1:0]          m_axi_gmem_araddr,
    output wire  [7:0]                                  m_axi_gmem_arlen,
    output wire  [2:0]                                  m_axi_gmem_arsize,
    output wire  [1:0]                                  m_axi_gmem_arburst,
    input  wire                                         m_axi_gmem_rvalid,
    output wire                                         m_axi_gmem_rready,
    input  wire  [C_M_AXI_GMEM_DATA_WIDTH-1:0]          m_axi_gmem_rdata,
    input  wire       					m_axi_gmem_rlast,

    // AXI Master Interface for Global Memory Reads for Weights
    output wire       					m_axi_wgmem_arvalid,
    input  wire                                         m_axi_wgmem_arready,
    output wire  [C_M_AXI_GMEM_ADDR_WIDTH-1:0]          m_axi_wgmem_araddr,
    output wire  [7:0]                                  m_axi_wgmem_arlen,
    output wire  [2:0]                                  m_axi_wgmem_arsize,
    output wire  [1:0]                                  m_axi_wgmem_arburst,
    input  wire                                         m_axi_wgmem_rvalid,
    output wire                                         m_axi_wgmem_rready,
    input  wire  [C_M_AXI_GMEM_DATA_WIDTH-1:0]          m_axi_wgmem_rdata,
    input  wire       					m_axi_wgmem_rlast,

    // AXI Master Interface for Global Memory Writes for Output Feature Map
    output wire       					m_axi_fo_gmem_awvalid,
    input  wire                                         m_axi_fo_gmem_awready,
    output wire  [C_M_AXI_GMEM_ADDR_WIDTH-1:0]          m_axi_fo_gmem_awaddr,
    output wire  [7:0]                                  m_axi_fo_gmem_awlen,
    output wire  [2:0]                                  m_axi_fo_gmem_awsize,
    output wire  [1:0]                                  m_axi_fo_gmem_awburst,
    output wire                                         m_axi_fo_gmem_wvalid,
    input  wire                                         m_axi_fo_gmem_wready,
    output wire  [C_M_AXI_GMEM_DATA_WIDTH-1:0]          m_axi_fo_gmem_wdata,
    output wire  [C_M_AXI_GMEM_DATA_WIDTH/8-1:0]        m_axi_fo_gmem_wstrb,
    output wire                                         m_axi_fo_gmem_wlast,
    input  wire  [1:0]                                	m_axi_fo_gmem_bresp,
    input  wire                                         m_axi_fo_gmem_bvalid,
    output wire                                         m_axi_fo_gmem_bready
); 

    // Start, ready and done signals
    wire start_process_internal; // Pulse from controller to start core logic
    wire process_done_internal; // Signal from core logic to controller
    reg ap_idle_reg;
    reg ap_done_reg;
    wire ap_idle = ap_idle_reg;
    wire ap_ready = ap_done_reg;
    always @(posedge ap_clk or negedge ap_rst_n) begin
	if (!ap_rst_n) begin
            ap_idle_reg <= 1'b1;
	end else begin
	    ap_idle_reg <= process_done_internal ? 1'b1 :
		   	   start_process_internal ? 1'b0 : ap_idle;
	end
    end

    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_feat_input_addr_base;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_input_addr_w;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_read_addr = hbm_feat_input_addr_base + (hbm_input_addr_w << 6);  // * 128 since data width is 1024 bits = 128 bytes per burst
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_weight_input_addr_base;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_winput_addr;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_read_w_addr = hbm_weight_input_addr_base + (hbm_winput_addr << 6);  // * 128 since data width is 1024 bits = 128 bytes per burst
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_output_addr_base;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_foutput_addr;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_write_f_addr = hbm_output_addr_base + (hbm_foutput_addr << 6);  // * 128 since data width is 1024 bits = 128 bytes per burst

    vla_accel_control_s_axi #(
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH),
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH)
    ) axi_controller_inst (
        .ACLK(ap_clk),
        .ARESET(ap_rst_n), // Provide active high reset
        .ACLK_EN(1'b1), // Assuming clock enable is always high
        .AWADDR(s_axi_control_awaddr), 
	.AWVALID(s_axi_control_awvalid),
	.AWREADY(s_axi_control_awready),
        .WDATA(s_axi_control_wdata), 
	.WSTRB(s_axi_control_wstrb), 
	.WVALID(s_axi_control_wvalid), 
	.WREADY(s_axi_control_wready),
        .BRESP(s_axi_control_bresp),
	.BVALID(s_axi_control_bvalid),
	.BREADY(s_axi_control_bready),
        .ARADDR(s_axi_control_araddr),
	.ARVALID(s_axi_control_arvalid),
	.ARREADY(s_axi_control_arready),
        .RDATA(s_axi_control_rdata),
	.RRESP(s_axi_control_rresp),
	.RVALID(s_axi_control_rvalid),
	.RREADY(s_axi_control_rready),
	.reg_feat_input_addr(hbm_feat_input_addr_base),
	.reg_weight_input_addr(hbm_weight_input_addr_base),
	.reg_feat_output_addr(hbm_output_addr_base),
        .user_start(start_process_internal), // Controller generates start pulse
        .user_done(process_done_internal),     // Core logic signals completion
	.user_idle(ap_idle),
	.user_ready(ap_ready)
    );

    wire read_master_data_valid;
    wire hbm_read_start;
    wire hbm_load_done;
    wire [C_M_AXI_GMEM_DATA_WIDTH-1:0] read_master_data_out;
    wire [15:0] feature_map_words;

    burst_axi_read_master #(
	.AXI_ADDR_WIDTH(C_M_AXI_GMEM_ADDR_WIDTH),
	.AXI_DATA_WIDTH(C_M_AXI_GMEM_DATA_WIDTH)
    ) hbm_feat_read_master_inst (
        .clk(ap_clk),
        .rst_n(ap_rst_n),
        .start(hbm_read_start),
	.base_address(hbm_read_addr),
        .m_axi_arvalid(m_axi_gmem_arvalid),
        .m_axi_arready(m_axi_gmem_arready),
        .m_axi_araddr(m_axi_gmem_araddr),
        .m_axi_arlen(m_axi_gmem_arlen),
        .m_axi_arsize(m_axi_gmem_arsize),
	.m_axi_arburst(m_axi_gmem_arburst),
        .m_axi_rvalid(m_axi_gmem_rvalid),
        .m_axi_rready(m_axi_gmem_rready),
        .m_axi_rdata(m_axi_gmem_rdata),
        .m_axi_rlast(m_axi_gmem_rlast),
	.total_words_to_read(feature_map_words),
        .data_valid(read_master_data_valid),
        .data_out(read_master_data_out),
        .done(hbm_load_done)
    );

    wire read_master_weights_data_valid;
    wire hbm_wread_start;
    wire hbm_wload_done;
    wire hbm_fwrite_start;
    wire [C_M_AXI_GMEM_DATA_WIDTH-1:0] read_master_weights_data_out;
    wire [15:0] weight_words;

    burst_axi_read_master #(
	.AXI_ADDR_WIDTH(C_M_AXI_GMEM_ADDR_WIDTH),
	.AXI_DATA_WIDTH(C_M_AXI_GMEM_DATA_WIDTH)
    ) hbm_weights_read_master_inst (
        .clk(ap_clk),
        .rst_n(ap_rst_n),
        .start(hbm_wread_start),
	.base_address(hbm_read_w_addr),
        .m_axi_arvalid(m_axi_wgmem_arvalid),
        .m_axi_arready(m_axi_wgmem_arready),
        .m_axi_araddr(m_axi_wgmem_araddr),
        .m_axi_arlen(m_axi_wgmem_arlen),
        .m_axi_arsize(m_axi_wgmem_arsize),
	.m_axi_arburst(m_axi_wgmem_arburst),
        .m_axi_rvalid(m_axi_wgmem_rvalid),
        .m_axi_rready(m_axi_wgmem_rready),
        .m_axi_rdata(m_axi_wgmem_rdata),
        .m_axi_rlast(m_axi_wgmem_rlast),
	.total_words_to_read(weight_words),
        .data_valid(read_master_weights_data_valid),
        .data_out(read_master_weights_data_out),
        .done(hbm_wload_done)
    );

    wire [15:0] output_feature_map_words;
    wire [C_M_AXI_GMEM_DATA_WIDTH-1:0] write_master_fmap_data_in;
    wire write_master_fmap_data_valid;
    wire write_master_fmap_data_ready;
    wire hbm_fwrite_done;
    burst_axi_write_master #(
	.AXI_ADDR_WIDTH(C_M_AXI_GMEM_ADDR_WIDTH),
	.AXI_DATA_WIDTH(C_M_AXI_GMEM_DATA_WIDTH)
    ) hbm_output_wr_master_inst (
	.clk(ap_clk),
	.rst_n(ap_rst_n),
	.start(hbm_fwrite_start),
	.base_address(hbm_write_f_addr),
	.total_words_to_write(output_feature_map_words),
	.data_in(write_master_fmap_data_in),
	.data_valid(write_master_fmap_data_valid),
	.data_ready(write_master_fmap_data_ready),
	.m_axi_awvalid(m_axi_fo_gmem_awvalid),
	.m_axi_awready(m_axi_fo_gmem_awready),
	.m_axi_awaddr(m_axi_fo_gmem_awaddr),
	.m_axi_awlen(m_axi_fo_gmem_awlen),
	.m_axi_awsize(m_axi_fo_gmem_awsize),
	.m_axi_awburst(m_axi_fo_gmem_awburst),
	.m_axi_wvalid(m_axi_fo_gmem_wvalid),
	.m_axi_wready(m_axi_fo_gmem_wready),
	.m_axi_wdata(m_axi_fo_gmem_wdata),
	.m_axi_wstrb(m_axi_fo_gmem_wstrb),
	.m_axi_wlast(m_axi_fo_gmem_wlast),
	.m_axi_bvalid(m_axi_fo_gmem_bvalid),
	.m_axi_bready(m_axi_fo_gmem_bready),
	.m_axi_bresp(m_axi_fo_gmem_bresp),
	.done(hbm_fwrite_done)
    );

    tile_manager #(
	.IC_PAR(IC_PAR),
	.OC_PAR(OC_PAR),
	.PP_PAR(PP_PAR),
	.DATA_WIDTH(DATA_WIDTH),
	.MAX_IMG_WIDTH(MAX_IMG_WIDTH),
	.TILE_HEIGHT(TILE_HEIGHT),
	.GMEM_DATA_WIDTH(C_M_AXI_GMEM_DATA_WIDTH)
    ) tile_manager_inst (
	.clk(ap_clk),
	.rst_n(ap_rst_n),
	.start(start_process_internal),

	.full_img_width_strips(IMG_W / PP_PAR),
	.full_img_height(IMG_H),
	.full_img_channels(IMG_C),
	.output_channels(OC),
	.feature_map_words(feature_map_words),
	.weight_words(weight_words),
	.fmap_out_words(output_feature_map_words),
	// FMap HBM interface	
	.hbm_data_in(read_master_data_out),
	.hbm_addr(hbm_input_addr_w),
	.hbm_ren(hbm_read_start),
	.hbm_rvalid(read_master_data_valid),
	// Weights HBM interface
	.hbm_data_in_w(read_master_weights_data_out),
	.hbm_addr_w(hbm_winput_addr),
	.hbm_ren_w(hbm_wread_start),
	.hbm_rvalid_w(read_master_weights_data_valid),
	// Output FMap HBM interface
	.hbm_fmap_out_data(write_master_fmap_data_in),
	.hbm_fmap_out_addr(hbm_foutput_addr),
	.hbm_fmap_out_wen(hbm_fwrite_start),
	.hbm_fmap_wvalid(write_master_fmap_data_valid),
	.hbm_fmap_out_ready(write_master_fmap_data_ready),
	.hbm_fmap_out_done(hbm_fwrite_done),
	.done(process_done_internal)
    );

endmodule
