`default_nettype none

module vla_accel_top #(
    parameter IC_PAR = 16,
    parameter OC_PAR = 16,
    parameter PP_PAR = 8,
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

    // AXI Master Interface for Global Memory Reads for Feature Map (and
    // Instructions)
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

    wire        sched_arvalid;
    wire [63:0] sched_araddr;  // Offset from Scheduler
    wire [7:0]  sched_arlen;
    wire [2:0]  sched_arsize;
    wire [1:0]  sched_arburst;
    wire        sched_rready;

    // --- Internal AXI Signals for Feature Map Reader ---
    // (Ensure your burst_axi_read_master connects to these, not top-level directly)
    wire        feat_arvalid;
    wire [63:0] feat_araddr;   // Physical address from Tile Manager logic
    wire [7:0]  feat_arlen;
    wire [2:0]  feat_arsize;
    wire [1:0]  feat_arburst;
    wire        feat_rready;

    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] heap_base_addr;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_input_addr_w;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_reg_a_addr;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_reg_b_addr;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_foutput_addr;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_weight_input_addr_base;
    wire [1:0] cfg_input_bank, cfg_output_bank;

    wire [63:0] cfg_input_offset;
    wire [63:0] cfg_output_offset;
    wire [63:0] cfg_weight_offset;

    wire [63:0] base_hbm0 = heap_base_addr;
    wire [63:0] base_hbm1 = hbm_reg_a_addr;
    wire [63:0] base_hbm2 = hbm_reg_b_addr;
    wire [63:0] base_hbm3 = hbm_weight_input_addr_base;

    wire [63:0] current_input_base  = (cfg_input_bank  == 2'b00) ? base_hbm0 :
	    			      (cfg_input_bank  == 2'b01) ? base_hbm1 :
				      (cfg_input_bank  == 2'b10) ? base_hbm2 : base_hbm0;
    wire [63:0] current_output_base = (cfg_output_bank == 2'b00) ? base_hbm0 : 
	    			      (cfg_output_bank == 2'b01) ? base_hbm1 :
				      (cfg_output_bank == 2'b10) ? base_hbm2 : base_hbm0;

    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_read_addr = current_input_base + cfg_input_offset + (hbm_input_addr_w << 6);  // * 64 since data width is 512 bits = 64 bytes per burst
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_winput_addr;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_read_w_addr = hbm_weight_input_addr_base + cfg_weight_offset + (hbm_winput_addr << 6);
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_write_f_addr = current_output_base + cfg_output_offset + (hbm_foutput_addr << 6);

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
	.reg_heap_addr(heap_base_addr),
	.reg_buff_a_addr(hbm_reg_a_addr),
	.reg_buff_b_addr(hbm_reg_b_addr),
	.reg_weight_input_addr(hbm_weight_input_addr_base),
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

        .m_axi_arvalid(feat_arvalid),
        .m_axi_arready(m_axi_gmem_arready),
        .m_axi_araddr(feat_araddr),
        .m_axi_arlen(feat_arlen),
        .m_axi_arsize(feat_arsize),
	.m_axi_arburst(feat_arburst),
        .m_axi_rvalid(m_axi_gmem_rvalid),
        .m_axi_rready(feat_rready),
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

    wire tm_start, tm_done;
    wire [15:0] img_width, img_height, img_channels;
    wire [15:0] out_channels;
    wire [4:0] quant_shift;
    wire is_conv;
    wire [63:0] sched_fetch_addr = heap_base_addr + sched_araddr;

    assign m_axi_gmem_arvalid = sched_arvalid | feat_arvalid;
    assign m_axi_gmem_araddr = sched_arvalid ? sched_fetch_addr : hbm_read_addr;
    assign m_axi_gmem_arlen   = sched_arvalid ? sched_arlen  : feat_arlen;
    assign m_axi_gmem_arsize  = sched_arvalid ? sched_arsize : feat_arsize;
    assign m_axi_gmem_arburst = sched_arvalid ? sched_arburst : feat_arburst;
    assign m_axi_gmem_rready  = sched_rready | feat_rready;

    wire relu_en;
    wire [1:0] stride;
    wire [2:0] log2_mem_tile_height;
    wire is_sparse;
    wire [31:0] ic_tile_mask;
    wire [31:0] oc_tile_mask;
    wire flatten;

    instruction_scheduler #(
	.ADDR_WIDTH(C_M_AXI_GMEM_ADDR_WIDTH),
	.DATA_WIDTH(C_M_AXI_GMEM_DATA_WIDTH)
    ) instr_scheduler_inst (
	.clk(ap_clk),
	.rst_n(ap_rst_n),
	.start(start_process_internal),
	.base_addr(64'd0),
	.done(process_done_internal),

	.m_axi_arvalid(sched_arvalid),
	.m_axi_arready(m_axi_gmem_arready),
	.m_axi_araddr(sched_araddr),
	.m_axi_arlen(sched_arlen),
	.m_axi_arsize(sched_arsize),
	.m_axi_arburst(sched_arburst),
	.m_axi_rvalid(m_axi_gmem_rvalid),
	.m_axi_rready(sched_rready),
	.m_axi_rdata(m_axi_gmem_rdata),

	.tm_start(tm_start),
	.tm_done(tm_done),
	.cfg_input_addr(cfg_input_offset),
	.cfg_output_addr(cfg_output_offset),
	.cfg_weight_addr(cfg_weight_offset),
	.cfg_img_width(img_width),
	.cfg_img_height(img_height),
	.cfg_in_channels(img_channels),
	.cfg_out_channels(out_channels),
	.cfg_quant_shift(quant_shift),
	.cfg_is_conv(is_conv),
	.cfg_relu_en(relu_en),
	.cfg_stride(stride),
	.cfg_flatten(flatten),
	.cfg_log2_mem_tile_height(log2_mem_tile_height),
	.cfg_is_sparse(is_sparse),
	.cfg_ic_tile_mask(ic_tile_mask),
	.cfg_oc_tile_mask(oc_tile_mask),
	.cfg_input_bank(cfg_input_bank),
	.cfg_output_bank(cfg_output_bank)
    );

    wire [15:0] img_width_strips = (img_width < PP_PAR) ? 1 : (img_width + PP_PAR - 1) / PP_PAR;

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
	.start(tm_start),

	.full_img_width_strips(img_width_strips),
	.full_img_height(img_height),
	.full_img_channels(img_channels),
	.output_channels(out_channels),
	.feature_map_words(feature_map_words),
	.weight_words(weight_words),
	.fmap_out_words(output_feature_map_words),
	.quant_shift(quant_shift),
	.relu_en(relu_en),
	.stride(stride),
	.flatten(flatten),
	.is_conv(is_conv),
	.log2_mem_tile_height(log2_mem_tile_height),
	.is_sparse(is_sparse),
	.ic_tile_mask(ic_tile_mask),
	.oc_tile_mask(oc_tile_mask),
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
	.done(tm_done)
    );

endmodule
