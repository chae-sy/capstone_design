//top
module SPRNN_Top#(
    parameter DATA_WIDTH = 8,
    parameter NUM_COLOR = 3,
    parameter NUM_CHNL = 16,
    parameter SIZE_BUFFER_H   = 3, 
    parameter SIZE_BUFFER_W   = 4,
    parameter AD_TR_NUM_INPUTS = 16,
    parameter AD_TR_SUM_WIDTH = DATA_WIDTH + $clog2(AD_TR_NUM_INPUTS),
    parameter BIAS_WIDTH =32
    
)(
    input   wire            clk,
    input   wire            rst_n,
    input   wire            start,
    input   wire            data_in,
    output  wire            fin,
    output  reg             total_done_o  
);

    wire    [8:0]           wmem_addr_o;
    wire    [15:0]          memA_addr_o;
    wire    [15:0]          memB_addr_o;

    wire                    wmem_wenb_o,
                            wmem_cenb_o,
                            memA_wenb_o,
                            memA_cenb_o,
                            memB_wenb_o,
                            memB_cenb_o;

    wire                    wei_buff_wren_o,
                            wei_buff_rden_o,
                            in_buf_wren_o,
                            in_buf_rden_o,
                            in_buf_sel_o,
                            out_buf_wren_o,
                            out_buf_rden_o;

    wire                    pe_en_o,
                            relu_en_o,
                            maxpool_en_o;

    wire                    pe_done_i,
                            relu_done_i,
                            maxpool_done_i;

    wire                    channel;

    // Controller
    Controller  u_controller(
        .clk                (clk),
        .rst_n              (rst_n),

        .initial_SRAMw_done (initial_SRAMw_done),
        .initial_weight_done(initial_weight_done),

        .wmem_addr_o        (wmem_addr_o),
        .wmem_wenb_o        (wmem_wenb_o),
        .wmem_cenb_o        (wmem_cenb_o),

        .memA_addr_o        (memA_addr_o),
        .memA_wenb_o        (memA_wenb_o),
        .memA_cenb_o        (memA_cenb_o),

        .memB_addr_o        (memB_addr_o),
        .memB_wenb_o        (memB_wenb_o),
        .memB_cenb_o        (memB_cenb_o),

        .wei_buff_wren_o    (wei_buff_wren_o),
        .wei_buff_rden_o    (wei_buff_rden_o),
        
        .in_buf_wren_o      (in_buf_wren_o),
        .in_buf_rden_o      (in_buf_rden_o),
        .in_buf_sel_o       (in_buf_sel_o),
        
        .out_buf_wren_o     (out_buf_wren_o),
        .out_buf_rden_o     (out_buf_rden_o),
        
        .pe_en_o            (pe_en_o),
        .pe_done_i          (pe_done_i),

        .relu_en_o          (relu_en_o),
        .relu_done_i        (relu_done_i),

        .maxpool_en_o       (maxpool_en_o),
        .maxpool_done_i     (maxpool_done_i),
        
        .channel            (channel),
        .total_done_o       (total_done_o)
    );

    // memory
    memory_w_v0 u_memW(  // Data Storage weight
        .CLK                (memW_clk),
        .CEB                (wmem_cenb_o),
        .WEB                (wmem_wenb_o),
        .A                  (wmem_addr_o),
    	.D                  (memW_din),
    	.Q                  (memW_dout)
    );

    memory_w_v0 u_memA(  // Data Storage A
    	.CLK                (memA_clk),
        .CEB                (memA_cenb_o),
        .WEB                (memA_wenb_o),
        .A                  (memA_addr_o),
    	.D                  (memA_din),
    	.Q                  (memA_dout)
    );
    
    memory_w_v0 u_memB(  // Data Storage B
        .CLK                (memB_clk),
        .CEB                (memB_cenb_o),
        .WEB                (memB_wenb_o),
        .A                  (memB_addr_o),
    	.D                  (memB_din),
    	.Q                  (memB_dout)
    );        
        
    
    register_file_single u_mem_bias(
        .clk                (clk),
        .rst_n              (rst_n),
        .wen                (wen_bias),
        .addr               (write_addr_bias),
        .wdata              (write_data_bias),
        .addr_flat          (read_addr_bias),
        .rdata_flat         (read_data_bias)
     );

    // we need register file for bias
    // since biases are 32bit (!= weight, input 8bits)
    // and their amount is small, we don't need SRAM
    wire is_A;
    wire buffer_in_data = is_A ? memA_dout : memB_dout;
    
    wire [4:0] buffer_mode_f[NUM_COLOR-1:0];  // R = 0, G = 1, B = 2
    wire buffer_load_f[NUM_COLOR-1:0];  // R = 0, G = 1, B = 2
    
    wire [$clog2(SIZE_BUFFER_H)-1:0] buffer_ptr_h_f[NUM_COLOR-1:0];  // R = 0, G = 1, B = 2
    wire [$clog2(SIZE_BUFFER_W)-1:0] buffer_ptr_w_f[NUM_COLOR-1:0];  // R = 0, G = 1, B = 2
    
    wire buffer_start_f[NUM_COLOR-1:0];  // R = 0, G = 1, B = 2
    wire shift_f[NUM_COLOR-1:0];  // R = 0, G = 1, B = 2
    wire pad_en_f[NUM_COLOR-1:0];  // R = 0, G = 1, B = 2
    
    wire [DATA_WIDTH*NUM_CHNL-1:0]  f_buffer_out_red ;
    wire [DATA_WIDTH*NUM_CHNL-1:0] f_buffer_out_green;
    wire [DATA_WIDTH*NUM_CHNL-1:0] f_buffer_out_blue;


    //버퍼 2차원으로 만들어줘야함. 그리고 shift 신호 받지말고 알아서 안에서 count 세서 shift하는 걸로로
    f_buffer_v1      u_in_buf_red
    (
        .clk(clk),
        .rst_n(rst_n),
        .buffer_mode(buffer_mode_f[0]),
        .buffer_load_f(buffer_load_f[0]),
        .buffer_ptr_h_f(buffer_ptr_h_f[0]),
        .buffer_ptr_w_f(buffer_ptr_w_f[0]),
        .buffer_start(buffer_start_f[0]),
        .shift(shift_f[0]),
        .pad_en(pad_en_f[0]),
        .f_data_in(buffer_in_data),
        .f_buffer_out(f_buffer_out_red)
    );
     f_buffer_v1     u_in_buf_green
    (
        .clk(clk),
        .rst_n(rst_n),
        .buffer_mode(buffer_mode_f[1]),
        .buffer_load_f(buffer_load_f[1]),
        .buffer_ptr_h_f(buffer_ptr_h_f[1]),
        .buffer_ptr_w_f(buffer_ptr_w_f[1]),
        .buffer_start(buffer_start_f[1]),
        .shift(shift_f[1]),
        .pad_en(pad_en_f[1]),
        .f_data_in(buffer_in_data),
        .f_buffer_out(f_buffer_out_green)
    );
     f_buffer_v1     u_in_buf_blue
    (
        .clk(clk),
        .rst_n(rst_n),
        .buffer_mode(buffer_mode_f[2]),
        .buffer_load_f(buffer_load_f[2]),
        .buffer_ptr_h_f(buffer_ptr_h_f[2]),
        .buffer_ptr_w_f(buffer_ptr_w_f[2]),
        .buffer_start(buffer_start_f[2]),
        .shift(shift_f[2]),
        .pad_en(pad_en_f[2]),
        .f_data_in(buffer_in_data),
        .f_buffer_out(f_buffer_out_blue)
    );
    
    
    
    wire w_data = memW_dout; //w_buffer input
    wire [DATA_WIDTH*NUM_CHNL-1:0] w_buffer_out; //w_buffer output

    w_buffer_v1 u_w_buf
    ( 
    .clk(clk),
    .rst_n(rst_n),
    .buffer_mode(buffer_mode_w),
    .buffer_load_w(buffer_load_w),
    .buffer_ptr_h_w(buffer_ptr_h_w),
    .buffer_ptr_w_w(buffer_ptr_w_w),
    .buffer_start(buffer_start_w),
    .w_data(w_data),
    .w_buffer_out(w_buffer_out)
    );


    wire valid_in[NUM_CHNL-1:0] ;
    wire valid_out[NUM_CHNL-1:0];
     // PE array 
    genvar ch;
    generate
    for (ch = 0; ch < NUM_CHNL; ch = ch + 1) begin : GEN_PE
    
        wire [DATA_WIDTH-1:0] rgb_lane [0:NUM_COLOR-1];

        assign rgb_lane[0] = f_buffer_out_red  [ch*DATA_WIDTH +: DATA_WIDTH];
        assign rgb_lane[1] = f_buffer_out_green[ch*DATA_WIDTH +: DATA_WIDTH];
        assign rgb_lane[2] = f_buffer_out_blue [ch*DATA_WIDTH +: DATA_WIDTH];
            
        mac_pipeline_superscalar #(
            .DATA_WIDTH (DATA_WIDTH),
            .LANE_NUM   (NUM_COLOR)
        ) u_PE_array (
            .clk            (clk),
            .rst_n          (rst_n),
            .data_in        (rgb_lane [ch]),     
            .weight_in      (w_buffer_out[ch*DATA_WIDTH +: DATA_WIDTH]),
            .valid_in       (valid_in [ch]),
            .valid_out      (valid_out[ch]),
            .result_out_flat(pe_out_flat[ch*NUM_COLOR*2*DATA_WIDTH +: NUM_COLOR*2*DATA_WIDTH])
        );
    end
    endgenerate

        
    adder_tree_nlane_flat u_adder_tree (
        .clk                (clk),
        .rst_n              (rst_n),
        .in_flat            (pe_out_flat),
        .sum_flat           (ad_tr_out_flat)  
    );
    
    wire [NUM_COLOR*AD_TR_SUM_WIDTH-1:0] ad_tr_out_flat; 
    wire [NUM_COLOR*BIAS_WIDTH-1:0] read_data_bias;
    wire [DATA_WIDTH-1:0] relu_out_R;
    wire [DATA_WIDTH-1:0] relu_out_G;
    wire [DATA_WIDTH-1:0] relu_out_B;
 
    bias_relu      u_bias_relu_R
    (
        .relu_on             (relu_on),
        .layer_state         (layer_state),
        .data_in_flat        (ad_tr_out_flat[0*AD_TR_SUM_WIDTH +: AD_TR_SUM_WIDTH]),
        .bias                (read_data_bias[0*BIAS_WIDTH +: BIAS_WIDTH]),
        .data_out            (relu_out_R)
    );
    
    bias_relu      u_bias_relu_G
    (
        .relu_on             (relu_on),
        .layer_state         (layer_state),
        .data_in_flat        (ad_tr_out_flat[1*AD_TR_SUM_WIDTH +: AD_TR_SUM_WIDTH]),
        .bias                (read_data_bias[1*BIAS_WIDTH +: BIAS_WIDTH]),
        .data_out            (relu_out_G)
    );
    
    bias_relu      u_bias_relu_B
    (
        .relu_on             (relu_on),
        .layer_state         (layer_state),
        .data_in_flat        (ad_tr_out_flat[2*AD_TR_SUM_WIDTH +: AD_TR_SUM_WIDTH]),
        .bias                (read_data_bias[2*BIAS_WIDTH +: BIAS_WIDTH]),
        .data_out            (relu_out_B)
    );

    maxpool u_maxpool_red (
        .clk                 (clk),
        .rst_n               (rst_n),
        .maxpool_en          (maxpool_en_o),
        .color               (0),// r=0 (4x2), g=1 (4x1), b=2 (4x2)
        .in_data             (),
        .maxpool_done_o      (maxpool_done[0]),
        .out_data_o          ()
    );
    
    maxpool u_maxpool_green (
        .clk                 (clk),
        .rst_n               (rst_n),
        .maxpool_en          (maxpool_en_o),
        .color               (1),// r=0 (4x2), g=1 (4x1), b=2 (4x2)
        .in_data             (),
        .maxpool_done_o      (maxpool_done[1]),
        .out_data_o          ()
    );

    maxpool u_maxpool_blue (
        .clk                 (clk),
        .rst_n               (rst_n),
        .maxpool_en          (maxpool_en_o),
        .color               (2),// r=0 (4x2), g=1 (4x1), b=2 (4x2)
        .in_data             (),
        .maxpool_done_o      (maxpool_done[2]),
        .out_data_o          ()
    );

    assign maxpool_done_i = (maxpool_done[0] & maxpool_done[1]) & maxpool_done[2];

    o_buffer_v1     u_out_buf
    (
        .clk                 (clk),
        .rst_n               (rst_n)
    );
    
   


endmodule
