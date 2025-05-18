//top
module SPRNN_Top#(
    parameter DATA_WIDTH = 8,
    parameter NUM_COLOR = 3,
    parameter NUM_CHNL = 16,
    parameter SIZE_BUFFER_H   = 3, 
    parameter SIZE_BUFFER_W   = 4
    
)(
    input   wire                clk,
    input   wire                rst_n,
    input   wire                start,
    output   wire                fin
);



    memory_w_v0 u_memA(  // Data Storage A
    	.CLK(memA_clk),
        .CEB(memA_ceb),
        .WEB(memA_web),
        .A(memA_addr),
    	.D(memA_din),
    	.Q(memA_dout)
    );
    
    memory_w_v0 u_memB(  // Data Storage B
        .CLK(memB_clk),
        .CEB(memB_ceb),
        .WEB(memB_web),
        .A(memB_addr),
    	.D(memB_din),
    	.Q(memB_dout)
    );        
        
    memory_w_v0 u_memW(  // Data Storage weight
        .CLK(memW_clk),
        .CEB(memW_ceb),
        .WEB(memW_web),
        .A(memW_addr),
    	.D(memW_din),
    	.Q(memW_dout)
    );
    
    register_file u_mem_bias(
    .clk(clk),
    .rst_n(rst_n),
    .wen(wen_bias),
    .addr(addr_bias),
    .wdata(wdata_bias),
    .rdata(rdata_bias)
    );
    // we need register file for bias
    // since biases are 32bit (!= weight, input 8bits)
    // and their amount is small, we don't need SRAM

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

    wire b_data =rdata_bias; //b_buffer input
    wire [DATA_WIDTH*NUM_CHNL-1:0] b_buffer_out; //b_buffer output
    w_buffer_v1 u_bias_buf
    ( 
    .clk(clk),
    .rst_n(rst_n),
    .buffer_mode(buffer_mode_b),
    .buffer_load_w(buffer_load_b),
    .buffer_ptr_h_w(buffer_ptr_h_b),
    .buffer_ptr_w_w(buffer_ptr_w_b),
    .buffer_start(buffer_start_b),
    .w_data(b_data),
    .w_buffer_out(b_buffer_out)
    
);
    wire valid_in[NUM_CHNL-1:0] ;
    wire valid_out[NUM_CHNL-1:0];
    wire [2*DATA_WIDTH-1:0] pe_out [0:NUM_CHNL-1];

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
      .clk       ( clk            ),
      .rst_n     ( rst_n          ),
      .data_in   ( rgb_lane [ch]    ),     
      .weight_in (  w_buffer_out[ ch*DATA_WIDTH +: DATA_WIDTH ]   ),
      .valid_in  ( valid_in [ch]),
      .valid_out ( valid_out[ch]  ),
      .result_out( pe_out[ch] )
        );
    end
    endgenerate
    
    // 1) flat 벡터 선언: NUM_CHNL 개의 2*DATA_WIDTH 비트를 연속으로
wire [NUM_CHNL*2*DATA_WIDTH-1:0] pe_out_flat;

// 2) unpacked 배열→packed 벡터 매핑
genvar chnl;
generate
  for (chnl = 0; chnl < NUM_CHNL; chnl = chnl + 1) begin : FLATTEN_PE_OUT
    // ch번째 element를 flat 버스의 해당 비트 슬라이스에 할당
    assign pe_out_flat[ chnl*2*DATA_WIDTH +: 2*DATA_WIDTH ] = pe_out[chnl];
  end
endgenerate
    
    bias_relu_16chnl        u_bias_relu
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .data_in_flat(pe_out_flat),
        .bias(b_buffer_out),
        .data_out(relu_out)
    );
    
    maxpool_16chnl u_maxpool (
    .clk(clk),
    .rst_n(rst_n),
    .maxpool_en(maxpool_en),
    .color(color),
    .in_data(relu_out),
    .maxpool_done_o(maxpool_done_o),
    .out_data_o(maxpool_out)
   
    ); 
    
    
    out_buf     u_out_buf
    (
        .clk                        (clk),
        .rst_n                      (rst_n)
    );
    
    
    controller  u_controller
    (
        .clk                        (clk),
        .rst_n                      (rst_n),
        .buffer_mode_f              (buffer_mode_f),
        .buffer_load_f              (buffer_load_f),
        .buffer_ptr_h_f             (buffer_ptr_h_f),
        .buffer_ptr_w_f             (buffer_ptr_w_f),
        .buffer_start               (buffer_start_f),
        .shift                      (shift_f),
        .pad_en                     (pad_en_f),
        .valid_in                   (valid_in),


    );


endmodule
