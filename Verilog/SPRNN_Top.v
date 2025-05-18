//top
module SPRNN_Top
(
    input   wire                clk,
    input   wire                rst_n; 
);


    mem       u_mem
    (
        .clk                        (clk),
        .rst_n                      (rst_n),
    );
    SRAM_W32_A64 u_memA(  // Data Storage A
    	.CLK                        (memA_clk),
        .CEB                        (memA_ceb),
        .WEB                        (memA_web),
        .A                          (memA_addr),
    	.D                          (memA_din),
    	.Q                          (memA_dout),
    );
    
    SRAM_W32_A64 u_memB(  // Data Storage B
        .CLK                        (memB_clk),
        .CEB                        (memB_ceb),
        .WEB                        (memB_web),
        .A                          (memB_addr),
    	.D                          (memB_din),
    	.Q                          (memB_dout),
    );        
        
    
    SRAM_W32_A64 u_memW(  // Data Storage weight
        .CLK                        (memW_clk),
        .CEB                        (memW_ceb),
        .WEB                        (memW_web),
        .A                          (memW_addr),
    	.D                          (memW_din),
    	.Q                          (memW_dout),
    );

    wire buffer_in_data;
    assign buffer_in_data = (is_A) ? memA_dout : memB_dout;
    wire buffer_mode_f[2:0];
    wire buffer_load_f[2:0];
    wire buffer_ptr_h_f[2:0];
    wire buffer_ptr_w_f[2:0];
    wire buffer_start_f[2:0];
    wire shift_f[2:0];
    wire pad_en_f[2:0];
    wire f_buffer_out_red;
    wire f_buffer_out_green;
    wire f_buffer_out_blue;

    in_buf      u_in_buf_red
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
        in_buf      u_in_buf_green
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
        in_buf      u_in_buf_blue
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
    
    out_buf     u_out_buf
    (
        .clk                        (clk),
        .rst_n                      (rst_n),
    );

     // 3개 채널을 위한 PE array
    genvar ch;
    generate
    for (ch = 0; ch < NUM_CHNL; ch = ch + 1) begin : GEN_PE
        // 3개 lane짜리 unpacked array 생성
        wire [DATA_WIDTH-1:0] rgb_lane [0:LANE_NUM-1];

        // 각 색 채널을 rgb_lane 배열에 매핑
        assign rgb_lane[0] = f_buffer_out_red  [ch*DATA_WIDTH +: DATA_WIDTH];
        assign rgb_lane[1] = f_buffer_out_green[ch*DATA_WIDTH +: DATA_WIDTH];
        assign rgb_lane[2] = f_buffer_out_blue [ch*DATA_WIDTH +: DATA_WIDTH];

        PE_array #(
        .DATA_WIDTH (DATA_WIDTH),
        .LANE_NUM   (LANE_NUM)
        ) u_PE_array (
        .clk     (clk),
        .rst_n   (rst_n),
        .data_in (rgb_lane),       // unpacked array 그대로 연결
        .weight_in ( weight_in[ch*WIDTH_W_DATA +: WIDTH_W_DATA] ),
        .valid_in  ( valid_in[ch]  ),
        .valid_out ( valid_out[ch] ),
        .result_out( result_out[ch*(2*DATA_WIDTH) +: (2*DATA_WIDTH)] )
        );
    end
    endgenerate
    
    ReLU        u_ReLU
    (
        .clk                        (clk),
        .rst_n                      (rst_n),
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
