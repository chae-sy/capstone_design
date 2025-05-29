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
    input   wire            memA_addr_i,
    input   wire            wmem_addr_i,
    input   wire  [127:0]   memA_d_i,
    input   wire  [127:0]   wmem_d_i,
    input   wire            initial_SRAMw_done,
    input   wire            initial_weight_done,   
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

    wire                    wmem_wenb,
                            wmem_cenb,
                            memA_wenb,
                            memA_cenb;

    wire                    wei_buff_wren_o,
                            wei_buff_rden_o,
                            in_buf_wren_o,
                            in_buf_rden_o,
                            in_buf_sel_o,
                            out_buf_wren_o,
                            out_buf_rden_o;

    wire                    pe_en_o,
                            addtree_en_o,
                            relu_en_o,
                            maxpool_en_o;

    wire                    pe_done_i,
                            addtree_done_i,
                            relu_done_i,
                            maxpool_done_i;

    wire                    channel;

    wire    [8:0]           wmem_addr,
    wire    [15:0]          memA_addr;

    wire    [127:0]         wmem_din,
                            memA_din,
                            memB_din,
                            wmem_qout,
                            memA_qout,
                            memB_qout;

    wire    [127:0]         memA_d,
                            memB_d;
    

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

        .addtree_en_o       (addtree_en_o),
        .addtree_done_i     (addtree_done_i),

        .relu_en_o          (relu_en_o),
        .relu_done_i        (relu_done_i),

        .maxpool_en_o       (maxpool_en_o),
        .maxpool_done_i     (maxpool_done_i),
        
        .channel            (channel),
        .total_done_o       (total_done_o)
    );


    // memory
    memory_w_v0 u_memW(  // Data Storage weight
        .CLK                (wmem_clk),
        .CEB                (wmem_cenb),
        .WEB                (wmem_wenb),
        .A                  (wmem_addr),
    	.D                  (wmem_din),
    	.Q                  (wmem_qout)
    );

    memory_w_v0 u_memA(  // Data Storage A
    	.CLK                (memA_clk),
        .CEB                (memA_cenb),
        .WEB                (memA_wenb),
        .A                  (memA_addr),
    	.D                  (memA_din),
    	.Q                  (memA_qout)
    );
    
    memory_w_v0 u_memB(  // Data Storage B
        .CLK                (memB_clk),
        .CEB                (memB_cenb_o),
        .WEB                (memB_wenb_o),
        .A                  (memB_addr_o),
    	.D                  (memB_din),
    	.Q                  (memB_qout)
    );        
    
    assign wmem_cenb = (initial_weight_done & start) ? wmem_cenb_o : (start ? 1 : 0);
    assign wmem_wenb = (initial_weight_done & start) ? wmem_wenb_o : (start ? 1 : 0);
    assign memA_cenb = (initial_SRAMw_done & start) ? memA_cenb_o : (start ? 1 : 0);
    assign memA_wenb = (initial_SRAMw_done & start) ? memA_wenb_o : (start ? 1 : 0);

    assign wmem_addr = (initial_weight_done & start) ? wmem_addr_o : (start ? wmem_addr_i : 0);
    assign memA_addr = (initial_SRAMw_done & start) ? memA_addr_o : (start ? memA_addr_i : 0);    

    assign wmem_din = wmem_d_i;
    assign memA_din = (initial_SRAMw_done & start) ? memA_d : (start ? memA_d_i : 0);
    assign memB_din = memB_d;

    assign memA_d = ((layer_num == 2) | (layer_num == 4)) ? out_data : 0;
    assign memB_d = ((layer_num == 2) | (layer_num == 4)) ? 0 : out_data;
 

    
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
        
    wire w_data = memW_dout; //w_buffer input
    wire [DATA_WIDTH*NUM_CHNL-1:0] w_buffer_out; //w_buffer output


    // input buffer( R, G, B )
    f_buffer_v1      u_in_buf_red
    (
        .clk                (clk),
        .rst_n              (rst_n),
        .color_sel          (in_buf_sel_o[0]),
        .wren               (in_buf_wren_o[0]),
        .data_in            (),
        .rden               (),
        .data_out           (stage1_in_output[0])
    );
    // 이런 느낌 

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

    // (memory -> buffer) => PE array 여기도 수정해야함
    // always_ff @(posedge clk) begin
    //     if (pe_en_o) begin
    //         stage2_in_input <= stage1_in_output;
    //     end
    //     if (pe_en_o) begin
    //         stage2_weight_input <= stage1_weight_output;
    //     end
    // end 

    wire valid_in[NUM_CHNL-1:0] ;
    wire valid_out[NUM_CHNL-1:0];
     // PE array 
    genvar ch;
    generate
        for (ch = 0; ch < NUM_CHNL; ch = ch + 1) begin : GEN_PE
        
            wire [DATA_WIDTH-1:0] rgb_lane [0:NUM_COLOR-1];

            assign rgb_lane[0] = stage2_in_input  [ch*DATA_WIDTH +: DATA_WIDTH];
            assign rgb_lane[1] = f_buffer_out_green[ch*DATA_WIDTH +: DATA_WIDTH];
            assign rgb_lane[2] = f_buffer_out_blue [ch*DATA_WIDTH +: DATA_WIDTH];
                
            mac_pipeline_superscalar #(
                .DATA_WIDTH (DATA_WIDTH),
                .LANE_NUM   (NUM_COLOR)
            ) u_PE_array (
                .clk            (clk),
                .rst_n          (rst_n),
                .data_in        (rgb_lane [ch]),     
                .weight_in      (stage2_weight_input[ch*DATA_WIDTH +: DATA_WIDTH]),
                .valid_in       (valid_in [ch]),
                .valid_out      (valid_out[ch]),
                .result_out_flat(pe_out_flat[ch*NUM_COLOR*2*DATA_WIDTH +: NUM_COLOR*2*DATA_WIDTH])
            );
        end
    endgenerate

    // PE => add tree
    always_ff @(posedge clk or negedge rst_n) begin
        if (stage2_valid) begin // valid == done
            stage3_input <= stage2_output;
        end
    end

        
    adder_tree_nlane_flat u_adder_tree (
        .clk                (clk),
        .rst_n              (rst_n),
        .in_flat            (pe_out_flat),
        .sum_flat           (ad_tr_out_flat) stage3_output 
    );
    
    wire [NUM_COLOR*AD_TR_SUM_WIDTH-1:0] ad_tr_out_flat; 
    wire [NUM_COLOR*BIAS_WIDTH-1:0] read_data_bias;
    wire [DATA_WIDTH-1:0] relu_out_R;
    wire [DATA_WIDTH-1:0] relu_out_G;
    wire [DATA_WIDTH-1:0] relu_out_B;

    // add tree => (1,2,3,4) relu/ (5) output buffer
    always_ff @(posedge clk or negedge rst_n) begin
        if (stage3_valid) begin // valid == done
            stage4_input <= stage3_output;
        end
    end

    register_file_single u_mem_bias(
        .clk                (clk),
        .rst_n              (rst_n),
        .wen                (wen_bias),
        .addr               (write_addr_bias),
        .wdata              (write_data_bias),
        .addr_flat          (read_addr_bias),
        .rdata_flat         (read_data_bias)
     );
 
    bias_relu      u_bias_relu_R
    (
        .relu_on             (relu_en_o),
        .layer_state         (layer_state),
        .data_in_flat        (stage3_in_input[0*AD_TR_SUM_WIDTH +: AD_TR_SUM_WIDTH]), stage3_input
        .bias                (read_data_bias[0*BIAS_WIDTH +: BIAS_WIDTH]),
        .data_out            (stage4_output[0])
    );
    
    bias_relu      u_bias_relu_G
    (
        .relu_on             (relu_en_o),
        .layer_state         (layer_state),
        .data_in_flat        (ad_tr_out_flat[1*AD_TR_SUM_WIDTH +: AD_TR_SUM_WIDTH]),
        .bias                (read_data_bias[1*BIAS_WIDTH +: BIAS_WIDTH]),
        .data_out            (stage4_output[1])
    );
    
    bias_relu      u_bias_relu_B
    (
        .relu_on             (relu_en_o),
        .layer_state         (layer_state),
        .data_in_flat        (ad_tr_out_flat[2*AD_TR_SUM_WIDTH +: AD_TR_SUM_WIDTH]),
        .bias                (read_data_bias[2*BIAS_WIDTH +: BIAS_WIDTH]),
        .data_out            (stage4_output[2])
    );
    // relu -> (1,2,4) output_buffer/(3) maxpool
    always_ff @( posedge clk or negedge rst_n) begin
        case (layer_num)
            1, 2, 3, 4: begin
                if (relu_done_i) begin
                    stage5_input <= stage4_output;
                end
            end
            default: begin
                
            end
        endcase
    end

    maxpool u_maxpool_red (
        .clk                 (clk),
        .rst_n               (rst_n),
        .maxpool_en          (maxpool_en_o),
        .color               (0),// r=0 (4x2), g=1 (4x1), b=2 (4x2)
        .in_data             (stage5_input[0]),
        .maxpool_done_o      (maxpool_done[0]),
        .out_data_o          (stage5_output[0])
    );
    
    maxpool u_maxpool_green (
        .clk                 (clk),
        .rst_n               (rst_n),
        .maxpool_en          (maxpool_en_o),
        .color               (1),// r=0 (4x2), g=1 (4x1), b=2 (4x2)
        .in_data             (stage5_input[1]),
        .maxpool_done_o      (maxpool_done[1]),
        .out_data_o          (stage5_output[1])
    );

    maxpool u_maxpool_blue (
        .clk                 (clk),
        .rst_n               (rst_n),
        .maxpool_en          (maxpool_en_o),
        .color               (2),// r=0 (4x2), g=1 (4x1), b=2 (4x2)
        .in_data             (stage5_input[2]),
        .maxpool_done_o      (maxpool_done[2]),
        .out_data_o          (stage5_output[2])
    );

    assign maxpool_done_i = (maxpool_done[0] & maxpool_done[1]) & maxpool_done[2];
    
    // (3) maxpool -> output_buffer
    // always_ff @( posedge clk or negedge rst_n) begin
    //     case (layer_num)
    //         3: begin
    //             if (maxpool_done_i) begin
    //                 stage5_input <= stage4_output;
    //             end
    //         end
    //         default: begin
    //         end
    //     endcase
    // end

    // (1,2,4) relu/(3) maxpool/(5) PE+adder == output buffer 여기 부분 타이밍 다시 계산해야함.
    always_ff @( posedge clk or negedge rst_n) begin
        case(layer_num)
            3: begin 
                for (int i=0; i < 3; i = i + 1) begin
                    if (maxpool_done[i]) begin 
                        out_data[i] = stage5_output[i];    
                    end
                    else begin
                        out_data = 0; // padding
                    end
                end
            end
            5: begin
                for (int i=0; i < 3; i = i + 1) begin
                    if (pe_done[i]) begin 
                        out_data[i] = stage3_output[i];    
                    end
                end
                if (addtree_done) begin 
                    out_data = stage3_output;
                end
            end
            default: begin 
                for (int i=0; i < 3; i = i + 1) begin
                    if (relu_done[i]) begin 
                        out_data[i] = stage4_output[i];    
                    end
                end
                if (relu_done) begin 
                    out_data = stage4_output;
                end
            end
        endcase
    end

    o_buffer_v1     u_out_buf
    (
        .clk                (clk),
        .rst_n              (rst_n),
        .color_sel          (out_buf_sel_o[0]),
        .wren               (out_buf_wren_o),
        .data_in_R          (out_data[0]),
        .data_in_G          (out_data[1]),
        .data_in_B          (out_data[2]),
        .rden               (out_buf_rden_o),
        .data_out           ()
    );





endmodule
