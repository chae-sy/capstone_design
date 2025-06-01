//top
module SPRNN_Top#(
    parameter DATA_WIDTH = 8,
    parameter NUM_COLOR = 3,
    parameter NUM_CHNL = 16,
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
                            in_buf_wren_o[0:NUM_COLOR-1],
                            in_buf_rden_o[0:NUM_COLOR-1],
                            is_initial,
                            out_buf_wren_o[0:NUM_COLOR-1],
                            out_buf_rden_o[0:NUM_COLOR-1];

    wire                    pe_en_o,
                            addtree_en_o,
                            relu_en_o,
                            maxpool_en_o,
                            color_o;

    wire                    pe_done_i,
                            addtree_done_i,
                            relu_done_i,
                            maxpool_done_i;

    wire                    channel;
    wire    [2:0]           layer_num;

    wire    [8:0]           wmem_addr;
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
        .is_initial         (is_initial),
        
        .out_buf_wren_o     (out_buf_wren_o),
        .out_buf_rden_o     (out_buf_rden_o),
        .out_buf_done_i     (out_buf_done_i),
        
        .pe_en_o            (pe_en_o),
        .pe_done_i          (pe_done_i),

        .addtree_en_o       (addtree_en_o),
        .addtree_done_i     (addtree_done_i),

        .relu_en_o          (relu_en_o),
        .relu_done_i        (relu_done_i),

        .maxpool_en_o       (maxpool_en_o),
        .maxpool_done_i     (maxpool_done_i),
        .color_o            (color_o),
        
        .channel            (channel),
        .total_done_o       (total_done_o),
        .layer_num          (layer_num)
    );


    /////////////////////// memory //////////////////////////

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

    //(4) maxpool -> memory //maxpool_output
    //(1,2,3,5,6) output buffer -> memory // data_out
    assign memA_d = (is_A_rd) ? 0 : ((layer_num == 4) ? maxpool_output : data_out);
    assign memB_d = (is_A_rd) ? ((layer_num == 4) ? maxpool_output : data_out): 0;
 
  
    wire is_A_rd;
    wire [DATA_WIDTH*NUM_CHNL-1:0] buf_in_data;

    assign is_A_rd = ((layer_num == 1)|(layer_num == 3)|(layer_num == 5)) ? 1'b1 : 1'b0;
    assign buf_in_data = is_A_rd ? memA_qout : memB_qout;
    
    
    wire [DATA_WIDTH*NUM_CHNL-1:0]  stage1_in_output[0:NUM_COLOR-1],
                                    stage1_weight_output;
    reg  [DATA_WIDTH*NUM_CHNL-1:0]  stage2_in_input[0:NUM_COLOR-1],
                                    stage2_weight_input;
    wire in_buf_done[0:NUM_CHNL-1];


    /////////////////////// buffer //////////////////////////

    f_buffer_v1      u_in_buf_red
    (
        .clk                (clk),
        .rst_n              (rst_n),
        .is_initial         (is_initial),
        .wren               (in_buf_wren_o[0]),
        .rden               (in_buf_rden_o[0]),
        .data_in            (buf_in_data),
        .data_out           (stage1_in_output[0]),
        .f_buffer_done      (in_buf_done[0])
    );

    f_buffer_v1      u_in_buf_green
    (
        .clk                (clk),
        .rst_n              (rst_n),
        .is_initial         (is_initial),
        .wren               (in_buf_wren_o[1]),
        .rden               (in_buf_rden_o[1]),
        .data_in            (buf_in_data),
        .data_out           (stage1_in_output[1]),
        .f_buffer_done      (in_buf_done[1])
    );

    f_buffer_v1      u_in_buf_blue
    (
        .clk                (clk),
        .rst_n              (rst_n),
        .is_initial         (is_initial),
        .wren               (in_buf_wren_o[2]),
        .rden               (in_buf_rden_o[2]),
        .data_in            (buf_in_data),
        .data_out           (stage1_in_output[2]),
        .f_buffer_done      (in_buf_done[2])
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
    
    //(memory -> buffer) => PE array 
    always_comb begin
        stage2_in_input = stage1_in_output;
        stage2_weight_input = stage1_weight_output;
    end 
    
    wire [DATA_WIDTH-1:0]       rgb_lane[0:NUM_COLOR-1];

    /////////////////////// PE array //////////////////////////
    
    genvar ch;
    generate // 16 번 불러옴.
        for (ch = 0; ch < NUM_CHNL; ch = ch + 1) begin : GEN_PE

            assign rgb_lane[0] = stage2_in_input[0][(ch+1)*DATA_WIDTH-1:ch*DATA_WIDTH];
            assign rgb_lane[1] = stage2_in_input[1][(ch+1)*DATA_WIDTH-1:ch*DATA_WIDTH];
            assign rgb_lane[2] = stage2_in_input[2][(ch+1)*DATA_WIDTH-1:ch*DATA_WIDTH];
                
            mac_pipeline_superscalar #(
                .DATA_WIDTH (DATA_WIDTH),
                .LANE_NUM   (NUM_COLOR)
            ) u_PE_array (
                .clk            (clk),
                .rst_n          (rst_n),
                .pe_en          (pe_en_o),
                .data_in        (rgb_lane),     
                .weight_in      (stage2_weight_input[(ch+1)*DATA_WIDTH-1:ch*DATA_WIDTH]),
                .pe_done        (pe_done[ch]),
                .result_out_flat(stage2_output[ch]) // r, g, b 다 포함, 한 채널
            );
        end
    endgenerate
    
    wire [19:0]             stage2_output[0:NUM_CHNL-1][0:NUM_COLOR-1];
    reg  [20*NUM_CHNL-1:0]  stage3_input[0:NUM_COLOR-1];

    wire pe_done[0:15];
    assign pe_done_i = pe_done[0] & pe_done[1] & pe_done[2] & pe_done[3] &
                   pe_done[4] & pe_done[5] & pe_done[6] & pe_done[7] &
                   pe_done[8] & pe_done[9] & pe_done[10] & pe_done[11] &
                   pe_done[12] & pe_done[13] & pe_done[14] & pe_done[15];

    // PE => add tree
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage3_input <= 0;
        end
        else if (pe_done_i) begin // valid == done
            stage3_input[0] <= {stage2_output[15][0], stage2_output[14][0], stage2_output[13][0], stage2_output[12][0],
                                stage2_output[11][0], stage2_output[10][0], stage2_output[9][0], stage2_output[8][0],
                                stage2_output[7][0], stage2_output[6][0], stage2_output[5][0], stage2_output[4][0],
                                stage2_output[3][0], stage2_output[2][0], stage2_output[1][0], stage2_output[0][0]};
                                
            stage3_input[1] <= {stage2_output[15][1], stage2_output[14][1], stage2_output[13][1], stage2_output[12][1],
                                stage2_output[11][1], stage2_output[10][1], stage2_output[9][1], stage2_output[8][1],
                                stage2_output[7][1], stage2_output[6][1], stage2_output[5][1], stage2_output[4][1],
                                stage2_output[3][1], stage2_output[2][1], stage2_output[1][1], stage2_output[0][1]};
                                
            stage3_input[2] <= {stage2_output[15][2], stage2_output[14][2], stage2_output[13][2], stage2_output[12][2],
                                stage2_output[11][2], stage2_output[10][2], stage2_output[9][2], stage2_output[8][2],
                                stage2_output[7][2], stage2_output[6][2], stage2_output[5][2], stage2_output[4][2],
                                stage2_output[3][2], stage2_output[2][2], stage2_output[1][2], stage2_output[0][2]};
        end
    end

    /////////////////////// adder tree ////////////////////////// 

    adder_tree u_adder_tree_r (
        .clk                (clk),
        .rst_n              (rst_n),
        .adder_tree_en      (addtree_en_o),
        .in_flat            (stage3_input[0]),
        .sum_out_flat       (stage3_output[0]),
        .adder_tree_done    (addtree_done[0])
    );

    adder_tree u_adder_tree_g (
        .clk                (clk),
        .rst_n              (rst_n),
        .adder_tree_en      (addtree_en_o),
        .in_flat            (stage3_input[1]),
        .sum_out_flat       (stage3_output[1]),
        .adder_tree_done    (addtree_done[1])
    );

    adder_tree u_adder_tree_b (
        .clk                (clk),
        .rst_n              (rst_n),
        .adder_tree_en      (addtree_en_o),
        .in_flat            (stage3_input[2]),
        .sum_out_flat       (stage3_output[2]), 
        .adder_tree_done    (addtree_done[2])
    );

    wire addtree_done[0:NUM_COLOR-1];
    assign addtree_done_i = addtree_done[0] & addtree_done[1] & addtree_done[2];

    wire [23:0] stage3_output[0:NUM_COLOR-1],
                stage4_input[0:NUM_COLOR-1];
    wire [DATA_WIDTH-1:0] stage4_output[0:NUM_COLOR-1];

    wire [NUM_COLOR*BIAS_WIDTH-1:0] read_data_bias;



    // add tree => (1,2,3,5,6) relu
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage4_input <= 0;
        end
        else if (addtree_done_i) begin // valid == done
            stage4_input <= stage3_output;
        end
    end
    

    // we need register file for bias
    // since biases are 32bit (!= weight, input 8bits)
    // and their amount is small, we don't need SRAM

    /////////////////////// relu & bias //////////////////////////

    register_file_single u_mem_bias(
        .clk                (clk),
        .rst_n              (rst_n),
        .we                 (wen_bias),
        .waddr              (write_addr_bias),
        .wdata              (write_data_bias),
        .raddr              (read_addr_bias),
        .rdata              (read_data_bias)
    );

    bias_relu      u_bias_relu_R
    (
        .relu_en             (relu_en_o),
        .relu_done           (relu_done[0]),
        .layer_state         (layer_num),
        .data_in             (stage4_input[0]),
        .bias                (read_data_bias[0*BIAS_WIDTH +: BIAS_WIDTH]),
        .data_out            (stage4_output[0])
    );
    
    bias_relu      u_bias_relu_G
    (
        .relu_en             (relu_en_o),
        .relu_done           (relu_done[1]),
        .layer_state         (layer_num),
        .data_in             (stage4_input[1]),
        .bias                (read_data_bias[1*BIAS_WIDTH +: BIAS_WIDTH]),
        .data_out            (stage4_output[1])
    );
    
    bias_relu      u_bias_relu_B
    (
        .relu_en             (relu_en_o),
        .relu_done           (relu_done[2]),
        .layer_state         (layer_num),
        .data_in             (stage4_input[2]),
        .bias                (read_data_bias[2*BIAS_WIDTH +: BIAS_WIDTH]),
        .data_out            (stage4_output[2])
    );

    // (1,2,3,5,6) relu ->  output_buffer
    always_comb begin
        case (layer_num)
            1,2,3,5,6: begin
                if (relu_done_i) begin
                    stage5_input = stage4_output;
                end
            end
        endcase
    end
     wire [DATA_WIDTH-1:0] stage5_input[0:NUM_COLOR-1];

    wire relu_done[0:NUM_COLOR-1];
    assign relu_done_i = (relu_done[0] & relu_done[1]) & relu_done[2];


    /////////////////////// maxpool //////////////////////////

    maxpool_16ch u_maxpool(
        .clk                 (clk),
        .rst_n               (rst_n),
        .maxpool_en          (maxpool_en_o),
        .color               (color_o),// r=0 (4x2), g=1 (4x1), b=2 (4x2)
        .in_data             (memA_qout),
        .maxpool_done_o      (maxpool_done_i),
        .out_data_o          (maxpool_output)
    );
    
    wire [DATA_WIDTH-1:0] maxpool_output;
    wire [DATA_WIDTH-1:0] data_out;


    /////////////////////// output buffer //////////////////////////

    output_buffer     u_out_buf
    (
        .clk                (clk),
        .rst_n              (rst_n),
        .wren               (out_buf_wren_o),
        .data_in_r          (stage5_input[0]),
        .data_in_g          (stage5_input[1]),
        .data_in_b          (stage5_input[2]),
        .rden               (out_buf_rden_o),
        .layer_num          (layer_num),
        .o_buffer_done      (out_buf_done_i)
        .data_out           (data_out)
    );





endmodule
