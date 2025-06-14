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
    input   wire  [15:0]    memA_addr_i,
    input   wire  [15:0]    memB_addr_i,
    input   wire  [9:0]     wmem_addr_i,
    input   wire  [127:0]   memA_d_i,
    input   wire  [127:0]   memB_d_i,
    input   wire  [127:0]   wmem_d_i,
    input   wire            wren_bias_i,
    input   wire  [2:0]     write_addr_bias_i,
    input   wire  [511:0]   write_data_bias_i,
    input   wire            initial_SRAMw_done,
    input   wire            initial_weight_done,
    output  wire  [2:0]     layer_num_o,
    output  wire            layer_done_o,   
    output  wire            total_done_o  
);

    wire    [9:0]           wmem_addr_o;
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
                            memA_cenb,
                            memB_wenb,
                            memB_cenb;

    wire                    wei_buf_wren_o,
                            wei_buf_rden_o,
                            in_buf_wren_o[0:NUM_COLOR-1],
                            in_buf_rden_o[0:NUM_COLOR-1],
                            is_initial,
                            out_buf_wren_o[0:NUM_COLOR-1],
                            out_buf_rden_o[0:NUM_COLOR-1];

    wire                    pe_en_o,
                            addtree_en_o,
                            relu_en_o,
                            maxpool_en_o;
    wire [1:0]              color_o;

    wire                    pe_done_i,
                            addtree_done_i,
                            relu_done_i,
                            maxpool_done_i;

    wire                    channel;
    wire    [2:0]           layer_num;

    wire    [9:0]           wmem_addr;
    wire    [15:0]          memA_addr;
    wire    [15:0]          memB_addr;
    
    wire    [127:0]         wmem_din,
                            memA_din,
                            memB_din,
                            wmem_qout,
                            memA_qout,
                            memB_qout;

    wire    [127:0]         memA_d,
                            memB_d;
    
    wire                    is_A_rd;
    wire [DATA_WIDTH*NUM_CHNL-1:0]  buf_in_data;

    wire [DATA_WIDTH*NUM_CHNL-1:0]  stage1_in_output[0:NUM_COLOR-1],
                                    stage1_weight_output;
    reg  [DATA_WIDTH*NUM_CHNL-1:0]  stage2_in_input[0:NUM_COLOR-1],
                                    stage2_weight_input;
    wire [19:0]                     stage2_output[0:NUM_CHNL-1][0:NUM_COLOR-1];
    reg  [20*NUM_CHNL-1:0]          stage3_input[0:NUM_COLOR-1];
    wire signed [23:0]              stage3_output[0:NUM_COLOR-1];
    reg  signed [23:0]              stage4_input[0:NUM_COLOR-1];
    wire [DATA_WIDTH-1:0]           stage4_output[0:NUM_COLOR-1];
    reg  [DATA_WIDTH-1:0]           stage5_input[0:NUM_COLOR-1];
                                    
    wire                            in_buf_done[0:NUM_CHNL-1],
                                    w_buf_done,
                                    addtree_done[0:NUM_COLOR-1],
                                    relu_done[0:NUM_COLOR-1],
                                    pe_done[0:NUM_CHNL-1];
    
    wire [DATA_WIDTH-1:0] rgb_lane     [0:NUM_CHNL-1][0:2]; 

        
    reg  [2:0]                      read_addr_bias;
    reg                             rden_bias;
    wire [BIAS_WIDTH-1:0]           read_data_bias;
    
    wire [DATA_WIDTH*NUM_CHNL-1:0]  maxpool_output;
    wire [DATA_WIDTH*NUM_CHNL-1:0]  data_out;
    wire                            layer_start;
    
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

        .wei_buff_wren_o    (wei_buf_wren_o),
        .wei_buff_rden_o    (wei_buf_rden_o),
        
        .in_buf_wren_r      (in_buf_wren_o[0]),
        .in_buf_wren_g      (in_buf_wren_o[1]),
        .in_buf_wren_b      (in_buf_wren_o[2]),
        .in_buf_rden_r      (in_buf_rden_o[0]),
        .in_buf_rden_g      (in_buf_rden_o[1]),
        .in_buf_rden_b      (in_buf_rden_o[2]),        
        .is_initial_o       (is_initial),
        
        .out_buf_wren_r     (out_buf_wren_o[0]),
        .out_buf_wren_g     (out_buf_wren_o[1]),
        .out_buf_wren_b     (out_buf_wren_o[2]),
        .out_buf_rden_r     (out_buf_rden_o[0]),
        .out_buf_rden_g     (out_buf_rden_o[1]),
        .out_buf_rden_b     (out_buf_rden_o[2]),
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
        
        .layer_done_o       (layer_done_o),
        .total_done_o       (total_done_o),
        .layer_num_o        (layer_num),
        .layer_start_o      (layer_start)
    );
    assign  layer_num_o = layer_num;

    /////////////////////// memory //////////////////////////

    memory_w_v0 #(
    .addr_width (10), 
    .data_width (128),
    .wr_delay (8)
   )   u_memW(  // Data Storage weight
        .CLK                (clk),
        .CEB                (wmem_cenb),
        .WEB                (wmem_wenb),
        .A                  (wmem_addr),
    	.D                  (wmem_din),
    	.Q                  (wmem_qout)
    );

    memory_w_v0 u_memA(  // Data Storage A
    	.CLK                (clk),
        .CEB                (memA_cenb),
        .WEB                (memA_wenb),
        .A                  (memA_addr),
    	.D                  (memA_din),
    	.Q                  (memA_qout)
    );
    
    memory_w_v0 u_memB(  // Data Storage B
        .CLK                (clk),
        .CEB                (memB_cenb),
        .WEB                (memB_wenb),
        .A                  (memB_addr),
    	.D                  (memB_din),
    	.Q                  (memB_qout)
    );        
    
    assign wmem_cenb = (initial_weight_done & start) ? wmem_cenb_o : (start ? 0 : 1);
    assign wmem_wenb = (initial_weight_done & start) ? wmem_wenb_o : (start ? 0 : 1);
    assign memA_cenb = (initial_SRAMw_done & start) ? memA_cenb_o : (start ? 0 : 1);
    assign memA_wenb = (initial_SRAMw_done & start) ? memA_wenb_o : (start ? 0 : 1);
    assign memB_cenb = (initial_SRAMw_done & start) ? memB_cenb_o : (start ? 0 : 1);
    assign memB_wenb = (initial_SRAMw_done & start) ? memB_wenb_o : (start ? 0 : 1);

    assign wmem_addr = (initial_weight_done & start) ? wmem_addr_o : (start ? wmem_addr_i : 0);
    assign memA_addr = (initial_SRAMw_done & start) ? memA_addr_o : (start ? memA_addr_i : 0);
    assign memB_addr = (initial_SRAMw_done & start) ? memB_addr_o : (start ? memB_addr_i : 0);       

    assign wmem_din = wmem_d_i;
    assign memA_din = (initial_SRAMw_done & start) ? memA_d : (start ? memA_d_i : 0);
    assign memB_din = (initial_SRAMw_done & start) ? memB_d : (start ? memB_d_i : 0);

    //(4) maxpool -> memory //maxpool_output
    //(1,2,3,5,6) output buffer -> memory // data_out
    assign memA_d = (is_A_rd) ? 0 : ((layer_num == 4) ? ((maxpool_done_i) ? maxpool_output : 128'd0 ): data_out);
    assign memB_d = (is_A_rd) ? ((layer_num == 4) ? maxpool_output : data_out): 0;

    assign is_A_rd = ((layer_num == 1)|(layer_num == 3)|(layer_num == 5)) ? 1'b1 : 1'b0;
    assign buf_in_data = is_A_rd ? memA_qout : memB_qout;
    
    /////////////////////// buffer //////////////////////////

    f_buffer      u_in_buf_red
    (
        .clk                (clk),
        .rst_n              (rst_n),
        .is_initial         (is_initial),
        .wren               (in_buf_wren_o[0]),
        .rden               (in_buf_rden_o[0]),
        .data_in            (buf_in_data),
        .layer_start        (layer_start),
        .data_out           (stage1_in_output[0]),
        .f_buffer_done      (in_buf_done[0])
    );

    f_buffer      u_in_buf_green
    (
        .clk                (clk),
        .rst_n              (rst_n),
        .is_initial         (is_initial),
        .wren               (in_buf_wren_o[1]),
        .rden               (in_buf_rden_o[1]),
        .data_in            (buf_in_data),
        .layer_start        (layer_start),
        .data_out           (stage1_in_output[1]),
        .f_buffer_done      (in_buf_done[1])
    );

    f_buffer      u_in_buf_blue
    (
        .clk                (clk),
        .rst_n              (rst_n),
        .is_initial         (is_initial),
        .wren               (in_buf_wren_o[2]),
        .rden               (in_buf_rden_o[2]),
        .data_in            (buf_in_data),
        .layer_start        (layer_start),
        .data_out           (stage1_in_output[2]),
        .f_buffer_done      (in_buf_done[2])
    );

    w_buffer u_w_buf
    ( 
        .clk                (clk),
        .rst_n              (rst_n),
        .wren               (wei_buf_wren_o),
        .rden               (wei_buf_rden_o),
        .data_in            (wmem_qout),
        .layer_start        (layer_start),
        .data_out           (stage1_weight_output),
        .w_buffer_done      (w_buf_done)
    );
    
    //(memory -> buffer) => PE array 
    always @(*) begin
        stage2_in_input[0] = stage1_in_output[0];
        stage2_in_input[1] = stage1_in_output[1];
        stage2_in_input[2] = stage1_in_output[2];
        stage2_weight_input = stage1_weight_output;
    end 
    
    /////////////////////// PE array //////////////////////////
    
    genvar ch;
    generate // 16 channel
        for (ch = 0; ch < NUM_CHNL; ch = ch + 1) begin : GEN_PE

            assign rgb_lane[ch][0] = stage2_in_input[0][(ch+1)*DATA_WIDTH-1:ch*DATA_WIDTH];
            assign rgb_lane[ch][1] = stage2_in_input[1][(ch+1)*DATA_WIDTH-1:ch*DATA_WIDTH];
            assign rgb_lane[ch][2] = stage2_in_input[2][(ch+1)*DATA_WIDTH-1:ch*DATA_WIDTH];
        
            mac_pipeline_superscalar #(
                .DATA_WIDTH (DATA_WIDTH),
                .LANE_NUM   (NUM_COLOR)
            ) u_PE_array (
                .clk                (clk),
                .rst_n              (rst_n),
                .pe_en              (pe_en_o),
                .data_in_r          (rgb_lane[ch][0]),     
                .data_in_g          (rgb_lane[ch][1]),
                .data_in_b          (rgb_lane[ch][2]),
                .weight_in          ($signed(stage2_weight_input[(ch+1)*DATA_WIDTH-1:ch*DATA_WIDTH])),
                .layer_start        (layer_start),
                .pe_done            (pe_done[ch]),
                .result_out_flat_r  (stage2_output[ch][0]),
                .result_out_flat_g  (stage2_output[ch][1]),
                .result_out_flat_b  (stage2_output[ch][2])
            );
        end
    endgenerate
    
    assign pe_done_i = pe_done[0] & pe_done[1] & pe_done[2] & pe_done[3] &
                   pe_done[4] & pe_done[5] & pe_done[6] & pe_done[7] &
                   pe_done[8] & pe_done[9] & pe_done[10] & pe_done[11] &
                   pe_done[12] & pe_done[13] & pe_done[14] & pe_done[15];

    integer i;
    // PE => add tree
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i=0; i < NUM_COLOR; i=i+1) begin
            stage3_input[i] <= {20*NUM_CHNL{1'b0}};
            end
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
        .layer_start        (layer_start),
        .sum_out            (stage3_output[0]),
        .adder_tree_done    (addtree_done[0])
    );

    adder_tree u_adder_tree_g (
        .clk                (clk),
        .rst_n              (rst_n),
        .adder_tree_en      (addtree_en_o),
        .in_flat            (stage3_input[1]),
        .layer_start        (layer_start),
        .sum_out            (stage3_output[1]),
        .adder_tree_done    (addtree_done[1])
    );

    adder_tree u_adder_tree_b (
        .clk                (clk),
        .rst_n              (rst_n),
        .adder_tree_en      (addtree_en_o),
        .in_flat            (stage3_input[2]),
        .layer_start        (layer_start),
        .sum_out            (stage3_output[2]), 
        .adder_tree_done    (addtree_done[2])
    );

    assign addtree_done_i = addtree_done[0] & addtree_done[1] & addtree_done[2];

    // add tree => (1,2,3,5,6) relu
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i=0; i < NUM_COLOR; i=i+1) begin
                stage4_input[i] <= {24{1'b0}};
            end
            rden_bias <= 1'b0;
        end
        else if (addtree_done_i) begin // valid == done
            for (i=0; i < NUM_COLOR; i=i+1) begin
                stage4_input[i] <= stage3_output[i];
            end
            rden_bias <= 1'b1;
        end
        else begin
            rden_bias <= 1'b0;
        end
        case (layer_num)
            1: read_addr_bias <= 0;
            2: read_addr_bias <= 1;
            3: read_addr_bias <= 2;
            5: read_addr_bias <= 3;
            6: read_addr_bias <= 4;
            default: read_addr_bias <= 0;
        endcase
    end

    /////////////////////// relu & bias //////////////////////////

    regfile_sync u_mem_bias(
        .clk                (clk),
        .rst_n              (rst_n),
        .we                 (wren_bias_i),
        .waddr              (write_addr_bias_i),
        .wdata              (write_data_bias_i),
        .raddr              (read_addr_bias),
        .rden               (rden_bias),
        .rdata              (read_data_bias),
        .layer_num          (layer_num)
    );

    bias_relu      u_bias_relu_R
    (
        .relu_en             (relu_en_o),
        .relu_done           (relu_done[0]),
        .layer_state         (layer_num),
        .data_in             (stage4_input[0]),
        .bias                (read_data_bias),
        .data_out            (stage4_output[0])
    );
    
    bias_relu      u_bias_relu_G
    (
        .relu_en             (relu_en_o),
        .relu_done           (relu_done[1]),
        .layer_state         (layer_num),
        .data_in             (stage4_input[1]),
        .bias                (read_data_bias),
        .data_out            (stage4_output[1])
    );
    
    bias_relu      u_bias_relu_B
    (
        .relu_en             (relu_en_o),
        .relu_done           (relu_done[2]),
        .layer_state         (layer_num),
        .data_in             (stage4_input[2]),
        .bias                (read_data_bias),
        .data_out            (stage4_output[2])
    );

    // (1,2,3,5,6) relu ->  output_buffer
    always @(*) begin
        case (layer_num)
            1,2,3,5,6: begin
                if (relu_done_i) begin
                    for (i=0; i < NUM_COLOR; i=i+1) begin
                        stage5_input[i] = stage4_output[i];
                    end
                end
            end
        endcase
    end

    assign relu_done_i = (relu_done[0] & relu_done[1]) & relu_done[2];


    /////////////////////// maxpool //////////////////////////

    maxpool_16ch u_maxpool(
        .clk                 (clk),
        .rst_n               (rst_n),
        .maxpool_en          (maxpool_en_o),
        .color               (color_o),// r=0 (4x2), g=1 (4x1), b=2 (4x2)
        .in_data             (memB_qout),
        .layer_start         (layer_start),
        .maxpool_done_o      (maxpool_done_i),
        .out_data_o          (maxpool_output)
    );
    
    /////////////////////// output buffer //////////////////////////

    output_buffer     u_out_buf
    (
        .clk                (clk),
        .rst_n              (rst_n),
        .wren_r             (out_buf_wren_o[0]),
        .wren_g             (out_buf_wren_o[1]),
        .wren_b             (out_buf_wren_o[2]),
        .data_in_r          (stage5_input[0]),
        .data_in_g          (stage5_input[1]),
        .data_in_b          (stage5_input[2]),
        .rden_r             (out_buf_rden_o[0]),
        .rden_g             (out_buf_rden_o[1]),
        .rden_b             (out_buf_rden_o[2]),
        .layer_num          (layer_num),
        .layer_start        (layer_start),
        .o_buffer_done      (out_buf_done_i),
        .data_out           (data_out)
    );


endmodule