`timescale 1ns / 1ps
//Controller
module Controller#(
    //parameter           MEM_BIT_LEN = $clog2(100),
    parameter           MA_BIT_LEN = $clog2(480000),
    parameter           MB_BIT_LEN = $clog2(480000),
    parameter           WMEM_BIT_LEN = $clog2(2304),
    parameter           NUM_CHANNEL = 16
)(
    input               clk,
    input               rst_n,
    
    input               initial_SRAMw_done,
    input               initial_weight_done,

    //Weight Memory
    output  wire [8:0]  wmem_addr_o,
    output  wire        wmem_wenb_o,
    output  wire        wmem_cenb_o,

    //Memory A 
    output  wire [15:0] memA_addr_o,
    output  wire        memA_wenb_o,
    output  wire        memA_cenb_o,
    //Memory B
    output  wire [15:0] memB_addr_o,
    output  wire        memB_wenb_o,
    output  wire        memB_cenb_o,

    //Weight Buffer
    output  wire        wei_buff_wren_o,
    output  wire        wei_buff_rden_o,
    
    // input buffer
    output  wire        in_buf_wren_o,
    output  wire        in_buf_rden_o,
    output  wire        in_buf_sel_o,
    
    // output buffer
    output  wire        out_buf_wren_o,
    output  wire        out_buf_rden_o,
    
    //PE array
    output  wire        pe_en_o,
    input   wire        pe_done_i,

    //add tree
    output  wire        addtree_en_o,
    input   wire        addtree_done_i,

    // ReLU
    output  wire        relu_en_o,
    input   wire        relu_done_i,

    //maxpool
    output  wire        maxpool_en_o,
    input   wire        maxpool_done_i,

    output  reg [4:0]   channel, //input 채널 개수
    output  wire        total_done_o // 최종 끝
    
);

    //FSM state
    localparam      S_IDLE          = 3'd0,
                    S_SRAM_W        = 3'd1,
                    S_Layer1        = 3'd2,
                    S_Layer2        = 3'd3,
                    S_Layer3        = 3'd4,
                    S_Layer4        = 3'd5,  
                    S_Layer5        = 3'd6;

    reg     [2:0]           state,      state_n;
    reg     [2:0]           layer_num,  layer_num_n;
    reg                     layer_start,  layer_start_n;
    reg     [15:0]          data_num,  data_num_n;
    reg     [8:0]           weight_num,  weight_num_n;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;   
            layer_num       <= 3'b0;
            layer_start     <= 1'b0;
            data_num        <= 'd10404;
            weight_num      <= 'd16;
        end
        else begin
            state           <= state_n;
            layer_num       <= layer_num_n;
            layer_start     <= layer_start_n;
            data_num        <= data_num_n;
            weight_num      <= weight_num_n;
        end
    end

    always_comb begin
        state_n                         = state;
        layer_num_n                     = layer_num;
        data_num_n                      = 'd10404; // 안쓰긴함
        channel                         = 'd16; 
        weight_num_n                    = 'd16;
        layer_start_n                   = 1'b0;
        case(state)
            S_IDLE: begin
                state_n                 = S_SRAM_W;
                layer_num_n             = 3'd0;
            end
            S_SRAM_W: begin
                if(initial_SRAMw_done & initial_weight_done) begin
                    data_num_n          = 'd10404; // cov1 102*102
                    channel             = 'd2;
                    weight_num_n        = 'd16;
                    layer_start_n       = 1'b1;
                    state_n             = S_Layer1;
                    layer_num_n         = 3'd1;
                end
            end
            S_Layer1: begin
                if(layer_done) begin
                    data_num_n          = 'd10404; // cov2 102*102
                    channel             = 'd16;
                    weight_num_n        = 'd16;
                    layer_start_n       = 1'b1;
                    state_n             = S_Layer2;
                    layer_num_n         = 3'd2;
                end
            end
            S_Layer2: begin
                if(layer_done) begin
                    data_num_n          = 'd10404; // cov3 102*102
                    channel             = 'd16;
                    weight_num_n        = 'd16;
                    layer_start_n       = 1'b1;
                    state_n             = S_Layer3;
                    layer_num_n         = 3'd3;
                end
            end
            S_Layer3: begin
                if(layer_done) begin
                    data_num_n          = 'd10404; // cov4 102*102
                    channel             = 'd16;
                    weight_num_n        = 'd16;
                    layer_start_n       = 1'b1;
                    state_n             = S_Layer4;
                    layer_num_n         = 3'd4;
                end
            end
            S_Layer4: begin
                if(layer_done) begin
                    data_num_n          = 'd10404; // cov5 102*102
                    channel             = 'd16;
                    weight_num_n        = 'd1;
                    layer_start_n       = 1'b1;
                    state_n             = S_Layer5;
                    layer_num_n         = 3'd5;
                end
            end
            S_Layer5: begin
                if(layer_done) begin
                    data_num_n          = 'd10404; // cov1 102*102
                    channel             = 'd2;
                    weight_num_n        = 'd16;
                    layer_start_n       = 1'b1;
                    state_n             = S_IDLE;
                    layer_num_n         = 3'd0;
                end
            end
        endcase
    end

    layer_pipeline  u_layer_pipeline(
        .clk                (clk),
        .rst_n              (rst_n),

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
        
        .layer_num          (layer_num),
        .weight_num         (weight_num),
        .channel            (channel),
        .layer_start        (layer_start),
        .layer_done_o       (layer_done)
    );

    assign total_done_o = (state == S_Layer5) & layer_done;

//     always_comb @(*) begin
//     // Default value
//     wmem_addr           = 0;
//     memA_addr           = 0;
//     memB_addr           = 0;
    
//     wmem_wenb           = 1;
//     wmem_enb            = 1;
    
//     memA_wenb           = 1;
//     memA_cenb           = 1;
//     memB_wenb           = 1;
//     memB_cenb           = 1;

//     wei_buff_en         = 0;
//     in_buf_en           = 0;
//     in_buf_rst          = 0;
//     in_buf_sel          = 0;
//     pe_en               = 0;
//     pe_rst              = 0;
//     relu_en             = 0;
//     out_buf_en          = 0;
//     out_buf_sel         = 0;
//     out_buf_rst         = 0;
//     pool_sel            = 0;
//     done                = 0;
    
//     case (state)
//         S_IDLE: begin

//         end
//         S_SRAM_W: begin
//             wmem_enb            = 0;
//             wmem_wenb           = 0;
//             memA_cenb           = 0;
//             memA_wenb           = 0;
//         end
//         S_Layer1, S_Layer2, S_Layer3, S_Layer4, S_Layer5: begin
//             memA_cenb           = 0;
//             memB_cenb           = 0;
//             pe_en               = 1;
//             wei_buff_en         = 1;
//             in_buf_en           = 1;
//             out_buf_en          = 1;
//             if ((state == S_Layer1) | (state == S_Layer3) | (state == S_Layer5) ) begin
//                 memB_wenb       = 0;
//                 if (state == S_Layer3) begin
//                     pool_sel    = 1;
//                 end
//                 else begin
//                     pool_sel    = 0;
//                 end
//             end
//             else begin
//                 memA_wenb       = 0;
//             end
//             if (state != S_Layer5) begin
//                 relu_en         = 1;
//             end
//             else begin
//                 relu_en         = 0;
//             end
//         end
//         S_data_mapping: begin
//             memB_cenb           = 0;
//             data_map_enb        = 0;
//         end
//     endcase
// end


endmodule
