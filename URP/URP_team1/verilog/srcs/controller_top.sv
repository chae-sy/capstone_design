`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/04 23:29:37
// Design Name: 
// Module Name: Controller Top
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module controller_top #(
    parameter IM_BIT_LEN = $clog2(100),
    parameter MA_BIT_LEN = $clog2(16),
    parameter MB_BIT_LEN = $clog2(20),
    parameter WM_BIT_LEN = $clog2(144),
    parameter NUM_CHANNEL = 32
)(
    // Global input
    input rstb,
    input clk, 
    // Control Signal for Weight Initialization    
    input wr_weight_on, // weight should be written before calculation starts
    input calc_en, // FE Done -> 


    //Weight Buffer Input/Output

    //Weight Memory
    output reg [WM_BIT_LEN-1:0] wm_addr_o,
    output reg wm_web_o,
    output reg wm_ceb_o,
    //Input Memory
    output reg [IM_BIT_LEN-1:0] im_addr_o,
    output reg im_web_o,
    output reg im_ceb_o,
    //Memory A 
    output reg [MA_BIT_LEN-1:0] ma_addr_o,
    output reg ma_web_o,
    output reg ma_ceb_o,
    //Memory B
    output reg [MB_BIT_LEN-1:0] mb_addr_o,
    output reg mb_web_o,
    output reg mb_ceb_o,
    //Other Module Blocks
    output reg pe_en_o,
    output reg pe_rst_o,
    output reg [31:0] out_buf_en_o,
    output reg out_buf_sel_o,
    output reg out_buf_rst_o,
    output reg pool_sel_o, //1 at layer 4, 0 for else
    output reg comp_start_o,
    //output reg rf_sel_o,
    output reg [2:0] layer,
    output done_o
);

    
    
    // Layer Num
    localparam WEIGHT = 0;
    localparam LAYER1 = 1;
    localparam LAYER2 = 2;
    localparam LAYER3 = 3;
    localparam LAYER4 = 4;
    localparam LAYER5 = 5;


    //reg [2:0] layer;
    reg [2:0] layer_n;

    reg layer2_en;
    reg layer2_en_n;

    reg layer3_en;
    reg layer3_en_n;

    reg layer4_en;
    reg layer4_en_n;

    reg layer5_en;
    reg layer5_en_n;

    //Weight Layer Parameter
    localparam WEIGHT_BIT = 8; 
    localparam MEM_LENGTH = 144; 
    
    
    // Layer1 Parameter
    localparam L1_INPUT_HORIZ = 10;
    localparam L1_INPUT_VERT = 29;
    localparam L1_WEIGHT_HORIZ = 4;
    localparam L1_WEIGHT_VERT = 10;
    localparam L1_STRIDE_HORIZ = 2;
    localparam L1_STRIDE_VERT = 1;
    localparam L1_INPUT_BIT_LEN = 8;
    localparam L1_WEIGHT_BIT_LEN = 32;
    localparam L1_NEXT_WEIGHT_VERT = 3;
    localparam L1_NEXET_INPUT_HORIZ = (L1_INPUT_HORIZ - L1_WEIGHT_HORIZ)/L1_STRIDE_HORIZ + 1;
    localparam L1_WEIGHT_VERT_BIT_LEN = $clog2(L1_WEIGHT_VERT);
    localparam L1_INPUT_VERT_BIT_LEN = $clog2(L1_INPUT_VERT);

    // Layer1 Wires
    wire [IM_BIT_LEN-1:0]im_addr_l1;
    wire [MA_BIT_LEN-1:0]ma_addr_l1;
    wire [WM_BIT_LEN-1:0]wm_addr_l1;
    wire im_ceb_l1;
    wire im_web_l1;
    wire ma_ceb_l1;
    wire ma_web_l1;
    wire wm_ceb_l1;
    wire wm_web_l1;
    wire pe_rst_l1;
    wire pe_en_l1;
    wire [31:0] out_buf_en_l1;
    wire out_buf_rst_l1;
    wire done_l1;
    wire init;
    wire [L1_INPUT_VERT_BIT_LEN-1:0] l1_line_cnt;
    wire l1_en;

    assign l1_en = calc_en && (!wr_weight_on);

    layer1 #(
        .INPUT_HORIZ(L1_INPUT_HORIZ),
        .INPUT_VERT(L1_INPUT_VERT),
        .WEIGHT_HORIZ(L1_WEIGHT_HORIZ),
        .WEIGHT_VERT(L1_WEIGHT_VERT),
        .STRIDE_HORIZ(L1_STRIDE_HORIZ),
        .STRIDE_VERT(L1_STRIDE_VERT),
        .INPUT_BIT_LEN(L1_INPUT_BIT_LEN),
        .WEIGHT_BIT_LEN(L1_WEIGHT_BIT_LEN),
        .NEXT_WEIGHT_VERT(L1_NEXT_WEIGHT_VERT),
        .IM_BIT_LEN(IM_BIT_LEN),
        .MA1_BIT_LEN(MA_BIT_LEN),
        .L1_WM_BIT_LEN(WM_BIT_LEN),
        .WEIGHT_VERT_BIT_LEN(L1_WEIGHT_VERT_BIT_LEN),
        .INPUT_VERT_BIT_LEN(L1_INPUT_VERT_BIT_LEN)
    ) layer1_control (
        .rstb(rstb),
        .clk(clk),
        .layer_en(l1_en),
        .im_addr(im_addr_l1),
        .ma_wr_addr(ma_addr_l1),
        .l1_wm_r_addr(wm_addr_l1),
        .im_cen(im_ceb_l1),
        .im_wen(im_web_l1),
        .ma_cen(ma_ceb_l1),
        .ma_wen(ma_web_l1),
        .wm_cen(wm_ceb_l1),
        .wm_wen(wm_web_l1),
        .pe_rst(pe_rst_l1),
        .pe_en(pe_en_l1),
        .output_buf_en(out_buf_en_l1),
        .output_buf_rst(out_buf_rst_l1),
        .layer_done(done_l1),
        .init(init),
        .l1_line_cnt(l1_line_cnt)
    );

    // Layer2 Parameter
    localparam L2_INPUT_HORIZ = 4;
    localparam L2_INPUT_VERT = 3;
    localparam L2_WEIGHT_HORIZ = 3;
    localparam L2_WEIGHT_VERT = 3;
    localparam L2_STRIDE_HORIZ = 1;
    localparam L2_STRIDE_VERT = 1;
    localparam L2_INPUT_BIT_LEN = 8;
    localparam L2_WEIGHT_BIT_LEN = 32;
    localparam L2_NEXT_WEIGHT_VERT = 1;
    localparam L2_NEXET_INPUT_HORIZ = (L2_INPUT_HORIZ - L2_WEIGHT_HORIZ)/L2_STRIDE_HORIZ + 1;
    localparam L2_WEIGHT_VERT_BIT_LEN = $clog2(L2_WEIGHT_VERT);

    // Layer2 wires 
    wire [MA_BIT_LEN-1:0]ma_addr_l2;
    wire [MB_BIT_LEN-1:0]mb_addr_l2;
    wire [WM_BIT_LEN-1:0]wm_addr_l2;
    wire ma_ceb_l2;
    wire ma_web_l2;
    wire mb_ceb_l2;
    wire mb_web_l2;
    wire wm_ceb_l2;
    wire wm_web_l2;
    wire pe_rst_l2;
    wire pe_en_l2;
    wire [31:0] out_buf_en_l2;
    wire out_buf_rst_l2;
    wire done_l2;
    

    layer2 #(
    .INPUT_HORIZ(L2_INPUT_HORIZ),
    .INPUT_VERT(L2_INPUT_VERT),
    .WEIGHT_HORIZ(L2_WEIGHT_HORIZ),
    .WEIGHT_VERT(L2_WEIGHT_VERT),
    .STRIDE_HORIZ(L2_STRIDE_HORIZ),
    .STRIDE_VERT(L2_STRIDE_VERT),
    .INPUT_BIT_LEN(L2_INPUT_BIT_LEN),
    .WEIGHT_BIT_LEN(L2_WEIGHT_BIT_LEN),
    .NEXT_WEIGHT_VERT(L2_NEXT_WEIGHT_VERT),
    .NEXET_INPUT_HORIZ(L2_NEXET_INPUT_HORIZ),
    .MA_BIT_LEN(MA_BIT_LEN),
    .MB1_BIT_LEN(MB_BIT_LEN),
    .L2_WM_BIT_LEN(WM_BIT_LEN),
    .WEIGHT_VERT_BIT_LEN(L2_WEIGHT_VERT_BIT_LEN)
    ) layer2_control (
    .rstb(rstb),
    .clk(clk),
    .layer_en(layer2_en),
    .ma_r_addr(ma_addr_l2),
    .mb_wr_addr(mb_addr_l2),
    .l2_wm_r_addr(wm_addr_l2),
    .ma_cen(ma_ceb_l2),
    .ma_wen(ma_web_l2),
    .mb_cen(mb_ceb_l2),
    .mb_wen(mb_web_l2),
    .wm_cen(wm_ceb_l2),
    .wm_wen(wm_web_l2),
    .pe_rst(pe_rst_l2),
    .pe_en(pe_en_l2),
    .output_buf_en(out_buf_en_l2),
    .output_buf_rst(out_buf_rst_l2),
    .layer_done(done_l2)
    );

    // Layer3 Parameter
    localparam L3_INPUT_HORIZ = 2;
    localparam L3_INPUT_VERT = 1;
    localparam L3_WEIGHT_HORIZ = 1;
    localparam L3_WEIGHT_VERT = 1;
    localparam L3_STRIDE_HORIZ = 1;
    localparam L3_STRIDE_VERT = 1;
    localparam L3_INPUT_BIT_LEN = 8; 
    localparam L3_WEIGHT_BIT_LEN = 32;
    localparam L3_WEIGHT_NUM = 32;
    localparam L3_NEXT_WEIGHT_VERT = 2;
    localparam L3_NEXT_INPUT_HORIZ = (L3_INPUT_HORIZ - L3_WEIGHT_HORIZ) / L3_STRIDE_HORIZ + 1;
    
    // Layer3 Wires
    wire [MA_BIT_LEN-1:0]ma_addr_l3;
    wire [MB_BIT_LEN-1:0]mb_addr_l3;
    wire [WM_BIT_LEN-1:0]wm_addr_l3;
    wire ma_ceb_l3;
    wire ma_web_l3;
    wire mb_ceb_l3;
    wire mb_web_l3;
    wire wm_ceb_l3;
    wire wm_web_l3;
    wire pe_rst_l3;
    wire pe_en_l3;
    wire [31:0] out_buf_en_l3;
    wire out_buf_rst_l3;
    wire pool_sel_l3;
    wire done_l3;

    layer3 #(
    .INPUT_HORIZ(L3_INPUT_HORIZ),
    .INPUT_VERT(L3_INPUT_VERT),
    .WEIGHT_HORIZ(L3_WEIGHT_HORIZ),
    .WEIGHT_VERT(L3_WEIGHT_VERT),
    .STRIDE_HORIZ(L3_STRIDE_HORIZ),
    .STRIDE_VERT(L3_STRIDE_VERT),
    .INPUT_BIT_LEN(L3_INPUT_BIT_LEN),
    .WEIGHT_BIT_LEN(L3_WEIGHT_BIT_LEN),
    .WEIGHT_NUM(L3_WEIGHT_NUM),
    .NEXT_WEIGHT_VERT(L3_NEXT_WEIGHT_VERT),
    .NEXT_INPUT_HORIZ(L3_NEXT_INPUT_HORIZ), // (INPUT_HORIZ - WEIGHT_HORIZ)/STRIDE_HORIZ + 1
    .MB_BIT_LEN(MB_BIT_LEN),       // $clog2(20) = 5
    .MA_BIT_LEN(MA_BIT_LEN),       // $clog2(16) = 4
    .L3_WM_BIT_LEN(WM_BIT_LEN)     // $clog2(144) = 8
    ) layer3_control (
    .rstb(rstb),
    .layer_en(layer3_en),
    .clk(clk),
    .ma_wr_addr(ma_addr_l3),
    .mb_r_addr(mb_addr_l3),
    .l3_wm_r_addr(wm_addr_l3),
    .ma_ceb(ma_ceb_l3),
    .ma_web(ma_web_l3),
    .mb_ceb(mb_ceb_l3),
    .mb_web(mb_web_l3),
    .wm_ceb(wm_ceb_l3),
    .wm_web(wm_web_l3),
    .output_buf_en(out_buf_en_l3),
    .pe_en(pe_en_l3),
    .pe_rst(pe_rst_l3),
    .PA_sel(pool_sel_l3),
    .output_buf_rst(out_buf_rst_l3),
    .done(done_l3)
    );


    // Layer4 Parameter
    localparam L4_INPUT_HORIZ = 2;
    localparam L4_INPUT_VERT = 2;
    localparam L4_WEIGHT_HORIZ = 2;
    localparam L4_WEIGHT_VERT = 2;
    localparam L4_STRIDE_HORIZ = 2;
    localparam L4_STRIDE_VERT = 2;
    localparam L4_INPUT_BIT_LEN = 8;
    localparam L4_WEIGHT_BIT_LEN = 32;
    localparam L4_NEXT_WEIGHT_VERT = 1;
    localparam L4_NEXET_INPUT_HORIZ = (L4_INPUT_HORIZ - L4_WEIGHT_HORIZ)/L4_STRIDE_HORIZ + 1;
    localparam L4_REDUCED_HORIZ = 1;
    localparam L4_REDUCED_VERT = 9;

    // Layer4 Wires
    wire [MA_BIT_LEN-1:0]ma_addr_l4;
    wire [MB_BIT_LEN-1:0]mb_addr_l4;
    wire ma_ceb_l4;
    wire ma_web_l4;
    wire mb_ceb_l4;
    wire mb_web_l4;
    wire pe_rst_l4;
    wire pe_en_l4;
    wire [31:0] out_buf_en_l4;
    wire out_buf_rst_l4;
    wire done_l4;

    layer4 #(
    .INPUT_HORIZ(L4_INPUT_HORIZ),
    .INPUT_VERT(L4_INPUT_VERT),
    .WEIGHT_HORIZ(L4_WEIGHT_HORIZ),
    .WEIGHT_VERT(L4_WEIGHT_VERT),
    .STRIDE_HORIZ(L4_STRIDE_HORIZ),
    .STRIDE_VERT(L4_STRIDE_VERT),
    .INPUT_BIT_LEN(L4_INPUT_BIT_LEN),
    .WEIGHT_BIT_LEN(L4_WEIGHT_BIT_LEN),
    .NEXT_WEIGHT_VERT(L4_NEXT_WEIGHT_VERT),
    .NEXET_INPUT_HORIZ(L4_NEXET_INPUT_HORIZ), // (INPUT_HORIZ - WEIGHT_HORIZ)/STRIDE_HORIZ + 1
    .MA2_BIT_LEN(MA_BIT_LEN),       // $clog2(16) = 4
    .REDUCED_HORIZ(L4_REDUCED_HORIZ),
    .REDUCED_VERT(L4_REDUCED_VERT),
    .MB2_BIT_LEN(MB_BIT_LEN)        // $clog2(20) = 5
    ) layer4_control (
    .rstb(rstb),
    .clk(clk),
    .layer_en(layer4_en),

    .ma_r_addr(ma_addr_l4),
    .mb_wr_addr(mb_addr_l4),
    .ma_cen(ma_ceb_l4),
    .ma_wen(ma_web_l4),
    .mb_cen(mb_ceb_l4),
    .mb_wen(mb_web_l4),
    .pe_rst(pe_rst_l4),
    .pe_en(pe_en_l4),
    .output_buf_en(out_buf_en_l4),
    .output_buf_rst(out_buf_rst_l4),
    .layer_done(done_l4)
    );

    // Layer5 Parameter
    localparam L5_INPUT_HORIZ = 1;
    localparam L5_INPUT_VERT = 9;
    localparam L5_WEIGHT_HORIZ = 7;
    localparam L5_WEIGHT_VERT = 9;
    localparam L5_STRIDE_HORIZ = 1;
    localparam L5_STRIDE_VERT = 1;
    localparam L5_INPUT_BIT_LEN = 8; // Yet to be used
    localparam L5_WEIGHT_BIT_LEN = 32;
    localparam L5_WEIGHT_NUM = L5_WEIGHT_HORIZ * L5_WEIGHT_VERT;
    localparam L5_OUTPUT_BUF_LEN = 32;
    
    // Layer5 Wires
    wire [MB_BIT_LEN-1:0]mb_addr_l5;
    wire [WM_BIT_LEN-1:0]wm_addr_l5;
    wire mb_ceb_l5;
    wire mb_web_l5;
    wire wm_ceb_l5;
    wire wm_web_l5;
    wire pe_rst_l5;
    wire pe_en_l5;
    wire pool_sel_l5;
    wire [31:0] out_buf_en_l5;
    wire out_buf_rst_l5;
    wire comparator_init_l5;
    wire done_l5;

    layer5 #(
    .INPUT_HORIZ(L5_INPUT_HORIZ),
    .INPUT_VERT(L5_INPUT_VERT),
    .WEIGHT_HORIZ(L5_WEIGHT_HORIZ),
    .WEIGHT_VERT(L5_WEIGHT_VERT),
    .STRIDE_HORIZ(L5_STRIDE_HORIZ),
    .STRIDE_VERT(L5_STRIDE_VERT),
    .INPUT_BIT_LEN(L5_INPUT_BIT_LEN),
    .WEIGHT_BIT_LEN(L5_WEIGHT_BIT_LEN),
    .WEIGHT_NUM(L5_WEIGHT_NUM),             // WEIGHT_HORIZ * WEIGHT_VERT
    .MB_BIT_LEN(MB_BIT_LEN),              // $clog2(WEIGHT_VERT * INPUT_HORIZ) = $clog2(9) = 7
    .OUTPUT_BUF_LEN(L5_OUTPUT_BUF_LEN),
    .L5_WM_BIT_LEN(WM_BIT_LEN),           // $clog2(256) = 8
    .L3_WEIGHT_CHANNEL(L3_WEIGHT_NUM),
    .L2_WEIGHT_HORIZ(L2_WEIGHT_HORIZ),
    .L2_WEIGHT_VERT(L2_WEIGHT_VERT),
    .L1_WEIGHT_HORIZ(L1_INPUT_HORIZ),
    .L1_WEIGHT_VERT(L1_WEIGHT_VERT)
    ) layer5_control (
    .rstb(rstb),
    .layer_en(layer5_en),
    .clk(clk),
    .mb_r_addr(mb_addr_l5),
    .l5_wm_r_addr(wm_addr_l5),
    .mb_ceb(mb_ceb_l5),
    .mb_web(mb_web_l5),
    .wm_ceb(wm_ceb_l5),
    .wm_web(wm_web_l5),
    .pe_en(pe_en_l5),
    .pe_rst(pe_rst_l5),
    .PA_sel(pool_sel_l5),
    .output_buf_en(out_buf_en_l5),
    .output_buf_rst(out_buf_rst_l5),
    .comparator_init(comparator_init_l5),
    .done(done_l5)
    );



    //Layer Assignment
    always @ (posedge clk or negedge rstb) begin
        if (!rstb) begin
            layer <= LAYER1;
            layer2_en <= 0;
            layer3_en <= 0;
            layer4_en <= 0;
            layer5_en <= 0;
        end
        else begin
            layer <= layer_n;
            layer2_en <= layer2_en_n;
            layer3_en <= layer3_en_n;
            layer4_en <= layer4_en_n;
            layer5_en <= layer5_en_n;
        end
    end

    always @ (*) begin
        layer_n = layer;
        layer2_en_n = layer2_en;
        layer3_en_n = layer3_en;
        layer4_en_n = layer4_en;
        layer5_en_n = layer5_en;
        case(layer) //synopsys full_case
            /*
            WEIGHT: begin
                if (weight_done) begin 
                    layer_n = LAYER1;
                end 
            end
            */
            LAYER1: begin
                if (done_l1) begin 
                    if(init && l1_line_cnt < L1_WEIGHT_VERT + L2_INPUT_VERT -1) begin
                        layer_n = LAYER1;
                    end
                    else begin
                        layer_n = LAYER2;
                        layer2_en_n = 1;
                    end
                end
            end
            LAYER2: begin
                layer2_en_n = 0;
                if (done_l2) begin 
                    layer_n = LAYER3;
                    layer3_en_n = 1;
                end
            end
            LAYER3: begin
                layer3_en_n = 0;
                if (done_l3) begin 
                    layer_n = LAYER4;
                    layer4_en_n = 1;
                end
            end
            LAYER4: begin
                layer4_en_n = 0;
                if (done_l4) begin 
                    if(init) begin
                        layer_n = LAYER1;
                    end
                    else begin
                        layer_n = LAYER5;
                        layer5_en_n = 1;
                    end
                end
            end
            LAYER5: begin
                layer5_en_n = 0;
                if (done_l5) begin 
                    layer_n = LAYER1;
                end
            end
        endcase
    end

    //Output MUX
    always @ (*) begin
        case(layer) //synopsys full_case
            WEIGHT: begin 
                wm_addr_o = 0;
                wm_ceb_o = 1;
                wm_web_o = 1;
                ma_addr_o = 0;
                ma_ceb_o = 1; 
                ma_web_o = 1;
                mb_addr_o = 0; 
                mb_ceb_o = 1;
                mb_web_o = 1;
                im_addr_o = 0;
                im_ceb_o = 1;
                im_web_o = 1; 
                pe_en_o = 0;
                pe_rst_o = 0;
                out_buf_en_o = 0;
                out_buf_rst_o = 0;
                out_buf_sel_o = 0;
                //rf_sel_o = 1;
                pool_sel_o = 0;
                comp_start_o = 0;
                
            end
            LAYER1: begin
                wm_addr_o = wm_addr_l1;
                wm_ceb_o = wm_ceb_l1;
                wm_web_o = wm_web_l1;
                ma_addr_o = ma_addr_l1;
                ma_ceb_o = ma_ceb_l1; 
                ma_web_o = ma_web_l1;
                mb_addr_o = 0; 
                mb_ceb_o = 1;
                mb_web_o = 1;
                im_addr_o = im_addr_l1;
                im_ceb_o = im_ceb_l1;
                im_web_o = im_web_l1; 
                pe_en_o = pe_en_l1;
                pe_rst_o = pe_rst_l1;
                out_buf_en_o = out_buf_en_l1;
                out_buf_rst_o = out_buf_rst_l1;
                out_buf_sel_o = 1;
                //rf_sel_o = 0;
                pool_sel_o = 0;
                comp_start_o = 0;
            end
            LAYER2: begin
                wm_addr_o = wm_addr_l2;
                wm_ceb_o = wm_ceb_l2;
                wm_web_o = wm_web_l2;
                ma_addr_o = ma_addr_l2;
                ma_ceb_o = ma_ceb_l2; 
                ma_web_o = ma_web_l2;
                mb_addr_o = mb_addr_l2; 
                mb_ceb_o = mb_ceb_l2;
                mb_web_o = mb_web_l2;
                im_addr_o = 0;
                im_ceb_o = 1;
                im_web_o = 1; 
                pe_en_o = pe_en_l2;
                pe_rst_o = pe_rst_l2;
                out_buf_en_o = out_buf_en_l2;
                out_buf_rst_o = out_buf_rst_l2;
                out_buf_sel_o = 1;
                //rf_sel_o = 0;
                pool_sel_o = 0;
                comp_start_o = 0;
            end
            LAYER3: begin
                wm_addr_o = wm_addr_l3;
                wm_ceb_o = wm_ceb_l3;
                wm_web_o = wm_web_l3;
                ma_addr_o = ma_addr_l3;
                ma_ceb_o = ma_ceb_l3; 
                ma_web_o = ma_web_l3;
                mb_addr_o = mb_addr_l3; 
                mb_ceb_o = mb_ceb_l3;
                mb_web_o = mb_web_l3;
                im_addr_o = 0;
                im_ceb_o = 1;
                im_web_o = 1; 
                pe_en_o = pe_en_l3;
                pe_rst_o = pe_rst_l3;
                out_buf_en_o = out_buf_en_l3;
                out_buf_rst_o = out_buf_rst_l3;
                out_buf_sel_o = 0;
                //rf_sel_o = 0;
                pool_sel_o = pool_sel_l3;
                comp_start_o = 0;
            end
            LAYER4: begin
                wm_addr_o = 0;
                wm_ceb_o = 1;
                wm_web_o = 1;
                ma_addr_o = ma_addr_l4;
                ma_ceb_o = ma_ceb_l4; 
                ma_web_o = ma_web_l4;
                mb_addr_o = mb_addr_l4; 
                mb_ceb_o = mb_ceb_l4;
                mb_web_o = mb_web_l4;
                im_addr_o = 0;
                im_ceb_o = 1;
                im_web_o = 1; 
                pe_en_o = pe_en_l4;
                pe_rst_o = pe_rst_l4;
                out_buf_en_o = out_buf_en_l4;
                out_buf_rst_o = out_buf_rst_l4;
                out_buf_sel_o = 1;
                //rf_sel_o = 0;
                pool_sel_o = 1;
                comp_start_o = 0;
            end
            LAYER5: begin
                wm_addr_o = wm_addr_l5;
                wm_ceb_o = wm_ceb_l5;
                wm_web_o = wm_web_l5;
                ma_addr_o = 0;
                ma_ceb_o = 1; 
                ma_web_o = 1;
                mb_addr_o = mb_addr_l5; 
                mb_ceb_o = mb_ceb_l5;
                mb_web_o = mb_web_l5;
                im_addr_o = 0;
                im_ceb_o = 1;
                im_web_o = 1; 
                pe_en_o = pe_en_l5;
                pe_rst_o = pe_rst_l5;
                out_buf_en_o = out_buf_en_l5;
                out_buf_rst_o = out_buf_rst_l5;
                out_buf_sel_o = 0;
                //rf_sel_o = 0;
                pool_sel_o = pool_sel_l5;
                comp_start_o = comparator_init_l5;
            end
        endcase
    end
    
    //assign weight_buf_done_o = weight_buf_done;
    //assign weight_buf_rst_o = weight_buf_rst;
    //assign weight_buf_en_o = weight_buf_en;
    assign done_o = done_l5;


    reg [15:0] cnt;

    initial cnt <= 0;

    always @ (posedge clk or negedge rstb) begin
        if(!rstb | calc_en) cnt <= 0;
        else begin
            if(layer != 0) cnt <= cnt + 1;
            else cnt <= cnt;
        end
    end 

endmodule
