`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Donghwan So
// 
// Create Date: 2024/09/30 14:25:54
// Design Name: 
// Module Name: CNN_HEAD_controller
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

// 1. Firstly controller needs to deal with the !!!!!!TIMING!!!!!!!!!
// 2. Clear or Reset Other Module
// 3. Control the Signal for Write of Read Data from SRAM
// +@ If We needs to store the data to the OUTPUT SRAM, controller should control those related signals
//                    - Future work ==> IF NOT - > we needs to carry those datas to some othe modules     


module layer_state_ctrl_v0
#(    
    parameter WIDTH_WSRAM_WL = 128,
    parameter WIDTH_FSRAM_WL = 128,
    parameter WIDTH_W_DATA = 8,
    parameter WIDTH_F_DATA = 8,
    parameter NUM_PE = 16,
    parameter WIDTH_FSRAM_ADDR = 10,
    parameter WIDTH_WSRAM_ADDR = 10,
    //L1
    parameter ADDR_START_L1_W = 0,
    parameter ADDR_START_L1_F = 0,
    parameter SIZE_L1_OUT_CHANNEL = 16,
    parameter SIZE_KERNEL_H = 3,
    parameter SIZE_KERNEL_W = 3,   
    // parameter for L2
    parameter ADDR_START_L2_W = 10,
    parameter ADDR_START_L2_F = 21,
    parameter NUM_L1_OUT_CHANNEL = 16,
    parameter NUM_L2_OUT_CHANNEL = 8,
   //L3
    parameter ADDR_START_L3_W = 19,
    parameter ADDR_START_L3_F = 39,   
    //L4
    parameter ADDR_START_L4_W = 35,
    parameter ADDR_START_L4_F = 48,
    parameter SIZE_L3_OUT_CHANNEL = 8,
    parameter NUM_L4_OUT_CHANNEL = 8,
    //L5
    parameter ADDR_START_L5_W = 36,
    parameter ADDR_START_L5_F = 60,
    //L6
    parameter ADDR_START_L6_W = 40,
    parameter ADDR_START_L6_F = 63,
    //L7
    parameter ADDR_START_L7_W = 48,
    parameter ADDR_START_L7_F = 48,
    //L8
    parameter ADDR_START_L8_W = 51,
    parameter ADDR_START_L8_F = 64,
    //L9
    parameter ADDR_START_L9_W = 52,
    parameter ADDR_START_L9_F = 69,
    //L10
    parameter ADDR_START_L10_W = 61,
    parameter ADDR_START_L10_F = 73,
    //L11
    parameter ADDR_START_L11_W = 73,
    parameter ADDR_START_L11_F = 81,
    parameter SIZE_L10_OUT_CHANNEL = 16,
    parameter NUM_L11_OUT_CHANNEL = 16,
    //L12
    parameter ADDR_START_L12_W = 74,
    parameter ADDR_START_L12_F = 90,
    //L13
    parameter ADDR_START_L13_W = 78,
    parameter ADDR_START_L13_F = 93,
    //L14 
    parameter ADDR_START_L14_W = 94,
    parameter ADDR_START_L14_F = 81,
    //L15
    parameter ADDR_START_L15_W = 97,
    parameter ADDR_START_L15_F = 94,
    //L16
    parameter ADDR_START_L16_W = 98,
    parameter ADDR_START_L16_F = 98,
    //L17
    parameter ADDR_START_L17_W = 132,
    parameter ADDR_START_L17_F = 101,
    //L18
    parameter ADDR_START_L18_W = 140,
    parameter ADDR_START_L18_F = 107,
    //L19
    parameter ADDR_START_L19_W = 148,
    parameter ADDR_START_L19_F = 113,
    //L20
    parameter ADDR_START_L20_W = 212, 
    parameter ADDR_START_L20_F = 107,
    //L21
    parameter ADDR_START_L21_W = 218,
    parameter ADDR_START_L21_F = 115,
    //L22
    parameter ADDR_START_L22_W = 219, 
    parameter ADDR_START_L22_F = 119, 
    parameter NUM_POOL = 22,    
    //L23
    parameter ADDR_START_LAVGPOOL = 121
)
(
    input clk,
    input rstb,
    input start_nn,
    
    output  reg                                pe_clear,
    output  reg                                pe_en,
    output  reg     [4:0]                      layer_state,
    
    output  reg                                buffer_start,
    output  reg    [4:0]                       buffer_mode_f,
    output  reg    [4:0]                       buffer_mode_w,
    output  reg    [3:0]                       buffer_loc_w,
    output  reg    [3:0]                       buffer_loc_f,
    output  reg                                buffer_load_w,
    output  reg                                buffer_load_f,
    output  reg                                shift,
    output  reg                                norm_on,
    output  reg                                relu_on,
    
    output reg [WIDTH_FSRAM_ADDR-1:0] c_f_addr,
    output reg [WIDTH_WSRAM_ADDR-1:0] c_w_addr,
    
    output reg c_w_ceb,
    output reg c_w_web,
    output reg c_f_ceb,
    output reg c_f_web,
    
    output reg clf_mode,
    output reg clf_en,
    output reg clf_clear,
    
    output reg lavg_done
    
);

    localparam              S_IDLE                  = 5'd0,
                            S_LAYER1                    = 5'd1,
                            S_LAYER2                    = 5'd2,
                            S_LAYER3                    = 5'd3,
                            S_LAYER4                    = 5'd4,
                            S_LAYER5                    = 5'd5,
                            S_LAYER6                    = 5'd6,
                            S_LAYER7                    = 5'd7,
                            S_LAYER8                    = 5'd8,
                            S_LAYER9                    = 5'd9,
                            S_LAYER10                   = 5'd10,
                            S_LAYER11                   = 5'd11,
                            S_LAYER12                   = 5'd12,
                            S_LAYER13                   = 5'd13,
                            S_LAYER14                   = 5'd14,
                            S_LAYER15                   = 5'd15,
                            S_LAYER16                   = 5'd16,
                            S_LAYER17                   = 5'd17,
                            S_LAYER18                   = 5'd18,
                            S_LAYER19                   = 5'd19,
                            S_LAYER20                   = 5'd20,
                            S_LAYER21                   = 5'd21,
                            S_LAYER22                   = 5'd22,
                            S_AVGPOOL                   = 5'd23,

                            TOTAL_LAYER_NUM = 23;
    
    
    
    reg [4:0] state;
    
    reg layer_start;
    // Changing start addr and some parameters -> so that we can use same config in different layer.
    // like layer 1's CONV & 2's CONV or 1's max RELU, 2's max RELU
    
    // wire for done signal
    wire l1_done;
    wire l2_done;
    wire l3_done;
    wire l4_done;
    wire l5_done;
    wire l6_done;
    wire l7_done;
    wire l8_done;
    wire l9_done;
    wire l10_done;
    wire l11_done;
    wire l12_done;
    wire l13_done;
    wire l14_done;
    wire l15_done;
    wire l16_done;
    wire l17_done;
    wire l18_done;
    wire l19_done;
    wire l20_done;
    wire l21_done;
    wire l22_done;


    
    // wire for arbitering signals from each ctrl
    wire                        buffer_start_l          [TOTAL_LAYER_NUM:0];
    wire [4:0]                  buffer_mode_f_l         [TOTAL_LAYER_NUM:0];
    wire [4:0]                  buffer_mode_w_l         [TOTAL_LAYER_NUM:0];
    wire [3:0]                  buffer_loc_w_l          [TOTAL_LAYER_NUM:0];
    wire [3:0]                  buffer_loc_f_l          [TOTAL_LAYER_NUM:0];
    wire                        buffer_load_w_l         [TOTAL_LAYER_NUM:0];
    wire                        buffer_load_f_l         [TOTAL_LAYER_NUM:0];
    
    wire [WIDTH_FSRAM_ADDR-1:0] c_f_addr_l              [TOTAL_LAYER_NUM:0];
    wire [WIDTH_WSRAM_ADDR-1:0] c_w_addr_l              [TOTAL_LAYER_NUM:0];
    wire                        c_w_ceb_l               [TOTAL_LAYER_NUM:0]; 
    wire                        c_w_web_l               [TOTAL_LAYER_NUM:0];
    wire                        c_f_ceb_l               [TOTAL_LAYER_NUM:0];
    wire                        c_f_web_l               [TOTAL_LAYER_NUM:0];
    
    wire                        pe_clear_l              [TOTAL_LAYER_NUM:0];
    wire                        pe_en_l                 [TOTAL_LAYER_NUM:0];
    wire                        shift_l                 [TOTAL_LAYER_NUM:0];
    
    wire                        norm_on_l               [TOTAL_LAYER_NUM:0];
    wire                        relu_on_l               [TOTAL_LAYER_NUM:0];
    
  
    reg [4:0] timing_cnt;
    
    always@(posedge clk or negedge rstb) begin
        if(!rstb)begin 
            state <= S_IDLE;
            layer_state <= 0;
            layer_start <= 0;
            timing_cnt <= 0;
        end
        else begin 
            case(state)
                S_IDLE: begin 
                    layer_start <= 0;
                    if (start_nn) begin
                        if (timing_cnt > 1) begin
                            state <= S_LAYER1;
                            layer_state <= 1;
                            layer_start <= 1;
                            if(timing_cnt < 31) timing_cnt <= timing_cnt + 1; 
                        end 
                        else begin 
                            state <= S_IDLE;
                            layer_state <= 0;
                            timing_cnt <= timing_cnt + 1;
                        end
                    end
                end
                
                S_LAYER1: begin
                    layer_start <= 0;
                    if (l1_done) begin
                        state <= S_LAYER2;
                        layer_state <= 2;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER2: begin
                    layer_start <= 0;
                    if (l2_done) begin
                        state <= S_LAYER3;
                        layer_state <= 3;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER3: begin
                    layer_start <= 0;
                    if (l3_done) begin
                        state <= S_LAYER4;
                        layer_state <= 4;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER4: begin
                    layer_start <= 0;
                    if (l4_done) begin
                        if (timing_cnt > 4) begin
                            state <= S_LAYER5;
                            layer_state <= 5;
                            layer_start <= 1;
                        end
                        else begin
                            state <= S_IDLE;
                            layer_state <= 0;
                        end
                    end
                end
                
                S_LAYER5: begin
                    layer_start <= 0;
                    if (l5_done) begin
                        state <= S_LAYER6;
                        layer_state <= 6;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER6: begin
                    layer_start <= 0;
                    if (l6_done) begin
                        state <= S_LAYER7;
                        layer_state <= 7;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER7: begin
                    layer_start <= 0;
                    if (l7_done) begin
                        state <= S_LAYER8;
                        layer_state <= 8;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER8: begin
                    layer_start <= 0;
                    if (l8_done) begin
                        state <= S_LAYER9;
                        layer_state <= 9;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER9: begin
                    layer_start <= 0;
                    if (l9_done) begin
                        state <= S_LAYER10;
                        layer_state <= 10;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER10: begin
                    layer_start <= 0;
                    if (l10_done) begin
                        state <= S_LAYER11;
                        layer_state <= 11;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER11: begin
                    layer_start <= 0;
                    if (l11_done) begin
                        if (timing_cnt > 6) begin
                            state <= S_LAYER12;
                            layer_state <= 12;
                            layer_start <= 1;
                        end
                        else begin 
                            state <= S_IDLE;
                            layer_state <= 0;   
                        end
                    end
                end
                
                S_LAYER12: begin
                    layer_start <= 0;
                    if (l12_done) begin
                        state <= S_LAYER13;
                        layer_state <= 13;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER13: begin
                    layer_start <= 0;
                    if (l13_done) begin
                        state <= S_LAYER14;
                        layer_state <= 14;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER14: begin
                    layer_start <= 0;
                    if (l14_done) begin
                        state <= S_LAYER15;
                        layer_state <= 15;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER15: begin
                    layer_start <= 0;
                    if (l15_done) begin
                        state <= S_LAYER16;
                        layer_state <= 16;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER16: begin
                    layer_start <= 0;
                    if (l16_done) begin
                        state <= S_LAYER17;
                        layer_state <= 17;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER17: begin
                    layer_start <= 0;
                    if (l17_done) begin
                        if (timing_cnt > 8) begin
                            state <= S_LAYER18;
                            layer_state <= 18;
                            layer_start <= 1;
                        end
                        else begin 
                            state <= S_IDLE;
                            layer_state <= 0;
                        end
                    end
                end
                
                S_LAYER18: begin
                    layer_start <= 0;
                    if (l18_done) begin
                        state <= S_LAYER19;
                        layer_state <= 19;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER19: begin
                    layer_start <= 0;
                    if (l19_done) begin
                        state <= S_LAYER20;
                        layer_state <= 20;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER20: begin
                    layer_start <= 0;
                    if (l20_done) begin
                        state <= S_LAYER21;
                        layer_state <= 21;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER21: begin
                    layer_start <= 0;
                    if (l21_done) begin
                        state <= S_LAYER22;
                        layer_state <= 22;
                        layer_start <= 1;
                    end
                end
                
                S_LAYER22: begin
                    layer_start <= 0;
                    if (l22_done) begin
                        if (timing_cnt > 29) begin
                            state <= S_AVGPOOL;
                            layer_state <= 23;
                            layer_start <= 1;
                        end
                        else begin
                            state <= S_IDLE;
                            layer_state <= 0;
                        end
                    end
                end
                
                S_AVGPOOL: begin
                    layer_start <= 0;
                    if (lavg_done) begin
                        state <= S_IDLE;
                        layer_state <= 0;
                        layer_start <= 0;
                    end
                end
                
            endcase
        end
    end
    
    stl1_ctrl_v0_1
    #(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L1_W(ADDR_START_L1_W), 
        .ADDR_START_L1_F(ADDR_START_L1_F), 
        .SIZE_L1_OUT_CHANNEL(SIZE_L1_OUT_CHANNEL),
        .SIZE_KERNEL_H(SIZE_KERNEL_H),
        .SIZE_KERNEL_W(SIZE_KERNEL_W)
    )
    xstl1_ctrl_v0_1(
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
        
        .buffer_start(buffer_start_l[1]),
        .buffer_mode_f(buffer_mode_f_l[1]),
        .buffer_mode_w(buffer_mode_w_l[1]),
        .buffer_loc_w(buffer_loc_w_l[1]),
        .buffer_loc_f(buffer_loc_f_l[1]),
        .buffer_load_w(buffer_load_w_l[1]),
        .buffer_load_f(buffer_load_f_l[1]),
        
        .fsram_addr(c_f_addr_l[1]),
        .wsram_addr(c_w_addr_l[1]),
        .c_w_ceb(c_w_ceb_l[1]), 
        .c_w_web(c_w_web_l[1]),
        .c_f_ceb(c_f_ceb_l[1]),
        .c_f_web(c_f_web_l[1]),
        
        .pe_clear(pe_clear_l[1]),
        .pe_en(pe_en_l[1]),
        .shift(shift_l[1]),
        
        .norm_on(norm_on_l[1]),
        .relu_on(relu_on_l[1]),
        
        .l1_done(l1_done)
    );
    
     stl2_ctrl_v0
    #(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L2_W(ADDR_START_L2_W), 
        .ADDR_START_L2_F(ADDR_START_L2_F),
        .NUM_L1_OUT_CHANNEL(NUM_L1_OUT_CHANNEL),
        .NUM_L2_OUT_CHANNEL(NUM_L2_OUT_CHANNEL)     
    )
    xstl2_ctrl_v0(
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
        
        .buffer_start(buffer_start_l[2]),
        .buffer_mode_f(buffer_mode_f_l[2]),
        .buffer_mode_w(buffer_mode_w_l[2]),
        .buffer_loc_w(buffer_loc_w_l[2]),
        .buffer_loc_f(buffer_loc_f_l[2]),
        .buffer_load_w(buffer_load_w_l[2]),
        .buffer_load_f(buffer_load_f_l[2]),
        
        .fsram_addr(c_f_addr_l[2]),
        .wsram_addr(c_w_addr_l[2]),
        .c_w_ceb(c_w_ceb_l[2]), 
        .c_w_web(c_w_web_l[2]),
        .c_f_ceb(c_f_ceb_l[2]),
        .c_f_web(c_f_web_l[2]),
        
        .pe_clear(pe_clear_l[2]),
        .pe_en(pe_en_l[2]),
        .shift(shift_l[2]),
        
        .norm_on(norm_on_l[2]),
        .relu_on(relu_on_l[2]),
        
        .l2_done(l2_done)
    );
    
    stl3_ctrl_v0
    #(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L3_W(ADDR_START_L3_W), 
        .ADDR_START_L3_F(ADDR_START_L3_F)
    )
    xstl3_ctrl_v0(
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
        
        .buffer_start(buffer_start_l[3]),
        .buffer_mode_f(buffer_mode_f_l[3]),
        .buffer_mode_w(buffer_mode_w_l[3]),
        .buffer_loc_w(buffer_loc_w_l[3]),
        .buffer_loc_f(buffer_loc_f_l[3]),
        .buffer_load_w(buffer_load_w_l[3]),
        .buffer_load_f(buffer_load_f_l[3]),
        
        .fsram_addr(c_f_addr_l[3]),
        .wsram_addr(c_w_addr_l[3]),
        .c_w_ceb(c_w_ceb_l[3]), 
        .c_w_web(c_w_web_l[3]),
        .c_f_ceb(c_f_ceb_l[3]),
        .c_f_web(c_f_web_l[3]),
        
        .pe_clear(pe_clear_l[3]),
        .pe_en(pe_en_l[3]),
        .shift(shift_l[3]),
        
        .norm_on(norm_on_l[3]),
        .relu_on(relu_on_l[3]),
        
        .l3_done(l3_done)
    );
    
    stl4_ctrl_v0
    #(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L4_W(ADDR_START_L4_W), 
        .ADDR_START_L4_F(ADDR_START_L4_F),
        .SIZE_L3_OUT_CHANNEL(SIZE_L3_OUT_CHANNEL),
        .NUM_L4_OUT_CHANNEL(NUM_L4_OUT_CHANNEL) 
    )
    xstl4_ctrl_v0(
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
        
        .buffer_start(buffer_start_l[4]),
        .buffer_mode_f(buffer_mode_f_l[4]),
        .buffer_mode_w(buffer_mode_w_l[4]),
        .buffer_loc_w(buffer_loc_w_l[4]),
        .buffer_loc_f(buffer_loc_f_l[4]),
        .buffer_load_w(buffer_load_w_l[4]),
        .buffer_load_f(buffer_load_f_l[4]),
        
        .fsram_addr(c_f_addr_l[4]),
        .wsram_addr(c_w_addr_l[4]),
        .c_w_ceb(c_w_ceb_l[4]), 
        .c_w_web(c_w_web_l[4]),
        .c_f_ceb(c_f_ceb_l[4]),
        .c_f_web(c_f_web_l[4]),
        
        .pe_clear(pe_clear_l[4]),
        .pe_en(pe_en_l[4]),
        .shift(shift_l[4]),
        
        .norm_on(norm_on_l[4]),
        .relu_on(relu_on_l[4]),
        
        .l4_done(l4_done)
    );
    
    stl5_ctrl_v0
    #(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L5_W(ADDR_START_L5_W), 
        .ADDR_START_L5_F(ADDR_START_L5_F)
    )
    xstl5_ctrl_v0(
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
        
        .buffer_start(buffer_start_l[5]),
        .buffer_mode_f(buffer_mode_f_l[5]),
        .buffer_mode_w(buffer_mode_w_l[5]),
        .buffer_loc_w(buffer_loc_w_l[5]),
        .buffer_loc_f(buffer_loc_f_l[5]),
        .buffer_load_w(buffer_load_w_l[5]),
        .buffer_load_f(buffer_load_f_l[5]),
        
        .fsram_addr(c_f_addr_l[5]),
        .wsram_addr(c_w_addr_l[5]),
        .c_w_ceb(c_w_ceb_l[5]), 
        .c_w_web(c_w_web_l[5]),
        .c_f_ceb(c_f_ceb_l[5]),
        .c_f_web(c_f_web_l[5]),
        
        .pe_clear(pe_clear_l[5]),
        .pe_en(pe_en_l[5]),
        .shift(shift_l[5]),
        
        .norm_on(norm_on_l[5]),
        .relu_on(relu_on_l[5]),
        
        .l5_done(l5_done)
    );
    
    stl6_ctrl_v0
	#(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L6_W(ADDR_START_L6_W), 
        .ADDR_START_L6_F(ADDR_START_L6_F)
	)
	xstl6_ctrl_v0(
		.clk(clk),
		.rstb(rstb),
		.layer_state(layer_state),
		.layer_start(layer_start),

		.buffer_start(buffer_start_l[6]),
		.buffer_mode_f(buffer_mode_f_l[6]),
		.buffer_mode_w(buffer_mode_w_l[6]),
		.buffer_loc_w(buffer_loc_w_l[6]),
		.buffer_loc_f(buffer_loc_f_l[6]),
		.buffer_load_w(buffer_load_w_l[6]),
		.buffer_load_f(buffer_load_f_l[6]),

		.fsram_addr(c_f_addr_l[6]),
		.wsram_addr(c_w_addr_l[6]),
		.c_w_ceb(c_w_ceb_l[6]),
		.c_w_web(c_w_web_l[6]),
		.c_f_ceb(c_f_ceb_l[6]),
		.c_f_web(c_f_web_l[6]),

		.pe_clear(pe_clear_l[6]),
		.pe_en(pe_en_l[6]),
		.shift(shift_l[6]),

		.norm_on(norm_on_l[6]),
		.relu_on(relu_on_l[6]),

		.l6_done(l6_done)
	);

	stl7_ctrl_v0
	#(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L7_W(ADDR_START_L7_W), 
        .ADDR_START_L7_F(ADDR_START_L7_F)
	)
	xstl7_ctrl_v0(
		.clk(clk),
		.rstb(rstb),
		.layer_state(layer_state),
		.layer_start(layer_start),

		.buffer_start(buffer_start_l[7]),
		.buffer_mode_f(buffer_mode_f_l[7]),
		.buffer_mode_w(buffer_mode_w_l[7]),
		.buffer_loc_w(buffer_loc_w_l[7]),
		.buffer_loc_f(buffer_loc_f_l[7]),
		.buffer_load_w(buffer_load_w_l[7]),
		.buffer_load_f(buffer_load_f_l[7]),

		.fsram_addr(c_f_addr_l[7]),
		.wsram_addr(c_w_addr_l[7]),
		.c_w_ceb(c_w_ceb_l[7]),
		.c_w_web(c_w_web_l[7]),
		.c_f_ceb(c_f_ceb_l[7]),
		.c_f_web(c_f_web_l[7]),

		.pe_clear(pe_clear_l[7]),
		.pe_en(pe_en_l[7]),
		.shift(shift_l[7]),

		.norm_on(norm_on_l[7]),
		.relu_on(relu_on_l[7]),

		.l7_done(l7_done)
	);

	stl8_ctrl_v0
	#(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L8_W(ADDR_START_L8_W), 
        .ADDR_START_L8_F(ADDR_START_L8_F)	
	)
	xstl8_ctrl_v0(
		.clk(clk),
		.rstb(rstb),
		.layer_state(layer_state),
		.layer_start(layer_start),

		.buffer_start(buffer_start_l[8]),
		.buffer_mode_f(buffer_mode_f_l[8]),
		.buffer_mode_w(buffer_mode_w_l[8]),
		.buffer_loc_w(buffer_loc_w_l[8]),
		.buffer_loc_f(buffer_loc_f_l[8]),
		.buffer_load_w(buffer_load_w_l[8]),
		.buffer_load_f(buffer_load_f_l[8]),

		.fsram_addr(c_f_addr_l[8]),
		.wsram_addr(c_w_addr_l[8]),
		.c_w_ceb(c_w_ceb_l[8]),
		.c_w_web(c_w_web_l[8]),
		.c_f_ceb(c_f_ceb_l[8]),
		.c_f_web(c_f_web_l[8]),

		.pe_clear(pe_clear_l[8]),
		.pe_en(pe_en_l[8]),
		.shift(shift_l[8]),

		.norm_on(norm_on_l[8]),
		.relu_on(relu_on_l[8]),

		.l8_done(l8_done)
	);

	stl9_ctrl_v0
	#(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L9_W(ADDR_START_L9_W), 
        .ADDR_START_L9_F(ADDR_START_L9_F)	
	)
	xstl9_ctrl_v0(
		.clk(clk),
		.rstb(rstb),
		.layer_state(layer_state),
		.layer_start(layer_start),

		.buffer_start(buffer_start_l[9]),
		.buffer_mode_f(buffer_mode_f_l[9]),
		.buffer_mode_w(buffer_mode_w_l[9]),
		.buffer_loc_w(buffer_loc_w_l[9]),
		.buffer_loc_f(buffer_loc_f_l[9]),
		.buffer_load_w(buffer_load_w_l[9]),
		.buffer_load_f(buffer_load_f_l[9]),

		.fsram_addr(c_f_addr_l[9]),
		.wsram_addr(c_w_addr_l[9]),
		.c_w_ceb(c_w_ceb_l[9]),
		.c_w_web(c_w_web_l[9]),
		.c_f_ceb(c_f_ceb_l[9]),
		.c_f_web(c_f_web_l[9]),

		.pe_clear(pe_clear_l[9]),
		.pe_en(pe_en_l[9]),
		.shift(shift_l[9]),

		.norm_on(norm_on_l[9]),
		.relu_on(relu_on_l[9]),

		.l9_done(l9_done)
	);
	
	stl10_ctrl_v0
    #(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L10_W(ADDR_START_L10_W), 
        .ADDR_START_L10_F(ADDR_START_L10_F)
    )
    xstl10_ctrl_v0(
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
        
        .buffer_start(buffer_start_l[10]),
        .buffer_mode_f(buffer_mode_f_l[10]),
        .buffer_mode_w(buffer_mode_w_l[10]),
        .buffer_loc_w(buffer_loc_w_l[10]),
        .buffer_loc_f(buffer_loc_f_l[10]),
        .buffer_load_w(buffer_load_w_l[10]),
        .buffer_load_f(buffer_load_f_l[10]),
        
        .fsram_addr(c_f_addr_l[10]),
        .wsram_addr(c_w_addr_l[10]),
        .c_w_ceb(c_w_ceb_l[10]), 
        .c_w_web(c_w_web_l[10]),
        .c_f_ceb(c_f_ceb_l[10]),
        .c_f_web(c_f_web_l[10]),
        
        .pe_clear(pe_clear_l[10]),
        .pe_en(pe_en_l[10]),
        .shift(shift_l[10]),
        
        .norm_on(norm_on_l[10]),
        .relu_on(relu_on_l[10]),
        
        .l10_done(l10_done)
    );    
    
	stl11_ctrl_v0
	#(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L11_W(ADDR_START_L11_W), 
        .ADDR_START_L11_F(ADDR_START_L11_F),
        .SIZE_L10_OUT_CHANNEL(SIZE_L10_OUT_CHANNEL),
        .NUM_L11_OUT_CHANNEL(NUM_L11_OUT_CHANNEL)
	)
	xstl11_ctrl_v0(
		.clk(clk),
		.rstb(rstb),
		.layer_state(layer_state),
		.layer_start(layer_start),

		.buffer_start(buffer_start_l[11]),
		.buffer_mode_f(buffer_mode_f_l[11]),
		.buffer_mode_w(buffer_mode_w_l[11]),
		.buffer_loc_w(buffer_loc_w_l[11]),
		.buffer_loc_f(buffer_loc_f_l[11]),
		.buffer_load_w(buffer_load_w_l[11]),
		.buffer_load_f(buffer_load_f_l[11]),

		.fsram_addr(c_f_addr_l[11]),
		.wsram_addr(c_w_addr_l[11]),
		.c_w_ceb(c_w_ceb_l[11]),
		.c_w_web(c_w_web_l[11]),
		.c_f_ceb(c_f_ceb_l[11]),
		.c_f_web(c_f_web_l[11]),

		.pe_clear(pe_clear_l[11]),
		.pe_en(pe_en_l[11]),
		.shift(shift_l[11]),

		.norm_on(norm_on_l[11]),
		.relu_on(relu_on_l[11]),

		.l11_done(l11_done)
	);

	stl12_ctrl_v0
	#(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L12_W(ADDR_START_L12_W), 
        .ADDR_START_L12_F(ADDR_START_L12_F)	
	)
	xstl12_ctrl_v0(
		.clk(clk),
		.rstb(rstb),
		.layer_state(layer_state),
		.layer_start(layer_start),

		.buffer_start(buffer_start_l[12]),
		.buffer_mode_f(buffer_mode_f_l[12]),
		.buffer_mode_w(buffer_mode_w_l[12]),
		.buffer_loc_w(buffer_loc_w_l[12]),
		.buffer_loc_f(buffer_loc_f_l[12]),
		.buffer_load_w(buffer_load_w_l[12]),
		.buffer_load_f(buffer_load_f_l[12]),

		.fsram_addr(c_f_addr_l[12]),
		.wsram_addr(c_w_addr_l[12]),
		.c_w_ceb(c_w_ceb_l[12]),
		.c_w_web(c_w_web_l[12]),
		.c_f_ceb(c_f_ceb_l[12]),
		.c_f_web(c_f_web_l[12]),

		.pe_clear(pe_clear_l[12]),
		.pe_en(pe_en_l[12]),
		.shift(shift_l[12]),

		.norm_on(norm_on_l[12]),
		.relu_on(relu_on_l[12]),

		.l12_done(l12_done)
	);

	stl13_ctrl_v0
	#(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L13_W(ADDR_START_L13_W), 
        .ADDR_START_L13_F(ADDR_START_L13_F)
	)
	xstl13_ctrl_v0(
		.clk(clk),
		.rstb(rstb),
		.layer_state(layer_state),
		.layer_start(layer_start),

		.buffer_start(buffer_start_l[13]),
		.buffer_mode_f(buffer_mode_f_l[13]),
		.buffer_mode_w(buffer_mode_w_l[13]),
		.buffer_loc_w(buffer_loc_w_l[13]),
		.buffer_loc_f(buffer_loc_f_l[13]),
		.buffer_load_w(buffer_load_w_l[13]),
		.buffer_load_f(buffer_load_f_l[13]),

		.fsram_addr(c_f_addr_l[13]),
		.wsram_addr(c_w_addr_l[13]),
		.c_w_ceb(c_w_ceb_l[13]),
		.c_w_web(c_w_web_l[13]),
		.c_f_ceb(c_f_ceb_l[13]),
		.c_f_web(c_f_web_l[13]),

		.pe_clear(pe_clear_l[13]),
		.pe_en(pe_en_l[13]),
		.shift(shift_l[13]),

		.norm_on(norm_on_l[13]),
		.relu_on(relu_on_l[13]),

		.l13_done(l13_done)
	);

	stl14_ctrl_v0
	#(
    .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
    .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
    .WIDTH_W_DATA(WIDTH_W_DATA),
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
    .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
    .ADDR_START_L14_W(ADDR_START_L14_W), // 94
    .ADDR_START_L14_F(ADDR_START_L14_F), // 81    
    .ADDR_START_L15_F(ADDR_START_L15_F) // 94 - for store ADDR	
	)
	xstl14_ctrl_v0(
		.clk(clk),
		.rstb(rstb),
		.layer_state(layer_state),
		.layer_start(layer_start),

		.buffer_start(buffer_start_l[14]),
		.buffer_mode_f(buffer_mode_f_l[14]),
		.buffer_mode_w(buffer_mode_w_l[14]),
		.buffer_loc_w(buffer_loc_w_l[14]),
		.buffer_loc_f(buffer_loc_f_l[14]),
		.buffer_load_w(buffer_load_w_l[14]),
		.buffer_load_f(buffer_load_f_l[14]),

		.fsram_addr(c_f_addr_l[14]),
		.wsram_addr(c_w_addr_l[14]),
		.c_w_ceb(c_w_ceb_l[14]),
		.c_w_web(c_w_web_l[14]),
		.c_f_ceb(c_f_ceb_l[14]),
		.c_f_web(c_f_web_l[14]),

		.pe_clear(pe_clear_l[14]),
		.pe_en(pe_en_l[14]),
		.shift(shift_l[14]),

		.norm_on(norm_on_l[14]),
		.relu_on(relu_on_l[14]),

		.l14_done(l14_done)
	);

     stl15_ctrl_v0
    #(    
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L15_W(ADDR_START_L15_W), 
        .ADDR_START_L15_F(ADDR_START_L15_F)
    )
    xstl15_ctrl_v0(
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
        
        .buffer_start(buffer_start_l[15]),
        .buffer_mode_f(buffer_mode_f_l[15]),
        .buffer_mode_w(buffer_mode_w_l[15]),
        .buffer_loc_w(buffer_loc_w_l[15]),
        .buffer_loc_f(buffer_loc_f_l[15]),
        .buffer_load_w(buffer_load_w_l[15]),
        .buffer_load_f(buffer_load_f_l[15]),
        
        .fsram_addr(c_f_addr_l[15]),
        .wsram_addr(c_w_addr_l[15]),
        .c_w_ceb(c_w_ceb_l[15]), 
        .c_w_web(c_w_web_l[15]),
        .c_f_ceb(c_f_ceb_l[15]),
        .c_f_web(c_f_web_l[15]),
        
        .pe_clear(pe_clear_l[15]),
        .pe_en(pe_en_l[15]),
        .shift(shift_l[15]),
        
        .norm_on(norm_on_l[15]),
        .relu_on(relu_on_l[15]),
        
        .l15_done(l15_done)
        );         

     stl16_ctrl_v0
    #(    
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_L16_W(ADDR_START_L16_W), 
        .ADDR_START_L16_F(ADDR_START_L16_F)
    )
    xstl16_ctrl_v0(
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
        
        .buffer_start(buffer_start_l[16]),
        .buffer_mode_f(buffer_mode_f_l[16]),
        .buffer_mode_w(buffer_mode_w_l[16]),
        .buffer_loc_w(buffer_loc_w_l[16]),
        .buffer_loc_f(buffer_loc_f_l[16]),
        .buffer_load_w(buffer_load_w_l[16]),
        .buffer_load_f(buffer_load_f_l[16]),
        
        .fsram_addr(c_f_addr_l[16]),
        .wsram_addr(c_w_addr_l[16]),
        .c_w_ceb(c_w_ceb_l[16]), 
        .c_w_web(c_w_web_l[16]),
        .c_f_ceb(c_f_ceb_l[16]),
        .c_f_web(c_f_web_l[16]),
        
        .pe_clear(pe_clear_l[16]),
        .pe_en(pe_en_l[16]),
        .shift(shift_l[16]),
        
        .norm_on(norm_on_l[16]),
        .relu_on(relu_on_l[16]),
        
        .l16_done(l16_done)
    );
    
    stl17_ctrl_v0
    #(    
    .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
    .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
    .WIDTH_W_DATA(WIDTH_W_DATA),
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
    .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
    .ADDR_START_L17_W(ADDR_START_L17_W), // 132
    .ADDR_START_L17_F(ADDR_START_L17_F) // 101   
    )
    xstl17_ctrl_v0(
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
        
        .buffer_start(buffer_start_l[17]),
        .buffer_mode_f(buffer_mode_f_l[17]),
        .buffer_mode_w(buffer_mode_w_l[17]),
        .buffer_loc_w(buffer_loc_w_l[17]),
        .buffer_loc_f(buffer_loc_f_l[17]),
        .buffer_load_w(buffer_load_w_l[17]),
        .buffer_load_f(buffer_load_f_l[17]),
        
        .fsram_addr(c_f_addr_l[17]),
        .wsram_addr(c_w_addr_l[17]),
        .c_w_ceb(c_w_ceb_l[17]), 
        .c_w_web(c_w_web_l[17]),
        .c_f_ceb(c_f_ceb_l[17]),
        .c_f_web(c_f_web_l[17]),
        
        .pe_clear(pe_clear_l[17]),
        .pe_en(pe_en_l[17]),
        .shift(shift_l[17]),
        
        .norm_on(norm_on_l[17]),
        .relu_on(relu_on_l[17]),
        
        .l17_done(l17_done)
    );         
    
     stl18_ctrl_v0
    #(    
    .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
    .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
    .WIDTH_W_DATA(WIDTH_W_DATA),
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
    .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
    .ADDR_START_L18_W(ADDR_START_L18_W), // 140
    .ADDR_START_L18_F(ADDR_START_L18_F) // 107 
    )
    xstl18_ctrl_v0(
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
        
        .buffer_start(buffer_start_l[18]),
        .buffer_mode_f(buffer_mode_f_l[18]),
        .buffer_mode_w(buffer_mode_w_l[18]),
        .buffer_loc_w(buffer_loc_w_l[18]),
        .buffer_loc_f(buffer_loc_f_l[18]),
        .buffer_load_w(buffer_load_w_l[18]),
        .buffer_load_f(buffer_load_f_l[18]),
        
        .fsram_addr(c_f_addr_l[18]),
        .wsram_addr(c_w_addr_l[18]),
        .c_w_ceb(c_w_ceb_l[18]), 
        .c_w_web(c_w_web_l[18]),
        .c_f_ceb(c_f_ceb_l[18]),
        .c_f_web(c_f_web_l[18]),
        
        .pe_clear(pe_clear_l[18]),
        .pe_en(pe_en_l[18]),
        .shift(shift_l[18]),
        
        .norm_on(norm_on_l[18]),
        .relu_on(relu_on_l[18]),
        
        .l18_done(l18_done)
    );         
    
     stl19_ctrl_v0
    #(    
    .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
    .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
    .WIDTH_W_DATA(WIDTH_W_DATA),
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
    .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR), 
    .ADDR_START_L19_W(ADDR_START_L19_W), // 148
    .ADDR_START_L19_F(ADDR_START_L19_F) // 113 
    )
    xstl19_ctrl_v0(
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
        
        .buffer_start(buffer_start_l[19]),
        .buffer_mode_f(buffer_mode_f_l[19]),
        .buffer_mode_w(buffer_mode_w_l[19]),
        .buffer_loc_w(buffer_loc_w_l[19]),
        .buffer_loc_f(buffer_loc_f_l[19]),
        .buffer_load_w(buffer_load_w_l[19]),
        .buffer_load_f(buffer_load_f_l[19]),
        
        .fsram_addr(c_f_addr_l[19]),
        .wsram_addr(c_w_addr_l[19]),
        .c_w_ceb(c_w_ceb_l[19]), 
        .c_w_web(c_w_web_l[19]),
        .c_f_ceb(c_f_ceb_l[19]),
        .c_f_web(c_f_web_l[19]),
        
        .pe_clear(pe_clear_l[19]),
        .pe_en(pe_en_l[19]),
        .shift(shift_l[19]),
        
        .norm_on(norm_on_l[19]),
        .relu_on(relu_on_l[19]),
        
        .l19_done(l19_done)
    );     
    
    // L20 -> new
     stl20_ctrl_v0
    #(    
    .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
    .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
    .WIDTH_W_DATA(WIDTH_W_DATA),
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
    .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
    .ADDR_START_L20_W(ADDR_START_L20_W), // 212
    .ADDR_START_L20_F(ADDR_START_L20_F), // 107 
    .ADDR_START_L21_F(ADDR_START_L21_F) // 115 - ADDR for store  
    )
    xstl20_ctrl_v0(
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
        
        .buffer_start(buffer_start_l[20]),
        .buffer_mode_f(buffer_mode_f_l[20]),
        .buffer_mode_w(buffer_mode_w_l[20]),
        .buffer_loc_w(buffer_loc_w_l[20]),
        .buffer_loc_f(buffer_loc_f_l[20]),
        .buffer_load_w(buffer_load_w_l[20]),
        .buffer_load_f(buffer_load_f_l[20]),
        
        .fsram_addr(c_f_addr_l[20]),
        .wsram_addr(c_w_addr_l[20]),
        .c_w_ceb(c_w_ceb_l[20]), 
        .c_w_web(c_w_web_l[20]),
        .c_f_ceb(c_f_ceb_l[20]),
        .c_f_web(c_f_web_l[20]),
        
        .pe_clear(pe_clear_l[20]),
        .pe_en(pe_en_l[20]),
        .shift(shift_l[20]),
        
        .norm_on(norm_on_l[20]),
        .relu_on(relu_on_l[20]),
        
        .l20_done(l20_done)
    );      
    
    
     stl21_ctrl_v0
    #(    
    .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
    .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
    .WIDTH_W_DATA(WIDTH_W_DATA),
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
    .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR), 
    .ADDR_START_L21_W(ADDR_START_L21_W), // 218
    .ADDR_START_L21_F(ADDR_START_L21_F) // 115  
    )
    xstl21_ctrl_v0(
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
        
        .buffer_start(buffer_start_l[21]),
        .buffer_mode_f(buffer_mode_f_l[21]),
        .buffer_mode_w(buffer_mode_w_l[21]),
        .buffer_loc_w(buffer_loc_w_l[21]),
        .buffer_loc_f(buffer_loc_f_l[21]),
        .buffer_load_w(buffer_load_w_l[21]),
        .buffer_load_f(buffer_load_f_l[21]),
        
        .fsram_addr(c_f_addr_l[21]),
        .wsram_addr(c_w_addr_l[21]),
        .c_w_ceb(c_w_ceb_l[21]), 
        .c_w_web(c_w_web_l[21]),
        .c_f_ceb(c_f_ceb_l[21]),
        .c_f_web(c_f_web_l[21]),
        
        .pe_clear(pe_clear_l[21]),
        .pe_en(pe_en_l[21]),
        .shift(shift_l[21]),
        
        .norm_on(norm_on_l[21]),
        .relu_on(relu_on_l[21]),
        
        .l21_done(l21_done)
    );         
    
     stl22_ctrl_v0
    #(    
    .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
    .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
    .WIDTH_W_DATA(WIDTH_W_DATA),
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
    .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
    .ADDR_START_L22_W(ADDR_START_L22_W), // 219
    .ADDR_START_L22_F(ADDR_START_L22_F), // 119
    .NUM_POOL(NUM_POOL)
    )
    xstl22_ctrl_v0(
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
        
        .buffer_start(buffer_start_l[TOTAL_LAYER_NUM-1]),
        .buffer_mode_f(buffer_mode_f_l[TOTAL_LAYER_NUM-1]),
        .buffer_mode_w(buffer_mode_w_l[TOTAL_LAYER_NUM-1]),
        .buffer_loc_w(buffer_loc_w_l[TOTAL_LAYER_NUM-1]),
        .buffer_loc_f(buffer_loc_f_l[TOTAL_LAYER_NUM-1]),
        .buffer_load_w(buffer_load_w_l[TOTAL_LAYER_NUM-1]),
        .buffer_load_f(buffer_load_f_l[TOTAL_LAYER_NUM-1]),
        
        .fsram_addr(c_f_addr_l[TOTAL_LAYER_NUM-1]),
        .wsram_addr(c_w_addr_l[TOTAL_LAYER_NUM-1]),
        .c_w_ceb(c_w_ceb_l[TOTAL_LAYER_NUM-1]), 
        .c_w_web(c_w_web_l[TOTAL_LAYER_NUM-1]),
        .c_f_ceb(c_f_ceb_l[TOTAL_LAYER_NUM-1]),
        .c_f_web(c_f_web_l[TOTAL_LAYER_NUM-1]),
        
        .pe_clear(pe_clear_l[TOTAL_LAYER_NUM-1]),
        .pe_en(pe_en_l[TOTAL_LAYER_NUM-1]),
        .shift(shift_l[TOTAL_LAYER_NUM-1]),
        
        .norm_on(norm_on_l[TOTAL_LAYER_NUM-1]),
        .relu_on(relu_on_l[TOTAL_LAYER_NUM-1]),
        
        .l22_done(l22_done)
    );
        
        
    // 30
    wire clf_mode_o, clf_en_o,clf_clear_o;
    reg clf_mode_n, clf_en_n, clf_clear_n;
        
     stAvgpool_ctrl_v1 #(
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .ADDR_START_LAVGPOOL(ADDR_START_LAVGPOOL), //121
        .NUM_POOL(NUM_POOL) //22
    ) 
    xstAvgpool_ctrl_v1 (
        .clk(clk),
        .rstb(rstb),
        .layer_state(layer_state),
        .layer_start(layer_start),
         
        .layer_done(lavg_done),
        .clf_mode(clf_mode_o),
        .en_avgpool(clf_en_o),
        
        .c_f_ceb(c_f_ceb_l[TOTAL_LAYER_NUM]),
        .c_f_web(c_f_web_l[TOTAL_LAYER_NUM]),
        
        .clear(clf_clear_o),
        .ADDR(c_f_addr_l[TOTAL_LAYER_NUM]),
        
        .wsram_addr(c_w_addr_l[TOTAL_LAYER_NUM]),
        .c_w_ceb(c_w_ceb_l[TOTAL_LAYER_NUM]),
        .c_w_web(c_w_web_l[TOTAL_LAYER_NUM]),
        .buffer_start(buffer_start_l[TOTAL_LAYER_NUM]),
        .buffer_mode_f(buffer_mode_f_l[TOTAL_LAYER_NUM]),
        .buffer_mode_w(buffer_mode_w_l[TOTAL_LAYER_NUM]),
        .buffer_loc_w(buffer_loc_w_l[TOTAL_LAYER_NUM]),
        .buffer_loc_f(buffer_loc_f_l[TOTAL_LAYER_NUM]),
        .buffer_load_w(buffer_load_w_l[TOTAL_LAYER_NUM]),
        .buffer_load_f(buffer_load_f_l[TOTAL_LAYER_NUM]),
        .pe_clear(pe_clear_l[TOTAL_LAYER_NUM]),
        .pe_en(pe_en_l[TOTAL_LAYER_NUM]),        
        .shift(shift_l[TOTAL_LAYER_NUM]),        
        .norm_on(norm_on_l[TOTAL_LAYER_NUM]),
        .relu_on(relu_on_l[TOTAL_LAYER_NUM])      
    );
    
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            clf_mode_n <= 0;
            clf_en_n <= 0;
            clf_clear_n <= 0;
        end else begin
            clf_mode_n <= clf_mode_o;
            clf_en_n <= clf_en_o;
            clf_clear_n <= clf_clear_o;
        end
    end     
     
    assign pe_clear_l[0] = 0;
    assign pe_en_l[0] = 0;
    assign buffer_start_l[0] = 0;
    assign buffer_mode_f_l[0] = 0;
    assign buffer_mode_w_l[0] = 0;
    assign buffer_loc_w_l[0] = 0;
    assign buffer_loc_f_l[0] = 0;
    assign buffer_load_w_l[0] = 0;
    assign buffer_load_f_l[0] = 0;
    assign shift_l[0] = 0;
    assign norm_on_l[0] = 0;
    assign relu_on_l[0] = 0;
    assign c_f_addr_l[0] = 0;
    assign c_w_addr_l[0] = 0;
    assign c_w_ceb_l[0] = 1;
    assign c_w_web_l[0] = 1;
    assign c_f_ceb_l[0] = 1;
    assign c_f_web_l[0] = 1;


        
        
    always @(*) begin 
        pe_clear = pe_clear_l[layer_state]; 
        pe_en = pe_en_l[layer_state];
        buffer_start = buffer_start_l[layer_state];
        buffer_mode_f = buffer_mode_f_l[layer_state];
        buffer_mode_w = buffer_mode_w_l[layer_state];
        buffer_loc_w = buffer_loc_w_l[layer_state];
        buffer_loc_f = buffer_loc_f_l[layer_state];
        buffer_load_w = buffer_load_w_l[layer_state];
        buffer_load_f = buffer_load_f_l[layer_state];
        shift = shift_l[layer_state];
        norm_on = norm_on_l[layer_state];
        relu_on = relu_on_l[layer_state];
        c_f_addr = c_f_addr_l[layer_state];
        c_w_addr = c_w_addr_l[layer_state];
        c_w_ceb = c_w_ceb_l[layer_state];
        c_w_web = c_w_web_l[layer_state];
        c_f_ceb = c_f_ceb_l[layer_state];
        c_f_web = c_f_web_l[layer_state];
        if(layer_state == 23) begin
            clf_mode = clf_mode_n;
            clf_en = clf_en_n;
            clf_clear = clf_clear_n;
        end
        else begin 
            clf_mode = 0;
            clf_en = 0;
            clf_clear = 0;
        end
    end
    
endmodule
