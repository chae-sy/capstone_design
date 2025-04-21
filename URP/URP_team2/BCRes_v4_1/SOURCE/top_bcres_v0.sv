`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Donghwan So
// 
// Create Date: 2024/09/30 22:32:51
// Design Name: 
// Module Name: top_BCRes
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


module top_BCRes
#(
    parameter WIDTH_WSRAM_WL = 128,
    parameter WIDTH_FSRAM_WL = 128,
    parameter WIDTH_W_DATA = 8,
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_FSRAM_ADDR = 8,
    parameter WIDTH_WSRAM_ADDR = 8,
    
    parameter WIDTH_PE_O_DATA = 20,
    parameter WIDTH_NORM_O_DATA = 21,   // WIDTH_NORM_O_DATA == WIDTH_PE_O_DATA + 1
    parameter WIDTH_O_DATA = 8,
    
    parameter WIDTH_L1_PE_IL = 8,
    parameter WIDTH_L1_B_IL = 0,
    parameter WIDTH_L1_NORM_IL = 9,     // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L1_O_IL = 1,
    
    parameter WIDTH_L2_PE_IL = 6,
    parameter WIDTH_L2_B_IL = 0,
    parameter WIDTH_L2_NORM_IL = 7,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L2_O_IL = 1,
    
    parameter WIDTH_L3_PE_IL = 11,
    parameter WIDTH_L3_B_IL = 1,
    parameter WIDTH_L3_NORM_IL = 12,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L3_O_IL = 1,
    
    parameter WIDTH_L4_PE_IL = 6,
    parameter WIDTH_L4_B_IL = 2, //Nan
    parameter WIDTH_L4_NORM_IL = 7,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L4_O_IL = 0,

    parameter WIDTH_L5_PE_IL = 7,
    parameter WIDTH_L5_B_IL = 1,
    parameter WIDTH_L5_NORM_IL = 8,     // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L5_O_IL = 0,
    
    parameter WIDTH_L6_PE_IL = 5,
    parameter WIDTH_L6_B_IL = 2, //nan
    parameter WIDTH_L6_NORM_IL = 6,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L6_O_IL = 0,
    
    parameter WIDTH_L7_PE_IL = 5,
    parameter WIDTH_L7_B_IL = 2, //nan
    parameter WIDTH_L7_NORM_IL = 6,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L7_O_IL = 1,
    
    parameter WIDTH_L8_PE_IL = 7,
    parameter WIDTH_L8_B_IL = 2, //nan
    parameter WIDTH_L8_NORM_IL = 8,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L8_O_IL = 1,
    
    parameter WIDTH_L9_PE_IL = 7,
    parameter WIDTH_L9_B_IL = 0,
    parameter WIDTH_L9_NORM_IL = 8,     // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L9_O_IL = 1,
    
    parameter WIDTH_L10_PE_IL = 9,
    parameter WIDTH_L10_B_IL = 1,
    parameter WIDTH_L10_NORM_IL = 10,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L10_O_IL = 1,
    
    parameter WIDTH_L11_PE_IL = 6,
    parameter WIDTH_L11_B_IL = 2, //nan
    parameter WIDTH_L11_NORM_IL = 7,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L11_O_IL = 0,
    
    parameter WIDTH_L12_PE_IL = 7,
    parameter WIDTH_L12_B_IL = 0,
    parameter WIDTH_L12_NORM_IL = 8,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L12_O_IL = 0,

    parameter WIDTH_L13_PE_IL = 5,
    parameter WIDTH_L13_B_IL = 2, //nan
    parameter WIDTH_L13_NORM_IL = 6,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L13_O_IL = 0,
    
    parameter WIDTH_L14_PE_IL = 5,
    parameter WIDTH_L14_B_IL = 2, //nan
    parameter WIDTH_L14_NORM_IL = 6,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L14_O_IL = 0,
                
    parameter WIDTH_L15_PE_IL = 6,
    parameter WIDTH_L15_B_IL = 2, //nan
    parameter WIDTH_L15_NORM_IL = 7,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L15_O_IL = 0,    
    
    parameter WIDTH_L16_PE_IL = 6,
    parameter WIDTH_L16_B_IL = 0,
    parameter WIDTH_L16_NORM_IL = 7,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L16_O_IL = 1,
    
    parameter WIDTH_L17_PE_IL = 9,
    parameter WIDTH_L17_B_IL = 1,
    parameter WIDTH_L17_NORM_IL = 10,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L17_O_IL = 1,
    
    parameter WIDTH_L18_PE_IL = 7,
    parameter WIDTH_L18_B_IL = 0,
    parameter WIDTH_L18_NORM_IL = 8,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L18_O_IL = 0,   
    
    parameter WIDTH_L19_PE_IL = 5,
    parameter WIDTH_L19_B_IL = 2, //nan
    parameter WIDTH_L19_NORM_IL = 6,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L19_O_IL = 0,

    parameter WIDTH_L20_PE_IL = 5,
    parameter WIDTH_L20_B_IL = 2, //nan
    parameter WIDTH_L20_NORM_IL = 6,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L20_O_IL = 0,

    parameter WIDTH_L21_PE_IL = 6,
    parameter WIDTH_L21_B_IL = 2, //nan
    parameter WIDTH_L21_NORM_IL = 7,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L21_O_IL = 0,

    parameter WIDTH_L22_PE_IL = 12,
    parameter WIDTH_L22_B_IL = 3,
    parameter WIDTH_L22_NORM_IL = 13,    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
    parameter WIDTH_L22_O_IL = 7,
        
    parameter ADDR_START_L1_W = 0,
    parameter ADDR_START_L1_F = 0,
    
    parameter SIZE_KERNEL_H = 3,
    parameter SIZE_KERNEL_W = 3,    
    
    parameter NUM_PE = 16,
    parameter WR_DELAY = 6,
    
    parameter RELU_MAX_VAL = 6,
    
    // ADDR
    parameter SIZE_L1_OUT_CHANNEL = 16,
    //L2
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
    //L23
    parameter ADDR_START_LAVGPOOL = 121,     
    parameter NUM_POOL = 22,
    
    parameter WIDTH_EXTEND = $clog2(NUM_POOL)
) 
(
    input       rstb,
    input       clk,
    input       start,
    
    //From OFF chip & for weight write
    input                               wr_wsram_sclk,              // clk generated by FPGA
    input                               wr_wsram_ss,               // ss  generated by FPGA
    input                               wr_wsram_sdata,
    input                               wr_weight_on,
    
    //From ON chip & for feature write
    input                               wr_fsram_clk,
    input   [WIDTH_FSRAM_WL-1:0]        wr_fsram_data,
    input   [WIDTH_FSRAM_ADDR-1:0]      wr_fsram_addr,
    input                               wr_fsram_ceb,
    input                               wr_fsram_web,
    input                               wr_fsram_mux,
    
    
    output  [3:0]                       max_index
);

    // sginal for controller output to each module input
    wire [WIDTH_WSRAM_ADDR-1:0] c_w_addr;
    wire [WIDTH_FSRAM_ADDR-1:0] c_f_addr;
    
    wire [4:0] buffer_mode_w;
    wire [4:0] buffer_mode_f;
    
    wire buffer_start; // signal for starting to send buffer output to PE
    
    wire [3:0] buffer_loc_w;  // signal for describe the location of data to be stored
    wire [3:0] buffer_loc_f;
    
    wire buffer_load_w;     // signal for starting to load data into buffer
    wire buffer_load_f;
    
    wire shift;
    
    wire norm_on;
    wire relu_on;
    
    wire c_w_ceb;
    wire c_w_web;
    
    wire c_f_ceb;
    wire c_f_web;
    
    wire [4:0] layer_state;
    
    wire pe_clear;
    wire pe_en;
    
    wire clf_mode;
    wire clf_en;
    wire clf_clear;
    
    wire lavg_done;
    
    layer_state_ctrl_v0 #(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .NUM_PE(NUM_PE),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        
        .SIZE_L1_OUT_CHANNEL(SIZE_L1_OUT_CHANNEL),
        .ADDR_START_L1_W(ADDR_START_L1_W),
        .ADDR_START_L1_F(ADDR_START_L1_F),
        .SIZE_KERNEL_H(SIZE_KERNEL_H),
        .SIZE_KERNEL_W(SIZE_KERNEL_W),
        
        .ADDR_START_L2_W(ADDR_START_L2_W), 
        .ADDR_START_L2_F(ADDR_START_L2_F),
        .NUM_L1_OUT_CHANNEL(NUM_L1_OUT_CHANNEL),
        .NUM_L2_OUT_CHANNEL(NUM_L2_OUT_CHANNEL),
        
        .ADDR_START_L3_W(ADDR_START_L3_W), 
        .ADDR_START_L3_F(ADDR_START_L3_F),
        
        .ADDR_START_L4_W(ADDR_START_L4_W), 
        .ADDR_START_L4_F(ADDR_START_L4_F),
        .SIZE_L3_OUT_CHANNEL(SIZE_L3_OUT_CHANNEL),
        .NUM_L4_OUT_CHANNEL(NUM_L4_OUT_CHANNEL),
        
        .ADDR_START_L5_W(ADDR_START_L5_W), 
        .ADDR_START_L5_F(ADDR_START_L5_F),
        
        .ADDR_START_L6_W(ADDR_START_L6_W), 
        .ADDR_START_L6_F(ADDR_START_L6_F),
        
        .ADDR_START_L7_W(ADDR_START_L7_W), 
        .ADDR_START_L7_F(ADDR_START_L7_F),
        
        .ADDR_START_L8_W(ADDR_START_L8_W), 
        .ADDR_START_L8_F(ADDR_START_L8_F),
        
        .ADDR_START_L9_W(ADDR_START_L9_W), 
        .ADDR_START_L9_F(ADDR_START_L9_F),
        
        .ADDR_START_L10_W(ADDR_START_L10_W), 
        .ADDR_START_L10_F(ADDR_START_L10_F),
        
        .ADDR_START_L11_W(ADDR_START_L11_W), 
        .ADDR_START_L11_F(ADDR_START_L11_F),
        .SIZE_L10_OUT_CHANNEL(SIZE_L10_OUT_CHANNEL),
        .NUM_L11_OUT_CHANNEL(NUM_L11_OUT_CHANNEL),
        
        .ADDR_START_L12_W(ADDR_START_L12_W), 
        .ADDR_START_L12_F(ADDR_START_L12_F),
        
        .ADDR_START_L13_W(ADDR_START_L13_W), 
        .ADDR_START_L13_F(ADDR_START_L13_F),
        
        .ADDR_START_L14_W(ADDR_START_L14_W), // 94
        .ADDR_START_L14_F(ADDR_START_L14_F), // 81
        
        .ADDR_START_L15_W(ADDR_START_L15_W),    
        .ADDR_START_L15_F(ADDR_START_L15_F), // 94
           
        .ADDR_START_L16_W(ADDR_START_L16_W), 
        .ADDR_START_L16_F(ADDR_START_L16_F),
           
        .ADDR_START_L17_W(ADDR_START_L17_W), // 132
        .ADDR_START_L17_F(ADDR_START_L17_F), // 10
        
        .ADDR_START_L18_W(ADDR_START_L18_W), // 140
        .ADDR_START_L18_F(ADDR_START_L18_F), // 107
           
        .ADDR_START_L19_W(ADDR_START_L19_W), // 148
        .ADDR_START_L19_F(ADDR_START_L19_F), // 113
         
        .ADDR_START_L20_W(ADDR_START_L20_W), // 212
        .ADDR_START_L20_F(ADDR_START_L20_F), // 107
           
        .ADDR_START_L21_W(ADDR_START_L21_W), // 218
        .ADDR_START_L21_F(ADDR_START_L21_F), // 115
        
        .ADDR_START_L22_W(ADDR_START_L22_W), // 219
        .ADDR_START_L22_F(ADDR_START_L22_F), // 119
        .NUM_POOL(NUM_POOL), //22
        .ADDR_START_LAVGPOOL(ADDR_START_LAVGPOOL) //121       
    )
    xlayer_state_ctrl_v0(
        .clk(clk),
        .rstb(rstb),
        .start_nn(start),
    
        .layer_state(layer_state),
        .pe_clear(pe_clear),
        .pe_en(pe_en),
        .buffer_start(buffer_start),
        .buffer_mode_f(buffer_mode_f),
        .buffer_mode_w(buffer_mode_w),
        .buffer_loc_w(buffer_loc_w),
        .buffer_loc_f   (buffer_loc_f),
        .buffer_load_w(buffer_load_w),
        .buffer_load_f(buffer_load_f),
        .shift(shift),
        
        .norm_on(norm_on),
        .relu_on(relu_on),
    
        .c_f_addr(c_f_addr),
        .c_w_addr(c_w_addr),
    
        .c_w_ceb(c_w_ceb),
        .c_w_web(c_w_web),
        .c_f_ceb(c_f_ceb),
        .c_f_web(c_f_web),
        
        .clf_mode(clf_mode),
        .clf_en(clf_en),
        .clf_clear(clf_clear),
        
        .lavg_done(lavg_done)
    );
    
    //signal for weight sram input 
    wire WSRAM_CLK;
    wire WSRAM_WEB;
    wire WSRAM_CEB;
    wire [WIDTH_WSRAM_ADDR-1:0] WSRAM_ADDR;
    wire [WIDTH_WSRAM_WL-1:0] WSRAM_out;
    
    //signal from write spi module output
    wire [WIDTH_WSRAM_ADDR-1:0] ww_wsram_addr;
    wire [WIDTH_WSRAM_WL-1:0] ww_wsram_data;
    wire ww_wsram_clk;
    wire ww_wsram_web;
    wire ww_wsram_ceb;
    
    //signal from weight buffer output ( PE input )
    wire [WIDTH_W_DATA*NUM_PE-1:0] PE_in_w;
    
    sram_weight_write_v0 #(
        .addr_width(WIDTH_WSRAM_ADDR),
        .data_width(WIDTH_WSRAM_WL),
        .wr_delay(WR_DELAY)
    )
    xsram_weight_write (
        .rstb(rstb),
        .sram_sclk(wr_wsram_sclk),           
        .sram_ss(wr_wsram_ss),        
        .sram_sdata(wr_wsram_sdata),
        .wr_weight_on(wr_weight_on),
        
        .sram_addr(ww_wsram_addr),
        .sram_data(ww_wsram_data),
        .sram_clk(ww_wsram_clk),           
        .sram_web(ww_wsram_web),
        .sram_ceb(ww_wsram_ceb)
    );
    
    // Signal Muxing between from write module and process controller
    assign WSRAM_CLK = (wr_weight_on)? ww_wsram_clk: clk;
    assign WSRAM_CEB = (wr_weight_on)? ww_wsram_ceb: c_w_ceb;
    assign WSRAM_WEB = (wr_weight_on)? ww_wsram_web: c_w_web;
    assign WSRAM_ADDR = (wr_weight_on)? ww_wsram_addr: c_w_addr;
    
    memory_wrapper_weight
    xW_SRAM
    (
        .CLK(WSRAM_CLK),
        .CEN(WSRAM_CEB),
        .WEN(WSRAM_WEB),
        .A(WSRAM_ADDR),
        .D(ww_wsram_data),
        .Q(WSRAM_out)
    );
    
    w_buffer_v0#(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .NUM_PE(NUM_PE),
        .SIZE_KERNEL_W(SIZE_KERNEL_W),
        .SIZE_KERNEL_H(SIZE_KERNEL_H)
    )
    xw_buffer_v0(
        .clk(clk),
        .rstb(rstb),
        .buffer_mode(buffer_mode_w),
        .buffer_load_w(buffer_load_w),
        .buffer_loc_w(buffer_loc_w),
        .buffer_start(buffer_start),
        .w_data(WSRAM_out),            
        .w_buffer_out(PE_in_w)
    );
    
    
    //signal for feature sram inpu
    wire FSRAM_CLK;
    wire [WIDTH_FSRAM_WL-1:0] FSRAM_D;
    wire [WIDTH_FSRAM_ADDR-1:0] FSRAM_ADDR;
    wire FSRAM_WEB;
    wire FSRAM_CEB;
    wire [WIDTH_FSRAM_WL-1:0] FSRAM_out;
    
    //signal for feature buffer output ( PE input )
    wire [WIDTH_F_DATA*NUM_PE-1:0] PE_in_f;
    
    //signal for layer output ( fsram input )
    wire [WIDTH_FSRAM_WL-1:0] fsram_layer_out;
    
    // Signal Muxing between from write input and process controller
    assign FSRAM_CLK = (wr_fsram_mux) ? wr_fsram_clk : clk ;
    assign FSRAM_D = (wr_fsram_mux) ? wr_fsram_data : fsram_layer_out ;
    assign FSRAM_ADDR = (wr_fsram_mux) ? wr_fsram_addr : c_f_addr ;
    assign FSRAM_WEB = (wr_fsram_mux) ? wr_fsram_web : c_f_web ;
    assign FSRAM_CEB = (wr_fsram_mux) ? wr_fsram_ceb : c_f_ceb ;
    
    
    
    memory_wrapper_input 
    xF_SRAM
    (
        .CLK(FSRAM_CLK),
        .CEN(FSRAM_CEB),
        .WEN(FSRAM_WEB),
        .A(FSRAM_ADDR),
        .D(FSRAM_D),                               // How do data get in to the SRAM? FE? or something? 
        .Q(FSRAM_out)
    );
    
    f_buffer_v0#(
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .NUM_PE(NUM_PE),
        .SIZE_KERNEL_W(SIZE_KERNEL_W),
        .SIZE_KERNEL_H(SIZE_KERNEL_H)
    )
    xf_buffer_v0(
        .clk(clk),
        .rstb(rstb),
        .buffer_mode(buffer_mode_f),
        .buffer_load_f(buffer_load_f),
        .buffer_loc_f(buffer_loc_f),
        .buffer_start(buffer_start),
        .shift(shift),
        .f_data1(FSRAM_out[WIDTH_FSRAM_WL-1:WIDTH_FSRAM_WL-WIDTH_F_DATA]),             // # of f_data means that how many size needs to be convolution 
        .f_data2(FSRAM_out[WIDTH_FSRAM_WL-WIDTH_F_DATA-1:WIDTH_FSRAM_WL-2*WIDTH_F_DATA]),
        .f_data3(FSRAM_out[WIDTH_FSRAM_WL-2*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-3*WIDTH_F_DATA]),
        .f_data4(FSRAM_out[WIDTH_FSRAM_WL-3*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-4*WIDTH_F_DATA]),
        .f_data5(FSRAM_out[WIDTH_FSRAM_WL-4*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-5*WIDTH_F_DATA]),
        .f_data6(FSRAM_out[WIDTH_FSRAM_WL-5*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-6*WIDTH_F_DATA]),
        .f_data7(FSRAM_out[WIDTH_FSRAM_WL-6*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-7*WIDTH_F_DATA]),
        .f_data8(FSRAM_out[WIDTH_FSRAM_WL-7*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-8*WIDTH_F_DATA]),
        .f_data9(FSRAM_out[WIDTH_FSRAM_WL-8*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-9*WIDTH_F_DATA]),
        .f_data10(FSRAM_out[WIDTH_FSRAM_WL-9*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-10*WIDTH_F_DATA]),
        .f_data11(FSRAM_out[WIDTH_FSRAM_WL-10*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-11*WIDTH_F_DATA]),
        .f_data12(FSRAM_out[WIDTH_FSRAM_WL-11*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-12*WIDTH_F_DATA]),
        .f_data13(FSRAM_out[WIDTH_FSRAM_WL-12*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-13*WIDTH_F_DATA]),
        .f_data14(FSRAM_out[WIDTH_FSRAM_WL-13*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-14*WIDTH_F_DATA]),
        .f_data15(FSRAM_out[WIDTH_FSRAM_WL-14*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-15*WIDTH_F_DATA]),
        .f_data16(FSRAM_out[WIDTH_FSRAM_WL-15*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-16*WIDTH_F_DATA]),
        
        .f_buffer_out(PE_in_f)
    );
    
  
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT1;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT2;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT3;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT4;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT5;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT6;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT7;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT8;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT9;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT10;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT11;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT12;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT13;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT14;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT15;
    wire [WIDTH_PE_O_DATA-1:0] PE_OUT16;
    
    wire [WIDTH_NORM_O_DATA-1:0] norm_out1;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out2;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out3;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out4;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out5;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out6;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out7;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out8;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out9;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out10;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out11;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out12;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out13;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out14;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out15;
    wire [WIDTH_NORM_O_DATA-1:0] norm_out16;
    
    wire [WIDTH_O_DATA-1:0] layer_out1;
    wire [WIDTH_O_DATA-1:0] layer_out2;
    wire [WIDTH_O_DATA-1:0] layer_out3;
    wire [WIDTH_O_DATA-1:0] layer_out4;
    wire [WIDTH_O_DATA-1:0] layer_out5;
    wire [WIDTH_O_DATA-1:0] layer_out6;
    wire [WIDTH_O_DATA-1:0] layer_out7;
    wire [WIDTH_O_DATA-1:0] layer_out8;
    wire [WIDTH_O_DATA-1:0] layer_out9;
    wire [WIDTH_O_DATA-1:0] layer_out10;
    wire [WIDTH_O_DATA-1:0] layer_out11;
    wire [WIDTH_O_DATA-1:0] layer_out12;
    wire [WIDTH_O_DATA-1:0] layer_out13;
    wire [WIDTH_O_DATA-1:0] layer_out14;
    wire [WIDTH_O_DATA-1:0] layer_out15;
    wire [WIDTH_O_DATA-1:0] layer_out16;
    
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_1
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-1:WIDTH_F_DATA*NUM_PE-WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-1:WIDTH_W_DATA*NUM_PE-WIDTH_W_DATA]),
        .PE_out(PE_OUT1)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_1
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT1),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-1:WIDTH_W_DATA*NUM_PE-WIDTH_W_DATA]),
    .norm_out(norm_out1)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj1
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out1),
    
        .layer_out(layer_out1)
    );
    
    
    
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_2
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-WIDTH_F_DATA-1:WIDTH_F_DATA*NUM_PE-2*WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-2*WIDTH_W_DATA]),
        .PE_out(PE_OUT2)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_2
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT2),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-2*WIDTH_W_DATA]),
    .norm_out(norm_out2)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj2
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out2),
    
        .layer_out(layer_out2)
    );
    
    
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_3
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-2*WIDTH_F_DATA-1:WIDTH_F_DATA*NUM_PE-3*WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-2*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-3*WIDTH_W_DATA]),
        .PE_out(PE_OUT3)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_3
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT3),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-2*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-3*WIDTH_W_DATA]),
    .norm_out(norm_out3)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj3
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out3),
    
        .layer_out(layer_out3)
    );
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_4
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-3*WIDTH_F_DATA-1:WIDTH_F_DATA*NUM_PE-4*WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-3*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-4*WIDTH_W_DATA]),
        .PE_out(PE_OUT4)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_4
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT4),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-3*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-4*WIDTH_W_DATA]),
    .norm_out(norm_out4)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj4
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out4),
    
        .layer_out(layer_out4)
    );
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_5
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-4*WIDTH_F_DATA-1:WIDTH_F_DATA*NUM_PE-5*WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-4*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-5*WIDTH_W_DATA]),
        .PE_out(PE_OUT5)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_5
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT5),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-4*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-5*WIDTH_W_DATA]),
    .norm_out(norm_out5)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj5
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out5),
    
        .layer_out(layer_out5)
    );
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_6
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-5*WIDTH_F_DATA-1:WIDTH_F_DATA*NUM_PE-6*WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-5*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-6*WIDTH_W_DATA]),
        .PE_out(PE_OUT6)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_6
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT6),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-5*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-6*WIDTH_W_DATA]),
    .norm_out(norm_out6)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj6
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out6),
    
        .layer_out(layer_out6)
    );
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_7
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-6*WIDTH_F_DATA-1:WIDTH_F_DATA*NUM_PE-7*WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-6*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-7*WIDTH_W_DATA]),
        .PE_out(PE_OUT7)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_7
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT7),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-6*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-7*WIDTH_W_DATA]),
    .norm_out(norm_out7)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj7
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out7),
    
        .layer_out(layer_out7)
    );
    
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_8
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-7*WIDTH_F_DATA-1:WIDTH_F_DATA*NUM_PE-8*WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-7*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-8*WIDTH_W_DATA]),
        .PE_out(PE_OUT8)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_8
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT8),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-7*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-8*WIDTH_W_DATA]),
    .norm_out(norm_out8)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj8
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out8),
    
        .layer_out(layer_out8)
    );
    
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_9
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-8*WIDTH_F_DATA-1:WIDTH_F_DATA*NUM_PE-9*WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-8*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-9*WIDTH_W_DATA]),
        .PE_out(PE_OUT9)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_9
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT9),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-8*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-9*WIDTH_W_DATA]),
    .norm_out(norm_out9)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj9
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out9),
    
        .layer_out(layer_out9)
    );
    
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_10
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-9*WIDTH_F_DATA-1:WIDTH_F_DATA*NUM_PE-10*WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-9*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-10*WIDTH_W_DATA]),
        .PE_out(PE_OUT10)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_10
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT10),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-9*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-10*WIDTH_W_DATA]),
    .norm_out(norm_out10)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj10
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out10),
    
        .layer_out(layer_out10)
    );
    
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_11
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-10*WIDTH_F_DATA-1:WIDTH_F_DATA*NUM_PE-11*WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-10*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-11*WIDTH_W_DATA]),
        .PE_out(PE_OUT11)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_11
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT11),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-10*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-11*WIDTH_W_DATA]),
    .norm_out(norm_out11)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj11
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out11),
    
        .layer_out(layer_out11)
    );
    
    
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_12
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-11*WIDTH_F_DATA-1:WIDTH_F_DATA*NUM_PE-12*WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-11*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-12*WIDTH_W_DATA]),
        .PE_out(PE_OUT12)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_12
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT12),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-11*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-12*WIDTH_W_DATA]),
    .norm_out(norm_out12)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj12
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out12),
    
        .layer_out(layer_out12)
    );
    
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_13
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-12*WIDTH_F_DATA-1:WIDTH_F_DATA*NUM_PE-13*WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-12*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-13*WIDTH_W_DATA]),
        .PE_out(PE_OUT13)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_13
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT13),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-12*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-13*WIDTH_W_DATA]),
    .norm_out(norm_out13)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj13
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out13),
    
        .layer_out(layer_out13)
    );
    
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_14
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-13*WIDTH_F_DATA-1:WIDTH_F_DATA*NUM_PE-14*WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-13*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-14*WIDTH_W_DATA]),
        .PE_out(PE_OUT14)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_14
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT14),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-13*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-14*WIDTH_W_DATA]),
    .norm_out(norm_out14)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj14
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out14),
    
        .layer_out(layer_out14)
    );
    
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_15
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-14*WIDTH_F_DATA-1:WIDTH_F_DATA*NUM_PE-15*WIDTH_F_DATA]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-14*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-15*WIDTH_W_DATA]),
        .PE_out(PE_OUT15)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_15
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT15),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-14*WIDTH_W_DATA-1:WIDTH_W_DATA*NUM_PE-15*WIDTH_W_DATA]),
    .norm_out(norm_out15)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)  
    )
    relu_numadj15
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out15),
    
        .layer_out(layer_out15)
    );
    
    
    
    
    
    PE
    #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA)
    )
    PE_16
    (
        .clk(clk),
        .rstb(rstb),
        .clear(pe_clear),
        .pe_en(pe_en),
        .f_data(PE_in_f[WIDTH_F_DATA*NUM_PE-15*WIDTH_F_DATA-1:0]),
        .w_data(PE_in_w[WIDTH_W_DATA*NUM_PE-15*WIDTH_W_DATA-1:0]),
        .PE_out(PE_OUT16)
    );
    
    norm_v0
    #(
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL),
        
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL),
        
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL),
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),

        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL),
        
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL),

        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL),
        
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL)
    )
    norm_16
    (
    .clk(clk),
    .rstb(rstb),
    .layer_state(layer_state),
    .norm_on(norm_on),
    .pe_out(PE_OUT16),
    .bias(WSRAM_out[WIDTH_W_DATA*NUM_PE-15*WIDTH_W_DATA-1:0]),
    .norm_out(norm_out16)
    );
    
    relu_numadj_v0
    #(
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
        
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
        
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
        
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
        
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),
        
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),

        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),
        
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
        
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),

        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
        
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL)                                    
    )
    relu_numadj16
    (
        .relu_on(relu_on),
        .layer_state(layer_state),
        .norm_out(norm_out16),
    
        .layer_out(layer_out16)
    );
    
    assign fsram_layer_out = {layer_out1,layer_out2,layer_out3,layer_out4,layer_out5,layer_out6,layer_out7,layer_out8,layer_out9,layer_out10,layer_out11,layer_out12,layer_out13,layer_out14,layer_out15,layer_out16};
    
    wire [WIDTH_F_DATA+WIDTH_EXTEND-1:0] clf_out1;
    wire [WIDTH_F_DATA+WIDTH_EXTEND-1:0] clf_out2;
    wire [WIDTH_F_DATA+WIDTH_EXTEND-1:0] clf_out3;
    wire [WIDTH_F_DATA+WIDTH_EXTEND-1:0] clf_out4;
    wire [WIDTH_F_DATA+WIDTH_EXTEND-1:0] clf_out5;
    wire [WIDTH_F_DATA+WIDTH_EXTEND-1:0] clf_out6;
    wire [WIDTH_F_DATA+WIDTH_EXTEND-1:0] clf_out7;
    wire [WIDTH_F_DATA+WIDTH_EXTEND-1:0] clf_out8;
    wire [WIDTH_F_DATA+WIDTH_EXTEND-1:0] clf_out9;
    wire [WIDTH_F_DATA+WIDTH_EXTEND-1:0] clf_out10;
    wire [WIDTH_F_DATA+WIDTH_EXTEND-1:0] clf_out11;
    wire [WIDTH_F_DATA+WIDTH_EXTEND-1:0] clf_out12;
    
   
    classifier 
    #(
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .NUM_POOL(NUM_POOL),
    .WIDTH_EXTEND(WIDTH_EXTEND)
    )
    xclassifier_1
    ( 
    .clk(clk),
    .rstb(rstb),
    .clear(clf_clear),
    
    .en_avgpool(clf_en),   
    .clf_mode(clf_mode),
    .data_in(FSRAM_out[WIDTH_FSRAM_WL-1:WIDTH_FSRAM_WL-WIDTH_F_DATA]),
    
    .sum(clf_out1)
    );
    
    classifier 
    #(
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .NUM_POOL(NUM_POOL),
    .WIDTH_EXTEND(WIDTH_EXTEND)
    )
    xclassifier_2
    ( 
    .clk(clk),
    .rstb(rstb),
    .clear(clf_clear),
    
    .en_avgpool(clf_en),   
    .clf_mode(clf_mode),
    .data_in(FSRAM_out[WIDTH_FSRAM_WL-WIDTH_F_DATA-1:WIDTH_FSRAM_WL-2*WIDTH_F_DATA]),
    
    .sum(clf_out2)
    );
    
    classifier 
    #(
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .NUM_POOL(NUM_POOL),
    .WIDTH_EXTEND(WIDTH_EXTEND)
    )
    xclassifier_3
    ( 
    .clk(clk),
    .rstb(rstb),
    .clear(clf_clear),
    
    .en_avgpool(clf_en),   
    .clf_mode(clf_mode),
    .data_in(FSRAM_out[WIDTH_FSRAM_WL-2*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-3*WIDTH_F_DATA]),
    
    .sum(clf_out3)
    );   
    
    classifier 
    #(
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .NUM_POOL(NUM_POOL),
    .WIDTH_EXTEND(WIDTH_EXTEND)
    )
    xclassifier_4
    ( 
    .clk(clk),
    .rstb(rstb),
    .clear(clf_clear),
    
    .en_avgpool(clf_en),   
    .clf_mode(clf_mode),
    .data_in(FSRAM_out[WIDTH_FSRAM_WL-3*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-4*WIDTH_F_DATA]),
    
    .sum(clf_out4)
    );    
 
     classifier 
    #(
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .NUM_POOL(NUM_POOL),
    .WIDTH_EXTEND(WIDTH_EXTEND)
    )
    xclassifier_5
    ( 
    .clk(clk),
    .rstb(rstb),
    .clear(clf_clear),
    
    .en_avgpool(clf_en),   
    .clf_mode(clf_mode),
    .data_in(FSRAM_out[WIDTH_FSRAM_WL-4*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-5*WIDTH_F_DATA]),
    
    .sum(clf_out5)
    );
    
    classifier 
    #(
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .NUM_POOL(NUM_POOL),
    .WIDTH_EXTEND(WIDTH_EXTEND)
    )
    xclassifier_6
    ( 
    .clk(clk),
    .rstb(rstb),
    .clear(clf_clear),
    
    .en_avgpool(clf_en),   
    .clf_mode(clf_mode),
    .data_in(FSRAM_out[WIDTH_FSRAM_WL-5*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-6*WIDTH_F_DATA]),
    
    .sum(clf_out6)
    );
    
    classifier 
    #(
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .NUM_POOL(NUM_POOL),
    .WIDTH_EXTEND(WIDTH_EXTEND)
    )
    xclassifier_7
    ( 
    .clk(clk),
    .rstb(rstb),
    .clear(clf_clear),
    
    .en_avgpool(clf_en),   
    .clf_mode(clf_mode),
    .data_in(FSRAM_out[WIDTH_FSRAM_WL-6*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-7*WIDTH_F_DATA]),
    
    .sum(clf_out7)
    );   
    
    classifier 
    #(
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .NUM_POOL(NUM_POOL),
    .WIDTH_EXTEND(WIDTH_EXTEND)
    )
    xclassifier_8
    ( 
    .clk(clk),
    .rstb(rstb),
    .clear(clf_clear),
    
    .en_avgpool(clf_en),   
    .clf_mode(clf_mode),
    .data_in(FSRAM_out[WIDTH_FSRAM_WL-7*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-8*WIDTH_F_DATA]),
    
    .sum(clf_out8)
    );    
    
      classifier 
    #(
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .NUM_POOL(NUM_POOL),
    .WIDTH_EXTEND(WIDTH_EXTEND)
    )
    xclassifier_9
    ( 
    .clk(clk),
    .rstb(rstb),
    .clear(clf_clear),
    
    .en_avgpool(clf_en),   
    .clf_mode(clf_mode),
    .data_in(FSRAM_out[WIDTH_FSRAM_WL-8*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-9*WIDTH_F_DATA]),
    
    .sum(clf_out9)
    );
    
    classifier 
    #(
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .NUM_POOL(NUM_POOL),
    .WIDTH_EXTEND(WIDTH_EXTEND)
    )
    xclassifier_10
    ( 
    .clk(clk),
    .rstb(rstb),
    .clear(clf_clear),
    
    .en_avgpool(clf_en),   
    .clf_mode(clf_mode),
    .data_in(FSRAM_out[WIDTH_FSRAM_WL-9*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-10*WIDTH_F_DATA]),
    
    .sum(clf_out10)
    );
    
    classifier 
    #(
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .NUM_POOL(NUM_POOL),
    .WIDTH_EXTEND(WIDTH_EXTEND)
    )
    xclassifier_11
    ( 
    .clk(clk),
    .rstb(rstb),
    .clear(clf_clear),
    
    .en_avgpool(clf_en),   
    .clf_mode(clf_mode),
    .data_in(FSRAM_out[WIDTH_FSRAM_WL-10*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-11*WIDTH_F_DATA]),
    
    .sum(clf_out11)
    );   
    
    classifier 
    #(
    .WIDTH_F_DATA(WIDTH_F_DATA),
    .NUM_POOL(NUM_POOL),
    .WIDTH_EXTEND(WIDTH_EXTEND)
    )
    xclassifier_12
    ( 
    .clk(clk),
    .rstb(rstb),
    .clear(clf_clear),
    
    .en_avgpool(clf_en),   
    .clf_mode(clf_mode),
    .data_in(FSRAM_out[WIDTH_FSRAM_WL-11*WIDTH_F_DATA-1:WIDTH_FSRAM_WL-12*WIDTH_F_DATA]),
    
    .sum(clf_out12)
    );      
    
    
    max_finder_v0#(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .NUM_POOL(NUM_POOL),
        .WIDTH_EXTEND(WIDTH_EXTEND)
    )
    xmax_finder
    (
        .clk(clk),
        .rstb(rstb),
        
        .lavg_done(lavg_done),
        
        .in0(clf_out1), 
        .in1(clf_out2), 
        .in2(clf_out3), 
        .in3(clf_out4),  
        .in4(clf_out5),  
        .in5(clf_out6), 
        .in6(clf_out7), 
        .in7(clf_out8),  
        .in8(clf_out9), 
        .in9(clf_out10),  
        .in10(clf_out11), 
        .in11(clf_out12),  
    
        .max_index_o(max_index) 
    );
    
endmodule
