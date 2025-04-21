module top_spi
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
    
    //From OFF chip & for weight write
    input                               wr_wsram_sclk,              // clk generated by FPGA
    input                               wr_wsram_ss,               // ss  generated by FPGA
    input                               wr_wsram_sdata,
    input                               wr_weight_on,
    
    //From ON chip & for feature write
    input                               wr_fsram_ss,
    input                               wr_fsram_sdata,
    input                               wr_feature_on,
    
    output  [3:0]                       max_index
);



    
    wire wr_fsram_clk;
    wire [WIDTH_FSRAM_WL-1:0] wr_fsram_data;
    wire [WIDTH_FSRAM_ADDR-1:0] wr_fsram_addr;
    wire wr_fsram_ceb;
    wire wr_fsram_web;
    wire wr_fsram_mux;
    
    wire start_nn;
    
    sram_feature_write_v0
    #(
        .total_width(168),
        .addr_width(WIDTH_FSRAM_ADDR),
        .data_width(WIDTH_FSRAM_WL),
        .wr_delay(WR_DELAY)
    )
    xsram_feature_write_v0
    (
        .rstb(rstb),
        .sram_sclk(wr_wsram_sclk),              // clk generated by FPGA
        .sram_ss(wr_fsram_ss),               // ss  generated by FPGA
        .sram_sdata(wr_fsram_sdata),
        .wr_feature_on(wr_feature_on),
        
        .sram_clk(wr_fsram_clk),  
        .sram_data(wr_fsram_data),
        .sram_addr(wr_fsram_addr),
        .sram_ceb(wr_fsram_ceb),
        .sram_web(wr_fsram_web),  
        .sram_mux(wr_fsram_mux),
        
        .start_nn(start_nn)
    );
    
    top_BCRes
    #(
        .WIDTH_WSRAM_WL(WIDTH_WSRAM_WL),
        .WIDTH_FSRAM_WL(WIDTH_FSRAM_WL),
        .WIDTH_W_DATA(WIDTH_W_DATA),
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .WIDTH_FSRAM_ADDR(WIDTH_FSRAM_ADDR),
        .WIDTH_WSRAM_ADDR(WIDTH_WSRAM_ADDR),
        
        .WIDTH_PE_O_DATA(WIDTH_PE_O_DATA),
        .WIDTH_NORM_O_DATA(WIDTH_NORM_O_DATA),   // WIDTH_NORM_O_DATA == WIDTH_PE_O_DATA + 1
        .WIDTH_O_DATA(WIDTH_O_DATA),
        
        .WIDTH_L1_PE_IL(WIDTH_L1_PE_IL),
        .WIDTH_L1_B_IL(WIDTH_L1_B_IL),
        .WIDTH_L1_NORM_IL(WIDTH_L1_NORM_IL),     // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L1_O_IL(WIDTH_L1_O_IL),
        
        .WIDTH_L2_PE_IL(WIDTH_L2_PE_IL),
        .WIDTH_L2_B_IL(WIDTH_L2_B_IL),
        .WIDTH_L2_NORM_IL(WIDTH_L2_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L2_O_IL(WIDTH_L2_O_IL),
    
        .WIDTH_L3_PE_IL(WIDTH_L3_PE_IL),
        .WIDTH_L3_B_IL(WIDTH_L3_B_IL),
        .WIDTH_L3_NORM_IL(WIDTH_L3_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L3_O_IL(WIDTH_L3_O_IL),
        
        .WIDTH_L4_PE_IL(WIDTH_L4_PE_IL),
        .WIDTH_L4_B_IL(WIDTH_L4_B_IL), //Nan
        .WIDTH_L4_NORM_IL(WIDTH_L4_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L4_O_IL(WIDTH_L4_O_IL),
    
        .WIDTH_L5_PE_IL(WIDTH_L5_PE_IL),
        .WIDTH_L5_B_IL(WIDTH_L5_B_IL),
        .WIDTH_L5_NORM_IL(WIDTH_L5_NORM_IL),     // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L5_O_IL(WIDTH_L5_O_IL),
        
        .WIDTH_L6_PE_IL(WIDTH_L6_PE_IL),
        .WIDTH_L6_B_IL(WIDTH_L6_B_IL), //nan
        .WIDTH_L6_NORM_IL(WIDTH_L6_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L6_O_IL(WIDTH_L6_O_IL),
        
        .WIDTH_L7_PE_IL(WIDTH_L7_PE_IL),
        .WIDTH_L7_B_IL(WIDTH_L7_B_IL), //nan
        .WIDTH_L7_NORM_IL(WIDTH_L7_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L7_O_IL(WIDTH_L7_O_IL),
        
        .WIDTH_L8_PE_IL(WIDTH_L8_PE_IL),
        .WIDTH_L8_B_IL(WIDTH_L8_B_IL), //nan
        .WIDTH_L8_NORM_IL(WIDTH_L8_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L8_O_IL(WIDTH_L8_O_IL),
        
        .WIDTH_L9_PE_IL(WIDTH_L9_PE_IL),
        .WIDTH_L9_B_IL(WIDTH_L9_B_IL),
        .WIDTH_L9_NORM_IL(WIDTH_L9_NORM_IL),     // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L9_O_IL(WIDTH_L9_O_IL),
        
        .WIDTH_L10_PE_IL(WIDTH_L10_PE_IL),
        .WIDTH_L10_B_IL(WIDTH_L10_B_IL),
        .WIDTH_L10_NORM_IL(WIDTH_L10_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L10_O_IL(WIDTH_L10_O_IL),
        
        .WIDTH_L11_PE_IL(WIDTH_L11_PE_IL),
        .WIDTH_L11_B_IL(WIDTH_L11_B_IL), //nan
        .WIDTH_L11_NORM_IL(WIDTH_L11_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L11_O_IL(WIDTH_L11_O_IL),
        
        .WIDTH_L12_PE_IL(WIDTH_L12_PE_IL),
        .WIDTH_L12_B_IL(WIDTH_L12_B_IL),
        .WIDTH_L12_NORM_IL(WIDTH_L12_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L12_O_IL(WIDTH_L12_O_IL),
    
        .WIDTH_L13_PE_IL(WIDTH_L13_PE_IL),
        .WIDTH_L13_B_IL(WIDTH_L13_B_IL), //nan
        .WIDTH_L13_NORM_IL(WIDTH_L13_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L13_O_IL(WIDTH_L13_O_IL),
        
        .WIDTH_L14_PE_IL(WIDTH_L14_PE_IL),
        .WIDTH_L14_B_IL(WIDTH_L14_B_IL), //nan
        .WIDTH_L14_NORM_IL(WIDTH_L14_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L14_O_IL(WIDTH_L14_O_IL),
                    
        .WIDTH_L15_PE_IL(WIDTH_L15_PE_IL),
        .WIDTH_L15_B_IL(WIDTH_L15_B_IL), //nan
        .WIDTH_L15_NORM_IL(WIDTH_L15_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L15_O_IL(WIDTH_L15_O_IL),    
        
        .WIDTH_L16_PE_IL(WIDTH_L16_PE_IL),
        .WIDTH_L16_B_IL(WIDTH_L16_B_IL),
        .WIDTH_L16_NORM_IL(WIDTH_L16_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L16_O_IL(WIDTH_L16_O_IL),
        
        .WIDTH_L17_PE_IL(WIDTH_L17_PE_IL),
        .WIDTH_L17_B_IL(WIDTH_L17_B_IL),
        .WIDTH_L17_NORM_IL(WIDTH_L17_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L17_O_IL(WIDTH_L17_O_IL),
        
        .WIDTH_L18_PE_IL(WIDTH_L18_PE_IL),
        .WIDTH_L18_B_IL(WIDTH_L18_B_IL),
        .WIDTH_L18_NORM_IL(WIDTH_L18_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L18_O_IL(WIDTH_L18_O_IL),   
        
        .WIDTH_L19_PE_IL(WIDTH_L19_PE_IL),
        .WIDTH_L19_B_IL(WIDTH_L19_B_IL), //nan
        .WIDTH_L19_NORM_IL(WIDTH_L19_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L19_O_IL(WIDTH_L19_O_IL),
    
        .WIDTH_L20_PE_IL(WIDTH_L20_PE_IL),
        .WIDTH_L20_B_IL(WIDTH_L20_B_IL), //nan
        .WIDTH_L20_NORM_IL(WIDTH_L20_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L20_O_IL(WIDTH_L20_O_IL),
    
        .WIDTH_L21_PE_IL(WIDTH_L21_PE_IL),
        .WIDTH_L21_B_IL(WIDTH_L21_B_IL), //nan
        .WIDTH_L21_NORM_IL(WIDTH_L21_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L21_O_IL(WIDTH_L21_O_IL),
    
        .WIDTH_L22_PE_IL(WIDTH_L22_PE_IL),
        .WIDTH_L22_B_IL(WIDTH_L22_B_IL),
        .WIDTH_L22_NORM_IL(WIDTH_L22_NORM_IL),    // WIDTH_NORM_IL == WIDTH_PE_IL + 1
        .WIDTH_L22_O_IL(WIDTH_L22_O_IL),
            
        .ADDR_START_L1_W(ADDR_START_L1_W),
        .ADDR_START_L1_F(ADDR_START_L1_F),
        
        .SIZE_KERNEL_H(SIZE_KERNEL_H),
        .SIZE_KERNEL_W(SIZE_KERNEL_W),    
        
        .NUM_PE(NUM_PE),
        .WR_DELAY(WR_DELAY),
        
        .RELU_MAX_VAL(RELU_MAX_VAL),
        
        // ADDR
        .SIZE_L1_OUT_CHANNEL(SIZE_L1_OUT_CHANNEL),
        //L2
        .ADDR_START_L2_W(ADDR_START_L2_W),
        .ADDR_START_L2_F(ADDR_START_L2_F),
        .NUM_L1_OUT_CHANNEL(NUM_L1_OUT_CHANNEL),
        .NUM_L2_OUT_CHANNEL(NUM_L2_OUT_CHANNEL),
        //L3
        .ADDR_START_L3_W(ADDR_START_L3_W),
        .ADDR_START_L3_F(ADDR_START_L3_F),   
        //L4
        .ADDR_START_L4_W(ADDR_START_L4_W),
        .ADDR_START_L4_F(ADDR_START_L4_F),
        .SIZE_L3_OUT_CHANNEL(SIZE_L3_OUT_CHANNEL),
        .NUM_L4_OUT_CHANNEL(NUM_L4_OUT_CHANNEL),
        //L5
        .ADDR_START_L5_W(ADDR_START_L5_W),
        .ADDR_START_L5_F(ADDR_START_L5_F),
        //L6
        .ADDR_START_L6_W(ADDR_START_L6_W),
        .ADDR_START_L6_F(ADDR_START_L6_F),
        //L7
        .ADDR_START_L7_W(ADDR_START_L7_W),
        .ADDR_START_L7_F(ADDR_START_L7_F),
        //L8
        .ADDR_START_L8_W(ADDR_START_L8_W),
        .ADDR_START_L8_F(ADDR_START_L8_F),
        //L9
        .ADDR_START_L9_W(ADDR_START_L9_W),
        .ADDR_START_L9_F(ADDR_START_L9_F),
        //L10
        .ADDR_START_L10_W(ADDR_START_L10_W),
        .ADDR_START_L10_F(ADDR_START_L10_F),
        //L11
        .ADDR_START_L11_W(ADDR_START_L11_W),
        .ADDR_START_L11_F(ADDR_START_L11_F),
        .SIZE_L10_OUT_CHANNEL(SIZE_L10_OUT_CHANNEL),
        .NUM_L11_OUT_CHANNEL(NUM_L11_OUT_CHANNEL),
        //L12
        .ADDR_START_L12_W(ADDR_START_L12_W),
        .ADDR_START_L12_F(ADDR_START_L12_F),
        //L13
        .ADDR_START_L13_W(ADDR_START_L13_W),
        .ADDR_START_L13_F(ADDR_START_L13_F),
        //L14 
        .ADDR_START_L14_W(ADDR_START_L14_W),
        .ADDR_START_L14_F(ADDR_START_L14_F),
        //L15
        .ADDR_START_L15_W(ADDR_START_L15_W),
        .ADDR_START_L15_F(ADDR_START_L15_F),
        //L16
        .ADDR_START_L16_W(ADDR_START_L16_W),
        .ADDR_START_L16_F(ADDR_START_L16_F),
        //L17
        .ADDR_START_L17_W(ADDR_START_L17_W),
        .ADDR_START_L17_F(ADDR_START_L17_F),
        //L18
        .ADDR_START_L18_W(ADDR_START_L18_W),
        .ADDR_START_L18_F(ADDR_START_L18_F),
        //L19
        .ADDR_START_L19_W(ADDR_START_L19_W),
        .ADDR_START_L19_F(ADDR_START_L19_F),
        //L20
        .ADDR_START_L20_W(ADDR_START_L20_W), 
        .ADDR_START_L20_F(ADDR_START_L20_F),
        //L21
        .ADDR_START_L21_W(ADDR_START_L21_W),
        .ADDR_START_L21_F(ADDR_START_L21_F),
        //L22
        .ADDR_START_L22_W(ADDR_START_L22_W), 
        .ADDR_START_L22_F(ADDR_START_L22_F),  
        //L23
        .ADDR_START_LAVGPOOL(ADDR_START_LAVGPOOL),     
        .NUM_POOL(NUM_POOL),
        
        .WIDTH_EXTEND(WIDTH_EXTEND)
    ) 
    xtop_BCRes
    (
        .rstb(rstb),
        .clk(clk),
        .start(start_nn),
        
        //From OFF chip & for weight write
        .wr_wsram_sclk(wr_wsram_sclk),              // clk generated by FPGA
        .wr_wsram_ss(wr_wsram_ss),               // ss  generated by FPGA
        .wr_wsram_sdata(wr_wsram_sdata),
        .wr_weight_on(wr_weight_on),
        
        //From ON chip & for feature write
        .wr_fsram_clk(wr_fsram_clk),
        .wr_fsram_data(wr_fsram_data),
        .wr_fsram_addr(wr_fsram_addr),
        .wr_fsram_ceb(wr_fsram_ceb),
        .wr_fsram_web(wr_fsram_web),
        .wr_fsram_mux(wr_fsram_mux),
        
        
        .max_index(max_index)
    );

endmodule