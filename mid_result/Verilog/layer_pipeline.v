`timescale 1ns / 1ps
module layer_pipeline#(

)(
    input               rst_n,
    input               clk,

    input               weight_done,
    input               FE_done,

    //Weight Memory
    output  wire        [WMEM_BIT_LEN-1:0] wmem_addr_o;
    output  wire        wmem_wenb_o,
    output  wire        wmem_enb_o,
    /*//Data Memory
    output  reg         [IM_BIT_LEN-1:0] mem_addr_o,
    output  reg         mem_wenb_o,
    output  reg         mem_enb_o,*/
    //Memory A 
    output  wire [MA_BIT_LEN-1:0] memA_addr_o,
    output  wire        memA_wenb_o,
    output  wire        memA_cenb_o,
    //Memory B
    output  wire [MB_BIT_LEN-1:0] memB_addr_o,
    output  wire        memB_wenb_o,
    output  wire        memB_cenb_o,

    //Weight Buffer
    output  wire        wei_buff_en_o,
    
    //PE array
    output  wire        pe_en_o,
    output  wire        pe_rst_o,

    // ReLU
    output  wire        relu_en_o,

    //Output Buffer
    output  wire [31:0] out_buf_en_o,
    output  wire        out_buf_sel_o,
    output  wire        out_buf_rst_o,
    //maxpool
    output  wire        pool_sel_o, //1 at layer 3

    //output  reg     comp_start_o,
    //output reg rf_sel_o,
    output  reg [2:0]   layer_state,
    output  wire        done_o

)


endmodule