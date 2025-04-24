`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Kang Research Group
// Engineer: Chanhong Jeon
// 
// Create Date: 2024/09/30
// Design Name: Processing Element Array
// Module Name: pe_arr
// Project Name: KWS Chip Tape-out
// Target Devices: Samsung 28nm
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


module pe_arr #(parameter DATA_WIDTH = 8, WEIGHT_WIDTH = 8, INT_EXTEND = 9, WEIGHT_INT_WIDTH = 1, PE_NUM = 32) (
    input clk, 
    input rstb, 
    input en, 
    input rst_local,
    input sel,
    input [DATA_WIDTH-1:0] data_i [0:PE_NUM-1],
    input [WEIGHT_WIDTH-1:0] weight_i [0:PE_NUM-1],
    output [(DATA_WIDTH+WEIGHT_WIDTH+INT_EXTEND)-1:0] data_o [0:PE_NUM-1]
);

    wire [WEIGHT_WIDTH-1:0] weight_mux_o [0:PE_NUM-1];

    genvar j;
    generate
        for (j = 0; j < PE_NUM; j = j + 1) begin: mux_out
            assign weight_mux_o[j] = (sel) ? weight_i[j] : (1 << (WEIGHT_WIDTH - 2 - WEIGHT_INT_WIDTH));
        end
    endgenerate

    genvar i;
    generate
        for (i = 0; i < PE_NUM; i = i + 1) begin: pe_loop
            pe #(DATA_WIDTH, WEIGHT_WIDTH, INT_EXTEND) 
                pe_n (clk, rstb, en, rst_local, data_i[i], weight_i[i], data_o[i]);
        end
    endgenerate

endmodule
