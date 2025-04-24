`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Kang Research Group
// Engineer: Chanhong Jeon
// 
// Create Date: 2024/10/11
// Design Name: Output Buffer
// Module Name: output_buf
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

// Output Buf Module
// Role 1: Activation
// Role 2: Store Intermediate Output from Adder Tree (Stage 3, 5)
module output_buf #(parameter INPUT_WIDTH = 12, WEIGHT_WIDTH = 12, INT_EXTEND = 9, PE_NUM = 32, INPUT_INT_WIDTH = 1, WEIGHT_INT_WIDTH = 1)(
    input clk, rstb, rst_local,
    input sel, // Selection bit in between pe input and adder input Adder Tree input = 0, PE input = 1
    input [PE_NUM-1:0] en_in,
    input signed [(INPUT_WIDTH+WEIGHT_WIDTH+INT_EXTEND)-1:0] data_i [0:PE_NUM-1],
    input signed [(INPUT_WIDTH+WEIGHT_WIDTH+INT_EXTEND)-1:0] adder_i, // Result of Adder Tree
    output [INPUT_WIDTH-1:0] data_o [0:PE_NUM-1]
    );
    
    localparam DATA_WIDTH = INPUT_WIDTH + WEIGHT_WIDTH + INT_EXTEND;

    reg signed [DATA_WIDTH-1:0] buffer [0:PE_NUM-1];
    wire signed [DATA_WIDTH-1:0] nxt_data [0:PE_NUM-1];
    wire signed [DATA_WIDTH-1:0] out_relu [0:PE_NUM-1];

    genvar i;
    genvar j;

    generate
        for (i=0; i<PE_NUM; i=i+1) begin
            assign nxt_data[i] = (en_in[i]) ? ((sel)? data_i[i] : adder_i) : buffer[i];
        end
    endgenerate
    
    always @ (posedge clk or negedge rstb)
    begin 
        if(!rstb) begin
            for(integer k=0; k<PE_NUM; k=k+1) buffer[k] <= 0;
        end    
        else begin
            if (rst_local) begin  
                for(integer k=0; k<PE_NUM; k=k+1) buffer[k] <= 0;
            end
            else begin
                buffer <= nxt_data;
            end
        end
    end
    
    generate
        for (j=0; j<PE_NUM; j=j+1) begin
            assign out_relu[j] = (!buffer[j][DATA_WIDTH-1]) ? buffer[j] : 0;
            assign data_o[j] = (|out_relu[j][DATA_WIDTH-1 -: WEIGHT_INT_WIDTH+INT_EXTEND+1]) ?
                {1'b0, {(INPUT_WIDTH-1){1'b1}}} : out_relu[j][INPUT_WIDTH+WEIGHT_WIDTH-WEIGHT_INT_WIDTH-1 -: INPUT_WIDTH];
        end
    endgenerate

endmodule