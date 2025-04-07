`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Donghwan So
// 
// Create Date: 2024/09/30 14:08:56
// Design Name: 
// Module Name: PE
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


module PE
#(
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_W_DATA = 8,
    parameter WIDTH_PE_O_DATA = 20
)
(
    input       clk,
    input       rstb,
    input       clear,
    input       pe_en,
    input       signed                          [WIDTH_F_DATA-1:0] f_data,
    input       signed                          [WIDTH_W_DATA-1:0] w_data,
    
    output  reg signed                          [WIDTH_PE_O_DATA-1:0] PE_out
);
    
    always @(posedge clk or negedge rstb) begin
        if(!rstb) begin
            PE_out <= 0;
        end
        else begin
            if(clear) begin
                PE_out <= 0;
            end
            else if(pe_en) begin
                PE_out <= PE_out + f_data * w_data;
            end
        end
    end



endmodule
