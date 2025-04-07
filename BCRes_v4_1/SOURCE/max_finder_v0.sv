`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Donghwan So
// 
// Create Date: 2024/11/20 20:19:55
// Design Name: 
// Module Name: nax_finder_v0
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


module max_finder_v0#(
    parameter WIDTH_F_DATA = 8,
    parameter NUM_POOL = 22,
    parameter WIDTH_EXTEND = $clog2(NUM_POOL)
)
(
    input clk,
    input rstb,
    
    input lavg_done,
    
    input signed [WIDTH_F_DATA+WIDTH_EXTEND-1:0] in0, 
    input signed [WIDTH_F_DATA+WIDTH_EXTEND-1:0] in1, 
    input signed [WIDTH_F_DATA+WIDTH_EXTEND-1:0] in2, 
    input signed [WIDTH_F_DATA+WIDTH_EXTEND-1:0] in3,  
    input signed [WIDTH_F_DATA+WIDTH_EXTEND-1:0] in4,  
    input signed [WIDTH_F_DATA+WIDTH_EXTEND-1:0] in5, 
    input signed [WIDTH_F_DATA+WIDTH_EXTEND-1:0] in6, 
    input signed [WIDTH_F_DATA+WIDTH_EXTEND-1:0] in7,  
    input signed [WIDTH_F_DATA+WIDTH_EXTEND-1:0] in8, 
    input signed [WIDTH_F_DATA+WIDTH_EXTEND-1:0] in9,  
    input signed [WIDTH_F_DATA+WIDTH_EXTEND-1:0] in10, 
    input signed [WIDTH_F_DATA+WIDTH_EXTEND-1:0] in11,  
    
    output reg [3:0] max_index_o 
);

    reg [3:0] max_index;
    reg signed [WIDTH_F_DATA+WIDTH_EXTEND-1:0] max_val; 

    always @(*) begin
        max_val = in0;
        max_index = 0;
    
        if (in1 > max_val) begin
            max_val = in1;
            max_index = 1;
        end
        if (in2 > max_val) begin
            max_val = in2;
            max_index = 2;
        end
        if (in3 > max_val) begin
            max_val = in3;
            max_index = 3;
        end
        if (in4 > max_val) begin
            max_val = in4;
            max_index = 4;
        end
        if (in5 > max_val) begin
            max_val = in5;
            max_index = 5;
        end
        if (in6 > max_val) begin
            max_val = in6;
            max_index = 6;
        end
        if (in7 > max_val) begin
            max_val = in7;
            max_index = 7;
        end
        if (in8 > max_val) begin
            max_val = in8;
            max_index = 8;
        end
        if (in9 > max_val) begin
            max_val = in9;
            max_index = 9;
        end
        if (in10 > max_val) begin
            max_val = in10;
            max_index = 10;
        end
        if (in11 > max_val) begin
            max_val = in11;
            max_index = 11;
        end
    end
    
    always @(posedge clk or negedge rstb) begin 
        if(!rstb) begin 
            max_index_o <= 0;
        end
        else begin 
            if(lavg_done) begin 
                max_index_o <= max_index;
            end
            else begin 
                max_index_o <= max_index_o;
            end
        end
    end
endmodule
