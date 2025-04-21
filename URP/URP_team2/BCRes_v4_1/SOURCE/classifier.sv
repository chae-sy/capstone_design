`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/31 14:49:18
// Design Name: 
// Module Name: classifier
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


module classifier
#(
    parameter WIDTH_F_DATA = 8,
    parameter NUM_POOL = 22,
    parameter WIDTH_EXTEND = $clog2(NUM_POOL)
)
( 
    input clk,
    input rstb,
    input clear,
    
    input en_avgpool,   
    input clf_mode,
    input [WIDTH_F_DATA-1:0]    data_in,
    
    output reg [WIDTH_F_DATA+WIDTH_EXTEND-1:0]  sum

    );
    
    always @(posedge clk or negedge rstb) begin
        if(!rstb) begin
            sum <= 0;
        end
        else begin
            if(clear) begin
                sum <= 0;
            end
            else begin
                if (en_avgpool) begin
                    if (clf_mode == 0) begin
                        sum <= sum + data_in;
                    end
                    else begin
                        sum <= sum - data_in;
                    end
                end
            end
        end
    end      
    
endmodule
