`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Kang Research Group
// Engineer: Chanhong Jeon
// 
// Create Date: 2024/09/30
// Design Name: Processing Element
// Module Name: pe
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


module pe #(parameter DATA_WIDTH = 12, WEIGHT_WIDTH = 8, INT_EXTEND = 9)(
    input clk, rstb, cal_en, rst_local,
    input signed [DATA_WIDTH-1:0] data_i,
    input signed [WEIGHT_WIDTH-1:0] weight_i,
    output signed [DATA_WIDTH-1+WEIGHT_WIDTH+INT_EXTEND:0] data_o
    );

    reg signed [DATA_WIDTH+WEIGHT_WIDTH+INT_EXTEND-1:0] buffer;    

    always @ (posedge clk or negedge rstb)
    begin 
        if(!rstb) begin
            buffer <= 0;
        end    
        else begin 
            if(rst_local) begin
                buffer <= 0;
            end
            else if(cal_en) begin
                buffer <= buffer + data_i * weight_i; 
            end
			else begin
				buffer <= buffer;
			end
        end
    end
    
    // Activation: ReLU
    assign data_o = buffer;

endmodule
