`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Donghwan So
// 
// Create Date: 2024/11/04 17:38:24
// Design Name: 
// Module Name: norn_v0
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


`timescale 1ns / 1ps

module norm_v0#(
    parameter WIDTH_PE_O_DATA = 20,
    parameter WIDTH_W_DATA = 8,
    parameter WIDTH_NORM_O_DATA = 21,
    
    parameter WIDTH_L1_PE_IL = 8,
    parameter WIDTH_L1_B_IL = 0,
    
    parameter WIDTH_L2_PE_IL = 6,
    parameter WIDTH_L2_B_IL = 0,
    
    parameter WIDTH_L3_PE_IL = 11,
    parameter WIDTH_L3_B_IL = 1,
    
    parameter WIDTH_L4_PE_IL = 6,
    parameter WIDTH_L4_B_IL = 2, //Nan
    
    parameter WIDTH_L5_PE_IL = 7,
    parameter WIDTH_L5_B_IL = 1,
    
    parameter WIDTH_L6_PE_IL = 5,
    parameter WIDTH_L6_B_IL = 2, //Nan
    
    parameter WIDTH_L7_PE_IL = 5,
    parameter WIDTH_L7_B_IL = 2, //Nan
    
    parameter WIDTH_L8_PE_IL = 7,
    parameter WIDTH_L8_B_IL = 2, //Nan
    
    parameter WIDTH_L9_PE_IL = 7,
    parameter WIDTH_L9_B_IL = 0,
    
    parameter WIDTH_L10_PE_IL = 9,
    parameter WIDTH_L10_B_IL = 1,
    
    parameter WIDTH_L11_PE_IL = 6,
    parameter WIDTH_L11_B_IL = 2, //Nan
    
    parameter WIDTH_L12_PE_IL = 7,
    parameter WIDTH_L12_B_IL = 0, //Nan
    
    parameter WIDTH_L13_PE_IL = 5,
    parameter WIDTH_L13_B_IL = 2, //Nan
    
    parameter WIDTH_L14_PE_IL = 5,
    parameter WIDTH_L14_B_IL = 2, //Nan
    
    parameter WIDTH_L15_PE_IL = 6,
    parameter WIDTH_L15_B_IL = 2, //Nan
    
    parameter WIDTH_L16_PE_IL = 6,
    parameter WIDTH_L16_B_IL = 0,
    
    parameter WIDTH_L17_PE_IL = 9,
    parameter WIDTH_L17_B_IL = 1, //Nan
    
    parameter WIDTH_L18_PE_IL = 7,
    parameter WIDTH_L18_B_IL = 0,
    
    parameter WIDTH_L19_PE_IL = 5,
    parameter WIDTH_L19_B_IL = 2, //Nan
    
    parameter WIDTH_L20_PE_IL = 5,
    parameter WIDTH_L20_B_IL = 2, //Nan
    
    parameter WIDTH_L21_PE_IL = 6,
    parameter WIDTH_L21_B_IL = 2, //Nan
    
    parameter WIDTH_L22_PE_IL = 12,
    parameter WIDTH_L22_B_IL = 3,
    
    parameter WIDTH_L23_PE_IL = 8, //Nan
    parameter WIDTH_L23_B_IL = 2 //Nan

)
(
    input                                   clk,
    input                                   rstb,
    input wire          [4:0]               layer_state,
    
    input                                   norm_on,
    
    input       signed [WIDTH_PE_O_DATA-1:0]          pe_out,  // Declare pe_out as signed
    input       signed [WIDTH_W_DATA-1:0]             bias,    // Declare bias as signed
    
    output reg  signed [WIDTH_NORM_O_DATA-1:0] norm_out  // Declare layer_out as signed
);                                                   
    
    localparam                                      WIDTH_L1_PE_FL = WIDTH_PE_O_DATA - WIDTH_L1_PE_IL - 1,
                                                    WIDTH_L1_B_FL = WIDTH_W_DATA - WIDTH_L1_B_IL - 1,
                                                    
                                                    WIDTH_L2_PE_FL = WIDTH_PE_O_DATA - WIDTH_L2_PE_IL - 1,
                                                    WIDTH_L2_B_FL = WIDTH_W_DATA - WIDTH_L2_B_IL - 1,
                                                    
                                                    WIDTH_L3_PE_FL = WIDTH_PE_O_DATA - WIDTH_L3_PE_IL - 1,
                                                    WIDTH_L3_B_FL = WIDTH_W_DATA - WIDTH_L3_B_IL - 1,
                                                    
                                                    WIDTH_L4_PE_FL = WIDTH_PE_O_DATA - WIDTH_L4_PE_IL - 1,
                                                    WIDTH_L4_B_FL = WIDTH_W_DATA - WIDTH_L4_B_IL - 1,
                                                    
                                                    WIDTH_L5_PE_FL = WIDTH_PE_O_DATA - WIDTH_L5_PE_IL - 1,
                                                    WIDTH_L5_B_FL = WIDTH_W_DATA - WIDTH_L5_B_IL - 1,
                                                    
                                                    WIDTH_L6_PE_FL = WIDTH_PE_O_DATA - WIDTH_L6_PE_IL - 1,
                                                    WIDTH_L6_B_FL = WIDTH_W_DATA - WIDTH_L6_B_IL - 1,
                                                    
                                                    WIDTH_L7_PE_FL = WIDTH_PE_O_DATA - WIDTH_L7_PE_IL - 1,
                                                    WIDTH_L7_B_FL = WIDTH_W_DATA - WIDTH_L7_B_IL - 1,
                                                    
                                                    WIDTH_L8_PE_FL = WIDTH_PE_O_DATA - WIDTH_L8_PE_IL - 1,
                                                    WIDTH_L8_B_FL = WIDTH_W_DATA - WIDTH_L8_B_IL - 1,
                                                    
                                                    WIDTH_L9_PE_FL = WIDTH_PE_O_DATA - WIDTH_L9_PE_IL - 1,
                                                    WIDTH_L9_B_FL = WIDTH_W_DATA - WIDTH_L9_B_IL - 1,
                                                    
                                                    WIDTH_L10_PE_FL = WIDTH_PE_O_DATA - WIDTH_L10_PE_IL - 1,
                                                    WIDTH_L10_B_FL = WIDTH_W_DATA - WIDTH_L10_B_IL - 1,
                                                    
                                                    WIDTH_L11_PE_FL = WIDTH_PE_O_DATA - WIDTH_L11_PE_IL - 1,
                                                    WIDTH_L11_B_FL = WIDTH_W_DATA - WIDTH_L11_B_IL - 1,
                                                    
                                                    WIDTH_L12_PE_FL = WIDTH_PE_O_DATA - WIDTH_L12_PE_IL - 1,
                                                    WIDTH_L12_B_FL = WIDTH_W_DATA - WIDTH_L12_B_IL - 1,
                                                    
                                                    WIDTH_L13_PE_FL = WIDTH_PE_O_DATA - WIDTH_L13_PE_IL - 1,
                                                    WIDTH_L13_B_FL = WIDTH_W_DATA - WIDTH_L13_B_IL - 1,
                                                    
                                                    WIDTH_L14_PE_FL = WIDTH_PE_O_DATA - WIDTH_L14_PE_IL - 1,
                                                    WIDTH_L14_B_FL = WIDTH_W_DATA - WIDTH_L14_B_IL - 1,
                                                    
                                                    WIDTH_L15_PE_FL = WIDTH_PE_O_DATA - WIDTH_L15_PE_IL - 1,
                                                    WIDTH_L15_B_FL = WIDTH_W_DATA - WIDTH_L15_B_IL - 1,
                                                    
                                                    WIDTH_L16_PE_FL = WIDTH_PE_O_DATA - WIDTH_L16_PE_IL - 1,
                                                    WIDTH_L16_B_FL = WIDTH_W_DATA - WIDTH_L16_B_IL - 1,
                                                    
                                                    WIDTH_L17_PE_FL = WIDTH_PE_O_DATA - WIDTH_L17_PE_IL - 1,
                                                    WIDTH_L17_B_FL = WIDTH_W_DATA - WIDTH_L17_B_IL - 1,
                                                    
                                                    WIDTH_L18_PE_FL = WIDTH_PE_O_DATA - WIDTH_L18_PE_IL - 1,
                                                    WIDTH_L18_B_FL = WIDTH_W_DATA - WIDTH_L18_B_IL - 1,
                                                    
                                                    WIDTH_L19_PE_FL = WIDTH_PE_O_DATA - WIDTH_L19_PE_IL - 1,
                                                    WIDTH_L19_B_FL = WIDTH_W_DATA - WIDTH_L19_B_IL - 1,
                                                    
                                                    WIDTH_L20_PE_FL = WIDTH_PE_O_DATA - WIDTH_L20_PE_IL - 1,
                                                    WIDTH_L20_B_FL = WIDTH_W_DATA - WIDTH_L20_B_IL - 1,
                                                    
                                                    WIDTH_L21_PE_FL = WIDTH_PE_O_DATA - WIDTH_L21_PE_IL - 1,
                                                    WIDTH_L21_B_FL = WIDTH_W_DATA - WIDTH_L21_B_IL - 1,
                                                    
                                                    WIDTH_L22_PE_FL = WIDTH_PE_O_DATA - WIDTH_L22_PE_IL - 1,
                                                    WIDTH_L22_B_FL = WIDTH_W_DATA - WIDTH_L22_B_IL - 1,
                                                    
                                                    WIDTH_L23_PE_FL = WIDTH_PE_O_DATA - WIDTH_L23_PE_IL - 1,
                                                    WIDTH_L23_B_FL = WIDTH_W_DATA - WIDTH_L23_B_IL - 1;

    
    
                                                    
    reg signed [WIDTH_PE_O_DATA-1:0] bias_extended;
    
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            norm_out <= 0;
        end
        else begin
            if(norm_on) begin
                norm_out <= pe_out + bias_extended;
            end
            else begin 
                norm_out <= pe_out;
            end
        end
    end
    
    always @(*) begin
        case(layer_state)
            5'b00001: begin // Layer 1
                bias_extended = bias <<< (WIDTH_L1_PE_FL - WIDTH_L1_B_FL);
            end
            5'b00010: begin // Layer 2
                bias_extended = bias <<< (WIDTH_L2_PE_FL - WIDTH_L2_B_FL);
            end
            5'b00011: begin // Layer 3
                bias_extended = bias <<< (WIDTH_L3_PE_FL - WIDTH_L3_B_FL);
            end
            5'b00100: begin // Layer 4
                bias_extended = bias <<< (WIDTH_L4_PE_FL - WIDTH_L4_B_FL);
            end
            5'b00101: begin // Layer 5
                bias_extended = bias <<< (WIDTH_L5_PE_FL - WIDTH_L5_B_FL);
            end
            5'b00110: begin // Layer 6
                bias_extended = bias <<< (WIDTH_L6_PE_FL - WIDTH_L6_B_FL);
            end
            5'b00111: begin // Layer 7
                bias_extended = bias <<< (WIDTH_L7_PE_FL - WIDTH_L7_B_FL);
            end
            5'b01000: begin // Layer 8
                bias_extended = bias <<< (WIDTH_L8_PE_FL - WIDTH_L8_B_FL);
            end
            5'b01001: begin // Layer 9
                bias_extended = bias <<< (WIDTH_L9_PE_FL - WIDTH_L9_B_FL);
            end
            5'b01010: begin // Layer 10
                bias_extended = bias <<< (WIDTH_L10_PE_FL - WIDTH_L10_B_FL);
            end
            5'b01011: begin // Layer 11
                bias_extended = bias <<< (WIDTH_L11_PE_FL - WIDTH_L11_B_FL);
            end
            5'b01100: begin // Layer 12
                bias_extended = bias <<< (WIDTH_L12_PE_FL - WIDTH_L12_B_FL);
            end
            5'b01101: begin // Layer 13
                bias_extended = bias <<< (WIDTH_L13_PE_FL - WIDTH_L13_B_FL);
            end
            5'b01110: begin // Layer 14
                bias_extended = bias <<< (WIDTH_L14_PE_FL - WIDTH_L14_B_FL);
            end
            5'b01111: begin // Layer 15
                bias_extended = bias <<< (WIDTH_L15_PE_FL - WIDTH_L15_B_FL);
            end
            5'b10000: begin // Layer 16
                bias_extended = bias <<< (WIDTH_L16_PE_FL - WIDTH_L16_B_FL);
            end
            5'b10001: begin // Layer 17
                bias_extended = bias <<< (WIDTH_L17_PE_FL - WIDTH_L17_B_FL);
            end
            5'b10010: begin // Layer 18
                bias_extended = bias <<< (WIDTH_L18_PE_FL - WIDTH_L18_B_FL);
            end
            5'b10011: begin // Layer 19
                bias_extended = bias <<< (WIDTH_L19_PE_FL - WIDTH_L19_B_FL);
            end
            5'b10100: begin // Layer 20
                bias_extended = bias <<< (WIDTH_L20_PE_FL - WIDTH_L20_B_FL);
            end
            5'b10101: begin // Layer 21
                bias_extended = bias <<< (WIDTH_L21_PE_FL - WIDTH_L21_B_FL);
            end
            5'b10110: begin // Layer 22
                bias_extended = bias <<< (WIDTH_L22_PE_FL - WIDTH_L22_B_FL);
            end
            5'b10111: begin // Layer 23
                bias_extended = bias <<< (WIDTH_L23_PE_FL - WIDTH_L23_B_FL);
            end
            default: begin
                bias_extended = bias; // Default behavior
            end
        endcase
    end
    
    
endmodule
