`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Donghwan So
// 
// Create Date: 2024/11/05 11:46:52
// Design Name: 
// Module Name: relu_numadj_v0
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


module relu_numadj_v0#(
    parameter WIDTH_NORM_O_DATA = 21,
    parameter WIDTH_O_DATA = 8,
    
    parameter RELU_MAX_VAL = 6,
    
    parameter WIDTH_L1_NORM_IL = 9,
    parameter WIDTH_L1_O_IL = 1,
    
    parameter WIDTH_L2_NORM_IL = 7,
    parameter WIDTH_L2_O_IL = 1,
    
    parameter WIDTH_L3_NORM_IL = 12,
    parameter WIDTH_L3_O_IL = 1,
    
    parameter WIDTH_L4_NORM_IL = 7,
    parameter WIDTH_L4_O_IL = 0,
    
    parameter WIDTH_L5_NORM_IL = 8,
    parameter WIDTH_L5_O_IL = 0,
    
    parameter WIDTH_L6_NORM_IL = 6,
    parameter WIDTH_L6_O_IL = 0,
    
    parameter WIDTH_L7_NORM_IL = 6,
    parameter WIDTH_L7_O_IL = 1,
    
    parameter WIDTH_L8_NORM_IL = 8,
    parameter WIDTH_L8_O_IL = 1,
    
    parameter WIDTH_L9_NORM_IL = 8,
    parameter WIDTH_L9_O_IL = 1,
    
    parameter WIDTH_L10_NORM_IL = 10,
    parameter WIDTH_L10_O_IL = 1,
    
    parameter WIDTH_L11_NORM_IL = 7,
    parameter WIDTH_L11_O_IL = 0,
    
    parameter WIDTH_L12_NORM_IL = 8,
    parameter WIDTH_L12_O_IL = 0,
    
    parameter WIDTH_L13_NORM_IL = 6,
    parameter WIDTH_L13_O_IL = 0,
    
    parameter WIDTH_L14_NORM_IL = 6,
    parameter WIDTH_L14_O_IL = 0,
    
    parameter WIDTH_L15_NORM_IL = 7,
    parameter WIDTH_L15_O_IL = 0,
    
    parameter WIDTH_L16_NORM_IL = 7,
    parameter WIDTH_L16_O_IL = 1,
    
    parameter WIDTH_L17_NORM_IL = 10,
    parameter WIDTH_L17_O_IL = 1,
    
    parameter WIDTH_L18_NORM_IL = 8,
    parameter WIDTH_L18_O_IL = 0,
    
    parameter WIDTH_L19_NORM_IL = 6,
    parameter WIDTH_L19_O_IL = 0,
    
    parameter WIDTH_L20_NORM_IL = 6,
    parameter WIDTH_L20_O_IL = 0,
    
    parameter WIDTH_L21_NORM_IL = 7,
    parameter WIDTH_L21_O_IL = 0,
    
    parameter WIDTH_L22_NORM_IL = 13,
    parameter WIDTH_L22_O_IL = 7,
    
    parameter WIDTH_L23_NORM_IL = 9, //Nan
    parameter WIDTH_L23_O_IL = 3 //Nan
)
(
    input                                           relu_on,
    input   [4:0]                                   layer_state,
    input   signed      [WIDTH_NORM_O_DATA-1:0]      norm_out,
    
    output  reg         [WIDTH_O_DATA-1:0]          layer_out
    );
    
    localparam                                      WIDTH_L1_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L1_NORM_IL-1,
                                                    WIDTH_L1_O_FL = WIDTH_O_DATA-WIDTH_L1_O_IL-1,
                                                    
                                                    WIDTH_L2_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L2_NORM_IL-1,
                                                    WIDTH_L2_O_FL = WIDTH_O_DATA-WIDTH_L2_O_IL-1,
                                                    
                                                    WIDTH_L3_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L3_NORM_IL-1,
                                                    WIDTH_L3_O_FL = WIDTH_O_DATA-WIDTH_L3_O_IL-1,
                                                    
                                                    WIDTH_L4_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L4_NORM_IL-1,
                                                    WIDTH_L4_O_FL = WIDTH_O_DATA-WIDTH_L4_O_IL-1,
                                                    
                                                    WIDTH_L5_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L5_NORM_IL-1,
                                                    WIDTH_L5_O_FL = WIDTH_O_DATA-WIDTH_L5_O_IL-1,
                                                    
                                                    WIDTH_L6_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L6_NORM_IL-1,
                                                    WIDTH_L6_O_FL = WIDTH_O_DATA-WIDTH_L6_O_IL-1,
                                                    
                                                    WIDTH_L7_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L7_NORM_IL-1,
                                                    WIDTH_L7_O_FL = WIDTH_O_DATA-WIDTH_L7_O_IL-1,
                                                    
                                                    WIDTH_L8_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L8_NORM_IL-1,
                                                    WIDTH_L8_O_FL = WIDTH_O_DATA-WIDTH_L8_O_IL-1,
                                                    
                                                    WIDTH_L9_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L9_NORM_IL-1,
                                                    WIDTH_L9_O_FL = WIDTH_O_DATA-WIDTH_L9_O_IL-1,
                                                    
                                                    WIDTH_L10_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L10_NORM_IL-1,
                                                    WIDTH_L10_O_FL = WIDTH_O_DATA-WIDTH_L10_O_IL-1,
                                                    
                                                    WIDTH_L11_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L11_NORM_IL-1,
                                                    WIDTH_L11_O_FL = WIDTH_O_DATA-WIDTH_L11_O_IL-1,
                                                    
                                                    WIDTH_L12_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L12_NORM_IL-1,
                                                    WIDTH_L12_O_FL = WIDTH_O_DATA-WIDTH_L12_O_IL-1,
                                                    
                                                    WIDTH_L13_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L13_NORM_IL-1,
                                                    WIDTH_L13_O_FL = WIDTH_O_DATA-WIDTH_L13_O_IL-1,
                                                    
                                                    WIDTH_L14_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L14_NORM_IL-1,
                                                    WIDTH_L14_O_FL = WIDTH_O_DATA-WIDTH_L14_O_IL-1,
                                                    
                                                    WIDTH_L15_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L15_NORM_IL-1,
                                                    WIDTH_L15_O_FL = WIDTH_O_DATA-WIDTH_L15_O_IL-1,
                                                    
                                                    WIDTH_L16_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L16_NORM_IL-1,
                                                    WIDTH_L16_O_FL = WIDTH_O_DATA-WIDTH_L16_O_IL-1,
                                                    
                                                    WIDTH_L17_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L17_NORM_IL-1,
                                                    WIDTH_L17_O_FL = WIDTH_O_DATA-WIDTH_L17_O_IL-1,
                                                    
                                                    WIDTH_L18_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L18_NORM_IL-1,
                                                    WIDTH_L18_O_FL = WIDTH_O_DATA-WIDTH_L18_O_IL-1,
                                                    
                                                    WIDTH_L19_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L19_NORM_IL-1,
                                                    WIDTH_L19_O_FL = WIDTH_O_DATA-WIDTH_L19_O_IL-1,
                                                    
                                                    WIDTH_L20_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L20_NORM_IL-1,
                                                    WIDTH_L20_O_FL = WIDTH_O_DATA-WIDTH_L20_O_IL-1,
                                                    
                                                    WIDTH_L21_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L21_NORM_IL-1,
                                                    WIDTH_L21_O_FL = WIDTH_O_DATA-WIDTH_L21_O_IL-1,
                                                    
                                                    WIDTH_L22_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L22_NORM_IL-1,
                                                    WIDTH_L22_O_FL = WIDTH_O_DATA-WIDTH_L22_O_IL-1,
                                                    
                                                    WIDTH_L23_NORM_FL = WIDTH_NORM_O_DATA-WIDTH_L23_NORM_IL-1,
                                                    WIDTH_L23_O_FL = WIDTH_O_DATA-WIDTH_L23_O_IL-1;
                                                    
                                                    
    always @(*) begin 
        case(layer_state) 
//            5'b00001: begin
//                if(relu_on) begin   //relu on
//                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;            // if norm_out is negative layer_out is 0
//                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L1_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L1_O_FL;
//                    else layer_out = {1'b0, norm_out[WIDTH_L1_NORM_FL+WIDTH_L1_O_IL-1:WIDTH_L1_NORM_FL], norm_out[WIDTH_L1_NORM_FL-1:WIDTH_L1_NORM_FL-WIDTH_L1_O_FL]};
//                end
//                else begin // no relu
//                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
//                        if(norm_out < -(2**(WIDTH_L1_O_IL+WIDTH_L1_NORM_FL))) layer_out = 8'b10000000;
//                        else layer_out <= {1'b1, norm_out[WIDTH_L1_NORM_FL+WIDTH_L1_O_IL-1:WIDTH_L1_NORM_FL], norm_out[WIDTH_L1_NORM_FL-1:WIDTH_L1_NORM_FL-WIDTH_L1_O_FL]};
//                    end
//                    else begin
//                        if(norm_out > 2**(WIDTH_L1_O_IL+WIDTH_L1_NORM_FL)-1) layer_out = 8'b01111111;
//                        else layer_out = {1'b0, norm_out[WIDTH_L1_NORM_FL+WIDTH_L1_O_IL-1:WIDTH_L1_NORM_FL], norm_out[WIDTH_L1_NORM_FL-1:WIDTH_L1_NORM_FL-WIDTH_L1_O_FL]};
//                    end
//                end
//            end
//            5'b00010: begin
//                if(relu_on) begin   //relu on
//                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;            // if norm_out is negative layer_out is 0
//                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L2_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L2_O_FL;
//                    else layer_out = {1'b0, norm_out[WIDTH_L2_NORM_FL+WIDTH_L2_O_IL-1:WIDTH_L2_NORM_FL], norm_out[WIDTH_L2_NORM_FL-1:WIDTH_L2_NORM_FL-WIDTH_L1_O_FL]};
//                end
//                else begin // no relu
//                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
//                        if(norm_out < -(2**(WIDTH_L2_O_IL+WIDTH_L2_NORM_FL))) layer_out = 8'b10000000;
//                        else layer_out <= {1'b1, norm_out[WIDTH_L2_NORM_FL+WIDTH_L2_O_IL-1:WIDTH_L2_NORM_FL], norm_out[WIDTH_L2_NORM_FL-1:WIDTH_L2_NORM_FL-WIDTH_L1_O_FL]};
//                    end
//                    else begin
//                        if(norm_out > 2**(WIDTH_L2_O_IL+WIDTH_L2_NORM_FL)-1) layer_out = 8'b01111111;
//                        else layer_out = {1'b0, norm_out[WIDTH_L2_NORM_FL+WIDTH_L2_O_IL-1:WIDTH_L2_NORM_FL], norm_out[WIDTH_L2_NORM_FL-1:WIDTH_L2_NORM_FL-WIDTH_L1_O_FL]};
//                    end
//                end
//            end
//            5'b00011: begin
//                if(relu_on) begin   //relu on
//                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;            // if norm_out is negative layer_out is 0
//                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L3_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L3_O_FL;
//                    else layer_out = {1'b0, norm_out[WIDTH_L3_NORM_FL+WIDTH_L3_O_IL-1:WIDTH_L3_NORM_FL], norm_out[WIDTH_L3_NORM_FL-1:WIDTH_L3_NORM_FL-WIDTH_L1_O_FL]};
//                end
//                else begin // no relu
//                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
//                        if(norm_out < -(2**(WIDTH_L3_O_IL+WIDTH_L3_NORM_FL))) layer_out = 8'b10000000;
//                        else layer_out <= {1'b1, norm_out[WIDTH_L3_NORM_FL+WIDTH_L3_O_IL-1:WIDTH_L3_NORM_FL], norm_out[WIDTH_L3_NORM_FL-1:WIDTH_L3_NORM_FL-WIDTH_L1_O_FL]};
//                    end
//                    else begin
//                        if(norm_out > 2**(WIDTH_L3_O_IL+WIDTH_L3_NORM_FL)-1) layer_out = 8'b01111111;
//                        else layer_out = {1'b0, norm_out[WIDTH_L3_NORM_FL+WIDTH_L3_O_IL-1:WIDTH_L3_NORM_FL], norm_out[WIDTH_L3_NORM_FL-1:WIDTH_L3_NORM_FL-WIDTH_L1_O_FL]};
//                    end
//                end
//            end
//            5'b00100: begin
                
//            end
//            5'b00101: begin
                
//            end
//            5'b00110: begin
                
//            end
//            5'b00111: begin
                
//            end
//            5'b01000: begin
                
//            end
//            5'b01001: begin
                
//            end
            5'b00001: begin //L1
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L1_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L1_O_FL;
                    else layer_out = {1'b0, norm_out[WIDTH_L1_NORM_FL+WIDTH_L1_O_IL-1:WIDTH_L1_NORM_FL], 
                                      norm_out[WIDTH_L1_NORM_FL-1:WIDTH_L1_NORM_FL-WIDTH_L1_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L1_O_IL+WIDTH_L1_NORM_FL))) layer_out = 8'b10000000;
                        else layer_out = {1'b1, norm_out[WIDTH_L1_NORM_FL+WIDTH_L1_O_IL-1:WIDTH_L1_NORM_FL], 
                                          norm_out[WIDTH_L1_NORM_FL-1:WIDTH_L1_NORM_FL-WIDTH_L1_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L1_O_IL+WIDTH_L1_NORM_FL)-1) layer_out = 8'b01111111;
                        else layer_out = {1'b0, norm_out[WIDTH_L1_NORM_FL+WIDTH_L1_O_IL-1:WIDTH_L1_NORM_FL], 
                                          norm_out[WIDTH_L1_NORM_FL-1:WIDTH_L1_NORM_FL-WIDTH_L1_O_FL]};
                    end
                end
            end
            5'b00010: begin //L2
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L2_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L2_O_FL;
                    else layer_out = {1'b0, norm_out[WIDTH_L2_NORM_FL+WIDTH_L2_O_IL-1:WIDTH_L2_NORM_FL], 
                                      norm_out[WIDTH_L2_NORM_FL-1:WIDTH_L2_NORM_FL-WIDTH_L2_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L2_O_IL+WIDTH_L2_NORM_FL))) layer_out = 8'b10000000;
                        else layer_out = {1'b1, norm_out[WIDTH_L2_NORM_FL+WIDTH_L2_O_IL-1:WIDTH_L2_NORM_FL], 
                                          norm_out[WIDTH_L2_NORM_FL-1:WIDTH_L2_NORM_FL-WIDTH_L2_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L2_O_IL+WIDTH_L2_NORM_FL)-1) layer_out = 8'b01111111;
                        else layer_out = {1'b0, norm_out[WIDTH_L2_NORM_FL+WIDTH_L2_O_IL-1:WIDTH_L2_NORM_FL], 
                                          norm_out[WIDTH_L2_NORM_FL-1:WIDTH_L2_NORM_FL-WIDTH_L2_O_FL]};
                    end
                end
            end
            5'b00011: begin //L3
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L3_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L3_O_FL;
                    else layer_out = {1'b0, norm_out[WIDTH_L3_NORM_FL+WIDTH_L3_O_IL-1:WIDTH_L3_NORM_FL], 
                                      norm_out[WIDTH_L3_NORM_FL-1:WIDTH_L3_NORM_FL-WIDTH_L3_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L3_O_IL+WIDTH_L3_NORM_FL))) layer_out = 8'b10000000;
                        else layer_out = {1'b1, norm_out[WIDTH_L3_NORM_FL+WIDTH_L3_O_IL-1:WIDTH_L3_NORM_FL], 
                                          norm_out[WIDTH_L3_NORM_FL-1:WIDTH_L3_NORM_FL-WIDTH_L3_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L3_O_IL+WIDTH_L3_NORM_FL)-1) layer_out = 8'b01111111;
                        else layer_out = {1'b0, norm_out[WIDTH_L3_NORM_FL+WIDTH_L3_O_IL-1:WIDTH_L3_NORM_FL], 
                                          norm_out[WIDTH_L3_NORM_FL-1:WIDTH_L3_NORM_FL-WIDTH_L3_O_FL]};
                    end
                end
            end
            5'b00100: begin //L4
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L4_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L4_O_FL;
                    else if (WIDTH_L4_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L4_NORM_FL-1:WIDTH_L4_NORM_FL-WIDTH_L4_O_FL]};
                    else layer_out = {1'b0, norm_out[WIDTH_L4_NORM_FL+WIDTH_L4_O_IL-1:WIDTH_L4_NORM_FL], 
                                      norm_out[WIDTH_L4_NORM_FL-1:WIDTH_L4_NORM_FL-WIDTH_L4_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L4_O_IL+WIDTH_L4_NORM_FL))) layer_out = 8'b10000000;
                        else if (WIDTH_L4_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L4_NORM_FL-1:WIDTH_L4_NORM_FL-WIDTH_L4_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L4_NORM_FL+WIDTH_L4_O_IL-1:WIDTH_L4_NORM_FL], 
                                        norm_out[WIDTH_L4_NORM_FL-1:WIDTH_L4_NORM_FL-WIDTH_L4_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L4_O_IL+WIDTH_L4_NORM_FL)-1) layer_out = 8'b01111111;
                        else if (WIDTH_L4_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L4_NORM_FL-1:WIDTH_L4_NORM_FL-WIDTH_L4_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L4_NORM_FL+WIDTH_L4_O_IL-1:WIDTH_L4_NORM_FL], 
                                        norm_out[WIDTH_L4_NORM_FL-1:WIDTH_L4_NORM_FL-WIDTH_L4_O_FL]};
                    end
                end
            end
            5'b00101: begin //L5
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L5_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L5_O_FL;
                    else if (WIDTH_L5_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L5_NORM_FL-1:WIDTH_L5_NORM_FL-WIDTH_L5_O_FL]};
                    else layer_out = {1'b0, norm_out[WIDTH_L5_NORM_FL+WIDTH_L5_O_IL-1:WIDTH_L5_NORM_FL], 
                                      norm_out[WIDTH_L5_NORM_FL-1:WIDTH_L5_NORM_FL-WIDTH_L5_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L5_O_IL+WIDTH_L5_NORM_FL))) layer_out = 8'b10000000;
                        else if (WIDTH_L5_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L5_NORM_FL-1:WIDTH_L5_NORM_FL-WIDTH_L5_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L5_NORM_FL+WIDTH_L5_O_IL-1:WIDTH_L5_NORM_FL], 
                                      norm_out[WIDTH_L5_NORM_FL-1:WIDTH_L5_NORM_FL-WIDTH_L5_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L5_O_IL+WIDTH_L5_NORM_FL)-1) layer_out = 8'b01111111;
                        else if (WIDTH_L5_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L5_NORM_FL-1:WIDTH_L5_NORM_FL-WIDTH_L5_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L5_NORM_FL+WIDTH_L5_O_IL-1:WIDTH_L5_NORM_FL], 
                                      norm_out[WIDTH_L5_NORM_FL-1:WIDTH_L5_NORM_FL-WIDTH_L5_O_FL]};
                    end
                end
            end
            5'b00110: begin //L6
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L6_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L6_O_FL;
                    else if (WIDTH_L6_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L6_NORM_FL-1:WIDTH_L6_NORM_FL-WIDTH_L6_O_FL]};
                    else layer_out = {1'b0, norm_out[WIDTH_L6_NORM_FL+WIDTH_L6_O_IL-1:WIDTH_L6_NORM_FL], 
                                      norm_out[WIDTH_L6_NORM_FL-1:WIDTH_L6_NORM_FL-WIDTH_L6_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L6_O_IL+WIDTH_L6_NORM_FL))) layer_out = 8'b10000000;
                        else if (WIDTH_L6_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L6_NORM_FL-1:WIDTH_L6_NORM_FL-WIDTH_L6_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L6_NORM_FL+WIDTH_L6_O_IL-1:WIDTH_L6_NORM_FL], 
                                      norm_out[WIDTH_L6_NORM_FL-1:WIDTH_L6_NORM_FL-WIDTH_L6_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L6_O_IL+WIDTH_L6_NORM_FL)-1) layer_out = 8'b01111111;
                        else if (WIDTH_L6_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L6_NORM_FL-1:WIDTH_L6_NORM_FL-WIDTH_L6_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L6_NORM_FL+WIDTH_L6_O_IL-1:WIDTH_L6_NORM_FL], 
                                      norm_out[WIDTH_L6_NORM_FL-1:WIDTH_L6_NORM_FL-WIDTH_L6_O_FL]};
                    end
                end
            end
            5'b00111: begin //L7
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L7_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L7_O_FL;
                    else layer_out = {1'b0, norm_out[WIDTH_L7_NORM_FL+WIDTH_L7_O_IL-1:WIDTH_L7_NORM_FL],    
                                      norm_out[WIDTH_L7_NORM_FL-1:WIDTH_L7_NORM_FL-WIDTH_L7_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L7_O_IL+WIDTH_L7_NORM_FL))) layer_out = 8'b10000000;
                        else layer_out = {1'b1, norm_out[WIDTH_L7_NORM_FL+WIDTH_L7_O_IL-1:WIDTH_L7_NORM_FL], 
                                          norm_out[WIDTH_L7_NORM_FL-1:WIDTH_L7_NORM_FL-WIDTH_L7_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L7_O_IL+WIDTH_L7_NORM_FL)-1) layer_out = 8'b01111111;
                        else layer_out = {1'b0, norm_out[WIDTH_L7_NORM_FL+WIDTH_L7_O_IL-1:WIDTH_L7_NORM_FL], 
                                          norm_out[WIDTH_L7_NORM_FL-1:WIDTH_L7_NORM_FL-WIDTH_L7_O_FL]};
                    end
                end
            end
            5'b01000: begin //L8
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L8_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L8_O_FL;
                    else layer_out = {1'b0, norm_out[WIDTH_L8_NORM_FL+WIDTH_L8_O_IL-1:WIDTH_L8_NORM_FL], 
                                      norm_out[WIDTH_L8_NORM_FL-1:WIDTH_L8_NORM_FL-WIDTH_L8_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L8_O_IL+WIDTH_L8_NORM_FL))) layer_out = 8'b10000000;
                        else layer_out = {1'b1, norm_out[WIDTH_L8_NORM_FL+WIDTH_L8_O_IL-1:WIDTH_L8_NORM_FL], 
                                          norm_out[WIDTH_L8_NORM_FL-1:WIDTH_L8_NORM_FL-WIDTH_L8_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L8_O_IL+WIDTH_L8_NORM_FL)-1) layer_out = 8'b01111111;
                        else layer_out = {1'b0, norm_out[WIDTH_L8_NORM_FL+WIDTH_L8_O_IL-1:WIDTH_L8_NORM_FL], 
                                          norm_out[WIDTH_L8_NORM_FL-1:WIDTH_L8_NORM_FL-WIDTH_L8_O_FL]};
                    end
                end
            end
            5'b01001: begin //L9
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L9_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L9_O_FL;
                    else layer_out = {1'b0, norm_out[WIDTH_L9_NORM_FL+WIDTH_L9_O_IL-1:WIDTH_L9_NORM_FL], 
                                      norm_out[WIDTH_L9_NORM_FL-1:WIDTH_L9_NORM_FL-WIDTH_L9_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L9_O_IL+WIDTH_L9_NORM_FL))) layer_out = 8'b10000000;
                        else layer_out = {1'b1, norm_out[WIDTH_L9_NORM_FL+WIDTH_L9_O_IL-1:WIDTH_L9_NORM_FL], 
                                          norm_out[WIDTH_L9_NORM_FL-1:WIDTH_L9_NORM_FL-WIDTH_L9_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L9_O_IL+WIDTH_L9_NORM_FL)-1) layer_out = 8'b01111111;
                        else layer_out = {1'b0, norm_out[WIDTH_L9_NORM_FL+WIDTH_L9_O_IL-1:WIDTH_L9_NORM_FL], 
                                          norm_out[WIDTH_L9_NORM_FL-1:WIDTH_L9_NORM_FL-WIDTH_L9_O_FL]};
                    end
                end
            end
            5'b01010: begin //L10
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L10_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L10_O_FL;
                    else layer_out = {1'b0, norm_out[WIDTH_L10_NORM_FL+WIDTH_L10_O_IL-1:WIDTH_L10_NORM_FL], 
                                      norm_out[WIDTH_L10_NORM_FL-1:WIDTH_L10_NORM_FL-WIDTH_L10_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L10_O_IL+WIDTH_L10_NORM_FL))) layer_out = 8'b10000000;
                        else layer_out = {1'b1, norm_out[WIDTH_L10_NORM_FL+WIDTH_L10_O_IL-1:WIDTH_L10_NORM_FL], 
                                          norm_out[WIDTH_L10_NORM_FL-1:WIDTH_L10_NORM_FL-WIDTH_L10_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L10_O_IL+WIDTH_L10_NORM_FL)-1) layer_out = 8'b01111111;
                        else layer_out = {1'b0, norm_out[WIDTH_L10_NORM_FL+WIDTH_L10_O_IL-1:WIDTH_L10_NORM_FL], 
                                          norm_out[WIDTH_L10_NORM_FL-1:WIDTH_L10_NORM_FL-WIDTH_L10_O_FL]};
                    end
                end
            end
            5'b01011: begin //L11
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L11_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L11_O_FL;
                    else if (WIDTH_L11_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L11_NORM_FL-1:WIDTH_L11_NORM_FL-WIDTH_L11_O_FL]};
                    else layer_out = {1'b0, norm_out[WIDTH_L11_NORM_FL+WIDTH_L11_O_IL-1:WIDTH_L11_NORM_FL], 
                                      norm_out[WIDTH_L11_NORM_FL-1:WIDTH_L11_NORM_FL-WIDTH_L11_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L11_O_IL+WIDTH_L11_NORM_FL))) layer_out = 8'b10000000;
                        else if (WIDTH_L11_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L11_NORM_FL-1:WIDTH_L11_NORM_FL-WIDTH_L11_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L11_NORM_FL+WIDTH_L11_O_IL-1:WIDTH_L11_NORM_FL], 
                                      norm_out[WIDTH_L11_NORM_FL-1:WIDTH_L11_NORM_FL-WIDTH_L11_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L11_O_IL+WIDTH_L11_NORM_FL)-1) layer_out = 8'b01111111;
                        else if (WIDTH_L11_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L11_NORM_FL-1:WIDTH_L11_NORM_FL-WIDTH_L11_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L11_NORM_FL+WIDTH_L11_O_IL-1:WIDTH_L11_NORM_FL], 
                                      norm_out[WIDTH_L11_NORM_FL-1:WIDTH_L11_NORM_FL-WIDTH_L11_O_FL]};
                    end
                end
            end
            5'b01100: begin //L12
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L12_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L12_O_FL;
                    else if (WIDTH_L12_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L12_NORM_FL-1:WIDTH_L12_NORM_FL-WIDTH_L12_O_FL]};
                    else layer_out = {1'b0, norm_out[WIDTH_L12_NORM_FL+WIDTH_L12_O_IL-1:WIDTH_L12_NORM_FL], 
                                      norm_out[WIDTH_L12_NORM_FL-1:WIDTH_L12_NORM_FL-WIDTH_L12_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L12_O_IL+WIDTH_L12_NORM_FL))) layer_out = 8'b10000000;
                        else if (WIDTH_L12_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L12_NORM_FL-1:WIDTH_L12_NORM_FL-WIDTH_L12_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L12_NORM_FL+WIDTH_L12_O_IL-1:WIDTH_L12_NORM_FL], 
                                      norm_out[WIDTH_L12_NORM_FL-1:WIDTH_L12_NORM_FL-WIDTH_L12_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L12_O_IL+WIDTH_L12_NORM_FL)-1) layer_out = 8'b01111111;
                        else if (WIDTH_L12_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L12_NORM_FL-1:WIDTH_L12_NORM_FL-WIDTH_L12_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L12_NORM_FL+WIDTH_L12_O_IL-1:WIDTH_L12_NORM_FL], 
                                      norm_out[WIDTH_L12_NORM_FL-1:WIDTH_L12_NORM_FL-WIDTH_L12_O_FL]};
                    end
                end
            end
            5'b01101: begin //L13
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L13_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L13_O_FL;
                    else if (WIDTH_L13_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L13_NORM_FL-1:WIDTH_L13_NORM_FL-WIDTH_L13_O_FL]};
                    else layer_out = {1'b0, norm_out[WIDTH_L13_NORM_FL+WIDTH_L13_O_IL-1:WIDTH_L13_NORM_FL], 
                                      norm_out[WIDTH_L13_NORM_FL-1:WIDTH_L13_NORM_FL-WIDTH_L13_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L13_O_IL+WIDTH_L13_NORM_FL))) layer_out = 8'b10000000;
                        else if (WIDTH_L13_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L13_NORM_FL-1:WIDTH_L13_NORM_FL-WIDTH_L13_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L13_NORM_FL+WIDTH_L13_O_IL-1:WIDTH_L13_NORM_FL], 
                                      norm_out[WIDTH_L13_NORM_FL-1:WIDTH_L13_NORM_FL-WIDTH_L13_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L13_O_IL+WIDTH_L13_NORM_FL)-1) layer_out = 8'b01111111;
                        else if (WIDTH_L13_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L13_NORM_FL-1:WIDTH_L13_NORM_FL-WIDTH_L13_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L13_NORM_FL+WIDTH_L13_O_IL-1:WIDTH_L13_NORM_FL], 
                                      norm_out[WIDTH_L13_NORM_FL-1:WIDTH_L13_NORM_FL-WIDTH_L13_O_FL]};
                    end
                end
            end
            5'b01110: begin //L14
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L14_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L14_O_FL;
                    else if (WIDTH_L14_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L14_NORM_FL-1:WIDTH_L14_NORM_FL-WIDTH_L14_O_FL]};
                    else layer_out = {1'b0, norm_out[WIDTH_L14_NORM_FL+WIDTH_L14_O_IL-1:WIDTH_L14_NORM_FL], 
                                      norm_out[WIDTH_L14_NORM_FL-1:WIDTH_L14_NORM_FL-WIDTH_L14_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L14_O_IL+WIDTH_L14_NORM_FL))) layer_out = 8'b10000000;
                        else if (WIDTH_L14_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L14_NORM_FL-1:WIDTH_L14_NORM_FL-WIDTH_L14_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L14_NORM_FL+WIDTH_L14_O_IL-1:WIDTH_L14_NORM_FL], 
                                        norm_out[WIDTH_L14_NORM_FL-1:WIDTH_L14_NORM_FL-WIDTH_L14_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L14_O_IL+WIDTH_L14_NORM_FL)-1) layer_out = 8'b01111111;
                        else if (WIDTH_L14_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L14_NORM_FL-1:WIDTH_L14_NORM_FL-WIDTH_L14_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L14_NORM_FL+WIDTH_L14_O_IL-1:WIDTH_L14_NORM_FL], 
                                        norm_out[WIDTH_L14_NORM_FL-1:WIDTH_L14_NORM_FL-WIDTH_L14_O_FL]};
                    end
                end
            end
            5'b01111: begin //L15
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L15_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L15_O_FL;
                    else if (WIDTH_L15_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L15_NORM_FL-1:WIDTH_L15_NORM_FL-WIDTH_L15_O_FL]};
                    else layer_out = {1'b0, norm_out[WIDTH_L15_NORM_FL+WIDTH_L15_O_IL-1:WIDTH_L15_NORM_FL], 
                                      norm_out[WIDTH_L15_NORM_FL-1:WIDTH_L15_NORM_FL-WIDTH_L15_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L15_O_IL+WIDTH_L15_NORM_FL))) layer_out = 8'b10000000;
                        else if (WIDTH_L15_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L15_NORM_FL-1:WIDTH_L15_NORM_FL-WIDTH_L15_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L15_NORM_FL+WIDTH_L15_O_IL-1:WIDTH_L15_NORM_FL], 
                                      norm_out[WIDTH_L15_NORM_FL-1:WIDTH_L15_NORM_FL-WIDTH_L15_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L15_O_IL+WIDTH_L15_NORM_FL)-1) layer_out = 8'b01111111;
                        else if (WIDTH_L15_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L15_NORM_FL-1:WIDTH_L15_NORM_FL-WIDTH_L15_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L15_NORM_FL+WIDTH_L15_O_IL-1:WIDTH_L15_NORM_FL], 
                                      norm_out[WIDTH_L15_NORM_FL-1:WIDTH_L15_NORM_FL-WIDTH_L15_O_FL]};
                    end
                end
            end
            5'b10000: begin //L16
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L16_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L16_O_FL;
                    else layer_out = {1'b0, norm_out[WIDTH_L16_NORM_FL+WIDTH_L16_O_IL-1:WIDTH_L16_NORM_FL], 
                                      norm_out[WIDTH_L16_NORM_FL-1:WIDTH_L16_NORM_FL-WIDTH_L16_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L16_O_IL+WIDTH_L16_NORM_FL))) layer_out = 8'b10000000;
                        else layer_out = {1'b1, norm_out[WIDTH_L16_NORM_FL+WIDTH_L16_O_IL-1:WIDTH_L16_NORM_FL], 
                                          norm_out[WIDTH_L16_NORM_FL-1:WIDTH_L16_NORM_FL-WIDTH_L16_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L16_O_IL+WIDTH_L16_NORM_FL)-1) layer_out = 8'b01111111;
                        else layer_out = {1'b0, norm_out[WIDTH_L16_NORM_FL+WIDTH_L16_O_IL-1:WIDTH_L16_NORM_FL], 
                                          norm_out[WIDTH_L16_NORM_FL-1:WIDTH_L16_NORM_FL-WIDTH_L16_O_FL]};
                    end
                end
            end
            5'b10001: begin //L17
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L17_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L17_O_FL;
                    else layer_out = {1'b0, norm_out[WIDTH_L17_NORM_FL+WIDTH_L17_O_IL-1:WIDTH_L17_NORM_FL], 
                                      norm_out[WIDTH_L17_NORM_FL-1:WIDTH_L17_NORM_FL-WIDTH_L17_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L17_O_IL+WIDTH_L17_NORM_FL))) layer_out = 8'b10000000;
                        else layer_out = {1'b1, norm_out[WIDTH_L17_NORM_FL+WIDTH_L17_O_IL-1:WIDTH_L17_NORM_FL], 
                                          norm_out[WIDTH_L17_NORM_FL-1:WIDTH_L17_NORM_FL-WIDTH_L17_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L17_O_IL+WIDTH_L17_NORM_FL)-1) layer_out = 8'b01111111;
                        else layer_out = {1'b0, norm_out[WIDTH_L17_NORM_FL+WIDTH_L17_O_IL-1:WIDTH_L17_NORM_FL], 
                                          norm_out[WIDTH_L17_NORM_FL-1:WIDTH_L17_NORM_FL-WIDTH_L17_O_FL]};
                    end
                end
            end
            5'b10010: begin //L18i
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L18_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L18_O_FL;
                    else if (WIDTH_L18_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L18_NORM_FL-1:WIDTH_L18_NORM_FL-WIDTH_L18_O_FL]};
                    else layer_out = {1'b0, norm_out[WIDTH_L18_NORM_FL+WIDTH_L18_O_IL-1:WIDTH_L18_NORM_FL], 
                                      norm_out[WIDTH_L18_NORM_FL-1:WIDTH_L18_NORM_FL-WIDTH_L18_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L18_O_IL+WIDTH_L18_NORM_FL))) layer_out = 8'b10000000;
                        else if (WIDTH_L18_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L18_NORM_FL-1:WIDTH_L18_NORM_FL-WIDTH_L18_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L18_NORM_FL+WIDTH_L18_O_IL-1:WIDTH_L18_NORM_FL], 
                                      norm_out[WIDTH_L18_NORM_FL-1:WIDTH_L18_NORM_FL-WIDTH_L18_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L18_O_IL+WIDTH_L18_NORM_FL)-1) layer_out = 8'b01111111;
                        else if (WIDTH_L18_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L18_NORM_FL-1:WIDTH_L18_NORM_FL-WIDTH_L18_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L18_NORM_FL+WIDTH_L18_O_IL-1:WIDTH_L18_NORM_FL], 
                                      norm_out[WIDTH_L18_NORM_FL-1:WIDTH_L18_NORM_FL-WIDTH_L18_O_FL]};
                    end
                end
            end
            5'b10011: begin //L19
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L19_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L19_O_FL;
                    else if (WIDTH_L19_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L19_NORM_FL-1:WIDTH_L19_NORM_FL-WIDTH_L19_O_FL]};
                    else layer_out = {1'b0, norm_out[WIDTH_L19_NORM_FL+WIDTH_L19_O_IL-1:WIDTH_L19_NORM_FL], 
                                      norm_out[WIDTH_L19_NORM_FL-1:WIDTH_L19_NORM_FL-WIDTH_L19_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L19_O_IL+WIDTH_L19_NORM_FL))) layer_out = 8'b10000000;
                        else if (WIDTH_L19_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L19_NORM_FL-1:WIDTH_L19_NORM_FL-WIDTH_L19_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L19_NORM_FL+WIDTH_L19_O_IL-1:WIDTH_L19_NORM_FL], 
                                      norm_out[WIDTH_L19_NORM_FL-1:WIDTH_L19_NORM_FL-WIDTH_L19_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L19_O_IL+WIDTH_L19_NORM_FL)-1) layer_out = 8'b01111111;
                        else if (WIDTH_L19_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L19_NORM_FL-1:WIDTH_L19_NORM_FL-WIDTH_L19_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L19_NORM_FL+WIDTH_L19_O_IL-1:WIDTH_L19_NORM_FL], 
                                      norm_out[WIDTH_L19_NORM_FL-1:WIDTH_L19_NORM_FL-WIDTH_L19_O_FL]};
                    end
                end
            end
            5'b10100: begin //L20
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L20_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L20_O_FL;
                    else if (WIDTH_L20_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L20_NORM_FL-1:WIDTH_L20_NORM_FL-WIDTH_L20_O_FL]};
                    else layer_out = {1'b0, norm_out[WIDTH_L20_NORM_FL+WIDTH_L20_O_IL-1:WIDTH_L20_NORM_FL], 
                                      norm_out[WIDTH_L20_NORM_FL-1:WIDTH_L20_NORM_FL-WIDTH_L20_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L20_O_IL+WIDTH_L20_NORM_FL))) layer_out = 8'b10000000;
                        else if (WIDTH_L20_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L20_NORM_FL-1:WIDTH_L20_NORM_FL-WIDTH_L20_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L20_NORM_FL+WIDTH_L20_O_IL-1:WIDTH_L20_NORM_FL], 
                                      norm_out[WIDTH_L20_NORM_FL-1:WIDTH_L20_NORM_FL-WIDTH_L20_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L20_O_IL+WIDTH_L20_NORM_FL)-1) layer_out = 8'b01111111;
                        else if (WIDTH_L20_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L20_NORM_FL-1:WIDTH_L20_NORM_FL-WIDTH_L20_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L20_NORM_FL+WIDTH_L20_O_IL-1:WIDTH_L20_NORM_FL], 
                                      norm_out[WIDTH_L20_NORM_FL-1:WIDTH_L20_NORM_FL-WIDTH_L20_O_FL]};
                    end
                end
            end
            5'b10101: begin //L21
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L21_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L21_O_FL;
                    else if (WIDTH_L21_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L21_NORM_FL-1:WIDTH_L21_NORM_FL-WIDTH_L21_O_FL]};
                    else layer_out = {1'b0, norm_out[WIDTH_L21_NORM_FL+WIDTH_L21_O_IL-1:WIDTH_L21_NORM_FL], 
                                      norm_out[WIDTH_L21_NORM_FL-1:WIDTH_L21_NORM_FL-WIDTH_L21_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L21_O_IL+WIDTH_L21_NORM_FL))) layer_out = 8'b10000000;
                        else if (WIDTH_L21_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L21_NORM_FL-1:WIDTH_L21_NORM_FL-WIDTH_L21_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L21_NORM_FL+WIDTH_L21_O_IL-1:WIDTH_L21_NORM_FL], 
                                      norm_out[WIDTH_L21_NORM_FL-1:WIDTH_L21_NORM_FL-WIDTH_L21_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L21_O_IL+WIDTH_L21_NORM_FL)-1) layer_out = 8'b01111111;
                        else if (WIDTH_L21_O_IL == 0) layer_out = {1'b0, norm_out[WIDTH_L21_NORM_FL-1:WIDTH_L21_NORM_FL-WIDTH_L21_O_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L21_NORM_FL+WIDTH_L21_O_IL-1:WIDTH_L21_NORM_FL], 
                                      norm_out[WIDTH_L21_NORM_FL-1:WIDTH_L21_NORM_FL-WIDTH_L21_O_FL]};
                    end
                end
            end
            5'b10110: begin //L22
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L22_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L22_O_FL;
                    else if (WIDTH_L22_O_FL == 0) layer_out = {1'b0, norm_out[WIDTH_L22_NORM_FL+WIDTH_L22_O_IL-1:WIDTH_L22_NORM_FL]};
                    else layer_out = {1'b0, norm_out[WIDTH_L22_NORM_FL+WIDTH_L22_O_IL-1:WIDTH_L22_NORM_FL], 
                                      norm_out[WIDTH_L22_NORM_FL-1:WIDTH_L22_NORM_FL-WIDTH_L22_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L22_O_IL+WIDTH_L22_NORM_FL))) layer_out = 8'b10000000;
                        else if (WIDTH_L22_O_FL == 0) layer_out = {1'b0, norm_out[WIDTH_L22_NORM_FL+WIDTH_L22_O_IL-1:WIDTH_L22_NORM_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L22_NORM_FL+WIDTH_L22_O_IL-1:WIDTH_L22_NORM_FL], 
                                      norm_out[WIDTH_L22_NORM_FL-1:WIDTH_L22_NORM_FL-WIDTH_L22_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L22_O_IL+WIDTH_L22_NORM_FL)-1) layer_out = 8'b01111111;
                        else if (WIDTH_L22_O_FL == 0) layer_out = {1'b0, norm_out[WIDTH_L22_NORM_FL+WIDTH_L22_O_IL-1:WIDTH_L22_NORM_FL]};
                        else layer_out = {1'b0, norm_out[WIDTH_L22_NORM_FL+WIDTH_L22_O_IL-1:WIDTH_L22_NORM_FL], 
                                      norm_out[WIDTH_L22_NORM_FL-1:WIDTH_L22_NORM_FL-WIDTH_L22_O_FL]};
                    end
                end
            end
            5'b10111: begin //L23
                if(relu_on) begin   // relu on
                    if(norm_out[WIDTH_NORM_O_DATA-1]) layer_out = 8'b00000000;  // if norm_out is negative layer_out is 0
                    else if(norm_out > RELU_MAX_VAL*(2**(WIDTH_L23_NORM_FL))) layer_out = RELU_MAX_VAL <<< WIDTH_L23_O_FL;
                    else layer_out = {1'b0, norm_out[WIDTH_L23_NORM_FL+WIDTH_L23_O_IL-1:WIDTH_L23_NORM_FL], 
                                      norm_out[WIDTH_L23_NORM_FL-1:WIDTH_L23_NORM_FL-WIDTH_L23_O_FL]};
                end
                else begin // no relu
                    if(norm_out[WIDTH_NORM_O_DATA-1]) begin
                        if(norm_out < -(2**(WIDTH_L23_O_IL+WIDTH_L23_NORM_FL))) layer_out = 8'b10000000;
                        else layer_out = {1'b1, norm_out[WIDTH_L23_NORM_FL+WIDTH_L23_O_IL-1:WIDTH_L23_NORM_FL], 
                                          norm_out[WIDTH_L23_NORM_FL-1:WIDTH_L23_NORM_FL-WIDTH_L23_O_FL]};
                    end
                    else begin
                        if(norm_out > 2**(WIDTH_L23_O_IL+WIDTH_L23_NORM_FL)-1) layer_out = 8'b01111111;
                        else layer_out = {1'b0, norm_out[WIDTH_L23_NORM_FL+WIDTH_L23_O_IL-1:WIDTH_L23_NORM_FL], 
                                          norm_out[WIDTH_L23_NORM_FL-1:WIDTH_L23_NORM_FL-WIDTH_L23_O_FL]};
                    end
                end
            end
            default : begin 
                layer_out = 8'b00000000;
            end
        endcase
    end
    
endmodule