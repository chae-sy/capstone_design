`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/16 23:29:37
// Design Name: 
// Module Name: adder_tree_v1
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


module adder_tree #(parameter INPUT_WIDTH = 4, WEIGHT_WIDTH = 4, INT_EXTEND = 9)(
    input sel,
    input signed [(INPUT_WIDTH + WEIGHT_WIDTH + INT_EXTEND)-1:0] data_i [0:31],
    output signed [(INPUT_WIDTH + WEIGHT_WIDTH + INT_EXTEND)-1:0] data_o
    );

    localparam DATA_WIDTH = INPUT_WIDTH + WEIGHT_WIDTH + INT_EXTEND;

    wire signed [(DATA_WIDTH)-1:0] stage0_sum [0:31];
    wire signed [(DATA_WIDTH)-1:0] stage1_sum [0:15];
    wire signed [(DATA_WIDTH)-1:0] stage2_sum [0:7];
    wire signed [(DATA_WIDTH)-1:0] stage3_sum [0:3];
    wire signed [(DATA_WIDTH)-1:0] stage4_sum [0:1];
    wire signed [(DATA_WIDTH)-1:0] stage5_sum;
    
    genvar s0, s1, s2, s3, s4;
    generate 
        for(s0=0; s0<32; s0=s0+1) begin
            assign stage0_sum[s0] = (sel) ? 0 : data_i[s0];
        end
    endgenerate
    
    generate 
        for(s1=0; s1<16; s1=s1+1) begin
            assign stage1_sum[s1] = stage0_sum[2*s1] + stage0_sum[2*s1+1];
        end
    endgenerate

    generate 
        for(s2=0; s2<8; s2=s2+1) begin
            assign stage2_sum[s2] = stage1_sum[2*s2] + stage1_sum[2*s2+1];
        end
    endgenerate

    generate 
        for(s3=0; s3<4; s3=s3+1) begin
            assign stage3_sum[s3] = stage2_sum[2*s3] + stage2_sum[2*s3+1];
        end
    endgenerate

    generate 
        for(s4=0; s4<2; s4=s4+1) begin
            assign stage4_sum[s4] = stage3_sum[2*s4] + stage3_sum[2*s4+1];
        end
    endgenerate

    assign stage5_sum = stage4_sum[0] + stage4_sum[1];

    // Controller should extract data_o at the appropriate clock timing.
    
    // Activation: ReLU
    assign data_o = stage5_sum;

endmodule
