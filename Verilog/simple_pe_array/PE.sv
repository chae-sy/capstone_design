`timescale 1ns / 1ps
module clk_gate (
input clk_in, en,
output clk_out
);
assign clk_out = clk_in & en;
endmodule


// (1) 원본 PE 모듈은 그대로 재사용
module PE_with_clock_gating #(
    parameter DATA_WIDTH  = 8
)(
    input  wire                       clk,
    input  wire                       enable,
    input  wire                       reset,
    input  wire [DATA_WIDTH-1:0]      data_in,
    input  wire [DATA_WIDTH-1:0]      weight,
    output reg  [2*DATA_WIDTH-1:0]    result
);

wire gated_clk;
clk_gate u_clk_gate(.clk_in(clk), .en(enable), .clk_out(gated_clk));

    integer i;
    always @(posedge gated_clk or posedge reset) begin
        if (reset) begin
                result <= {2*DATA_WIDTH{1'b0}};
        end else begin
                result <= result + data_in * weight;
        end
    end
endmodule


//// (2) COLOR_WIDTH를 그대로 superscalar 폭으로 쓴 PE_super_scalar
//module PE_super_scalar #(
//    parameter DATA_WIDTH  = 8,
//    parameter COLOR_WIDTH = 3   //R, G, B superscalar 폭으로 사용
//)(
//    input  wire                       clk,
//    input  wire                       enable,
//    input  wire                       reset,
//    // superscalar lanes = COLOR_WIDTH
//    input  wire [DATA_WIDTH-1:0]      data_in  [0:COLOR_WIDTH-1],
//    input  wire [DATA_WIDTH-1:0]      weight,
//    output wire [2*DATA_WIDTH-1:0]    result   [0:COLOR_WIDTH-1]
//);

//    genvar lane;
//    generate
//        for (lane = 0; lane < COLOR_WIDTH; lane = lane + 1) begin : GEN_PE
//            // 각 lane마다 PE 인스턴스화
//            PE_with_clock_gating pe_inst (
//                .clk     (clk),
//                .enable  (enable),
//                .reset   (reset),
//                .data_in (data_in[lane]),   // [0:COLOR_WIDTH-1]
//                .weight  (weight),     // scalar
//                .result  (result[lane])      // [0:COLOR_WIDTH-1]
//            );
//        end
//    endgenerate

//endmodule
