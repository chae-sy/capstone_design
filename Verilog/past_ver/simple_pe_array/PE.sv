`timescale 1ns / 1ps
module clk_gate (
input clk_in, en,
output clk_out
);
assign clk_out = clk_in & en;
endmodule


// (1) ?›ë³? PE ëª¨ë“ˆ?? ê·¸ë?ë¡? ?¬?‚¬?š©
module PE_with_clock_gating #(
    parameter DATA_WIDTH  = 8
)(
    input  wire                       clk,
    input  wire                       enable,
    input  wire                       rstb,
    input  wire [DATA_WIDTH-1:0]      data_in,
    input  wire [DATA_WIDTH-1:0]      weight,
    output reg  [2*DATA_WIDTH-1:0]    result
);

wire gated_clk;
clk_gate u_clk_gate(.clk_in(clk), .en(enable), .clk_out(gated_clk));

    integer i;
    always @(posedge gated_clk or negedge rstb) begin
        if (!rstb) begin
                result <= {2*DATA_WIDTH{1'b0}};
        end else begin
                result <= result + data_in * weight;
        end
    end
endmodule


