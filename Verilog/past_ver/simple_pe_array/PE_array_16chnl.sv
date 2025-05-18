`timescale 1ns / 1ps

module systolic_array_16ch #(
    parameter NUM_CHANNELS = 16,
    parameter ARRAY_WIDTH   = 3,
    parameter ARRAY_HEIGHT  = 3,
    parameter DATA_WIDTH    = 8,
    parameter COLOR_WIDTH   = 3
)(
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             pe_en,
    // 채널 × 높이 × 컬러 × 데이터폭
    input  wire [DATA_WIDTH-1:0]            data_in    [0:NUM_CHANNELS-1]
                                                      [0:ARRAY_HEIGHT-1]
                                                      [0:COLOR_WIDTH-1],
    // 채널 × 너비 × 데이터폭
    input  wire [DATA_WIDTH-1:0]            weight_in  [0:NUM_CHANNELS-1]
                                                      [0:ARRAY_WIDTH-1],
    // 채널 × 컬러 × (2*데이터폭)
    output wire [2*DATA_WIDTH-1:0]          result_out [0:NUM_CHANNELS-1]
                                                      [0:COLOR_WIDTH-1],
    // 채널별 완료 플래그
    output wire                             pe_done    [0:NUM_CHANNELS-1]
);

    genvar ch;
    generate
        for (ch = 0; ch < NUM_CHANNELS; ch = ch + 1) begin : CHANNELS
            systolic_array #(
                .ARRAY_WIDTH (ARRAY_WIDTH),
                .ARRAY_HEIGHT(ARRAY_HEIGHT),
                .DATA_WIDTH  (DATA_WIDTH),
                .COLOR_WIDTH (COLOR_WIDTH)
            ) u_systolic_array (
                .clk        (clk),
                .rst_n      (rst_n),
                .pe_en      (pe_en),
                .data_in    (data_in[ch]),
                .weight_in  (weight_in[ch]),
                .result_out (result_out[ch]),
                .pe_done    (pe_done[ch])
            );
        end
    endgenerate

endmodule
