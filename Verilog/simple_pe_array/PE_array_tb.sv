`timescale 1ns / 1ps

module tb_systolic_array;
    // parameters match DUT
    localparam ARRAY_WIDTH   = 3;
    localparam ARRAY_HEIGHT  = 3;
    localparam DATA_WIDTH    = 8;
    localparam COLOR_WIDTH   = 3;

    // clock and reset
    reg clk;
    reg rst;
    reg enable;

    // DUT inputs
    reg [DATA_WIDTH-1:0] data_in    [0:ARRAY_HEIGHT-1][0:COLOR_WIDTH-1];
    reg [DATA_WIDTH-1:0] weight_in  [0:ARRAY_WIDTH-1];

    // DUT outputs
    wire [2*DATA_WIDTH-1:0] result_out [0:COLOR_WIDTH-1];

    integer i, j, k;

    // DUT instantiation
    systolic_array #(
        .ARRAY_WIDTH(ARRAY_WIDTH),
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .DATA_WIDTH(DATA_WIDTH),
        .COLOR_WIDTH(COLOR_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .enable(enable),
        .data_in(data_in),
        .weight_in(weight_in),
        .result_out(result_out)
    );

    // clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // stimulus
    initial begin
        // initialize signals
        rst = 1;
        enable = 0;
        for (j = 0; j < ARRAY_HEIGHT; j = j + 1)
            for (k = 0; k < COLOR_WIDTH; k = k + 1)
                data_in[j][k] = 0;
        for (i = 0; i < ARRAY_WIDTH; i = i + 1)
            weight_in[i] = 0;

        // hold reset
        #20;
        rst = 0;
        #10;
        rst = 1;
        #10;

        // apply weights
        // example: weights = {1,2,3}
        weight_in[0] = 1;
        weight_in[1] = 2;
        weight_in[2] = 3;

        // apply data waves on rows
        enable = 1;
        // for each row, send a cycle of color values
        for (j = 0; j < ARRAY_HEIGHT; j = j + 1) begin
            for (k = 0; k < COLOR_WIDTH; k = k + 1) begin
                data_in[j][k] = j*10 + k + 1;  // simple numeric pattern
            end
            #10; // wait one clock cycle
        end

        // wait for pipeline to flush
        #((ARRAY_WIDTH + ARRAY_HEIGHT) * 10);

        // display results
        $display("Results:\n");
        for (k = 0; k < COLOR_WIDTH; k = k + 1) begin
            $display("Lane %0d: result_out = %0d", k, result_out[k]);
        end

        #20;
        $finish;
    end

endmodule