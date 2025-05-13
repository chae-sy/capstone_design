`timescale 1ns / 1ps

module tb_systolic_array;
    // Parameters
    localparam ARRAY_WIDTH   = 3;
    localparam ARRAY_HEIGHT  = 3;
    localparam DATA_WIDTH    = 8;
    localparam COLOR_WIDTH   = 3;

    // Clock, reset, enable
    reg clk;
    reg rstb;
    reg enable;

    // DUT inputs (shift per cycle)
    reg [DATA_WIDTH-1:0] data_in    [0:ARRAY_HEIGHT-1][0:COLOR_WIDTH-1];
    reg [DATA_WIDTH-1:0] weight_in  [0:ARRAY_WIDTH-1];

    // DUT outputs
    wire [2*DATA_WIDTH-1:0] result_out [0:COLOR_WIDTH-1];

    integer rr, ll, cycle;

    // Instantiate DUT
    systolic_array #(
        .ARRAY_WIDTH (ARRAY_WIDTH),
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .DATA_WIDTH  (DATA_WIDTH),
        .COLOR_WIDTH (COLOR_WIDTH)
    ) dut (
        .clk        (clk),
        .rstb        (rstb),
        .enable     (enable),
        .data_in    (data_in),
        .weight_in  (weight_in),
        .result_out (result_out)
    );

    // Clock: 10ns period
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Waveform dump
    initial begin
        $dumpfile("tb_systolic_array.vcd");
        $dumpvars(0, tb_systolic_array);
    end

    initial begin
        // Initialize inputs
        rstb = 1; enable = 0;
        for (rr = 0; rr < ARRAY_HEIGHT; rr = rr + 1)
            for (ll = 0; ll < COLOR_WIDTH; ll = ll + 1)
                data_in[rr][ll] = 0;
        for (rr = 0; rr < ARRAY_WIDTH; rr = rr + 1)
            weight_in[rr] = 0;

        // Reset pulse
        #15; rstb = 0; #10; rstb = 1;

        // Set weights
        weight_in[0] = 8'd1;
        weight_in[1] = 8'd2;
        weight_in[2] = 8'd3;

        // Enable shifting
        enable = 1;
        // Feed new data each cycle
        for (cycle = 0; cycle < 6; cycle = cycle + 1) begin
            for (rr = 0; rr < ARRAY_HEIGHT; rr = rr + 1)
                for (ll = 0; ll < COLOR_WIDTH; ll = ll + 1)
                    data_in[rr][ll] = cycle*10 + rr*3 + ll;
            @(posedge clk);
        end

        // Disable input, let pipeline flush
        enable = 0;
        repeat (ARRAY_WIDTH) @(posedge clk);

        // Display outputs
        $display("=== Final Results ===");
        for (ll = 0; ll < COLOR_WIDTH; ll = ll + 1)
            $display("Lane %0d: %0d", ll, result_out[ll]);

        #20 $finish;
    end
endmodule
