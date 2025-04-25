`timescale 1ns / 1ps

module systolic_array_4x4_tb;

    // Clock & control
    logic clk, rst, enable;

    // Array inputs & outputs
    logic [7:0] data_in   [0:3];
    logic [7:0] weight_in [0:3];
    logic [15:0] result_out [0:3][0:3];

    // Instantiate DUT
    systolic_array_4x4 dut (
        .clk(clk),
        .rst(rst),
        .enable(enable),
        .data_in(data_in),
        .weight_in(weight_in),
        .result_out(result_out)
    );

    // Clock generation
    always #5 clk = ~clk;

    initial begin
        // Init
        clk = 0;
        rst = 1;
        enable = 0;

        // Clear inputs
        foreach (data_in[i])   data_in[i] = 0;
        foreach (weight_in[i]) weight_in[i] = 0;

        // Reset pulse
        #12;
        rst = 0;

        // Provide test inputs
        data_in[0] = 8'd1;
        data_in[1] = 8'd2;
        data_in[2] = 8'd3;
        data_in[3] = 8'd4;

        weight_in[0] = 8'd5;
        weight_in[1] = 8'd6;
        weight_in[2] = 8'd7;
        weight_in[3] = 8'd8;

        enable = 1;

        // Run for a few cycles
        #100;

        // Display results
        $display("===== Systolic Array Output =====");
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                $display("result_out[%0d][%0d] = %0d", i, j, result_out[i][j]);
            end
        end

        $finish;
    end

endmodule
