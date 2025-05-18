`timescale 1ns / 1ps

// Testbench for mac_pipeline_superscalar
module tb_mac_pipeline_superscalar;
    // Parameters matching the DUT
    parameter DATA_WIDTH  = 8;
    parameter CHANNEL_NUM = 9;
    parameter LANE_NUM    = 3;
    localparam PERIOD     = 10;

    // Clock & reset
    reg clk;
    reg rst_n;

    // Inputs
    reg                     valid_in;
    reg  [DATA_WIDTH-1:0]   data_in   [0:LANE_NUM-1];
    reg  [DATA_WIDTH-1:0]   weight_in;

    // Outputs
    wire                    valid_out;
    wire [2*DATA_WIDTH-1:0] result_out[0:LANE_NUM-1];

    integer i;

    // Instantiate DUT
    mac_pipeline_superscalar #(
        .DATA_WIDTH  (DATA_WIDTH),
        .CHANNEL_NUM (CHANNEL_NUM),
        .LANE_NUM    (LANE_NUM)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .data_in   (data_in),
        .weight_in (weight_in),
        .valid_out (valid_out),
        .result_out(result_out)
    );

    // Clock generation
    initial begin
        clk = 1'b0;
        forever #(PERIOD/2) clk = ~clk;
    end

    // Test stimulus
    initial begin
        // Initialize signals
        rst_n     = 1'b0;
        valid_in  = 1'b0;
        weight_in = 'd1;
        for (i = 0; i < LANE_NUM; i = i + 1)
            data_in[i] = 'd0;

        // Release reset
        #(PERIOD*2);
        rst_n = 1'b1;

        // Apply CHANNEL_NUM cycles of valid data
        #(PERIOD);
        valid_in = 1'b1;
        
        for (i = 0; i < CHANNEL_NUM; i = i + 1) begin
        
            // For lane 0: data = i
            // lane 1: data = i + 1
            // lane 2: data = i + 2
            data_in[0] = i;
            data_in[1] = i + 1;
            data_in[2] = i + 2;
            #(PERIOD);
            valid_in = 1'b0;
        end

        // Deassert valid
        valid_in = 1'b0;

        // Wait for pipeline to flush
        #(PERIOD * (CHANNEL_NUM + 2));

        // Check result
        $display("Expected lane0 = %0d, lane1 = %0d, lane2 = %0d", 
                 (0 + CHANNEL_NUM - 1)*CHANNEL_NUM/2, 
                 ((0 + CHANNEL_NUM - 1)*CHANNEL_NUM/2) + CHANNEL_NUM, 
                 ((0 + CHANNEL_NUM - 1)*CHANNEL_NUM/2) + 2*CHANNEL_NUM);
        $display("Got      lane0 = %0d, lane1 = %0d, lane2 = %0d", 
                 result_out[0], result_out[1], result_out[2]);

        if (valid_out) begin
            $display("[PASS] valid_out asserted at time %0t", $time);
        end else begin
            $display("[FAIL] valid_out not asserted");
        end

        $finish;
    end

endmodule
