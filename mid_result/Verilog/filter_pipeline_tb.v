`timescale 1ns/1ps

module filter_pipeline_tb;

    // Inputs
    reg clk;
    reg reset;
    reg [7:0] pixel_in;

    // Output
    wire [7:0] pixel_out;

    // Instantiate the Unit Under Test (UUT)
    filter_pipeline uut (
        .clk(clk),
        .reset(reset),
        .pixel_in(pixel_in),
        .pixel_out(pixel_out)
    );

    // Clock generation: 10ns period
    always #5 clk = ~clk;

    // Mock normalize function
    function [7:0] normalize(input [7:0] value);
        begin
            normalize = value >> 1; // simple divide by 2 for test
        end
    endfunction

    // Initial block
    initial begin
        // Initialize inputs
        clk = 0;
        reset = 1;
        pixel_in = 0;

        // Reset pulse
        #12 reset = 0;

        // Send test pixels
        #10 pixel_in = 8'd10;
        #10 pixel_in = 8'd20;
        #10 pixel_in = 8'd30;
        #10 pixel_in = 8'd40;
        #10 pixel_in = 8'd50;

        // Wait for pipeline to flush
        #100;

        $display("Finished.");
        $finish;
    end

endmodule
