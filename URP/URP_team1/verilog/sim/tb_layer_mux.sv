`timescale 1ns / 1ps

module tb_layer_mux;

    // Parameters
    parameter WORD_LENGTH = 128;

    // Inputs
    reg [2:0] layer;
    reg [WORD_LENGTH-1:0] im_q;
    reg [WORD_LENGTH-1:0] ma_q;
    reg [WORD_LENGTH-1:0] mb_q;

    // Output
    wire [WORD_LENGTH-1:0] data_o;

    // Instantiate the Unit Under Test (UUT)
    layer_mux #(
        .WORD_LENGTH(WORD_LENGTH)
    ) uut (
        .layer(layer),
        .im_q(im_q),
        .ma_q(ma_q),
        .mb_q(mb_q),
        .data_o(data_o)
    );

    // Test procedure
    initial begin
        // Initialize inputs
        im_q = 128'hA1A1A1A1A1A1A1A1A1A1A1A1A1A1A1A1; // Test pattern for im_q
        ma_q = 128'hB2B2B2B2B2B2B2B2B2B2B2B2B2B2B2B2; // Test pattern for ma_q
        mb_q = 128'hC3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3; // Test pattern for mb_q

        // Test Case 1: layer = 1, expect data_o = im_q
        layer = 3'd1;
        #10;
        $display("Test Case 1: layer = %d, data_o = %h (expected: %h)", layer, data_o, im_q);

        // Test Case 2: layer = 2, expect data_o = ma_q
        layer = 3'd2;
        #10;
        $display("Test Case 2: layer = %d, data_o = %h (expected: %h)", layer, data_o, ma_q);

        // Test Case 3: layer = 3, expect data_o = mb_q
        layer = 3'd3;
        #10;
        $display("Test Case 3: layer = %d, data_o = %h (expected: %h)", layer, data_o, mb_q);

        // Test Case 4: layer = 4, expect data_o = ma_q
        layer = 3'd4;
        #10;
        $display("Test Case 4: layer = %d, data_o = %h (expected: %h)", layer, data_o, ma_q);

        // Test Case 5: layer = 5, expect data_o = mb_q
        layer = 3'd5;
        #10;
        $display("Test Case 5: layer = %d, data_o = %h (expected: %h)", layer, data_o, mb_q);

        // Test Case 6: layer = 0 (default case), expect data_o = 0
        layer = 3'd0;
        #10;
        $display("Test Case 6: layer = %d, data_o = %h (expected: 0)", layer, data_o);

        // Finish simulation
        $finish;
    end

endmodule
