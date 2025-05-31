`timescale 1ns/1ps

module maxpool_tb;

    parameter DATA_WIDTH = 8;
    parameter CHANNELS   = 16;

    // Testbench signals
    reg clk;
    reg rst_n;
    reg maxpool_en;
    reg [1:0] color;
    reg signed [DATA_WIDTH-1:0] in_data_arr [0:CHANNELS-1];   // Testbench internal array
    wire maxpool_done;
    wire signed [DATA_WIDTH-1:0] out_data_arr [0:CHANNELS-1]; // Testbench internal array

    // Packed vector signals for DUT
    wire signed [DATA_WIDTH*CHANNELS-1:0] in_data_packed;
    wire signed [DATA_WIDTH*CHANNELS-1:0] out_data_packed;

    // DUT instance
    maxpool_16chnl #(
        .DATA_WIDTH(DATA_WIDTH),
        .CHANNELS(CHANNELS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .maxpool_en(maxpool_en),
        .color(color),
        .in_data(in_data_packed),
        .maxpool_done_o(maxpool_done),
        .out_data_o(out_data_packed)
    );

    // Connect arrays to packed vectors
    generate
        genvar ch;
        for (ch = 0; ch < CHANNELS; ch = ch + 1) begin
            assign in_data_packed[((ch+1)*DATA_WIDTH)-1 : ch*DATA_WIDTH] = in_data_arr[ch];
            assign out_data_arr[ch] = out_data_packed[((ch+1)*DATA_WIDTH)-1 : ch*DATA_WIDTH];
        end
    endgenerate

    // Clock generation
    always #5 clk = ~clk;

    // Test data and expected results
    reg signed [DATA_WIDTH-1:0] test_data [0:7][0:CHANNELS-1];
    reg signed [DATA_WIDTH-1:0] expected_max [0:CHANNELS-1];

    // Initialization
    task init_signals;
        begin
            clk = 0; rst_n = 0; maxpool_en = 0; color = 2'b00;
            for (int i = 0; i < CHANNELS; i++) in_data_arr[i] = 0;
        end
    endtask

    // Send one line of data
    task send_line(input int row);
        begin
            @(posedge clk); maxpool_en = 1;
            for (int ch = 0; ch < CHANNELS; ch++) in_data_arr[ch] = test_data[row][ch];
        end
    endtask

    // Disable input
    task disable_input;
        begin
            @(posedge clk); maxpool_en = 0;
        end
    endtask

    // Check output
    task check_output(input string tag);
        begin
            wait (maxpool_done);
            @(posedge clk);
            for (int ch = 0; ch < CHANNELS; ch++) begin
                $display("[%s] Channel %0d: out_data = %0d (expected = %0d)", tag, ch, out_data_arr[ch], expected_max[ch]);
                assert(out_data_arr[ch] == expected_max[ch])
                    else $fatal("[%s] Channel %0d FAILED", tag, ch);
            end
        end
    endtask

    // Main test sequence
    initial begin
        init_signals();
        #10; rst_n = 1; #10;

        // Test 1: RED 4x2
        $display("\n[TEST 1] RED 4x2 (8 inputs per channel)");
        color = 2'b00;
        for (int ch = 0; ch < CHANNELS; ch++) expected_max[ch] = -128;
        for (int row = 0; row < 8; row++) begin
            for (int ch = 0; ch < CHANNELS; ch++) begin
                test_data[row][ch] = $urandom_range(-50, 127);
                if (row == 0 || test_data[row][ch] > expected_max[ch])
                    expected_max[ch] = test_data[row][ch];
            end
            send_line(row);
        end
        disable_input(); check_output("RED");

        // Test 2: GREEN 2x2
        $display("\n[TEST 2] GREEN 2x2 (4 inputs per channel)");
        color = 2'b01;
        for (int ch = 0; ch < CHANNELS; ch++) expected_max[ch] = -128;
        for (int row = 0; row < 4; row++) begin
            for (int ch = 0; ch < CHANNELS; ch++) begin
                test_data[row][ch] = $urandom_range(-20, 100);
                if (row == 0 || test_data[row][ch] > expected_max[ch])
                    expected_max[ch] = test_data[row][ch];
            end
            send_line(row);
        end
        disable_input(); check_output("GREEN");

        // Test 3: BLUE 4x2
        $display("\n[TEST 3] BLUE 4x2 (8 inputs per channel)");
        color = 2'b10;
        for (int ch = 0; ch < CHANNELS; ch++) expected_max[ch] = -128;
        for (int row = 0; row < 8; row++) begin
            for (int ch = 0; ch < CHANNELS; ch++) begin
                test_data[row][ch] = $urandom_range(-64, 64);
                if (row == 0 || test_data[row][ch] > expected_max[ch])
                    expected_max[ch] = test_data[row][ch];
            end
            send_line(row);
        end
        disable_input(); check_output("BLUE");

        // Test 4: RED 4x2 repeat
        $display("\n[TEST 4] RED 4x2 repeat");
        for (int iter = 0; iter < 2; iter++) begin
            color = 2'b00;
            for (int ch = 0; ch < CHANNELS; ch++) expected_max[ch] = -128;
            for (int row = 0; row < 8; row++) begin
                for (int ch = 0; ch < CHANNELS; ch++) begin
                    test_data[row][ch] = $urandom_range(0, 50) + (iter * 10);
                    if (row == 0 || test_data[row][ch] > expected_max[ch])
                        expected_max[ch] = test_data[row][ch];
                end
                send_line(row);
            end
            disable_input(); check_output($sformatf("RED repeat #%0d", iter+1));
        end

        // Test 5: GREEN 2x2 repeat
        $display("\n[TEST 5] GREEN 2x2 repeat");
        for (int iter = 0; iter < 2; iter++) begin
            color = 2'b01;
            for (int ch = 0; ch < CHANNELS; ch++) expected_max[ch] = -128;
            for (int row = 0; row < 4; row++) begin
                for (int ch = 0; ch < CHANNELS; ch++) begin
                    test_data[row][ch] = $urandom_range(-30, 30) + iter * 5;
                    if (row == 0 || test_data[row][ch] > expected_max[ch])
                        expected_max[ch] = test_data[row][ch];
                end
                send_line(row);
            end
            disable_input(); check_output($sformatf("GREEN repeat #%0d", iter+1));
        end

        // Test 6: BLUE 4x2 repeat
        $display("\n[TEST 6] BLUE 4x2 repeat");
        for (int iter = 0; iter < 2; iter++) begin
            color = 2'b10;
            for (int ch = 0; ch < CHANNELS; ch++) expected_max[ch] = -128;
            for (int row = 0; row < 8; row++) begin
                for (int ch = 0; ch < CHANNELS; ch++) begin
                    test_data[row][ch] = $urandom_range(-40, 40) - iter * 5;
                    if (row == 0 || test_data[row][ch] > expected_max[ch])
                        expected_max[ch] = test_data[row][ch];
                end
                send_line(row);
            end
            disable_input(); check_output($sformatf("BLUE repeat #%0d", iter+1));
        end

        $display("\n✅ ALL TESTS PASSED!\n");
        #20; $finish;
    end

endmodule
