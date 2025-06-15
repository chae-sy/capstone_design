`timescale 1ns/1ps

module maxpool_tb;

    // Parameters
    parameter DATA_WIDTH = 8;
    parameter DATA_num   = 8;

    // DUT I/O
    reg                         clk;
    reg                         rst_n;
    reg                         maxpool_en;
    reg       [1:0]             color;
    reg signed [DATA_WIDTH-1:0] in_data;
    wire                        maxpool_done;
    wire signed [DATA_WIDTH-1:0] out_data;

    // 테스트용 배열 선언 (모듈 상단에서!)
    reg signed [DATA_WIDTH-1:0] green_data [0:3];
    reg signed [DATA_WIDTH-1:0] red_data   [0:7];
    reg signed [DATA_WIDTH-1:0] blue_data  [0:7];

    // Instantiate the DUT
    maxpool #(
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .maxpool_en(maxpool_en),
        .color(color),
        .in_data(in_data),
        .maxpool_done_o(maxpool_done),
        .out_data_o(out_data)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Task to send inputs one by one on clock edge
    task send_inputs;
        input [1:0] mode;
        input signed [DATA_WIDTH-1:0] values[];
        integer i;
        begin
            color = mode;
            for (i = 0; i < values.size(); i++) begin
                @(posedge clk);
                maxpool_en = 1;
                in_data    = values[i];
            end
            @(posedge clk);
            maxpool_en = 0;
            in_data    = 0;
        end
    endtask

    // Test Sequence
    initial begin
        // 초기화
        clk         = 0;
        rst_n       = 0;
        maxpool_en  = 0;
        in_data     = 0;
        color       = 0;

        // 리셋
        #20;
        rst_n = 1;
        $display("\n=== Maxpool Testbench Start ===\n");

        // === Test 1: Green (4x1) ===
        $display("[TEST 1] GREEN (4x1)");
        green_data[0] = -10;
        green_data[1] =  45;  // max
        green_data[2] =  12;
        green_data[3] =  20;

        send_inputs(2'b01, green_data);
        wait (maxpool_done);
        @(posedge clk);
        $display("Result: out_data = %0d", out_data);
        assert(out_data == 45) else $fatal("GREEN maxpool failed!");

        // === Test 2: Red (4x2) ===
        $display("\n[TEST 2] RED (4x2)");
        red_data[0] =  3;
        red_data[1] = -2;
        red_data[2] = 14;
        red_data[3] =  7;
        red_data[4] = 29;  // max
        red_data[5] = -8;
        red_data[6] = 10;
        red_data[7] = 18;

        send_inputs(2'b00, red_data);
        wait (maxpool_done);
        @(posedge clk);
        $display("Result: out_data = %0d", out_data);
        assert(out_data == 29) else $fatal("RED maxpool failed!");

        // === Test 3: Blue (4x2) ===
        $display("\n[TEST 3] BLUE (4x2)");
        blue_data[0] = -5;
        blue_data[1] = -1;
        blue_data[2] = -20;
        blue_data[3] = -2;
        blue_data[4] = -7;
        blue_data[5] = -15;
        blue_data[6] = -3;
        blue_data[7] = -11;  // max = -1

        send_inputs(2'b10, blue_data);
        wait (maxpool_done);
        @(posedge clk);
        $display("Result: out_data = %0d", out_data);
        assert(out_data == -1) else $fatal("BLUE maxpool failed!");
        
        // === Test 4: Blue (4x2) ===
        $display("\n[TEST 4] BLUE (4x2)");
        blue_data[0] = 0;
        blue_data[1] = 0;
        blue_data[2] = 8;
        blue_data[3] = 8;
        blue_data[4] = 17;
        blue_data[5] = 2;
        blue_data[6] = -10;
        blue_data[7] = -1;  // max = -1
        
        send_inputs(2'b10, blue_data);
        wait (maxpool_done);
        @(posedge clk);
        $display("Result: out_data = %0d", out_data);
        assert(out_data == 17) else $fatal("BLUE maxpool failed!");
        
        // === Test 5: Blue (4x2) ===
        $display("\n[TEST 5] BLUE (4x2)");
        blue_data[0] = 40;
        blue_data[1] = 25;
        blue_data[2] = -8;
        blue_data[3] = 41;
        blue_data[4] = 4;
        blue_data[5] = 2;
        blue_data[6] = -15;
        blue_data[7] = 0;
        
        send_inputs(2'b10, blue_data);
        wait (maxpool_done);
        @(posedge clk);
        $display("Result: out_data = %0d", out_data);
        assert(out_data == 41) else $fatal("BLUE maxpool failed!");
        
        // === Test 6: Green (4x1) ===
        $display("\n[TEST 6] GREEN (4x1)");
        green_data[0] = 45;
        green_data[1] = 12; 
        green_data[2] = -4;
        green_data[3] = 40;

        send_inputs(2'b01, green_data);
        wait (maxpool_done);
        @(posedge clk);
        $display("Result: out_data = %0d", out_data);
        assert(out_data == 45) else $fatal("GREEN maxpool failed!");

        $display("\n All tests passed successfully!");
        #20;
        $finish;
    end

endmodule
