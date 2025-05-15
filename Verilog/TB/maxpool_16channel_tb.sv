`timescale 1ns/1ps

module maxpool_tb;

    parameter DATA_WIDTH = 8;
    parameter CHANNELS   = 16;

    // DUT I/O
    reg                         clk;
    reg                         rst_n;
    reg                         maxpool_en;
    reg  [1:0]                  color;
    reg  signed [DATA_WIDTH-1:0] in_data   [0:CHANNELS-1];
    wire                        maxpool_done;
    wire signed [DATA_WIDTH-1:0] out_data  [0:CHANNELS-1];

    // DUT 인스턴스
    maxpool #(
        .DATA_WIDTH(DATA_WIDTH),
        .CHANNELS(CHANNELS)
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

    // 테스트용 데이터
    reg signed [DATA_WIDTH-1:0] test_data [0:7][0:CHANNELS-1]; // 최대 8줄까지 지원

    // 기대 결과
    reg signed [DATA_WIDTH-1:0] expected_max [0:CHANNELS-1];

    task init_signals;
        begin
            clk         = 0;
            rst_n       = 0;
            maxpool_en  = 0;
            color       = 2'b00;
            for (int i = 0; i < CHANNELS; i++) begin
                in_data[i] = 0;
            end
        end
    endtask

    // 입력 전송 (한 줄씩)
    task send_line(input int row);
        begin
            @(posedge clk);
            maxpool_en = 1;
            for (int ch = 0; ch < CHANNELS; ch++) begin
                in_data[ch] = test_data[row][ch];
            end
        end
    endtask

    task disable_input;
        begin
            @(posedge clk);
            maxpool_en = 0;
            for (int ch = 0; ch < CHANNELS; ch++) begin
                in_data[ch] = 0;
            end
        end
    endtask

    // 검증 루틴
    task check_output(input string tag);
        begin
            wait (maxpool_done);
            @(posedge clk);
            for (int ch = 0; ch < CHANNELS; ch++) begin
                $display("[%s] Channel %0d: out_data = %0d (expected = %0d)", tag, ch, out_data[ch], expected_max[ch]);
                assert(out_data[ch] == expected_max[ch])
                    else $fatal("[%s] Channel %0d FAILED", tag, ch);
            end
        end
    endtask

    // 메인 테스트 시퀀스
    initial begin
        init_signals();

        // Reset
        #10; rst_n = 1;
        #10;

        // === TEST 1: RED (4x2) ===
        $display("\n[TEST 1] RED 4x2 (8 inputs per channel)");
        color = 2'b00;
        for (int row = 0; row < 8; row++) begin
            for (int ch = 0; ch < CHANNELS; ch++) begin
                test_data[row][ch] = $urandom_range(-50, 127);
                if (row == 0 || test_data[row][ch] > expected_max[ch])
                    expected_max[ch] = test_data[row][ch];
            end
            send_line(row);
        end
        disable_input();
        check_output("RED");

        // === TEST 2: GREEN (4x1) ===
        $display("\n[TEST 2] GREEN 4x1 (4 inputs per channel)");
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
        disable_input();
        check_output("GREEN");

        // === TEST 3: BLUE (4x2) ===
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
        disable_input();
        check_output("BLUE");
        
        // === TEST 4: RED 연속 2회 ===
        $display("\n[TEST 4] RED 연속 2회 (4x2)");

        for (int iter = 0; iter < 2; iter++) begin
            color = 2'b00;
            for (int ch = 0; ch < CHANNELS; ch++) expected_max[ch] = -128;

            for (int row = 0; row < 8; row++) begin
                for (int ch = 0; ch < CHANNELS; ch++) begin
                    test_data[row][ch] = $urandom_range(0, 50) + (iter * 10);  // 연속된 두 테스트지만 약간 다른 값
                    if (row == 0 || test_data[row][ch] > expected_max[ch])
                        expected_max[ch] = test_data[row][ch];
                end
                send_line(row);
            end
            disable_input();
            check_output($sformatf("RED repeat #%0d", iter+1));
        end

        // === TEST 5: GREEN 연속 2회 ===
        $display("\n[TEST 5] GREEN 연속 2회 (4x1)");

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
            disable_input();
            check_output($sformatf("GREEN repeat #%0d", iter+1));
        end

        // === TEST 6: BLUE 연속 2회 ===
        $display("\n[TEST 6] BLUE 연속 2회 (4x2)");

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
            disable_input();
            check_output($sformatf("BLUE repeat #%0d", iter+1));
        end


        $display("\n? ALL TESTS PASSED!\n");
        #20;
        $finish;
    end

endmodule
