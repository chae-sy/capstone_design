`timescale 1ns / 1ps

module tb_maxPooling();

    // DUT 입력 포트
    reg clk;
    reg signed [7:0] input1, input2, input3, input4;
    reg enable;

    // DUT 출력 포트
    wire signed [7:0] output1;
    wire maxPoolingDone;

    // DUT 인스턴스
    maxPooling uut (
        .clk(clk),
        .input1(input1),
        .input2(input2),
        .input3(input3),
        .input4(input4),
        .enable(enable),
        .output1(output1),
        .maxPoolingDone(maxPoolingDone)
    );

    // 클럭 생성 (10ns 주기)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // 테스트 시나리오
    initial begin
        // 초기화
        input1 = 0; input2 = 0; input3 = 0; input4 = 0; enable = 0;
        #10;

        // Case 1: input1이 최대
        input1 = 8'd50;
        input2 = 8'd30;
        input3 = 8'd20;
        input4 = 8'd10;
        enable = 1;
        #10;

        // Case 2: input4가 최대
        input1 = 8'd5;
        input2 = 8'd15;
        input3 = 8'd25;
        input4 = 8'd60;
        enable = 1;
        #10;

        // Case 3: input2가 최대 (음수 섞임)
        input1 = -8'sd10;
        input2 = 8'd55;
        input3 = -8'sd5;
        input4 = 8'd0;
        enable = 1;
        #10;

        // Case 4: 모두 음수일 때
        input1 = -8'sd10;
        input2 = -8'sd20;
        input3 = -8'sd5;
        input4 = -8'sd15;
        enable = 1;
        #10;

        // Case 5: enable=0일 때 (출력 0)
        enable = 0;
        #10;

        $finish;
    end

    // 출력 모니터링
    initial begin
        $monitor("Time=%0t | enable=%b | input1=%d input2=%d input3=%d input4=%d | output1=%d | done=%b",
                  $time, enable, input1, input2, input3, input4, output1, maxPoolingDone);
    end

endmodule
