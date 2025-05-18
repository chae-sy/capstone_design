`timescale 1ns/1ps

module tb_bias_relu;
  // DUT 파라미터(기본값 그대로 사용)
  parameter NUM_ACCUMULATE   = 9;
  parameter WIDTH_BITWIDTH   = 8;
  // WIDTH_IN_DATA = 16 + $clog2(9) = 20
  parameter WIDTH_IN_DATA    = WIDTH_BITWIDTH * 2 + $clog2(NUM_ACCUMULATE);
  parameter WIDTH_BIAS       = 32;
  parameter WIDTH_OUT_DATA   = 8;

  // 입력/출력 신호 선언
  reg                   relu_on;
  reg       [2:0]       layer_state;
  reg signed [WIDTH_IN_DATA-1:0] data_in;
  reg signed [WIDTH_BIAS-1:0]    bias;
  wire      [WIDTH_OUT_DATA-1:0] data_out;

  // DUT 인스턴스
  bias_relu #(
    .NUM_ACCUMULATE(NUM_ACCUMULATE),
    .WIDTH_BITWIDTH(WIDTH_BITWIDTH),
    .WIDTH_IN_DATA (WIDTH_IN_DATA),
    .WIDTH_BIAS    (WIDTH_BIAS),
    .WIDTH_OUT_DATA(WIDTH_OUT_DATA)
  ) dut (
    .relu_on     (relu_on),
    .layer_state (layer_state),
    .data_in     (data_in),
    .bias        (bias),
    .data_out    (data_out)
  );

  // 파형 덤프 (GTKWave 등으로 확인할 때)
  

  // 시간, 레이어, ReLU 모드, 입력, 출력 모니터링
  initial begin
    $display("time ns | layer | ReLU |    data_in    | data_out");
    $monitor("%8t |  %b   |   %b  | %11d |   %3d",
             $time, layer_state, relu_on, data_in, data_out);
  end

  // 테스트 시퀀스
  initial begin
    bias = 10;  // bias는 현재 사용되지 않으므로 0으로 고정

    // 레이어별로, ReLU on/off 로 3가지 입력 조합을 시험
    foreach_test: for (integer ls = 1; ls <= 5; ls = ls + 1) begin
      layer_state = ls;
      // ReLU 모드
      for (integer r = 0; r <= 1; r = r + 1) begin
        relu_on = r;
        // 대표 입력값 3가지
        data_in = 0;         #10;  // zero
        data_in = 20'sd15;   #10;  // 작은 양수
        data_in = -20'sd15;  #10;  // 작은 음수
        // (원한다면 더 큰/작은 값도 추가 테스트)
      end
    end

    #10 $finish;
  end

endmodule
