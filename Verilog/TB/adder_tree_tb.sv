`timescale 1ns / 1ps

module tb_adder_tree16_2stage;

  // DUT 파라미터
  localparam DATA_WIDTH = 8;
  localparam NUM_INPUTS = 16;
  localparam SUM_WIDTH  = DATA_WIDTH + $clog2(NUM_INPUTS);

  // 클록 & 리셋
  reg clk;
  reg rst_n;

  // TB 내부 배열 및 flat 버스
  reg [DATA_WIDTH-1:0] in_array [0:NUM_INPUTS-1];
  reg [DATA_WIDTH*NUM_INPUTS-1:0] in_flat;

  // DUT 출력
  wire [SUM_WIDTH-1:0] sum_out;

  integer i, j;
  integer exp_sum;

  // DUT 인스턴스
  adder_tree16_2stage #(
    .DATA_WIDTH(DATA_WIDTH),
    .NUM_INPUTS(NUM_INPUTS)
  ) dut (
    .clk     (clk),
    .rst_n   (rst_n),
    .in_flat (in_flat),
    .sum_out (sum_out)
  );

  // 클록 생성 (10ns 주기)
  initial clk = 0;
  always #5 clk = ~clk;

  // flat 패킹 태스크
  task pack_flat;
    for (i = 0; i < NUM_INPUTS; i = i + 1)
      in_flat[i*DATA_WIDTH +: DATA_WIDTH] = in_array[i];
  endtask

  initial begin
    // 리셋
    rst_n = 0;
    #20;
    rst_n = 1;
    #10;

    // 벡터 1: 0..15
    for (i = 0; i < NUM_INPUTS; i = i + 1)
      in_array[i] = i;
    pack_flat();
    #10; #10;
    exp_sum = 0;
    for (j = 0; j < NUM_INPUTS; j = j + 1)
      exp_sum = exp_sum + in_array[j];
    $display("[%0t] EXPECTED=%0d, GOT=%0d", $time, exp_sum, sum_out);

    // 벡터 2: all 8
    for (i = 0; i < NUM_INPUTS; i = i + 1)
      in_array[i] = 8;
    pack_flat();
    #10; #10;
    exp_sum = 8 * NUM_INPUTS;
    $display("[%0t] EXPECTED=%0d, GOT=%0d", $time, exp_sum, sum_out);

    // 랜덤 테스트 5회
    repeat (5) begin
      exp_sum = 0;
      for (i = 0; i < NUM_INPUTS; i = i + 1) begin
        in_array[i] = $random;
        exp_sum = exp_sum + in_array[i];
      end
      pack_flat();
      #10; #10;
      $display("[%0t] RANDOM EXPECTED=%0d, GOT=%0d", $time, exp_sum, sum_out);
    end

    #20;
    $finish;
  end

endmodule
