`timescale 1ns / 1ps

module adder_tree_tb;

  // DUT 파라미터와 동일하게 선언
  parameter DATA_WIDTH = 20;
  parameter NUM_INPUTS = 16;
  parameter SUM_WIDTH  = DATA_WIDTH + $clog2(NUM_INPUTS);

  // 신호 선언
  reg                               clk;
  reg                               rst_n;
  reg                               adder_tree_en;
  reg  [NUM_INPUTS*DATA_WIDTH-1:0] in_flat;
  wire [SUM_WIDTH-1:0]             sum_out;
  wire                              adder_tree_done;

  // DUT 인스턴스
  adder_tree #(
    .DATA_WIDTH(DATA_WIDTH),
    .NUM_INPUTS(NUM_INPUTS),
    .SUM_WIDTH(SUM_WIDTH)
  ) DUT (
    .clk(clk),
    .rst_n(rst_n),
    .adder_tree_en(adder_tree_en),
    .in_flat(in_flat),
    .sum_out(sum_out),
    .adder_tree_done(adder_tree_done)
  );

  // 클럭 생성 (10 ns 주기)
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  integer i, bit_idx;

  // Stimulus
  initial begin
    // 초기화
    rst_n = 0;
    adder_tree_en = 0;
    in_flat = 0;

    // 리셋 해제
    #20;
    rst_n = 1;

    // ===========================
    // 케이스 1) 모든 입력을 '1'로
    // ===========================
    for (i = 0; i < NUM_INPUTS; i = i + 1) begin
      bit_idx = i * DATA_WIDTH;
      in_flat[bit_idx +: DATA_WIDTH] = 20'd1;
    end
    #15;
    adder_tree_en = 1;
    #10;
    adder_tree_en = 0;
    wait (adder_tree_done == 1);
    
    #10;
    $display("===== 케이스 1: 모든 입력 1 =====");
    $display("Time = %0t ns", $time);
    $display("sum_out = %0d (decimal), 0x%h (hex)", sum_out, sum_out);
    $display("============================");

    // ===========================
    // 케이스 2) 입력을 16,15,...,1로
    // ===========================
    for (i = 0; i < NUM_INPUTS; i = i + 1) begin
      bit_idx = i * DATA_WIDTH;
      in_flat[bit_idx +: DATA_WIDTH] = NUM_INPUTS - i;  // 16,15,...,1
    end
    #10;
    adder_tree_en = 1;
    #10;
    adder_tree_en = 0;
    wait (adder_tree_done == 1);

    #10;
    $display("===== 케이스 2: 입력 16,15,...,1 =====");
    $display("Time = %0t ns", $time);
    $display("sum_out = %0d (decimal), 0x%h (hex)", sum_out, sum_out);
    $display("============================");

    #10;
    $finish;
  end

endmodule
