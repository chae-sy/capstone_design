`timescale 1ns / 1ps

module tb_adder_tree_nlane_flat;

  // DUT 파라미터와 동일하게 선언
  parameter DATA_WIDTH = 20;
  parameter NUM_INPUTS = 16;
  parameter NUM_LANE   = 3;
  parameter SUM_WIDTH  = DATA_WIDTH + $clog2(NUM_INPUTS);

  // 클럭, 리셋, 인에이블, 입력 버스, 출력 버스, 완료 펄스
  reg                                clk;
  reg                                rst_n;
  reg                                adder_tree_en;
  reg  [NUM_LANE*NUM_INPUTS*DATA_WIDTH-1:0] in_flat;
  wire [NUM_LANE*SUM_WIDTH-1:0]             sum_out_flat;
  wire                               adder_tree_done;

  // DUT 인스턴스화
  adder_tree_nlane_flat #(
    .DATA_WIDTH (DATA_WIDTH),
    .NUM_INPUTS (NUM_INPUTS),
    .NUM_LANE   (NUM_LANE)
  ) DUT (
    .clk           (clk),
    .rst_n         (rst_n),
    .adder_tree_en (adder_tree_en),
    .in_flat       (in_flat),
    .sum_out_flat  (sum_out_flat),
    .adder_tree_done (adder_tree_done)
  );

  // =============================================================================
  // 1) 클럭 생성 (10 ns 주기)
  // =============================================================================
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  integer l, i, bit_idx;

  // =============================================================================
  // 2) Stimulus: 리셋, 입력값 설정, 인에이블 펄스, 완료 모니터링
  // =============================================================================
  initial begin
    // 초기값
    rst_n         = 0;
    adder_tree_en = 0;
    in_flat       = 0;

    // 리셋 해제 시점: 20ns 후
    #20;
    rst_n = 1;

    //-------------------------------------------------------------------------
    // 2-1) 모든 입력값을 '1'로 채워준다.
    //       - 각 lane(NUM_LANE)별로 NUM_INPUTS개씩, DATA_WIDTH 폭으로 1을 할당
    //       - in_flat[(lane*NUM_INPUTS + idx)*DATA_WIDTH +: DATA_WIDTH] 형태로 슬라이스
    //-------------------------------------------------------------------------
    for (l = 0; l < NUM_LANE; l = l + 1) begin
      for (i = 0; i < NUM_INPUTS; i = i + 1) begin
        bit_idx = (l*NUM_INPUTS + i)*DATA_WIDTH;
        in_flat[bit_idx +: DATA_WIDTH] = 20'd1;
      end
    end

    //-------------------------------------------------------------------------
    // 2-2) 10ns 후에 adder_tree_en을 1로 펄스 주고, 다시 10ns 뒤 0으로
    //-------------------------------------------------------------------------
    #10;
    adder_tree_en = 1;
    #10;
    adder_tree_en = 0;

    //-------------------------------------------------------------------------
    // 2-3) adder_tree_done 신호가 올라올 때까지 대기
    //       - done이 1이 된 이후 10ns 뒤에 결과 출력하고 시뮬레이션 종료
    //-------------------------------------------------------------------------
    wait (adder_tree_done == 1);
    #10;
    $display("===== Simulation 결과 =====");
    $display("Time = %0t ns", $time);
    $display("sum_out_flat = %h  (hex, 각 lane별 SUM 값)");
    $display("----------------------------");
    // sum_out_flat은 [lane*SUM_WIDTH +: SUM_WIDTH] 형식으로 각 lane별로 출력
    for (l = 0; l < NUM_LANE; l = l + 1) begin
      $display("  Lane %0d => %0d (decimal), 0x%h (hex)", 
               l,
               sum_out_flat[l*SUM_WIDTH +: SUM_WIDTH], 
               sum_out_flat[l*SUM_WIDTH +: SUM_WIDTH]);
    end
    $display("============================");
    #10;
    $finish;
  end

endmodule
