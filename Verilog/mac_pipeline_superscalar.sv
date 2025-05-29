`timescale 1ns / 1ps

module mac_pipeline_superscalar #(
  parameter int DATA_WIDTH = 8,
  parameter int NUM_STAGE  = 9,
  parameter int LANE_NUM   = 3
) (
  input  logic                       clk,
  input  logic                       rst_n,
  input  logic                       pe_en,
  input  logic [DATA_WIDTH-1:0]      data_in   [LANE_NUM],
  input  logic [DATA_WIDTH-1:0]      weight_in,
  output logic                       pe_done,
  output logic [LANE_NUM*2*DATA_WIDTH-1:0] result_out_flat
);

  // 1) 파이프라인 레지스터
  logic [2*DATA_WIDTH-1:0] pipe     [NUM_STAGE][LANE_NUM];
  logic                   val_pipe [NUM_STAGE];

  // 2) 펄스 감지를 위한 이전 valid 상태 저장
  logic prev_valid_stage;

  // --- 메인 파이프라인: shift & accumulate ---
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int s = 0; s < NUM_STAGE; s++) begin
        val_pipe[s] <= 1'b0;
        for (int l = 0; l < LANE_NUM; l++)
          pipe[s][l] <= '0;
      end
    end else begin
      if (pe_en) begin
        // stage-0
        val_pipe[0] <= 1'b1;
        for (int l = 0; l < LANE_NUM; l++)
          pipe[0][l] <= data_in[l] * weight_in;

        // stage 1..NUM_STAGE-1
        for (int s = 1; s < NUM_STAGE; s++) begin
          val_pipe[s] <= val_pipe[s-1];
          for (int l = 0; l < LANE_NUM; l++) begin
            if (val_pipe[s-1])
              pipe[s][l] <= pipe[s-1][l] + data_in[l] * weight_in;
            else
              pipe[s][l] <= '0;
          end
        end
      end else begin
        // idle 시 파이프라인 flush
        val_pipe[0] <= 1'b0;
        for (int l = 0; l < LANE_NUM; l++)
          pipe[0][l] <= '0;
        for (int s = 1; s < NUM_STAGE; s++)
          val_pipe[s] <= 1'b0;
      end
    end
  end

  // --- 출력 블록: pe_done은 1사이클 펄스, result도 그때만 찍어줌 ---
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prev_valid_stage   <= 1'b0;
      pe_done            <= 1'b0;
      result_out_flat    <= '0;
    end else begin
      // 1) 이전 사이클의 val_pipe[NUM_STAGE-2] 저장
      prev_valid_stage <= val_pipe[NUM_STAGE-2];

      // 2) 상승 에지(0→1)일 때만 pe_done 펄스
      if (val_pipe[NUM_STAGE-2] && !prev_valid_stage) begin
        pe_done <= 1'b1;
        // 바로 그 사이클에 최신 결과 계산해서 기록
        for (int l = 0; l < LANE_NUM; l++) begin
          result_out_flat[l*2*DATA_WIDTH +: 2*DATA_WIDTH] <=
            // stage NUM_STAGE-1 의 accumulate 식
            pipe[NUM_STAGE-2][l] + data_in[l] * weight_in;
        end
      end else begin
        pe_done         <= 1'b0;
        // result_out_flat은 마지막 펄스 때 찍힌 값 그대로 보존됩니다.
      end
    end
  end

endmodule
