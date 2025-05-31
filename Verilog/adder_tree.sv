`timescale 1ns / 1ps
module adder_tree_nlane_flat #(
  parameter DATA_WIDTH = 20,
  parameter NUM_INPUTS = 16,  // per lane
  parameter NUM_LANE   = 3,   // number of parallel lanes = NUM_COLOR
  parameter SUM_WIDTH  = DATA_WIDTH + $clog2(NUM_INPUTS)
)(
  input  wire                          clk,
  input  wire                          rst_n,
  input  wire                          adder_tree_en,     // 연산 시작 인에이블
  input  wire [NUM_LANE*NUM_INPUTS*DATA_WIDTH-1:0] in_flat, // [NUM_LANE*NUM_INPUTS*DATA_WIDTH - 1 : 0]
  output reg  [NUM_LANE*SUM_WIDTH-1:0] sum_out_flat,       // [NUM_LANE*SUM_WIDTH - 1 : 0]
  output reg                           adder_tree_done    // 연산 완료 펄스
);

  // Stage1 parameters
  localparam GROUP_SIZE = 4;
  localparam NUM_GROUPS = NUM_INPUTS / GROUP_SIZE;
  localparam ST1_WIDTH  = DATA_WIDTH + $clog2(GROUP_SIZE);

  // Stage1: combinational partial sums [lane][group]
  wire [ST1_WIDTH-1:0] stage1_comb [0:NUM_LANE-1][0:NUM_GROUPS-1];
  // Stage1 registers
  reg  [ST1_WIDTH-1:0] stage1_reg  [0:NUM_LANE-1][0:NUM_GROUPS-1];

  genvar lane, grp;
  generate
    for (lane = 0; lane < NUM_LANE; lane = lane + 1) begin : GEN_LANE
      for (grp = 0; grp < NUM_GROUPS; grp = grp + 1) begin : GEN_STAGE1
        // 각 그룹별로 DATA_WIDTH 길이의 입력 4개를 더한 값(콤비네이션)
        assign stage1_comb[lane][grp] =
             in_flat[(lane*NUM_INPUTS + grp*GROUP_SIZE + 0)*DATA_WIDTH +: DATA_WIDTH]
           + in_flat[(lane*NUM_INPUTS + grp*GROUP_SIZE + 1)*DATA_WIDTH +: DATA_WIDTH]
           + in_flat[(lane*NUM_INPUTS + grp*GROUP_SIZE + 2)*DATA_WIDTH +: DATA_WIDTH]
           + in_flat[(lane*NUM_INPUTS + grp*GROUP_SIZE + 3)*DATA_WIDTH +: DATA_WIDTH];
      end
    end
  endgenerate

  // Pipeline valid flag: en_stage1 = adder_tree_en delayed by 1clk
  //                     en_stage2 = en_stage1 delayed by 1clk → 최종 출력을 위한 done 펄스 생성 시점
  reg en_stage1, en_stage2;

  // valid flag 레지스터: 파이프라인 단계별로 en_delay 생성
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      en_stage1 <= 1'b0;
      en_stage2 <= 1'b0;
    end else begin
      en_stage1 <= adder_tree_en;
      en_stage2 <= en_stage1;
    end
  end

  integer i, j;
  // Stage1 register: adder_tree_en이 1일 때만 갱신
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (i = 0; i < NUM_LANE; i = i + 1) begin
        for (j = 0; j < NUM_GROUPS; j = j + 1) begin
          stage1_reg[i][j] <= {ST1_WIDTH{1'b0}};
        end
      end
    end else begin
      if (adder_tree_en) begin
        for (i = 0; i < NUM_LANE; i = i + 1) begin
          for (j = 0; j < NUM_GROUPS; j = j + 1) begin
            stage1_reg[i][j] <= stage1_comb[i][j];
          end
        end
      end
      // adder_tree_en == 0 이면 이전 값을 그대로 홀드
    end
  end

  // Stage2: sum each lane's groups
  wire [SUM_WIDTH-1:0] stage2_comb [0:NUM_LANE-1];
  generate
    for (lane = 0; lane < NUM_LANE; lane = lane + 1) begin : GEN_STAGE2
      // Stage1 결과(stage1_reg)를 모두 더해서 SUM_WIDTH 길이로 확장
      assign stage2_comb[lane] =
           {{(SUM_WIDTH-ST1_WIDTH){1'b0}}}
         + stage1_reg[lane][0]
         + stage1_reg[lane][1]
         + stage1_reg[lane][2]
         + stage1_reg[lane][3];
    end
  endgenerate

  // Stage2 register → flat output 및 done 생성
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sum_out_flat     <= {(NUM_LANE*SUM_WIDTH){1'b0}};
      adder_tree_done  <= 1'b0;
    end else begin
      // en_stage1 (adder_tree_en delayed by 1clk) == 1일 때만 출력 갱신
      if (en_stage1) begin
        for (i = 0; i < NUM_LANE; i = i + 1) begin
          sum_out_flat[i*SUM_WIDTH +: SUM_WIDTH] <= stage2_comb[i];
        end
      end
      // done 펄스는 여기서 발생: stage2 결과가 유효해지는 시점 = en_stage1 == 1인 사이클
      adder_tree_done <= en_stage1;
    end
  end

endmodule
