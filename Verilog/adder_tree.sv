`timescale 1ns / 1ps
module adder_tree_nlane_flat #(
  parameter DATA_WIDTH = 20,
  parameter NUM_INPUTS = 16,  // per lane
  parameter NUM_LANE   = 3,   // number of parallel lanes = NUM_COLOR
  parameter SUM_WIDTH  = DATA_WIDTH + $clog2(NUM_INPUTS)
)(
  input  wire                            clk,
  input  wire                            rst_n,
  // [NUM_LANE*NUM_INPUTS*DATA_WIDTH - 1 : 0]
  input  wire [NUM_LANE*NUM_INPUTS*DATA_WIDTH-1:0] in_flat,
  // now fully flat output: [NUM_LANE*SUM_WIDTH - 1 : 0]
  output reg  [NUM_LANE*SUM_WIDTH-1:0]   sum_out_flat
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
        assign stage1_comb[lane][grp] =
             in_flat[(lane*NUM_INPUTS + grp*GROUP_SIZE + 0)*DATA_WIDTH +: DATA_WIDTH]
           + in_flat[(lane*NUM_INPUTS + grp*GROUP_SIZE + 1)*DATA_WIDTH +: DATA_WIDTH]
           + in_flat[(lane*NUM_INPUTS + grp*GROUP_SIZE + 2)*DATA_WIDTH +: DATA_WIDTH]
           + in_flat[(lane*NUM_INPUTS + grp*GROUP_SIZE + 3)*DATA_WIDTH +: DATA_WIDTH];
      end
    end
  endgenerate

  integer i, j;
  // Stage1 register
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (i = 0; i < NUM_LANE; i = i + 1)
        for (j = 0; j < NUM_GROUPS; j = j + 1)
          stage1_reg[i][j] <= {ST1_WIDTH{1'b0}};
    end else begin
      for (i = 0; i < NUM_LANE; i = i + 1)
        for (j = 0; j < NUM_GROUPS; j = j + 1)
          stage1_reg[i][j] <= stage1_comb[i][j];
    end
  end

  // Stage2: sum each lane's groups
  wire [SUM_WIDTH-1:0] stage2_comb [0:NUM_LANE-1];
  generate
    for (lane = 0; lane < NUM_LANE; lane = lane + 1) begin : GEN_STAGE2
      assign stage2_comb[lane] =
           {{(SUM_WIDTH-ST1_WIDTH){1'b0}}}
         + stage1_reg[lane][0]
         + stage1_reg[lane][1]
         + stage1_reg[lane][2]
         + stage1_reg[lane][3];
    end
  endgenerate

  // Stage2 register ¡æ flat output
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sum_out_flat <= {(NUM_LANE*SUM_WIDTH){1'b0}};
    end else begin
      for (i = 0; i < NUM_LANE; i = i + 1) begin
        sum_out_flat[
          i*SUM_WIDTH +: SUM_WIDTH
        ] <= stage2_comb[i];
      end
    end
  end

endmodule
