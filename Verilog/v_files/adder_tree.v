`timescale 1ns / 1ps
module adder_tree #(
  parameter DATA_WIDTH = 20,
  parameter NUM_INPUTS = 16,  // ?ã®?ùº lane?ùò ?ûÖ?†• Í∞úÏàò
  parameter SUM_WIDTH  = DATA_WIDTH + $clog2(NUM_INPUTS)
)(
  input  wire                        clk,
  input  wire                        rst_n,
  input  wire                        adder_tree_en,     // ?ó∞?Ç∞ ?ãú?ûë ?ù∏?óê?ù¥Î∏?
  input  wire [NUM_INPUTS*DATA_WIDTH-1:0] in_flat,      // ?ã®?ùº lane ?ûÖ?†•
  input  wire                        layer_start,
  output reg  signed [SUM_WIDTH-1:0] sum_out,      // ?ã®?ùº lane Ï∂úÎ†•
  output reg                         adder_tree_done // ?ó∞?Ç∞ ?ôÑÎ£? ?éÑ?ä§
);

  // Stage1 parameters
  localparam GROUP_SIZE = 4;
  localparam NUM_GROUPS = NUM_INPUTS / GROUP_SIZE; //4
  localparam ST1_WIDTH  = DATA_WIDTH + $clog2(GROUP_SIZE); //22

  // Stage1: combinational partial sums [group]
  wire signed [ST1_WIDTH-1:0] stage1_comb [0:NUM_GROUPS-1];
  reg  signed [ST1_WIDTH-1:0] stage1_reg  [0:NUM_GROUPS-1];
  genvar grp;
  generate
    for (grp = 0; grp < NUM_GROUPS; grp = grp + 1) begin : GEN_STAGE1
      assign stage1_comb[grp] = $signed(
          $signed(in_flat[(grp*GROUP_SIZE + 0)*DATA_WIDTH + DATA_WIDTH - 1 : (grp*GROUP_SIZE + 0)*DATA_WIDTH])
        + $signed(in_flat[(grp*GROUP_SIZE + 1)*DATA_WIDTH + DATA_WIDTH - 1 : (grp*GROUP_SIZE + 1)*DATA_WIDTH])
        + $signed(in_flat[(grp*GROUP_SIZE + 2)*DATA_WIDTH + DATA_WIDTH - 1 : (grp*GROUP_SIZE + 2)*DATA_WIDTH])
        + $signed(in_flat[(grp*GROUP_SIZE + 3)*DATA_WIDTH + DATA_WIDTH - 1 : (grp*GROUP_SIZE + 3)*DATA_WIDTH]));
    end
  endgenerate


  reg en_stage1, en_stage2;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n | layer_start) begin
      en_stage1 <= 1'b0;
      en_stage2 <= 1'b0;
    end else begin
      en_stage1 <= adder_tree_en;
      en_stage2 <= en_stage1;
    end
  end

  integer j;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (j = 0; j < NUM_GROUPS; j = j + 1) begin
        stage1_reg[j] <= {ST1_WIDTH{1'b0}};
      end
    end else if (adder_tree_en) begin
      for (j = 0; j < NUM_GROUPS; j = j + 1) begin
        stage1_reg[j] <= stage1_comb[j];
      end
    end
  end

  // Stage2: sum all groups
  wire [SUM_WIDTH-1:0] stage2_comb;
  assign stage2_comb = $signed(
       {{(SUM_WIDTH-ST1_WIDTH){1'b0}}}
     + stage1_reg[0]
     + stage1_reg[1]
     + stage1_reg[2]
     + stage1_reg[3]);

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sum_out         <= {SUM_WIDTH{1'b0}};
      adder_tree_done <= 1'b0;
    end else begin
      if (en_stage1) begin
        sum_out <= $signed(stage2_comb);
      end
      adder_tree_done <= en_stage1;
    end
  end

endmodule