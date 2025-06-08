//------------------------------------------------------------------------------
// Progect : COMPASS (adapted for generic adder tree)
// DATE    : converted to Verilog-2001
// IP      : adder_tree
//------------------------------------------------------------------------------

`timescale 1ns / 1ps

module adder_tree #(
  parameter DATA_WIDTH = 20,
  parameter NUM_INPUTS = 16,  // total number of inputs
  // compute how many bits you need to sum NUM_INPUTS values of width DATA_WIDTH:
  // ceil(log2(NUM_INPUTS)) = $clog2(NUM_INPUTS) in SV, here we write it out:
  parameter SUM_WIDTH  = DATA_WIDTH + 4  // if NUM_INPUTS=16, clog2=4
)(
  input  wire                    clk,
  input  wire                    rst_n,
  input  wire                    adder_tree_en,
  input  wire [NUM_INPUTS*DATA_WIDTH-1:0] in_flat,
  output reg  [SUM_WIDTH-1:0]    sum_out,
  output reg                     adder_tree_done
);

  // split into 4‐wide groups
  localparam GROUP_SIZE = 4;
  localparam NUM_GROUPS = NUM_INPUTS / GROUP_SIZE;
  // width of each partial sum:
  localparam ST1_WIDTH  = DATA_WIDTH + 2;  // clog2(4)=2

  // partial sums for each group (flattened out explicitly)
  wire [ST1_WIDTH-1:0] stage1_comb0;
  wire [ST1_WIDTH-1:0] stage1_comb1;
  wire [ST1_WIDTH-1:0] stage1_comb2;
  wire [ST1_WIDTH-1:0] stage1_comb3;

  reg  [ST1_WIDTH-1:0] stage1_reg0;
  reg  [ST1_WIDTH-1:0] stage1_reg1;
  reg  [ST1_WIDTH-1:0] stage1_reg2;
  reg  [ST1_WIDTH-1:0] stage1_reg3;

  // compute each group’s sum
  // group 0: inputs 0–3
  assign stage1_comb0 =
       in_flat[  DATA_WIDTH*0 + DATA_WIDTH -1 :  DATA_WIDTH*0         ]
     + in_flat[  DATA_WIDTH*1 + DATA_WIDTH -1 :  DATA_WIDTH*1         ]
     + in_flat[  DATA_WIDTH*2 + DATA_WIDTH -1 :  DATA_WIDTH*2         ]
     + in_flat[  DATA_WIDTH*3 + DATA_WIDTH -1 :  DATA_WIDTH*3         ];

  // group 1: inputs 4–7
  assign stage1_comb1 =
       in_flat[  DATA_WIDTH*4 + DATA_WIDTH -1 :  DATA_WIDTH*4         ]
     + in_flat[  DATA_WIDTH*5 + DATA_WIDTH -1 :  DATA_WIDTH*5         ]
     + in_flat[  DATA_WIDTH*6 + DATA_WIDTH -1 :  DATA_WIDTH*6         ]
     + in_flat[  DATA_WIDTH*7 + DATA_WIDTH -1 :  DATA_WIDTH*7         ];

  // group 2: inputs 8–11
  assign stage1_comb2 =
       in_flat[  DATA_WIDTH*8 + DATA_WIDTH -1 :  DATA_WIDTH*8         ]
     + in_flat[  DATA_WIDTH*9 + DATA_WIDTH -1 :  DATA_WIDTH*9         ]
     + in_flat[ DATA_WIDTH*10 + DATA_WIDTH -1 : DATA_WIDTH*10        ]
     + in_flat[ DATA_WIDTH*11 + DATA_WIDTH -1 : DATA_WIDTH*11        ];

  // group 3: inputs 12–15
  assign stage1_comb3 =
       in_flat[ DATA_WIDTH*12 + DATA_WIDTH -1 : DATA_WIDTH*12        ]
     + in_flat[ DATA_WIDTH*13 + DATA_WIDTH -1 : DATA_WIDTH*13        ]
     + in_flat[ DATA_WIDTH*14 + DATA_WIDTH -1 : DATA_WIDTH*14        ]
     + in_flat[ DATA_WIDTH*15 + DATA_WIDTH -1 : DATA_WIDTH*15        ];

  // two-cycle enable pipeline
  reg en_stage1, en_stage2;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      en_stage1 <= 1'b0;
      en_stage2 <= 1'b0;
    end else begin
      en_stage1 <= adder_tree_en;
      en_stage2 <= en_stage1;
    end
  end

  // register partial sums
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      stage1_reg0 <= {ST1_WIDTH{1'b0}};
      stage1_reg1 <= {ST1_WIDTH{1'b0}};
      stage1_reg2 <= {ST1_WIDTH{1'b0}};
      stage1_reg3 <= {ST1_WIDTH{1'b0}};
    end else if (adder_tree_en) begin
      stage1_reg0 <= stage1_comb0;
      stage1_reg1 <= stage1_comb1;
      stage1_reg2 <= stage1_comb2;
      stage1_reg3 <= stage1_comb3;
    end
  end

  // Stage2: sum the four group results
  wire [SUM_WIDTH-1:0] stage2_comb;
  assign stage2_comb =
       {{(SUM_WIDTH-ST1_WIDTH){1'b0}}}
     + stage1_reg0
     + stage1_reg1
     + stage1_reg2
     + stage1_reg3;

  // final output & done pulse
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sum_out         <= {SUM_WIDTH{1'b0}};
      adder_tree_done <= 1'b0;
    end else begin
      if (en_stage1) begin
        sum_out <= stage2_comb;
      end
      adder_tree_done <= en_stage1;
    end
  end

endmodule
