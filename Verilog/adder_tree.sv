`timescale 1ns / 1ps
module adder_tree16_2stage #(
    parameter DATA_WIDTH  = 8,
    parameter NUM_INPUTS  = 16,                      // number of inputs (channels)
    parameter SUM_WIDTH   = DATA_WIDTH + $clog2(NUM_INPUTS)  // bits needed for sum of NUM_INPUTS full-scale values
)(
    input  wire                          clk,
    input  wire                          rst_n,
    // Flattened input bus: NUM_INPUTS of DATA_WIDTH bits
    input  wire [DATA_WIDTH*NUM_INPUTS-1:0] in_flat,
    // Registered output: SUM_WIDTH bits
    output reg  [SUM_WIDTH-1:0]          sum_out
);

    // Stage 1: group inputs in sets of 4
    localparam ST1_WIDTH    = DATA_WIDTH + 2;          // max sum of 4 values fits in DATA_WIDTH+2 bits
    localparam GROUP_SIZE   = 4;
    localparam NUM_GROUPS   = NUM_INPUTS / GROUP_SIZE; // should equal 4 for 16 inputs

    // Registers to hold stage1 partial sums
    reg [ST1_WIDTH-1:0] stage1_reg [0:NUM_GROUPS-1];

    // Combinational partial sums for stage1
    wire [ST1_WIDTH-1:0] stage1_comb [0:NUM_GROUPS-1];
    genvar gi;
    generate
      for (gi = 0; gi < NUM_GROUPS; gi = gi + 1) begin : GEN_STAGE1
        assign stage1_comb[gi] = in_flat[(gi*GROUP_SIZE + 0)*DATA_WIDTH +: DATA_WIDTH]
                              + in_flat[(gi*GROUP_SIZE + 1)*DATA_WIDTH +: DATA_WIDTH]
                              + in_flat[(gi*GROUP_SIZE + 2)*DATA_WIDTH +: DATA_WIDTH]
                              + in_flat[(gi*GROUP_SIZE + 3)*DATA_WIDTH +: DATA_WIDTH];
      end
    endgenerate

    integer i;
    // Stage 1 registering
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        for (i = 0; i < NUM_GROUPS; i = i + 1)
          stage1_reg[i] <= {ST1_WIDTH{1'b0}};
      end else begin
        for (i = 0; i < NUM_GROUPS; i = i + 1)
          stage1_reg[i] <= stage1_comb[i];
      end
    end

    // Stage 2: sum all partial results
    wire [SUM_WIDTH-1:0] stage2_comb;
    assign stage2_comb = {SUM_WIDTH{1'b0}}
                         + stage1_reg[0]
                         + stage1_reg[1]
                         + stage1_reg[2]
                         + stage1_reg[3];

    // Stage 2 registering
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n)
        sum_out <= {SUM_WIDTH{1'b0}};
      else
        sum_out <= stage2_comb;
    end

endmodule
