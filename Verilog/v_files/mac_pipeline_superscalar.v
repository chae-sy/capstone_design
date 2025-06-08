`timescale 1ns / 1ps
// mac_pipeline_superscalar converted to Verilog-2001: no packed/unpacked arrays, no SystemVerilog types

module mac_pipeline_superscalar #(
  parameter DATA_WIDTH = 8,
  parameter NUM_STAGE  = 9,
  parameter LANE_NUM   = 3
)(
  input  wire                     clk,
  input  wire                     rst_n,
  input  wire                     pe_en,
  input  wire [LANE_NUM*DATA_WIDTH-1:0] data_in_flat,
  input  wire [DATA_WIDTH-1:0]    weight_in,
  output reg                      pe_done,
  output reg [LANE_NUM*20-1:0]    result_out_flat
);

  // flatten a 2-D pipeline into a 1-D array of depth NUM_STAGE*LANE_NUM
  // each entry holds a 2*DATA_WIDTH-bit partial sum
  reg [2*DATA_WIDTH-1:0] pipe [0:NUM_STAGE*LANE_NUM-1];
  // valid flag for each stage
  reg val_pipe [0:NUM_STAGE-1];
  // previous valid to form a one-cycle pulse
  reg prev_valid_stage;

  integer s, l;

  // Helper to extract input lane l
  function [DATA_WIDTH-1:0] data_in;
    input integer idx;
    begin
      data_in = data_in_flat[idx*DATA_WIDTH +: DATA_WIDTH];
    end
  endfunction

  //----------------------------------------------------------------
  // 1) Main pipeline: shift & accumulate
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // reset all pipeline registers and valids
      for (s = 0; s < NUM_STAGE; s = s + 1) begin
        val_pipe[s] <= 1'b0;
        for (l = 0; l < LANE_NUM; l = l + 1) begin
          pipe[s*LANE_NUM + l] <= {2*DATA_WIDTH{1'b0}};
        end
      end
    end else begin
      if (pe_en) begin
        // Stage 0
        val_pipe[0] <= 1'b1;
        for (l = 0; l < LANE_NUM; l = l + 1)
          pipe[l] <= data_in(l) * weight_in;

        // Stages 1 .. NUM_STAGE-1
        for (s = 1; s < NUM_STAGE; s = s + 1) begin
          val_pipe[s] <= val_pipe[s-1];
          for (l = 0; l < LANE_NUM; l = l + 1) begin
            if (val_pipe[s-1])
              pipe[s*LANE_NUM + l] <= pipe[(s-1)*LANE_NUM + l]
                                   + data_in(l) * weight_in;
            else
              pipe[s*LANE_NUM + l] <= {2*DATA_WIDTH{1'b0}};
          end
        end
      end else begin
        // flush pipeline on idle
        val_pipe[0] <= 1'b0;
        for (l = 0; l < LANE_NUM; l = l + 1)
          pipe[l] <= {2*DATA_WIDTH{1'b0}};
        for (s = 1; s < NUM_STAGE; s = s + 1)
          val_pipe[s] <= 1'b0;
      end
    end
  end

  //----------------------------------------------------------------
  // 2) Output pulse and result
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prev_valid_stage   <= 1'b0;
      pe_done            <= 1'b0;
      result_out_flat    <= {LANE_NUM*20{1'b0}};
    end else begin
      prev_valid_stage <= val_pipe[NUM_STAGE-2];
      if (val_pipe[NUM_STAGE-2] && !prev_valid_stage) begin
        // generate one-cycle pe_done pulse
        pe_done <= 1'b1;
        // compute final output: last stage sum + one more MAC
        for (l = 0; l < LANE_NUM; l = l + 1) begin
          // each result is 2*DATA_WIDTH + DATA_WIDTH = 3*DATA_WIDTH bits;
          // we clamp/truncate to 20 bits here
          result_out_flat[l*20 +: 20] <= 
            pipe[(NUM_STAGE-2)*LANE_NUM + l]
            + data_in(l) * weight_in;
        end
      end else begin
        pe_done <= 1'b0;
        result_out_flat <= {LANE_NUM*20{1'b0}};
      end
    end
  end

endmodule
