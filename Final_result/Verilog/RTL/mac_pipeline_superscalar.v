`timescale 1ns / 1ps

module mac_pipeline_superscalar #(
  parameter DATA_WIDTH = 8,
  parameter NUM_STAGE  = 9,
  parameter LANE_NUM   = 3
) (
  input  wire                       clk,
  input  wire                       rst_n,
  input  wire                       pe_en,
  input  wire signed [DATA_WIDTH-1:0] data_in_r,
  input  wire signed [DATA_WIDTH-1:0] data_in_g,
  input  wire signed [DATA_WIDTH-1:0] data_in_b,
  input  wire signed [DATA_WIDTH-1:0]weight_in,
  input  wire                       layer_start,
  output reg                        pe_done,
  output reg signed [19:0]          result_out_flat_r,
  output reg signed [19:0]          result_out_flat_g,
  output reg signed [19:0]          result_out_flat_b
);

  reg signed [19:0] pipe[NUM_STAGE-1:0][LANE_NUM-1:0];
  reg [3:0] cnt, cnt_n;
  reg signed [DATA_WIDTH-1:0] data_in[0:LANE_NUM-1];
  reg signed [19:0] result_out_flat[0:LANE_NUM-1];
  integer l;
  
  always @(*) begin
        data_in[0] = data_in_r;
        data_in[1] = data_in_g;
        data_in[2] = data_in_b;
        result_out_flat_r = result_out_flat[0];
        result_out_flat_g = result_out_flat[1];
        result_out_flat_b = result_out_flat[2];
  end
    
   always @(posedge clk or negedge rst_n) begin
    if (!rst_n) cnt <= 0;
    else begin
        if (layer_start ) cnt <= 0;
        if (pe_en) begin
          if (cnt == 8) cnt <= 0;
          else          cnt <= cnt + 1;
        end
    end
 end

  always @(*) begin
    pe_done = 1'b0;
    if (pe_en) begin
      if (cnt == 1) begin
          for (l = 0; l < LANE_NUM; l = l + 1) begin
              pipe[1][l] = $signed($signed(pipe[0][l]) + data_in[l] * weight_in);
          end
      end

      for (l = 0; l < LANE_NUM; l = l + 1) begin
          pipe[0][l] = (data_in[l] * weight_in);
      end
      if (cnt >= 2)begin
          for (l = 0; l < LANE_NUM; l = l + 1) begin
            pipe[cnt][l] = $signed($signed(pipe[cnt-1][l]) + data_in[l] * weight_in);
          end
      end
        
      if (cnt == 8) begin
        pe_done = 1'b1;
        for (l = 0; l < LANE_NUM; l = l + 1) begin
            result_out_flat[l] = pipe[cnt][l];
        end
        for (l = 0; l < LANE_NUM; l = l + 1)
            pipe[0][l] = 0;
      end
    end
   end

endmodule
