`timescale 1ns / 1ps

module mac_pipeline_superscalar #(
  parameter int DATA_WIDTH = 8,
  parameter int NUM_STAGE  = 9,
  parameter int LANE_NUM   = 3
) (
  input  logic                       clk,
  input  logic                       rst_n,
  input  logic                       pe_en,
  input  logic [DATA_WIDTH-1:0]      data_in[0:LANE_NUM-1],
  input  logic [DATA_WIDTH-1:0]      weight_in,
  input  wire                        layer_start,
  output logic                       pe_done,
  output logic [19:0]                result_out_flat[0:LANE_NUM-1]
);

  logic [19:0] pipe[NUM_STAGE][LANE_NUM];

  reg [3:0] cnt, cnt_n;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n | layer_start) cnt <= 0;
    else if (pe_en) begin
      if (cnt == 8) cnt <= 0;
      else          cnt <= cnt + 1;
    end
  end

  always_comb begin
    pe_done = 1'b0;
    if (pe_en) begin
      if (cnt == 1) begin
          for (int l = 0; l < LANE_NUM; l++) begin
              pipe[1][l] =  pipe[0][l] + data_in[l] * weight_in;
          end
      end

      for (int l = 0; l < LANE_NUM; l++) begin
          pipe[0][l] = data_in[l] * weight_in;
      end
      if (cnt >= 2)begin
          for (int l = 0; l < LANE_NUM; l++) begin
            pipe[cnt][l] = pipe[cnt-1][l] + data_in[l] * weight_in;
          end
      end
        
      if (cnt == 8) begin
        pe_done = 1'b1;
        for (int l = 0; l < LANE_NUM; l++) begin
            result_out_flat[l] = pipe[cnt][l];
        end
        for (int l = 0; l < LANE_NUM; l++)
            pipe[0][l] = '0;
      end
    end
   end

endmodule