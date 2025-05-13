// DUT: systolic_array.v
`timescale 1ns / 1ps

module systolic_array #(
    parameter ARRAY_WIDTH  = 3,
    parameter ARRAY_HEIGHT = 3,
    parameter DATA_WIDTH   = 8,
    parameter COLOR_WIDTH  = 3
)(
    input  wire                             clk,
    input  wire                             rstb,        
    input  wire                             enable,
    input  wire [DATA_WIDTH-1:0]            data_in    [0:ARRAY_HEIGHT-1][0:COLOR_WIDTH-1],
    input  wire [DATA_WIDTH-1:0]            weight_in  [0:ARRAY_WIDTH-1],
    output reg  [2*DATA_WIDTH-1:0]          result_out [0:COLOR_WIDTH-1]
);

    // shift registers for data across width
    reg [DATA_WIDTH-1:0] data_reg [0:ARRAY_HEIGHT-1][0:ARRAY_WIDTH-1][0:COLOR_WIDTH-1];
    // PE outputs
    wire [2*DATA_WIDTH-1:0] pe_out [0:ARRAY_HEIGHT-1][0:ARRAY_WIDTH-1][0:COLOR_WIDTH-1];
    // horizontal sums per row and lane
    reg [2*DATA_WIDTH-1:0] sum_w [0:ARRAY_HEIGHT-1][0:COLOR_WIDTH-1];

    integer rr, cc, ll;
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            for (rr = 0; rr < ARRAY_HEIGHT; rr = rr + 1)
              for (cc = 0; cc < ARRAY_WIDTH;  cc = cc + 1)
                for (ll = 0; ll < COLOR_WIDTH; ll = ll + 1)
                  data_reg[rr][cc][ll] <= {DATA_WIDTH{1'b0}};
        end else if (enable) begin
            for (rr = 0; rr < ARRAY_HEIGHT; rr = rr + 1) begin
                // shift right
                for (cc = ARRAY_WIDTH-1; cc > 0; cc = cc - 1)
                  for (ll = 0; ll < COLOR_WIDTH; ll = ll + 1)
                    data_reg[rr][cc][ll] <= data_reg[rr][cc-1][ll];
                // new data
                for (ll = 0; ll < COLOR_WIDTH; ll = ll + 1)
                    data_reg[rr][0][ll] <= data_in[rr][ll];
            end
        end
    end

    genvar i, j, k;
    generate
        // instantiate PEs
        for (j = 0; j < ARRAY_HEIGHT; j = j + 1) begin : ROWS
          for (i = 0; i < ARRAY_WIDTH; i = i + 1) begin : COLS
            for (k = 0; k < COLOR_WIDTH; k = k + 1) begin : LANES
              PE_with_clock_gating #(.DATA_WIDTH(DATA_WIDTH)) pe_inst (
                  .clk    (clk),
                  .enable (enable),
                  .rstb  (rstb),
                  .data_in(data_reg[j][i][k]),
                  .weight (weight_in[i]),
                  .result (pe_out[j][i][k])
              );
            end
          end
          // horizontal reduction
          for (k = 0; k < COLOR_WIDTH; k = k + 1) begin : REDUCE_WIDTH
            always @(posedge clk or negedge rstb) begin
                if (!rstb)
                  sum_w[j][k] <= {2*DATA_WIDTH{1'b0}};
                else if (enable)
                  sum_w[j][k] <= pe_out[j][0][k] + pe_out[j][1][k] + pe_out[j][2][k];
            end
          end
        end
        // vertical reduction
        for (k = 0; k < COLOR_WIDTH; k = k + 1) begin : REDUCE_HEIGHT
          always @(posedge clk or negedge rstb) begin
              if (!rstb)
                result_out[k] <= {2*DATA_WIDTH{1'b0}};
              else if (enable)
                result_out[k] <= sum_w[0][k] + sum_w[1][k] + sum_w[2][k];
          end
        end
    endgenerate

endmodule
