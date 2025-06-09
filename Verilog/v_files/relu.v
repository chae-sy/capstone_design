module bias_relu (
    input  wire           relu_en,
    output reg            relu_done,
    input  wire [2:0]     layer_state,
    input  wire signed [WIDTH_IN_DATA-1:0]  data_in,
    input  wire signed [WIDTH_BIAS-1:0]     bias,
    output reg         [WIDTH_OUT_DATA-1:0] data_out
);

// =============================================
// Parameter declarations
// =============================================
parameter NUM_CHNL       = 16;
parameter NUM_ACCUMULATE = 9;
parameter NUM_ADDER_TREE_INPUTS = 16;
parameter WIDTH_BITWIDTH = 8;

parameter WIDTH_IN_DATA  = 16 + 4 + 4; // WIDTH_BITWIDTH*2 + clog2(9) + clog2(16)
parameter WIDTH_BIAS     = 32;
parameter WIDTH_OUT_DATA = 8;
parameter RELU_MAX_VAL   = 6;

parameter WIDTH_L1_IN_IL  = 9;
parameter WIDTH_L1_OUT_IL = 1;
parameter WIDTH_L1_ZERO_POINT = -1;
parameter WIDTH_L1_SCALE_INV = 5;  // 1 / 0.2 = 5

parameter WIDTH_L2_IN_IL = 7;
parameter WIDTH_L2_OUT_IL = 1;
parameter WIDTH_L2_ZERO_POINT = 0;
parameter WIDTH_L2_SCALE_INV = 5;

parameter WIDTH_L3_IN_IL = 12;
parameter WIDTH_L3_OUT_IL = 1;
parameter WIDTH_L3_ZERO_POINT = 0;
parameter WIDTH_L3_SCALE_INV = 5;

parameter WIDTH_L4_IN_IL = 7;
parameter WIDTH_L4_OUT_IL = 0;
parameter WIDTH_L4_ZERO_POINT = 0;
parameter WIDTH_L4_SCALE_INV = 5;

parameter WIDTH_L5_IN_IL = 8;
parameter WIDTH_L5_OUT_IL = 0;
parameter WIDTH_L5_ZERO_POINT = 0;
parameter WIDTH_L5_SCALE_INV = 5;

parameter WIDTH_L6_IN_IL = 8;
parameter WIDTH_L6_OUT_IL = 0;
parameter WIDTH_L6_ZERO_POINT = 0;
parameter WIDTH_L6_SCALE_INV = 5;

// =============================================
// Derived constants
// =============================================
integer relu_q_l1, relu_q_l2, relu_q_l3, relu_q_l4, relu_q_l5, relu_q_l6;

initial begin
  relu_q_l1 = (RELU_MAX_VAL * (1 << (WIDTH_IN_DATA - WIDTH_L1_IN_IL - 1)) * WIDTH_L1_SCALE_INV) + WIDTH_L1_ZERO_POINT;
  relu_q_l2 = (RELU_MAX_VAL * (1 << (WIDTH_IN_DATA - WIDTH_L2_IN_IL - 1)) * WIDTH_L2_SCALE_INV) + WIDTH_L2_ZERO_POINT;
  relu_q_l3 = (RELU_MAX_VAL * (1 << (WIDTH_IN_DATA - WIDTH_L3_IN_IL - 1)) * WIDTH_L3_SCALE_INV) + WIDTH_L3_ZERO_POINT;
  relu_q_l4 = (RELU_MAX_VAL * (1 << (WIDTH_IN_DATA - WIDTH_L4_IN_IL - 1)) * WIDTH_L4_SCALE_INV) + WIDTH_L4_ZERO_POINT;
  relu_q_l5 = (RELU_MAX_VAL * (1 << (WIDTH_IN_DATA - WIDTH_L5_IN_IL - 1)) * WIDTH_L5_SCALE_INV) + WIDTH_L5_ZERO_POINT;
  relu_q_l6 = (RELU_MAX_VAL * (1 << (WIDTH_IN_DATA - WIDTH_L6_IN_IL - 1)) * WIDTH_L6_SCALE_INV) + WIDTH_L6_ZERO_POINT;
end

// =============================================
// Internal signals
// =============================================
wire signed [WIDTH_BIAS-1:0] data_extended;
assign data_extended = {{(WIDTH_BIAS - WIDTH_IN_DATA){data_in[WIDTH_IN_DATA-1]}}, data_in} << (WIDTH_BIAS - WIDTH_IN_DATA);

reg signed [WIDTH_BIAS-1:0] bias_added;

always @(*) begin
  data_out   = 0;
  relu_done  = 0;
  bias_added = 0;

  if (relu_en) begin
    case (layer_state)
      3'd1: begin
        bias_added = data_extended + bias;
        if (bias_added[WIDTH_BIAS-1])
          data_out = WIDTH_L1_ZERO_POINT;
        else if (bias_added > relu_q_l1)
          data_out = relu_q_l1;
        else
          data_out = bias_added[WIDTH_IN_DATA - WIDTH_L1_IN_IL - 1 +: WIDTH_OUT_DATA];
        relu_done = 1;
      end

      3'd2: begin
        bias_added = data_extended + bias;
        if (bias_added[WIDTH_BIAS-1])
          data_out = WIDTH_L2_ZERO_POINT;
        else if (bias_added > relu_q_l2)
          data_out = relu_q_l2;
        else
          data_out = bias_added[WIDTH_IN_DATA - WIDTH_L2_IN_IL - 1 +: WIDTH_OUT_DATA];
        relu_done = 1;
      end

      3'd3: begin
        bias_added = data_extended + bias;
        if (bias_added[WIDTH_BIAS-1])
          data_out = WIDTH_L3_ZERO_POINT;
        else if (bias_added > relu_q_l3)
          data_out = relu_q_l3;
        else
          data_out = bias_added[WIDTH_IN_DATA - WIDTH_L3_IN_IL - 1 +: WIDTH_OUT_DATA];
        relu_done = 1;
      end

      3'd4: begin
        if (data_extended[WIDTH_BIAS-1])
          data_out = WIDTH_L4_ZERO_POINT;
        else if (data_extended > relu_q_l4)
          data_out = relu_q_l4;
        else
          data_out = data_extended[WIDTH_IN_DATA - WIDTH_L4_IN_IL - 1 +: WIDTH_OUT_DATA];
        relu_done = 1;
      end

      3'd5: begin
        bias_added = data_extended + bias;
        if (bias_added[WIDTH_BIAS-1])
          data_out = WIDTH_L5_ZERO_POINT;
        else if (bias_added > relu_q_l5)
          data_out = relu_q_l5;
        else
          data_out = bias_added[WIDTH_IN_DATA - WIDTH_L5_IN_IL - 1 +: WIDTH_OUT_DATA];
        relu_done = 1;
      end

      3'd6: begin
        bias_added = data_extended + bias;
        if (bias_added[WIDTH_BIAS-1]) begin
          if (bias_added < -(1 << (WIDTH_OUT_DATA-1 + WIDTH_BIAS - WIDTH_L6_IN_IL - 1)))
            data_out = WIDTH_L6_ZERO_POINT;
          else
            data_out = {1'b1,
                        bias_added[WIDTH_BIAS - WIDTH_L6_IN_IL - 1 + WIDTH_L6_OUT_IL:
                                   WIDTH_BIAS - WIDTH_L6_IN_IL - 1],
                        bias_added[WIDTH_BIAS - WIDTH_L6_IN_IL - 1:
                                   WIDTH_BIAS - WIDTH_L6_IN_IL - 1 - (WIDTH_OUT_DATA - WIDTH_L6_OUT_IL - 1)]};
        end else begin
          if (bias_added > ((1 << (WIDTH_OUT_DATA-1)) - 1) << (WIDTH_BIAS - WIDTH_L6_IN_IL - 1 - (WIDTH_OUT_DATA - WIDTH_L6_OUT_IL - 1)))
            data_out = WIDTH_L6_ZERO_POINT + ((1 << WIDTH_OUT_DATA) - 1);
          else
            data_out = {1'b0,
                        bias_added[WIDTH_BIAS - WIDTH_L6_IN_IL - 1 + WIDTH_L6_OUT_IL:
                                   WIDTH_BIAS - WIDTH_L6_IN_IL - 1],
                        bias_added[WIDTH_BIAS - WIDTH_L6_IN_IL - 1:
                                   WIDTH_BIAS - WIDTH_L6_IN_IL - 1 - (WIDTH_OUT_DATA - WIDTH_L6_OUT_IL - 1)]};
        end
        relu_done = 1;
      end

      default: begin
        data_out = 0;
        relu_done = 0;
      end
    endcase
  end
end

endmodule
