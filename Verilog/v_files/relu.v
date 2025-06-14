`timescale 1ns / 1ps

module bias_relu #(
    parameter NUM_CHNL       = 16,
    parameter NUM_ACCUMULATE = 9,
    parameter NUM_ADDER_TREE_INPUTS = 16,
    parameter WIDTH_BITWIDTH = 8,
    parameter WIDTH_IN_DATA = WIDTH_BITWIDTH * 2 + $clog2(NUM_ACCUMULATE) + $clog2(NUM_ADDER_TREE_INPUTS), //24
    parameter WIDTH_BIAS = 32,
    parameter WIDTH_BIAS_ADDED = 33,
    parameter WIDTH_OUT_DATA = 8,
    
    parameter signed [WIDTH_OUT_DATA-1:0] MAX_VAL = (2**(WIDTH_OUT_DATA-1)) - 1,
    parameter signed [WIDTH_OUT_DATA-1:0] MIN_VAL = -(2**(WIDTH_OUT_DATA-1)),

    parameter WIDTH_L1_IN_IL = 9,  parameter WIDTH_L1_OUT_IL = 2, parameter WIDTH_L1_B_IL = 1,
    parameter WIDTH_L2_IN_IL = 9,  parameter WIDTH_L2_OUT_IL = 2, parameter WIDTH_L2_B_IL = 1,
    parameter WIDTH_L3_IN_IL = 9,  parameter WIDTH_L3_OUT_IL = 2, parameter WIDTH_L3_B_IL = 1,
    parameter WIDTH_L4_IN_IL = 9,  parameter WIDTH_L4_OUT_IL = 2, parameter WIDTH_L4_B_IL = 1,
    parameter WIDTH_L5_IN_IL = 9,  parameter WIDTH_L5_OUT_IL = 2, parameter WIDTH_L5_B_IL = 1,
    parameter WIDTH_L6_IN_IL = 9,  parameter WIDTH_L6_OUT_IL = 2, parameter WIDTH_L6_B_IL = 1
)(
    input   wire                                   relu_en,
    output  reg                                    relu_done,
    input   wire [2:0]                             layer_state,
    input   wire signed [WIDTH_IN_DATA-1:0]        data_in,
    input   wire signed [WIDTH_BIAS-1:0]           bias,
    output  reg signed [WIDTH_OUT_DATA-1:0]        data_out
);

  // Verilog에서는 localparam 계산을 위해 매크로보다는 직접 선언을 선호합니다.
  // 예시: L1 layer용 internal fixed-point 파라미터 계산

  wire signed [WIDTH_BIAS-1:0] data_extended;
  assign data_extended = $signed({{(WIDTH_BIAS - WIDTH_IN_DATA){data_in[WIDTH_IN_DATA-1]}}, data_in} <<< (WIDTH_BIAS - WIDTH_IN_DATA));

  reg signed [WIDTH_BIAS_ADDED-1:0] bias_added;
  reg signed [WIDTH_BIAS-1:0] bias_ext;
  reg signed [WIDTH_BITWIDTH-1:0] clipped;

  always @(*) begin
    data_out   = {WIDTH_OUT_DATA{1'b0}};
    relu_done  = 1'b0;
    bias_added = {WIDTH_BIAS_ADDED{1'b0}};
    bias_ext   = {WIDTH_BIAS{1'b0}};

    if (relu_en) begin
      case (layer_state)
        3'd1: begin
          bias_ext = {{(WIDTH_BIAS - WIDTH_BITWIDTH){bias[WIDTH_BITWIDTH-1]}}, bias} >>> 11;
          bias_added = data_extended + bias_ext;
          clipped = $signed({bias_added[31],bias_added[20:14]});
          if (bias_added < 0)
            data_out = 0;
          else if (bias_added > 127)
            data_out = 8'd127;
          else
            data_out = clipped;
          relu_done = 1'b1;
        end

        3'd2: begin
          bias_ext = {{(WIDTH_BIAS - WIDTH_BITWIDTH){bias[WIDTH_BITWIDTH-1]}}, bias} << 5;
          bias_added = data_extended + bias_ext;
          if (bias_added < 0)
            data_out = 0;
          else if (bias_added > 4064)
            data_out = 8'd127;
          else
            data_out = bias_added[12:5];
          relu_done = 1'b1;
        end

        3'd3: begin
          bias_ext = {{(WIDTH_BIAS - WIDTH_BITWIDTH){bias[WIDTH_BITWIDTH-1]}}, bias} << 5;
          bias_added = data_extended + bias_ext;
          if (bias_added < 0)
            data_out = 0;
          else if (bias_added > 4064)
            data_out = 8'd127;
          else
            data_out = bias_added[12:5];
          relu_done = 1'b1;
        end

        3'd4: begin
          if (data_extended < 0)
            data_out = 0;
          else if (data_extended > 4064)
            data_out = 8'd127;
          else
            data_out = data_extended[12:5];
          relu_done = 1'b1;
        end

        3'd5: begin
          bias_ext = {{(WIDTH_BIAS - WIDTH_BITWIDTH){bias[WIDTH_BITWIDTH-1]}}, bias} << 5;
          bias_added = data_extended + bias_ext;
          if (bias_added < 0)
            data_out = 0;
          else if (bias_added > 4064)
            data_out = 8'd127;
          else
            data_out = bias_added[12:5];
          relu_done = 1'b1;
        end

        3'd6: begin
          bias_ext = {{(WIDTH_BIAS - WIDTH_BITWIDTH){bias[WIDTH_BITWIDTH-1]}}, bias} << 5;
          bias_added = data_extended + bias_ext;
          if (bias_added < -4096)
            data_out = 0;
          else if (bias_added > 4064)
            data_out = 8'd127;
          else
            data_out = bias_added[12:5];
          relu_done = 1'b1;
        end

        default: begin
          data_out = {WIDTH_OUT_DATA{1'b0}};
          relu_done = 1'b0;
        end
      endcase
    end
  end

endmodule
