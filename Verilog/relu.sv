module bias_relu #(
    parameter NUM_CHNL       = 16,
    parameter NUM_ACCUMULATE = 9,
    parameter NUM_ADDER_TREE_INPUTS = 16,
    parameter WIDTH_BITWIDTH = 8,
    parameter WIDTH_IN_DATA = WIDTH_BITWIDTH * 2 + $clog2(NUM_ACCUMULATE) + $clog2(NUM_ADDER_TREE_INPUTS),
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

  // Layer별 파생 파라미터 정의
  `define LAYER_PARAMS(IN_IL, OUT_IL, B_IL) \
    localparam IN_FL  = WIDTH_IN_DATA - IN_IL - 1; \
    localparam OUT_FL = WIDTH_OUT_DATA - OUT_IL - 1; \
    localparam EXT_FL = WIDTH_BIAS - IN_IL - 1; \
    localparam EXT_IL = WIDTH_BIAS - EXT_FL - 1; \
    localparam B_FL   = WIDTH_BIAS - B_IL - 1;

  `LAYER_PARAMS(WIDTH_L1_IN_IL, WIDTH_L1_OUT_IL, WIDTH_L1_B_IL)
  localparam WIDTH_L1_IN_FL = IN_FL, WIDTH_L1_OUT_FL = OUT_FL, WIDTH_L1_EXT_FL = EXT_FL, WIDTH_L1_B_FL = B_FL;

  `LAYER_PARAMS(WIDTH_L2_IN_IL, WIDTH_L2_OUT_IL, WIDTH_L2_B_IL)
  localparam WIDTH_L2_IN_FL = IN_FL, WIDTH_L2_OUT_FL = OUT_FL, WIDTH_L2_EXT_FL = EXT_FL, WIDTH_L2_B_FL = B_FL;

  `LAYER_PARAMS(WIDTH_L3_IN_IL, WIDTH_L3_OUT_IL, WIDTH_L3_B_IL)
  localparam WIDTH_L3_IN_FL = IN_FL, WIDTH_L3_OUT_FL = OUT_FL, WIDTH_L3_EXT_FL = EXT_FL, WIDTH_L3_B_FL = B_FL;

  `LAYER_PARAMS(WIDTH_L4_IN_IL, WIDTH_L4_OUT_IL, WIDTH_L4_B_IL)
  localparam WIDTH_L4_IN_FL = IN_FL, WIDTH_L4_OUT_FL = OUT_FL, WIDTH_L4_EXT_FL = EXT_FL, WIDTH_L4_B_FL = B_FL;

  `LAYER_PARAMS(WIDTH_L5_IN_IL, WIDTH_L5_OUT_IL, WIDTH_L5_B_IL)
  localparam WIDTH_L5_IN_FL = IN_FL, WIDTH_L5_OUT_FL = OUT_FL, WIDTH_L5_EXT_FL = EXT_FL, WIDTH_L5_B_FL = B_FL;

  `LAYER_PARAMS(WIDTH_L6_IN_IL, WIDTH_L6_OUT_IL, WIDTH_L6_B_IL)
  localparam WIDTH_L6_IN_FL = IN_FL, WIDTH_L6_OUT_FL = OUT_FL, WIDTH_L6_EXT_FL = EXT_FL, WIDTH_L6_B_FL = B_FL;

  // Sign-extend and left shift for fixed-point alignment
  wire signed [WIDTH_BIAS-1:0] data_extended = 
        {{WIDTH_BIAS-WIDTH_IN_DATA{data_in[WIDTH_IN_DATA-1]}}, data_in} << (WIDTH_BIAS - WIDTH_IN_DATA);

  reg signed [WIDTH_BIAS_ADDED-1:0] bias_added;
  reg signed [WIDTH_BIAS-1:0] bias_ext;

  always @(*) begin
    data_out   = {WIDTH_OUT_DATA{1'b0}};
    relu_done  = 1'b0;
    bias_added = {WIDTH_BIAS_ADDED{1'b0}};
    bias_ext   = {WIDTH_BIAS{1'b0}};

    if (relu_en) begin
      case (layer_state)
        3'd1: begin
          bias_ext = {{WIDTH_BIAS-WIDTH_L1_B_IL-WIDTH_L1_EXT_FL{bias[WIDTH_BIAS-1]}}, bias >> (WIDTH_L1_B_FL - WIDTH_L1_EXT_FL)};
          bias_added = data_extended + bias_ext;

          if (bias_added[WIDTH_BIAS_ADDED-1]) data_out = {WIDTH_OUT_DATA{1'b0}};
          else if (bias_added > (2**(WIDTH_L1_OUT_IL + WIDTH_L1_EXT_FL) - 1)) data_out = MAX_VAL;
          else data_out = {1'b0, bias_added[WIDTH_L1_EXT_FL + WIDTH_L1_OUT_IL - 1:WIDTH_L1_EXT_FL],
                                  bias_added[WIDTH_L1_EXT_FL - 1:WIDTH_L1_EXT_FL - WIDTH_L1_OUT_FL]};
          relu_done = 1'b1;
        end

        3'd2: begin
          bias_ext = {{WIDTH_BIAS-WIDTH_L2_B_IL-WIDTH_L2_EXT_FL{bias[WIDTH_BIAS-1]}}, bias >> (WIDTH_L2_B_FL - WIDTH_L2_EXT_FL)};
          bias_added = data_extended + bias_ext;

          if (bias_added[WIDTH_BIAS_ADDED-1]) data_out = {WIDTH_OUT_DATA{1'b0}};
          else if (bias_added > (2**(WIDTH_L2_OUT_IL + WIDTH_L2_EXT_FL) - 1)) data_out = MAX_VAL;
          else data_out = {1'b0, bias_added[WIDTH_L2_EXT_FL + WIDTH_L2_OUT_IL - 1:WIDTH_L2_EXT_FL],
                                  bias_added[WIDTH_L2_EXT_FL - 1:WIDTH_L2_EXT_FL - WIDTH_L2_OUT_FL]};
          relu_done = 1'b1;
        end

        3'd3: begin
          bias_ext = {{WIDTH_BIAS-WIDTH_L3_B_IL-WIDTH_L3_EXT_FL{bias[WIDTH_BIAS-1]}}, bias >> (WIDTH_L3_B_FL - WIDTH_L3_EXT_FL)};
          bias_added = data_extended + bias_ext;

          if (bias_added[WIDTH_BIAS_ADDED-1]) data_out = {WIDTH_OUT_DATA{1'b0}};
          else if (bias_added > (2**(WIDTH_L3_OUT_IL + WIDTH_L3_EXT_FL) - 1)) data_out = MAX_VAL;
          else data_out = {1'b0, bias_added[WIDTH_L3_EXT_FL + WIDTH_L3_OUT_IL - 1:WIDTH_L3_EXT_FL],
                                  bias_added[WIDTH_L3_EXT_FL - 1:WIDTH_L3_EXT_FL - WIDTH_L3_OUT_FL]};
          relu_done = 1'b1;
        end

        3'd4: begin // maxpool
          if (data_extended[WIDTH_BIAS-1]) data_out = {WIDTH_OUT_DATA{1'b0}};
          else if (data_extended > (2**(WIDTH_L4_OUT_IL + WIDTH_L4_EXT_FL) - 1)) data_out = MAX_VAL;
          else data_out = {1'b0, data_extended[WIDTH_L4_EXT_FL + WIDTH_L4_OUT_IL - 1:WIDTH_L4_EXT_FL],
                                 data_extended[WIDTH_L4_EXT_FL - 1:WIDTH_L4_EXT_FL - WIDTH_L4_OUT_FL]};
          relu_done = 1'b1;
        end

        3'd5: begin
          bias_ext = {{WIDTH_BIAS-WIDTH_L5_B_IL-WIDTH_L5_EXT_FL{bias[WIDTH_BIAS-1]}}, bias >> (WIDTH_L5_B_FL - WIDTH_L5_EXT_FL)};
          bias_added = data_extended + bias_ext;

          if (bias_added[WIDTH_BIAS_ADDED-1]) data_out = {WIDTH_OUT_DATA{1'b0}};
          else if (bias_added > (2**(WIDTH_L5_OUT_IL + WIDTH_L5_EXT_FL) - 1)) data_out = MAX_VAL;
          else data_out = {1'b0, bias_added[WIDTH_L5_EXT_FL + WIDTH_L5_OUT_IL - 1:WIDTH_L5_EXT_FL],
                                  bias_added[WIDTH_L5_EXT_FL - 1:WIDTH_L5_EXT_FL - WIDTH_L5_OUT_FL]};
          relu_done = 1'b1;
        end

        3'd6: begin // no_relu 모드
          bias_ext = {{WIDTH_BIAS-WIDTH_L6_B_IL-WIDTH_L6_EXT_FL{bias[WIDTH_BIAS-1]}}, bias >>> (WIDTH_L6_B_FL - WIDTH_L6_EXT_FL)};
          bias_added = data_extended + bias_ext;

          if (bias_added[WIDTH_BIAS_ADDED-1]) begin
            if (bias_added < -((2**(WIDTH_L6_OUT_IL + WIDTH_L6_EXT_FL)))) data_out = MIN_VAL;
            else if (WIDTH_L6_OUT_IL == 0) data_out = {1'b1, bias_added[WIDTH_L6_EXT_FL - 1:WIDTH_L6_EXT_FL - WIDTH_L6_OUT_FL]};
            else data_out = {1'b1, bias_added[WIDTH_L6_EXT_FL + WIDTH_L6_OUT_IL - 1:WIDTH_L6_EXT_FL],
                                  bias_added[WIDTH_L6_EXT_FL - 1:WIDTH_L6_EXT_FL - WIDTH_L6_OUT_FL]};
          end else begin
            if (bias_added > (2**(WIDTH_L6_OUT_IL + WIDTH_L6_EXT_FL) - 1)) data_out = MAX_VAL;
            else if (WIDTH_L6_OUT_IL == 0) data_out = {1'b0, bias_added[WIDTH_L6_EXT_FL - 1:WIDTH_L6_EXT_FL - WIDTH_L6_OUT_FL]};
            else data_out = {1'b0, bias_added[WIDTH_L6_EXT_FL + WIDTH_L6_OUT_IL - 1:WIDTH_L6_EXT_FL],
                                  bias_added[WIDTH_L6_EXT_FL - 1:WIDTH_L6_EXT_FL - WIDTH_L6_OUT_FL]};
          end
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
