module bias_relu #(
    parameter NUM_CHNL              = 16,
    parameter NUM_ACCUMULATE        = 9,
    parameter NUM_ADDER_TREE_INPUTS = 16,
    parameter WIDTH_BITWIDTH        = 8,
    parameter WIDTH_IN_DATA         = WIDTH_BITWIDTH * 2 + $clog2(NUM_ACCUMULATE) + $clog2(NUM_ADDER_TREE_INPUTS),
    parameter WIDTH_BIAS            = 32,
    parameter WIDTH_BIAS_ADDED      = 33,
    parameter WIDTH_OUT_DATA        = 8,

    parameter MAX_VAL = (1 << (WIDTH_OUT_DATA - 1)) - 1,
    parameter MIN_VAL = -(1 << (WIDTH_OUT_DATA - 1)),

    parameter WIDTH_L1_IN_IL = 9,  parameter WIDTH_L1_OUT_IL = 2, parameter WIDTH_L1_B_IL = 1,
    parameter WIDTH_L2_IN_IL = 9,  parameter WIDTH_L2_OUT_IL = 2, parameter WIDTH_L2_B_IL = 1,
    parameter WIDTH_L3_IN_IL = 9,  parameter WIDTH_L3_OUT_IL = 2, parameter WIDTH_L3_B_IL = 1,
    parameter WIDTH_L4_IN_IL = 9,  parameter WIDTH_L4_OUT_IL = 2, parameter WIDTH_L4_B_IL = 1,
    parameter WIDTH_L5_IN_IL = 9,  parameter WIDTH_L5_OUT_IL = 2, parameter WIDTH_L5_B_IL = 1,
    parameter WIDTH_L6_IN_IL = 9,  parameter WIDTH_L6_OUT_IL = 2, parameter WIDTH_L6_B_IL = 1
)(
    input  wire                        relu_en,
    output reg                         relu_done,
    input  wire [2:0]                  layer_state,
    input  wire signed [WIDTH_IN_DATA-1:0]  data_in,
    input  wire signed [WIDTH_BIAS-1:0]     bias,
    output reg signed [WIDTH_OUT_DATA-1:0]  data_out
);

wire signed [WIDTH_BIAS-1:0] data_extended;
assign data_extended = {{(WIDTH_BIAS-WIDTH_IN_DATA){data_in[WIDTH_IN_DATA-1]}}, data_in} << (WIDTH_BIAS - WIDTH_IN_DATA);

reg signed [WIDTH_BIAS_ADDED-1:0] bias_added;
reg signed [WIDTH_BIAS-1:0]       bias_ext;

always @(*) begin
    data_out   = 0;
    relu_done  = 0;
    bias_added = 0;
    bias_ext   = 0;

    if (relu_en) begin
        case (layer_state)
            3'd1: begin
                // L1
                bias_ext = {{(WIDTH_BIAS - WIDTH_L1_B_IL - (WIDTH_BIAS - WIDTH_L1_B_IL - 1)){bias[WIDTH_BIAS-1]}}, bias >>> (WIDTH_L1_B_IL - 1)};
                bias_added = data_extended + bias_ext;

                if (bias_added[WIDTH_BIAS_ADDED-1]) data_out = 0;
                else if (bias_added > (1 << (WIDTH_L1_OUT_IL + WIDTH_BIAS - WIDTH_L1_IN_IL - 1)) - 1) data_out = MAX_VAL;
                else data_out = bias_added[WIDTH_BIAS_ADDED-2 -: WIDTH_OUT_DATA];
                relu_done = 1;
            end

            3'd2: begin
                // L2
                bias_ext = {{(WIDTH_BIAS - WIDTH_L2_B_IL - (WIDTH_BIAS - WIDTH_L2_B_IL - 1)){bias[WIDTH_BIAS-1]}}, bias >>> (WIDTH_L2_B_IL - 1)};
                bias_added = data_extended + bias_ext;

                if (bias_added[WIDTH_BIAS_ADDED-1]) data_out = 0;
                else if (bias_added > (1 << (WIDTH_L2_OUT_IL + WIDTH_BIAS - WIDTH_L2_IN_IL - 1)) - 1) data_out = MAX_VAL;
                else data_out = bias_added[WIDTH_BIAS_ADDED-2 -: WIDTH_OUT_DATA];
                relu_done = 1;
            end

            3'd3: begin
                // L3
                bias_ext = {{(WIDTH_BIAS - WIDTH_L3_B_IL - (WIDTH_BIAS - WIDTH_L3_B_IL - 1)){bias[WIDTH_BIAS-1]}}, bias >>> (WIDTH_L3_B_IL - 1)};
                bias_added = data_extended + bias_ext;

                if (bias_added[WIDTH_BIAS_ADDED-1]) data_out = 0;
                else if (bias_added > (1 << (WIDTH_L3_OUT_IL + WIDTH_BIAS - WIDTH_L3_IN_IL - 1)) - 1) data_out = MAX_VAL;
                else data_out = bias_added[WIDTH_BIAS_ADDED-2 -: WIDTH_OUT_DATA];
                relu_done = 1;
            end

            3'd4: begin
                // L4 (ReLU only)
                if (data_extended[WIDTH_BIAS-1]) data_out = 0;
                else if (data_extended > (1 << (WIDTH_L4_OUT_IL + WIDTH_BIAS - WIDTH_L4_IN_IL - 1)) - 1) data_out = MAX_VAL;
                else data_out = data_extended[WIDTH_BIAS-2 -: WIDTH_OUT_DATA];
                relu_done = 1;
            end

            3'd5: begin
                // L5
                bias_ext = {{(WIDTH_BIAS - WIDTH_L5_B_IL - (WIDTH_BIAS - WIDTH_L5_B_IL - 1)){bias[WIDTH_BIAS-1]}}, bias >>> (WIDTH_L5_B_IL - 1)};
                bias_added = data_extended + bias_ext;

                if (bias_added[WIDTH_BIAS_ADDED-1]) data_out = 0;
                else if (bias_added > (1 << (WIDTH_L5_OUT_IL + WIDTH_BIAS - WIDTH_L5_IN_IL - 1)) - 1) data_out = MAX_VAL;
                else data_out = bias_added[WIDTH_BIAS_ADDED-2 -: WIDTH_OUT_DATA];
                relu_done = 1;
            end

            3'd6: begin
                // L6 (no_relu mode)
                bias_ext = {{(WIDTH_BIAS - WIDTH_L6_B_IL - (WIDTH_BIAS - WIDTH_L6_B_IL - 1)){bias[WIDTH_BIAS-1]}}, bias >>> (WIDTH_L6_B_IL - 1)};
                bias_added = data_extended + bias_ext;

                if (bias_added[WIDTH_BIAS_ADDED-1]) begin
                    if (bias_added < -(1 << (WIDTH_L6_OUT_IL + WIDTH_BIAS - WIDTH_L6_IN_IL - 1))) data_out = MIN_VAL;
                    else data_out = bias_added[WIDTH_BIAS_ADDED-2 -: WIDTH_OUT_DATA];
                end else begin
                    if (bias_added > (1 << (WIDTH_L6_OUT_IL + WIDTH_BIAS - WIDTH_L6_IN_IL - 1)) - 1) data_out = MAX_VAL;
                    else data_out = bias_added[WIDTH_BIAS_ADDED-2 -: WIDTH_OUT_DATA];
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
