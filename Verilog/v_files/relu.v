module bias_relu #(
    parameter NUM_CHNL       = 16,
    parameter NUM_ACCUMULATE = 9,
    parameter NUM_ADDER_TREE_INPUTS = 16,
    parameter WIDTH_BITWIDTH = 8,
    parameter WIDTH_IN_DATA = 24, // 16 + 4 + 4, $clog2 제거 후 직접 계산
    parameter WIDTH_BIAS = 32,
    parameter WIDTH_OUT_DATA = 8,

    parameter RELU_MAX_VAL = 6,

    parameter WIDTH_L1_IN_IL = 9,
    parameter WIDTH_L1_OUT_IL = 2,
    parameter WIDTH_L2_IN_IL = 9,
    parameter WIDTH_L2_OUT_IL = 2,
    parameter WIDTH_L3_IN_IL = 9,
    parameter WIDTH_L3_OUT_IL = 2,
    parameter WIDTH_L4_IN_IL = 9,
    parameter WIDTH_L4_OUT_IL = 2,
    parameter WIDTH_L5_IN_IL = 9,
    parameter WIDTH_L5_OUT_IL = 2,
    parameter WIDTH_L6_IN_IL = 9,
    parameter WIDTH_L6_OUT_IL = 2
)(
    input  wire                                relu_en,
    output reg                                 relu_done,
    input  wire [2:0]                          layer_state,
    input  wire signed [WIDTH_IN_DATA-1:0]     data_in,
    input  wire signed [WIDTH_BIAS-1:0]        bias,
    output reg [WIDTH_OUT_DATA-1:0]            data_out
);

    // Fractional bit width 계산
    localparam WIDTH_L1_IN_FL  = WIDTH_IN_DATA - WIDTH_L1_IN_IL - 1;
    localparam WIDTH_L1_OUT_FL = WIDTH_OUT_DATA - WIDTH_L1_OUT_IL - 1;

    localparam WIDTH_L2_IN_FL  = WIDTH_IN_DATA - WIDTH_L2_IN_IL - 1;
    localparam WIDTH_L2_OUT_FL = WIDTH_OUT_DATA - WIDTH_L2_OUT_IL - 1;

    localparam WIDTH_L3_IN_FL  = WIDTH_IN_DATA - WIDTH_L3_IN_IL - 1;
    localparam WIDTH_L3_OUT_FL = WIDTH_OUT_DATA - WIDTH_L3_OUT_IL - 1;

    localparam WIDTH_L5_IN_FL  = WIDTH_IN_DATA - WIDTH_L5_IN_IL - 1;
    localparam WIDTH_L5_OUT_FL = WIDTH_OUT_DATA - WIDTH_L5_OUT_IL - 1;

    localparam WIDTH_L6_IN_FL  = WIDTH_IN_DATA - WIDTH_L6_IN_IL - 1;
    localparam WIDTH_L6_OUT_FL = WIDTH_OUT_DATA - WIDTH_L6_OUT_IL - 1;

    wire signed [WIDTH_BIAS-1:0] data_extended;
    reg  signed [WIDTH_BIAS-1:0] bias_added;

    assign data_extended = { { (WIDTH_BIAS - WIDTH_IN_DATA){data_in[WIDTH_IN_DATA-1]} }, data_in };

    always @(*) begin
        data_out   = {WIDTH_OUT_DATA{1'b0}};
        relu_done  = 1'b0;
        bias_added = {WIDTH_BIAS{1'b0}};

        if (relu_en) begin
            case (layer_state)
                3'd1: begin
                    bias_added = data_extended + bias;
                    if (bias_added[WIDTH_BIAS-1])
                        data_out = 8'd0;
                    else if (bias_added > (RELU_MAX_VAL << WIDTH_L1_OUT_FL))
                        data_out = RELU_MAX_VAL << WIDTH_L1_OUT_FL;
                    else
                        data_out = {1'b0, bias_added[WIDTH_L1_OUT_FL+WIDTH_L1_OUT_IL-1:WIDTH_L1_OUT_FL]};
                    relu_done = 1'b1;
                end
                3'd2: begin
                    bias_added = data_extended + bias;
                    if (bias_added[WIDTH_BIAS-1])
                        data_out = 8'd0;
                    else if (bias_added > (RELU_MAX_VAL << WIDTH_L2_OUT_FL))
                        data_out = RELU_MAX_VAL << WIDTH_L2_OUT_FL;
                    else
                        data_out = {1'b0, bias_added[WIDTH_L2_OUT_FL+WIDTH_L2_OUT_IL-1:WIDTH_L2_OUT_FL]};
                    relu_done = 1'b1;
                end
                3'd3: begin
                    bias_added = data_extended + bias;
                    if (bias_added[WIDTH_BIAS-1])
                        data_out = 8'd0;
                    else if (bias_added > (RELU_MAX_VAL << WIDTH_L3_OUT_FL))
                        data_out = RELU_MAX_VAL << WIDTH_L3_OUT_FL;
                    else
                        data_out = {1'b0, bias_added[WIDTH_L3_OUT_FL+WIDTH_L3_OUT_IL-1:WIDTH_L3_OUT_FL]};
                    relu_done = 1'b1;
                end
                3'd4: begin // maxpool: bias 없이
                    if (data_extended[WIDTH_BIAS-1])
                        data_out = 8'd0;
                    else if (data_extended > (RELU_MAX_VAL << WIDTH_L3_OUT_FL))
                        data_out = RELU_MAX_VAL << WIDTH_L3_OUT_FL;
                    else
                        data_out = {1'b0, data_extended[WIDTH_L3_OUT_FL+WIDTH_L3_OUT_IL-1:WIDTH_L3_OUT_FL]};
                    relu_done = 1'b1;
                end
                3'd5: begin
                    bias_added = data_extended + bias;
                    if (bias_added[WIDTH_BIAS-1])
                        data_out = 8'd0;
                    else if (bias_added > (RELU_MAX_VAL << WIDTH_L5_OUT_FL))
                        data_out = RELU_MAX_VAL << WIDTH_L5_OUT_FL;
                    else
                        data_out = {1'b0, bias_added[WIDTH_L5_OUT_FL+WIDTH_L5_OUT_IL-1:WIDTH_L5_OUT_FL]};
                    relu_done = 1'b1;
                end
                3'd6: begin // no_relu
                    bias_added = data_extended + bias;
                    if (bias_added[WIDTH_BIAS-1]) begin
                        if (bias_added < -(1 << (WIDTH_L6_OUT_IL + WIDTH_L6_OUT_FL)))
                            data_out = 8'b10000000;
                        else if (WIDTH_L6_OUT_IL == 0)
                            data_out = {1'b0, bias_added[WIDTH_L6_OUT_FL-1:0]};
                        else
                            data_out = {1'b0, bias_added[WIDTH_L6_OUT_FL+WIDTH_L6_OUT_IL-1:WIDTH_L6_OUT_FL]};
                    end else begin
                        if (bias_added > ((1 << (WIDTH_L6_OUT_IL + WIDTH_L6_OUT_FL)) - 1))
                            data_out = 8'b01111111;
                        else if (WIDTH_L6_OUT_IL == 0)
                            data_out = {1'b0, bias_added[WIDTH_L6_OUT_FL-1:0]};
                        else
                            data_out = {1'b0, bias_added[WIDTH_L6_OUT_FL+WIDTH_L6_OUT_IL-1:WIDTH_L6_OUT_FL]};
                    end
                    relu_done = 1'b1;
                end
                default: begin
                    data_out  = 8'd0;
                    relu_done = 1'b0;
                end
            endcase
        end
    end

endmodule
