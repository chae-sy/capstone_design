`timescale 1ns/1ps

// ---------------------------------------------------------------------
// Mini SPRNN Example (Single-Layer, 4-Channel Dot Product + Bias + ReLU)
// ---------------------------------------------------------------------
module gcd #(
    parameter DATA_WIDTH = 8,
    parameter NUM_CHNL   = 4,
    parameter PROD_WIDTH = 2*DATA_WIDTH,
    parameter SUM_WIDTH  = PROD_WIDTH + $clog2(NUM_CHNL)
)(
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             start,

    // 4-channel activation and weight (flattened)
    input  wire [DATA_WIDTH*NUM_CHNL-1:0]   act_in,
    input  wire [DATA_WIDTH*NUM_CHNL-1:0]   weight_in,

    // single 32-bit bias (for this mini example)
    input  wire [31:0]                      bias_in,

    // output
    output reg                              done,
    output reg  [DATA_WIDTH-1:0]            out_act
);

    // -----------------------------
    // 1. Unpack inputs & multiply
    // -----------------------------
    wire signed [DATA_WIDTH-1:0]  act_ch   [0:NUM_CHNL-1];
    wire signed [DATA_WIDTH-1:0]  w_ch     [0:NUM_CHNL-1];
    wire signed [PROD_WIDTH-1:0]  prod_ch  [0:NUM_CHNL-1];
    wire signed [PROD_WIDTH*NUM_CHNL-1:0] prod_flat;

    genvar i;
    generate
        for (i = 0; i < NUM_CHNL; i = i + 1) begin : GEN_INPUTS
            assign act_ch[i]   = act_in[   (i+1)*DATA_WIDTH-1 : i*DATA_WIDTH ];
            assign w_ch[i]     = weight_in[(i+1)*DATA_WIDTH-1 : i*DATA_WIDTH ];
            assign prod_ch[i]  = act_ch[i] * w_ch[i];
            assign prod_flat[ (i+1)*PROD_WIDTH-1 : i*PROD_WIDTH ] = prod_ch[i];
        end
    endgenerate

    // -----------------------------
    // 2. Adder tree: sum of 4 prods
    // -----------------------------
    wire signed [SUM_WIDTH-1:0] sum_out;

    adder_tree_4 #(
        .DATA_WIDTH (PROD_WIDTH),
        .NUM_INPUTS (NUM_CHNL),
        .SUM_WIDTH  (SUM_WIDTH)
    ) u_adder_tree_4 (
        .in_flat (prod_flat),
        .sum_out(sum_out)
    );

    // -----------------------------
    // 3. Bias + ReLU + saturation
    // -----------------------------
    wire [DATA_WIDTH-1:0] relu_out;

    bias_relu_mini #(
        .SUM_WIDTH (SUM_WIDTH),
        .OUT_WIDTH (DATA_WIDTH)
    ) u_bias_relu_mini (
        .sum_in (sum_out),
        .bias   (bias_in),
        .act_out(relu_out)
    );

    // -----------------------------
    // 4. Simple FSM: IDLE → RUN → DONE
    // -----------------------------
    localparam S_IDLE = 2'd0;
    localparam S_RUN  = 2'd1;
    localparam S_DONE = 2'd2;

    reg [1:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            done    <= 1'b0;
            out_act <= {DATA_WIDTH{1'b0}};
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        state <= S_RUN;
                    end
                end

                // combinational datapath is always “on”,
                // we just latch the result in this state
                S_RUN: begin
                    out_act <= relu_out;
                    done    <= 1'b1;
                    state   <= S_DONE;
                end

                S_DONE: begin
                    // done is a 1-cycle pulse; clear it here
                    done <= 1'b0;
                    // wait for start to go low, then back to IDLE
                    if (!start) begin
                        state <= S_IDLE;
                    end
                end

                default: begin
                    state <= S_IDLE;
                    done  <= 1'b0;
                end
            endcase
        end
    end

endmodule

// ---------------------------------------------------------------------
// 4-input adder tree (combinational)
// ---------------------------------------------------------------------
module adder_tree_4 #(
    parameter DATA_WIDTH = 16,
    parameter NUM_INPUTS = 4,
    parameter SUM_WIDTH  = DATA_WIDTH + $clog2(NUM_INPUTS)
)(
    input  wire [NUM_INPUTS*DATA_WIDTH-1:0] in_flat,
    output wire signed [SUM_WIDTH-1:0]      sum_out
);
    wire signed [DATA_WIDTH-1:0] in0, in1, in2, in3;

    assign in0 = in_flat[1*DATA_WIDTH-1 : 0*DATA_WIDTH];
    assign in1 = in_flat[2*DATA_WIDTH-1 : 1*DATA_WIDTH];
    assign in2 = in_flat[3*DATA_WIDTH-1 : 2*DATA_WIDTH];
    assign in3 = in_flat[4*DATA_WIDTH-1 : 3*DATA_WIDTH];

    assign sum_out = $signed(in0) + $signed(in1) + $signed(in2) + $signed(in3);

endmodule
// ---------------------------------------------------------------------
// Bias + ReLU + simple saturation (combinational)
//   - sum_in : signed SUM_WIDTH
//   - bias   : signed 32-bit
//   - act_out: signed 8-bit with ReLU
// ---------------------------------------------------------------------
module bias_relu_mini #(
    parameter SUM_WIDTH = 18,
    parameter OUT_WIDTH = 8
)(
    input  wire signed [SUM_WIDTH-1:0]  sum_in,
    input  wire signed [31:0]          bias,
    output reg  signed [OUT_WIDTH-1:0] act_out
);
    // sign-extend sum_in to 32 bits
    wire signed [31:0] sum_ext = {{(32-SUM_WIDTH){sum_in[SUM_WIDTH-1]}}, sum_in};

    // combine with bias
    wire signed [31:0] sum32 = sum_ext + bias;

    // pick some mid bits as “activation scale”
    wire signed [7:0] raw = sum32[15:8];

    // 내부 임시 값
    reg signed [7:0] tmp;

    // 완전 조합 논리, 래치 없음
    always @(*) begin
        // 기본값 설정 (모든 경로에서 값이 정해지도록)
        tmp     = 8'sd0;
        act_out = {OUT_WIDTH{1'b0}};

        // 1) 8비트 범위로 saturation
        if (raw > 8'sd127)
            tmp = 8'sd127;
        else if (raw < -8'sd128)
            tmp = -8'sd128;
        else
            tmp = raw;

        // 2) ReLU (0 ~ 127로 clamp)
        if (tmp <= 0)
            act_out = 0;
        else
            act_out = tmp;
    end

endmodule
