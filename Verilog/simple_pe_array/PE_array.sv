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
    output wire [2*DATA_WIDTH-1:0]          result_out [0:COLOR_WIDTH-1]
);

    // Stage 1 registers: shifted data and PE outputs
    reg [DATA_WIDTH-1:0]   data_reg [0:ARRAY_HEIGHT-1][0:ARRAY_WIDTH-1][0:COLOR_WIDTH-1];
    wire [2*DATA_WIDTH-1:0] pe_out_stage1 [0:ARRAY_HEIGHT-1][0:ARRAY_WIDTH-1][0:COLOR_WIDTH-1];

    // Stage 2 registers: width reduction results
    reg [2*DATA_WIDTH-1:0] sum_w_stage2 [0:ARRAY_HEIGHT-1][0:COLOR_WIDTH-1];

    // Stage 3 registers: height reduction results
    reg [2*DATA_WIDTH-1:0] result_reg_stage3 [0:COLOR_WIDTH-1];

    integer rr, cc, ll;
    // Stage 1: data shift and PE compute
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            for (rr = 0; rr < ARRAY_HEIGHT; rr = rr + 1)
                for (cc = 0; cc < ARRAY_WIDTH; cc = cc + 1)
                    for (ll = 0; ll < COLOR_WIDTH; ll = ll + 1)
                        data_reg[rr][cc][ll] <= {DATA_WIDTH{1'b0}}; //reset
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

    // PE instantiation uses data_reg and weight_in, outputs pe_out_stage1
    genvar i, j, k;
    generate
        for (j = 0; j < ARRAY_HEIGHT; j = j + 1) begin : ROWS
            for (i = 0; i < ARRAY_WIDTH; i = i + 1) begin : COLS
                for (k = 0; k < COLOR_WIDTH; k = k + 1) begin : LANES
                    PE_with_clock_gating #(.DATA_WIDTH(DATA_WIDTH)) pe_inst (
                        .clk    (clk),
                        .enable (enable),
                        .rstb   (rstb),
                        .data_in(data_reg[j][i][k]),
                        .weight (weight_in[i]),
                        .result (pe_out_stage1[j][i][k])
                    );
                end
            end
        end
    endgenerate

    // Stage 2: width reduction pipeline register
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            for (rr = 0; rr < ARRAY_HEIGHT; rr = rr + 1)
                for (ll = 0; ll < COLOR_WIDTH; ll = ll + 1)
                    sum_w_stage2[rr][ll] <= {2*DATA_WIDTH{1'b0}};
        end else if (enable) begin
            for (rr = 0; rr < ARRAY_HEIGHT; rr = rr + 1) begin
                for (ll = 0; ll < COLOR_WIDTH; ll = ll + 1) begin
                    sum_w_stage2[rr][ll] <= pe_out_stage1[rr][0][ll]
                                         + pe_out_stage1[rr][1][ll]
                                         + pe_out_stage1[rr][2][ll];
                end
            end
        end
    end

    // Stage 3: height reduction pipeline register
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            for (ll = 0; ll < COLOR_WIDTH; ll = ll + 1)
                result_reg_stage3[ll] <= {2*DATA_WIDTH{1'b0}};
        end else if (enable) begin
            for (ll = 0; ll < COLOR_WIDTH; ll = ll + 1) begin
                result_reg_stage3[ll] <= sum_w_stage2[0][ll]
                                       + sum_w_stage2[1][ll]
                                       + sum_w_stage2[2][ll];
            end
        end
    end

    // Output from stage 3 register
    assign result_out = result_reg_stage3;

endmodule
