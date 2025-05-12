module systolic_array #(
    parameter ARRAY_WIDTH  = 3,  // filter width
    parameter ARRAY_HEIGHT = 3,  // filter height
    parameter DATA_WIDTH   = 8,
    parameter COLOR_WIDTH  = 3   // R, G, B lanes
)(
    input  wire                                clk,
    input  wire                                rst,
    input  wire                                enable,
    // data inputs into first column for each row (COLOR_WIDTH lanes)
    input  wire [DATA_WIDTH-1:0]               data_in    [0:ARRAY_HEIGHT-1][0:COLOR_WIDTH-1],
    // weight inputs into each column (shared across color lanes)
    input  wire [DATA_WIDTH-1:0]               weight_in  [0:ARRAY_WIDTH-1],
    // output: summed result per color lane
    output wire [2*DATA_WIDTH-1:0]             result_out [0:COLOR_WIDTH-1]
);

    // internal buses
    wire [DATA_WIDTH-1:0]  data_bus   [0:ARRAY_HEIGHT-1][0:ARRAY_WIDTH-1][0:COLOR_WIDTH-1];
    wire [DATA_WIDTH-1:0]  weight_bus [0:ARRAY_HEIGHT-1][0:ARRAY_WIDTH-1];
    wire [2*DATA_WIDTH-1:0] pe_mult   [0:ARRAY_HEIGHT-1][0:ARRAY_WIDTH-1][0:COLOR_WIDTH-1];
    wire [2*DATA_WIDTH-1:0] sum_width [0:ARRAY_HEIGHT-1][0:COLOR_WIDTH-1];

    genvar i, j, k;
    generate
        // instantiate and shift PEs, accumulate across width
        for (j = 0; j < ARRAY_HEIGHT; j = j + 1) begin : ROWS
            for (i = 0; i < ARRAY_WIDTH; i = i + 1) begin : COLS
                for (k = 0; k < COLOR_WIDTH; k = k + 1) begin : LANES
                    // feed or shift data
                    if (i == 0) begin
                        assign data_bus[j][i][k] = data_in[j][k];
                    end else begin
                        assign data_bus[j][i][k] = data_bus[j][i-1][k];
                    end

                    // feed weight (constant per column)
                    assign weight_bus[j][i] = weight_in[i];

                    // PE multiply-accumulate per lane
                    PE_with_clock_gating #(
                        .DATA_WIDTH(DATA_WIDTH)
                    ) pe_inst (
                        .clk     (clk),
                        .enable  (enable),
                        .reset   (rst),
                        .data_in (data_bus[j][i][k]),
                        .weight  (weight_bus[j][i]),
                        .result  (pe_mult[j][i][k])
                    );

                    // accumulate across width for this row and lane
                    if (i == 0) begin
                        assign sum_width[j][k] = pe_mult[j][i][k];
                    end else begin
                        assign sum_width[j][k] = sum_width[j][k] + pe_mult[j][i][k];
                    end
                end
            end
        end

        // final reduction across height
        for (k = 0; k < COLOR_WIDTH; k = k + 1) begin : REDUCE_HEIGHT
            // sum sum_width[0..ARRAY_HEIGHT-1][k]
            if (ARRAY_HEIGHT == 3) begin
                assign result_out[k] = sum_width[0][k]
                                     + sum_width[1][k]
                                     + sum_width[2][k];
            end else begin
                // generic reduction for other heights
                wire [2*DATA_WIDTH-1:0] temp_sum [0:ARRAY_HEIGHT-1];
                assign temp_sum[0] = sum_width[0][k];
                for (j = 1; j < ARRAY_HEIGHT; j = j + 1) begin : SUM_LOOP
                    assign temp_sum[j] = temp_sum[j-1] + sum_width[j][k];
                end
                assign result_out[k] = temp_sum[ARRAY_HEIGHT-1];
            end
        end
    endgenerate

endmodule