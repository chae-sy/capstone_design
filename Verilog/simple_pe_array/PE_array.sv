module systolic_array_4x4 (
    input clk,
    input rst,
    input [7:0] data_in [0:3],   // Inputs to the first column (one per row)
    input [7:0] weight_in [0:3], // Inputs to the first row (one per column)
    input enable,
    output [15:0] result_out [0:3][0:3] // Result from each PE
);

    // Internal signals for data and weight propagation
    wire [7:0] data [0:3][0:4];    // 4 rows, 5 columns (one extra for input)
    wire [7:0] weight [0:4][0:3];  // 5 rows, 4 columns (one extra for input)
    wire [15:0] result [0:3][0:3];

    genvar i, j, x;

    // Initialize leftmost column and top row
    generate
        for (x = 0; x < 4; x = x + 1) begin
            assign data[x][0] = data_in[x];      // Inputs from the left
            assign weight[0][x] = weight_in[x];  // Inputs from the top
        end
    endgenerate

    // Instantiate PE_with_clock_gating
    generate
        for (i = 0; i < 4; i = i + 1) begin : row
            for (j = 0; j < 4; j = j + 1) begin : col
                PE_with_clock_gating pe_inst (
                            .clk(clk),
                            .enable(enable),
                            .reset(rst),            
                            .data_in(data[i][j]),
                            .weight(weight[i][j]),
                            .result(result[i][j])
                             );

                // Forward data and weight
                assign data[i][j+1] = data[i][j];       // Right propagation
                assign weight[i+1][j] = weight[i][j];   // Down propagation

                assign result_out[i][j] = result[i][j];
            end
        end
    endgenerate

endmodule
