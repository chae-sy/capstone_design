
module weight_buffer #(
    parameter DATA_WIDTH = 8,
    parameter IN_CHANNELS = 3,
    parameter FILTER_SIZE = 9,  // 3x3 filter
    parameter OUT_CHANNELS = 16
)(
    input clk,
    input rstn,
    input load_en,
    input [$clog2(IN_CHANNELS * FILTER_SIZE * OUT_CHANNELS)-1:0] load_addr,
    input [DATA_WIDTH-1:0] load_data,

    input [$clog2(IN_CHANNELS)-1:0] in_ch,
    input [$clog2(FILTER_SIZE)-1:0] weight_idx,
    input [$clog2(OUT_CHANNELS)-1:0] out_ch,
    output reg [DATA_WIDTH-1:0] weight_out
);

    // weights[in_ch][weight_idx][out_ch]
    reg [DATA_WIDTH-1:0] weights [0:IN_CHANNELS-1][0:FILTER_SIZE-1][0:OUT_CHANNELS-1];

    integer i, j, k;

    // Load weights
    always @(posedge clk) begin
        if (load_en) begin
            // Flattened address to 3D indices
            i = load_addr / (FILTER_SIZE * OUT_CHANNELS);
            j = (load_addr / OUT_CHANNELS) % FILTER_SIZE;
            k = load_addr % OUT_CHANNELS;
            weights[i][j][k] <= load_data;
        end
    end

    // Read weight
    always @(posedge clk) begin
        weight_out <= weights[in_ch][weight_idx][out_ch];
    end

endmodule
