module input_buffer (
    input clk,
    input rst,

    // write interface
    input wren, // write enable (1일때 wr_data 저장)
    input [7:0] wr_data,
    input [6:0] wr_row,   // up to 127
    input [6:0] wr_col,   // up to 127
    input [3:0] wr_chn,   // 0~15

    // read interface
    input [6:0] start_row,
    input [6:0] start_col,
    input rden, // read enable (1일때 데이터 읽어옴)
    output reg [7:0] rd_patch [0:2][0:4][0:15]  // 3x5x16 patch
);

    // 3 rows × 102 cols × 16 channels
    reg [7:0] mem [0:127][0:127][0:15];  // row, col, channel

    // Write logic
    always @(posedge clk) begin
        if (wren) begin
        // wren이 1일때, mem에 데이터 저장
            mem[wr_row][wr_col][wr_chn] <= wr_data;
        end
    end

    // Read logic (extract 3x5x16 patch)
    integer i, j, c;
    always @(posedge clk) begin
        if (rden) begin
        // rden이 1일 때, window를 읽어서 rd_patch에 저장
            for (i = 0; i < 3; i = i + 1) begin
                for (j = 0; j < 5; j = j + 1) begin
                    for (c = 0; c < 16; c = c + 1) begin
                        rd_patch[i][j][c] <= mem[start_row + i][start_col + j][c];
                    end
                end
            end
        end
    end

endmodule
