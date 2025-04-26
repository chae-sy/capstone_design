module w_buf (
    input clk,
    input rst,
    input wren, // write enable 신호
    input [7:0] wr_data, // 메모리에서 가져온 weight 데이터 (8비트)
    input [4:0] wr_row,  // 쓰기용 행 주소 (0~2)
    input [4:0] wr_col,  // 쓰기용 열 주소 (0~2)
    input [3:0] wr_chn,  // 쓰기용 채널 번호 (0~15)

    input rden, // read enable 신호
    output reg [7:0] rd_weight [0:2][0:2][0:15] // 읽어낸 3x3x16 weight 데이터
);

    // 내부 메모리 정의
    // 3 rows × 3 cols × 16 channels
    reg [7:0] mem [0:2][0:2][0:15];

    // Write logic
    always @(posedge clk) begin
        if (rst) begin
        end else if (wren) begin
            mem[wr_row][wr_col][wr_chn] <= wr_data;
        end
    end

    // Read logic
    integer i, j, c;
    always @(posedge clk) begin
        if (rden) begin
            for (i = 0; i < 3; i = i + 1) begin
                for (j = 0; j < 3; j = j + 1) begin
                    for (c = 0; c < 16; c = c + 1) begin
                        rd_weight[i][j][c] <= mem[i][j][c];
                    end
                end
            end
        end
    end

endmodule
