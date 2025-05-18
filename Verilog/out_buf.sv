module out_buf (
    input clk,
    input rst,

    // write interface
    input wren, // write enable
    input [7:0] wr_data, // 저장할 데이터
    input [6:0] wr_row,  // 저장할 행 번호 (0~127)
    input [6:0] wr_col,  // 저장할 열 번호 (0~127)
    input [3:0] wr_chn,  // 저장할 채널 번호 (0~15)

    // read interface
    input rden, // read enable
    input [6:0] rd_row,  // 읽을 행 번호
    input [6:0] rd_col,  // 읽을 열 번호
    input [3:0] rd_chn,  // 읽을 채널 번호
    output reg [7:0] rd_data // 읽은 데이터 출력
);

    // 내부 메모리 정의
    reg [7:0] mem [0:127][0:127][0:15];

    // Write logic
    always @(posedge clk) begin
        if (rst) begin
        end else if (wren) begin
            mem[wr_row][wr_col][wr_chn] <= wr_data;
        end
    end

    // Read logic
    always @(posedge clk) begin
        if (rden) begin
            rd_data <= mem[rd_row][rd_col][rd_chn];
        end
    end

endmodule
