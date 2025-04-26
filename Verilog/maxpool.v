//========================================================================
// Module: maxpool_2x4_16ch
// - Input : 100×100×16 (stream, valid_in)
// - Output: 50×25×16  (stream, valid_out)
// - Pool  : KH=4, KW=2, stride_h=2, stride_w=4
// - Assumes cropping(102→100) & conv3+ReLU가 끝난 직후에 삽입
//========================================================================
module maxpool_2x4_16ch #(
    parameter DATA_WIDTH = 8,
    parameter H_IN       = 100,
    parameter W_IN       = 100,
    parameter CH         = 16
)(
    input  wire                        clk,
    input  wire                        rstb,
    input  wire                        valid_in,
    input  wire signed [DATA_WIDTH-1:0] in_data [0:CH-1],  // ReLU 출력
    output reg                         valid_out,
    output reg signed [DATA_WIDTH-1:0] out_data[0:CH-1]    // Pool 결과
);

    // 두 줄(Row) buffer: [채널][가로좌표]
    reg signed [DATA_WIDTH-1:0] linebuf0 [0:CH-1][0:W_IN-1];
    reg signed [DATA_WIDTH-1:0] linebuf1 [0:CH-1][0:W_IN-1];

    // 현재 쓰는 row가 짝수인지(0→linebuf0 / 1→linebuf1)
    reg row_select;
    // 입력 스트림 좌표 카운터
    reg [6:0] row_cnt;   // 0..99
    reg [6:0] col_cnt;   // 0..99

    integer c;
    // 비교 함수: 4개 중 최대
    function signed [DATA_WIDTH-1:0] max4;
        input signed [DATA_WIDTH-1:0] a, b, c, d;
        reg   signed [DATA_WIDTH-1:0] m;
        begin
            m = a;
            if (b > m) m = b;
            if (c > m) m = c;
            if (d > m) m = d;
            max4 = m;
        end
    endfunction

    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            row_cnt   <= 0;
            col_cnt   <= 0;
            row_select<= 0;
            valid_out <= 0;
        end else if (valid_in) begin
            //—— 1) 스트림 좌표 업데이트 —————————————————————————————
            if (col_cnt == W_IN-1) begin
                col_cnt <= 0;
                row_cnt <= (row_cnt == H_IN-1) ? 0 : row_cnt + 1;
                row_select <= ~row_select;
            end else
                col_cnt <= col_cnt + 1;
            //———————————————————————————————————————————————

            //—— 2) 현재 row buffer에 입력 저장 —————————————————————
            if (row_select == 0) begin
                for (c = 0; c < CH; c = c+1)
                    linebuf0[c][col_cnt] <= in_data[c];
            end else begin
                for (c = 0; c < CH; c = c+1)
                    linebuf1[c][col_cnt] <= in_data[c];
            end
            //———————————————————————————————————————————————

            //—— 3) Pooling 출력 타이밍:  
            //    - 세로: row_cnt[0] == 1 → 짝수번째 row가 다 채워진 후 (2-row 윈도우 완성)
            //    - 가로: col_cnt % 4 == 3 → 4-col 윈도우 완성후 
            //———————————————————————————————————————————————
            if (col_cnt[0] == 1 && row_cnt % 4 == 3) begin
                valid_out <= 1;
                // 각 채널별로 2×4 window에서 max 추출
                for (c = 0; c < CH; c = c+1) begin
                    // 현재 row / 이전 row 선택
                    //    curr_buf = linebuf(row_select)
                    //    prev_buf = linebuf(~row_select)
                    // window columns: col_cnt-3, col_cnt-2, col_cnt-1, col_cnt
                    reg signed [DATA_WIDTH-1:0] a0, a1, a2, a3;
                    reg signed [DATA_WIDTH-1:0] b0, b1, b2, b3;
                    if (row_select == 0) begin
                        // 현재 row → buf0, 이전 row → buf1
                        a0 = linebuf0[c][col_cnt-3];
                        a1 = linebuf0[c][col_cnt-2];
                        a2 = linebuf0[c][col_cnt-1];
                        a3 = linebuf0[c][col_cnt];
                        b0 = linebuf1[c][col_cnt-3];
                        b1 = linebuf1[c][col_cnt-2];
                        b2 = linebuf1[c][col_cnt-1];
                        b3 = linebuf1[c][col_cnt];
                    end else begin
                        // 현재 row → buf1, 이전 row → buf0
                        a0 = linebuf1[c][col_cnt-3];
                        a1 = linebuf1[c][col_cnt-2];
                        a2 = linebuf1[c][col_cnt-1];
                        a3 = linebuf1[c][col_cnt];
                        b0 = linebuf0[c][col_cnt-3];
                        b1 = linebuf0[c][col_cnt-2];
                        b2 = linebuf0[c][col_cnt-1];
                        b3 = linebuf0[c][col_cnt];
                    end
                    // 2단계 비교로 8-way max
                    out_data[c] <= max4( max4(a0,a1,a2,a3),
                                         max4(b0,b1,b2,b3),
                                         0, 0 /* 두 그룹 비교만 하면 되므로 마지막 0,0 은 무시 */ );
                end
            end else begin
                valid_out <= 0;
            end
        end else begin
            valid_out <= 0;
        end
    end

endmodule
