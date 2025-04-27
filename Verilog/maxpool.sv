`timescale 1ns/1ps

module maxpool_2x4_16ch_circular #(
    parameter DATA_WIDTH = 8,
    parameter H_IN       = 100,
    parameter W_IN       = 100,
    parameter CH         = 16
)(
    input  wire                        clk,
    input  wire                        rstb,
    input  wire                        valid_in,
    input  wire signed [DATA_WIDTH-1:0] in_data [0:CH-1],
    output reg                         valid_out,
    output reg signed [DATA_WIDTH-1:0] out_data [0:CH-1]
);

    reg signed [DATA_WIDTH-1:0] linebuf0 [0:CH-1][0:7];
    reg signed [DATA_WIDTH-1:0] linebuf1 [0:CH-1][0:7];

    reg row_select;
    reg [6:0] row_cnt;
    reg [6:0] col_cnt;

    reg [2:0] write_ptr;
    integer c;

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
            row_cnt    <= 0;
            col_cnt    <= 0;
            row_select <= 0;
            write_ptr  <= 0;
            valid_out  <= 0;
        end else if (valid_in) begin

            // 스트림 위치 업데이트
            if (col_cnt == W_IN-1) begin
                col_cnt <= 0;
                row_cnt <= (row_cnt == H_IN-1) ? 0 : row_cnt + 1;
                row_select <= ~row_select;
            end else begin
                col_cnt <= col_cnt + 1;
            end

            // write_ptr 업데이트
            write_ptr <= write_ptr + 1;

            // 현재 row buffer에 입력 저장
            for (c = 0; c < CH; c = c+1) begin
                if (row_select == 0)
                    linebuf0[c][write_ptr] <= in_data[c];
                else
                    linebuf1[c][write_ptr] <= in_data[c];
            end

            // pooling 타이밍 체크
            if (col_cnt[1:0] == 2'b11 && row_cnt[1:0] == 2'b11) begin
                valid_out <= 1;

                for (c = 0; c < CH; c = c+1) begin
                    reg signed [DATA_WIDTH-1:0] a0, a1, a2, a3;
                    reg signed [DATA_WIDTH-1:0] b0, b1, b2, b3;
                    wire [2:0] idx0 = (write_ptr - 3) & 3'b111;
                    wire [2:0] idx1 = (write_ptr - 2) & 3'b111;
                    wire [2:0] idx2 = (write_ptr - 1) & 3'b111;
                    wire [2:0] idx3 = (write_ptr    ) & 3'b111;

                    if (row_select == 0) begin
                        a0 = linebuf0[c][idx0];
                        a1 = linebuf0[c][idx1];
                        a2 = linebuf0[c][idx2];
                        a3 = linebuf0[c][idx3];

                        b0 = linebuf1[c][idx0];
                        b1 = linebuf1[c][idx1];
                        b2 = linebuf1[c][idx2];
                        b3 = linebuf1[c][idx3];
                    end else begin
                        a0 = linebuf1[c][idx0];
                        a1 = linebuf1[c][idx1];
                        a2 = linebuf1[c][idx2];
                        a3 = linebuf1[c][idx3];

                        b0 = linebuf0[c][idx0];
                        b1 = linebuf0[c][idx1];
                        b2 = linebuf0[c][idx2];
                        b3 = linebuf0[c][idx3];
                    end

                    out_data[c] <= max4( max4(a0,a1,a2,a3),
                                         max4(b0,b1,b2,b3),
                                         0, 0 );
                end
            end else begin
                valid_out <= 0;
            end
        end else begin
            valid_out <= 0;
        end
    end

endmodule
