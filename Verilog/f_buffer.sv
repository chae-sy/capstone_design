`timescale 1ns / 1ps
module f_buffer #(
    parameter DATA_WIDTH      = 8,     // 한 채널당 데이터 폭
    parameter NUM_CHNL        = 16,    // 채널 수
    parameter SIZE_BUFFER_H   = 3,     // 버퍼 세로 크기
    parameter SIZE_BUFFER_W   = 4,     // 버퍼 가로 크기
    parameter SIZE_KERNEL_H   = 3,     // 커널 세로 크기
    parameter SIZE_KERNEL_W   = 3      // 커널 가로 크기
)(
    input  wire                                clk,
    input  wire                                rst_n,
    input  wire                                is_initial,   // 초기 로드 플래그
    input  wire                                wren,         // 쓰기 enable
    input  wire                                rden,         // 읽기 enable
    input  wire [NUM_CHNL*DATA_WIDTH-1:0]      data_in,      // 한 번에 NUM_CHNL * DATA_WIDTH 입력
    output reg  [NUM_CHNL*DATA_WIDTH-1:0]      data_out,     // NUM_CHNL * DATA_WIDTH 출력
    output reg                                 f_buffer_done // 완료 펄스
);

    // 2D 배열 선언, 각 요소는 NUM_CHNL*DATA_WIDTH 비트
    reg [NUM_CHNL*DATA_WIDTH-1:0] buffer_data [0:SIZE_BUFFER_H-1][0:SIZE_BUFFER_W-1];

    reg [5:0] load_cnt;
    reg [5:0] out_cnt, out_cnt_n;
    reg write_done, read_done;

    integer r, c;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (r = 0; r < SIZE_BUFFER_H; r = r + 1)
                for (c = 0; c < SIZE_BUFFER_W; c = c + 1)
                    buffer_data[r][c] <= {NUM_CHNL*DATA_WIDTH{1'b0}};
            load_cnt      <= 0;
            out_cnt       <= 0;
            data_out      <= {NUM_CHNL*DATA_WIDTH{1'b0}};
            f_buffer_done <= 1'b0;
             write_done <= 1'b0;
        end else begin
            f_buffer_done <= 1'b0;
            out_cnt <= out_cnt_n;
            // 쓰기 로직
            if (wren) begin
                if (is_initial) begin
                    if (load_cnt < SIZE_KERNEL_H * SIZE_KERNEL_W) begin
                        buffer_data[ load_cnt / SIZE_KERNEL_W ][ load_cnt % SIZE_KERNEL_W ] <= data_in;
                        load_cnt <= load_cnt + 1;
                    end
                    if (load_cnt == (SIZE_KERNEL_H * SIZE_KERNEL_W - 1)) begin
                        f_buffer_done <= 1'b1;
                        load_cnt <= 0;
                    end
                end else begin
                    if (load_cnt < SIZE_BUFFER_H) begin
                        buffer_data[ load_cnt ][ SIZE_BUFFER_W-1 ] <= data_in;
                        load_cnt <= load_cnt + 1;
                    end
                    if (load_cnt == (SIZE_BUFFER_H - 1)) begin
                        f_buffer_done <= 1'b1;
                        write_done <= 1'b1;
                        load_cnt <= 0;
                    end
                end
            end else begin
                load_cnt <= 0;
            end
            
            // shift
            if (write_done & read_done)begin
                for (r = 0; r < SIZE_BUFFER_H; r = r + 1) begin
                    for (c = 0; c < SIZE_BUFFER_W-1; c = c + 1) begin
                        buffer_data[r][c] <= buffer_data[r][c+1];
                    end
                end
                write_done <= 1'b0;
            end
        end
    end
    always_comb begin
        out_cnt_n = out_cnt;
        read_done = 1'b0;
        if (rden) begin
            if (out_cnt < SIZE_KERNEL_H * SIZE_KERNEL_W) begin
                data_out = buffer_data[ out_cnt / SIZE_KERNEL_W ][ out_cnt % SIZE_KERNEL_W ];
                out_cnt_n = out_cnt + 1;
            end
            if (out_cnt == (SIZE_KERNEL_H * SIZE_KERNEL_W - 1)) begin
                read_done = 1'b1;
                out_cnt_n = 0;
            end
        end else begin
            out_cnt_n  = 0;
            data_out = {NUM_CHNL*DATA_WIDTH{1'b0}};
        end
    end

endmodule
