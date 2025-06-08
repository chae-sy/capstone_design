`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
//
// Create Date: 2025/05/18
// Design Name: 
// Module Name: w_buffer_v1 (fixed & cleaned)
// 
// Description: 
//   INPUT_BUFFER (w_buffer_v0) 을 루프와 3차원 배열로 리팩토링한 버전.
//   ? 초기 로딩 (3×3 데이터): is_initial=1 일 때 9 사이클 동안 데이터를 채운 뒤 w_buffer_done=1
//   ? 이후 로딩 (shift + 3개 데이터): is_initial=0 이면 왼쪽으로 shift 후 마지막 열에 3개 행(row) 씩 데이터 로드
//   ? rden=1 이면, 3×3 데이터(tap)를 순차적으로 data_out 으로 출력 (각 채널마다 DATA_WIDTH)
//////////////////////////////////////////////////////////////////////////////////

module w_buffer #(
    parameter WIDTH_FSRAM_WL  = 128,   // SRAM에서 한 번에 읽어오는 비트 폭 (예: 8bit×16채널=128)
    parameter DATA_WIDTH      = 8,     // 한 채널당 데이터 폭
    parameter NUM_CHNL        = 16,    // 채널 수
    parameter SIZE_BUFFER_H   = 3,     // 버퍼 세로 크기 (행 개수)
    parameter SIZE_BUFFER_W   = 3,     // 버퍼 가로 크기 (열 개수)
    parameter SIZE_KERNEL_H   = 3,     // 커널 세로 크기 (예: 3)
    parameter SIZE_KERNEL_W   = 3      // 커널 가로 크기 (예: 3)
)(
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire                           wren,           // 외부에서 feature 로드 허용
    input  wire                           rden,           // 데이터 출력 허용
    input  wire [WIDTH_FSRAM_WL-1:0]      data_in,        // SRAM 에서 읽어온 128bit
    output reg [DATA_WIDTH*NUM_CHNL-1:0]  data_out,       // (출력) 각 채널별로 8bit씩 묶음
    output reg                            w_buffer_done   // 리턴: 초기 또는 후속 로드/출력이 끝났음을 알림
);

    //================================================================
    // 1) 내부 버퍼: 3D 배열 선언 (SIZE_BUFFER_H × SIZE_BUFFER_W × NUM_CHNL)
    //    → buffer_data[row][col][channel]
    //================================================================
    reg [NUM_CHNL*DATA_WIDTH-1:0] buffer_data [0:SIZE_BUFFER_H-1][0:SIZE_BUFFER_W-1];
    reg [WIDTH_FSRAM_WL-1:0] data_in_reg;

    //================================================================
    // 2) 읽기/쓰기 카운터: 초기 로딩, 후속 로딩, 출력 시 각각 따로 관리
    //    load_cnt : initial=1이면 0~8 (3×3), is_initial=0 이면 0~(SIZE_BUFFER_H-1)
    //    out_cnt  : 0~(SIZE_KERNEL_H*SIZE_KERNEL_W-1) 동안 tap 출력
    //================================================================
    reg [5:0] load_cnt;  // 최대 9 또는 3 까지 카운팅(6비트면 충분)
    reg [5:0] out_cnt, out_cnt_n;   // 최대 9까지 카운팅
    reg wren_d;
    // 정수 반복문용 변수
    integer r, c;

    //================================================================
    // 4) 메인 always 블록: 리셋, 쓰기(wren), 읽기(rden) 순으로 분기
    //================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (r = 0; r < SIZE_BUFFER_H; r = r + 1)
                for (c = 0; c < SIZE_BUFFER_W; c = c + 1)
                    buffer_data[r][c] <= {NUM_CHNL*DATA_WIDTH{1'b0}};
            load_cnt      <= 0;
            out_cnt       <= 0;
            data_out      <= {NUM_CHNL*DATA_WIDTH{1'b0}};
            w_buffer_done  <= 1'b0;
        end else begin
            w_buffer_done  <= 1'b0;
            out_cnt <= out_cnt_n;
            wren_d <= wren;
            if (wren) begin
                data_in_reg <= data_in;
            end
            // 쓰기 로직
            if (wren_d) begin
                if (load_cnt < SIZE_KERNEL_H * SIZE_KERNEL_W) begin
                    buffer_data[ load_cnt / SIZE_KERNEL_W ][ load_cnt % SIZE_KERNEL_W ] <= data_in_reg;
                    load_cnt <= load_cnt + 1;
                end
                if (load_cnt == (SIZE_KERNEL_H * SIZE_KERNEL_W - 1)) begin
                    w_buffer_done  <= 1'b1;
                    load_cnt <= 0;
                end
            end else begin
                load_cnt <= 0;
            end
            
        end
    end
    
    always_comb begin
        out_cnt_n = out_cnt;
        if (rden) begin
            if (out_cnt < SIZE_KERNEL_H * SIZE_KERNEL_W) begin
                data_out = buffer_data[ out_cnt / SIZE_KERNEL_W ][ out_cnt % SIZE_KERNEL_W ];
                out_cnt_n = out_cnt + 1;
            end
            if (out_cnt == (SIZE_KERNEL_H * SIZE_KERNEL_W - 1)) begin
                out_cnt_n = 0;
            end
        end else begin
            out_cnt_n  = 0;
            data_out = {NUM_CHNL*DATA_WIDTH{1'b0}};
        end
    end

endmodule