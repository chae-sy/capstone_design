`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
//
// Create Date: 2025/05/18
// Design Name: 
// Module Name: f_buffer_v1 (fixed & cleaned)
// 
// Description: 
//   INPUT_BUFFER (f_buffer_v0) 을 루프와 3차원 배열로 리팩토링한 버전.
//   ? 초기 로딩 (3×3 데이터): is_initial=1 일 때 9 사이클 동안 데이터를 채운 뒤 f_buffer_done=1
//   ? 이후 로딩 (shift + 3개 데이터): is_initial=0 이면 왼쪽으로 shift 후 마지막 열에 3개 행(row) 씩 데이터 로드
//   ? rden=1 이면, 3×3 데이터(tap)를 순차적으로 data_out 으로 출력 (각 채널마다 DATA_WIDTH)
//////////////////////////////////////////////////////////////////////////////////

module f_buffer_v1 #(
    parameter WIDTH_FSRAM_WL  = 128,   // SRAM에서 한 번에 읽어오는 비트 폭 (예: 8bit×16채널=128)
    parameter DATA_WIDTH      = 8,     // 한 채널당 데이터 폭
    parameter NUM_CHNL        = 16,    // 채널 수
    parameter SIZE_BUFFER_H   = 3,     // 버퍼 세로 크기 (행 개수)
    parameter SIZE_BUFFER_W   = 4,     // 버퍼 가로 크기 (열 개수)
    parameter SIZE_KERNEL_H   = 3,     // 커널 세로 크기 (예: 3)
    parameter SIZE_KERNEL_W   = 3      // 커널 가로 크기 (예: 3)
)(
    input                                 clk,
    input                                 rst_n,
    input                                 is_initial,     // 초기 로드 플래그 (3×3 로드)
    input                                 wren,           // 외부에서 feature 로드 허용
    input                                 rden,           // 데이터 출력 허용
    input  [WIDTH_FSRAM_WL-1:0]           data_in,        // SRAM 에서 읽어온 128bit
    output reg [DATA_WIDTH*NUM_CHNL-1:0]  data_out,       // (출력) 각 채널별로 8bit씩 묶음
    output reg                            f_buffer_done   // 리턴: 초기 또는 후속 로드/출력이 끝났음을 알림
);

    //================================================================
    // 1) 내부 버퍼: 3D 배열 선언 (SIZE_BUFFER_H × SIZE_BUFFER_W × NUM_CHNL)
    //    → buffer_data[row][col][channel]
    //================================================================
    reg [DATA_WIDTH-1:0] buffer_data [0:SIZE_BUFFER_H-1][0:SIZE_BUFFER_W-1][0:NUM_CHNL-1];

    //================================================================
    // 2) 읽기/쓰기 카운터: 초기 로딩, 후속 로딩, 출력 시 각각 따로 관리
    //    load_cnt : initial=1이면 0~8 (3×3), is_initial=0 이면 0~(SIZE_BUFFER_H-1)
    //    out_cnt  : 0~(SIZE_KERNEL_H*SIZE_KERNEL_W-1) 동안 tap 출력
    //================================================================
    reg [5:0] load_cnt;  // 최대 9 또는 3 까지 카운팅(6비트면 충분)
    reg [5:0] out_cnt;   // 최대 9까지 카운팅

    //================================================================
    // 3) data_in を NUM_CHNL × DATA_WIDTH 로 분리하기 위한 wire 배열
    //    (예: 128bit → 16채널 × 8bit)
    //================================================================
    wire [DATA_WIDTH-1:0] f_data [0:NUM_CHNL-1];
    genvar a;
    generate
        for (a = 0; a < NUM_CHNL; a = a + 1) begin : GEN_UNPACK
            assign f_data[a] = data_in[DATA_WIDTH*a +: DATA_WIDTH];
        end
    endgenerate

    // 정수 반복문용 변수
    integer i, r, c;

    //================================================================
    // 4) 메인 always 블록: 리셋, 쓰기(wren), 읽기(rden) 순으로 분기
    //================================================================
 always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // ─── 1) RESET ────────────────────────────────────────────
        for (r = 0; r < SIZE_BUFFER_H; r = r + 1) begin
            for (c = 0; c < SIZE_BUFFER_W; c = c + 1) begin
                for (i = 0; i < NUM_CHNL; i = i + 1) begin
                    buffer_data[r][c][i] <= {DATA_WIDTH{1'b0}};
                end
            end
        end
        load_cnt       <= 0;
        out_cnt        <= 0;
        data_out       <= {DATA_WIDTH*NUM_CHNL{1'b0}};
        f_buffer_done  <= 1'b0;

    end else begin
        // ─── 2) 기본: 매 사이클 done 신호는 0에서 시작 ──────────
        f_buffer_done <= 1'b0;

        // ─── 3) 쓰기 로직 (wren) ─────────────────────────────────
        if (wren) begin
            // 읽기 카운터는 쓰기 시 리셋하지 않는다 (동시 동작 허용)
            // → out_cnt, data_out은 읽기 로직에서 따로 처리

            if (is_initial) begin
                // 초기 로드 모드: 3×3 윈도우를 채운다
                if (load_cnt < SIZE_KERNEL_H * SIZE_KERNEL_W) begin
                    // buffer_data[row][col][i] <= f_data[i];
                    for (i = 0; i < NUM_CHNL; i = i + 1) begin
                        buffer_data[ load_cnt / SIZE_KERNEL_W ]
                                    [ load_cnt % SIZE_KERNEL_W ]
                                    [ i ] <= f_data[i];
                    end
                    load_cnt <= load_cnt + 1;
                end
                if (load_cnt == (SIZE_KERNEL_H * SIZE_KERNEL_W - 1)) begin
                    f_buffer_done <= 1'b1;
                    load_cnt       <= 0;
                end

            end else begin
                // 후속 로드 모드: 왼쪽으로 shift + 마지막 열에 한 행씩 삽입
                if (load_cnt == 0) begin
                    // 한 사이클에 한 번만 shift 수행
                    for (r = 0; r < SIZE_BUFFER_H; r = r + 1) begin
                        for (c = 0; c < SIZE_BUFFER_W-1; c = c + 1) begin
                            for (i = 0; i < NUM_CHNL; i = i + 1) begin
                                buffer_data[r][c][i] <= buffer_data[r][c+1][i];
                            end
                        end
                    end
                end

                if (load_cnt < SIZE_BUFFER_H) begin
                    // 마지막 열(col=SIZE_BUFFER_W-1)에 새 데이터 삽입
                    for (i = 0; i < NUM_CHNL; i = i + 1) begin
                        buffer_data[ load_cnt ]
                                    [ SIZE_BUFFER_W-1 ]
                                    [ i ] <= f_data[i];
                    end
                end
                if (load_cnt == (SIZE_BUFFER_H - 1)) begin
                    // SIZE_BUFFER_H개 삽입이 끝나면 완료
                    f_buffer_done <= 1'b1;
                    load_cnt       <= 0;
                end else begin
                    load_cnt <= load_cnt + 1;
                end
            end

        end else begin
            // wren이 0이면, 쓰기 카운터를 0으로 유지
            load_cnt <= 0;
        end


        // ─── 4) 읽기 로직 (rden) ─────────────────────────────────
        if (rden) begin
            // 쓰기 시 load_cnt가 영향을 안 받도록 이미 분리함

            if (out_cnt < SIZE_KERNEL_H * SIZE_KERNEL_W) begin
                // buffer_data[row][col][i] → data_out
                for (i = 0; i < NUM_CHNL; i = i + 1) begin
                    data_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] <=
                        buffer_data[ out_cnt / SIZE_KERNEL_W ]
                                    [ out_cnt % SIZE_KERNEL_W ]
                                    [ i ];
                end
            end

            if (out_cnt == (SIZE_KERNEL_H * SIZE_KERNEL_W - 1)) begin
                f_buffer_done <= 1'b1;
                out_cnt       <= 0;
            end else begin
                out_cnt <= out_cnt + 1;
            end

        end else begin
            // rden이 0이면, 읽기 카운터와 출력값은 초기화
            out_cnt  <= 0;
            data_out <= {DATA_WIDTH*NUM_CHNL{1'b0}};
        end

        // ─── 5) wren=0, rden=0 일 때도 해야 할 별도 동작 없다 ───
        // (이미 load_cnt와 out_cnt가 각각 reset 블록에서 처리됨)

    end
end


endmodule
