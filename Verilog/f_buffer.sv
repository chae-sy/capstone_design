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
//   • 초기 로딩 (3×3 데이터): is_initial=1 일 때 9 사이클 동안 데이터를 채운 뒤 f_buffer_done=1
//   • 이후 로딩 (shift + 3개 데이터): is_initial=0 이면 왼쪽으로 shift 후 마지막 열에 3개 행(row) 씩 데이터 로드
//   • rden=1 이면, 3×3 데이터(tap)를 순차적으로 data_out 으로 출력 (각 채널마다 DATA_WIDTH)
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
            // ─── 4.1) RESET 섹션 ─────────────────────────────────────────
            // buffer_data 전부 0으로 초기화
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
            // ─── 4.2) 기본값: 매 클럭마다 완료 신호 끄기 (동작 끝날 때만 1로 올림) ───
            f_buffer_done <= 1'b0;

            // ─── 4.3) 쓰기 동작 (wren=1) ─────────────────────────────────────────
            if (wren) begin
                // 출력 카운터는 쓰기 시점에 항상 0으로 리셋
                out_cnt <= 0;

                // ----- ① 초기 로딩 (is_initial=1) → 3×3 데이터 채우기 -----
                if (is_initial) begin
                    // load_cnt 가 0..8 (총 9) 일 때만 데이터를 채움
                    if (load_cnt < SIZE_KERNEL_H * SIZE_KERNEL_W) begin
                        // 행(row) = load_cnt / SIZE_KERNEL_W
                        // 열(col) = load_cnt % SIZE_KERNEL_W
                        for (i = 0; i < NUM_CHNL; i = i + 1) begin
                            buffer_data[ load_cnt / SIZE_KERNEL_W ]
                                        [ load_cnt % SIZE_KERNEL_W ]
                                        [ i ] <= f_data[i];
                        end

                        // 마지막 카운트가 채워질 때 f_buffer_done=1
                        if (load_cnt == (SIZE_KERNEL_H * SIZE_KERNEL_W - 1)) begin
                            f_buffer_done <= 1'b1;
                        end

                        // 카운터 증가
                        load_cnt <= load_cnt + 1;
                    end

                end else begin
                    // ----- ② 후속 로딩 (is_initial=0) → 왼쪽(열)으로 shift + 마지막 열에 SIZE_BUFFER_H개 삽입 -----
                    // load_cnt==0 이면 한 번만 shift 를 수행
                    if (load_cnt == 0) begin
                        // 모든 행(r)에 대해, col=0~(SIZE_BUFFER_W-2) ← col=1~(SIZE_BUFFER_W-1) 복사
                        for (r = 0; r < SIZE_BUFFER_H; r = r + 1) begin
                            for (c = 0; c < SIZE_BUFFER_W-1; c = c + 1) begin
                                for (i = 0; i < NUM_CHNL; i = i + 1) begin
                                    buffer_data[r][c][i] <= buffer_data[r][c+1][i];
                                end
                            end
                        end
                    end

                    // load_cnt 가 0..(SIZE_BUFFER_H-1) 동안 마지막 열(col=SIZE_BUFFER_W-1)에 새 데이터 삽입
                    if (load_cnt < SIZE_BUFFER_H) begin
                        for (i = 0; i < NUM_CHNL; i = i + 1) begin
                            buffer_data[ load_cnt ]
                                        [ SIZE_BUFFER_W-1 ]
                                        [ i ] <= f_data[i];
                        end

                        // 마지막 행이 채워질 때 f_buffer_done=1
                        if (load_cnt == (SIZE_BUFFER_H - 1)) begin
                            f_buffer_done <= 1'b1;
                        end

                        // 카운터 증가
                        load_cnt <= load_cnt + 1;
                    end else begin
                        // SIZE_BUFFER_H 만큼 다 채운 뒤에는 카운터만 0으로 리셋 (다음 wren 에 대비)
                        load_cnt <= 0;
                    end
                end

            // ─── 4.4) 읽기 동작 (rden=1) ─────────────────────────────────────────
            end else if (rden) begin
                // 쓰기 카운터는 읽기 시 리셋
                load_cnt <= 0;

                // out_cnt 가 0..(3×3-1) 동안 순차적으로 tap 값 뽑기
                if (out_cnt < SIZE_KERNEL_H * SIZE_KERNEL_W) begin
                    // 예를 들어 3×3 윈도우 (SIZE_KERNEL_H=3, SIZE_KERNEL_W=3)
                    // 행(row)   = out_cnt / SIZE_KERNEL_W
                    // 열(col)   = out_cnt % SIZE_KERNEL_W
                    for (i = 0; i < NUM_CHNL; i = i + 1) begin
                        data_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] 
                            <= buffer_data[ out_cnt / SIZE_KERNEL_W ]
                                          [ out_cnt % SIZE_KERNEL_W ]
                                          [ i ];
                    end

                    // 마지막 카운트가 출력될 때 f_buffer_done=1
                    if (out_cnt == (SIZE_KERNEL_H * SIZE_KERNEL_W - 1)) begin
                        f_buffer_done <= 1'b1;
                        out_cnt      <= 0;
                    end else begin
                        // 아직 다 뽑아내지 않았으면 카운터만 증가
                        out_cnt <= out_cnt + 1;
                    end
                end

            // ─── 4.5) wren=0 & rden=0 → 버퍼 유지, 출력은 0 ──────────────────────────
            end else begin
                load_cnt  <= 0;
                out_cnt   <= 0;
                data_out  <= {DATA_WIDTH*NUM_CHNL{1'b0}};
            end

        end // rst_n else
    end // always

endmodule
