`timescale 1ns / 1ps

module regfile_sync #(
    parameter BITWIDTH            = 32,                        // 각 워드의 비트 폭
    parameter NUM_WORD            = 16,                        // 워드 개수
    parameter LAYER_1_NUM_WORD    = 2,                         // layer_num == 1일 때만 사용할 워드 개수
    parameter DATA_WIDTH          = NUM_WORD * BITWIDTH,       // 전체 메모리 폭
    parameter ADDR_WIDTH          = 4                          // 2^4 = 16 개의 워드
)(
    input  wire                     clk,        // 클록
    input  wire                     rst_n,      // 비동기 리셋 (low active)
    // --- Write Port ---
    input  wire                     we,         // write enable
    input  wire [ADDR_WIDTH-1:0]    waddr,      // write address
    input  wire [DATA_WIDTH-1:0]    wdata,      // write data (한 번에 NUM_WORD * BITWIDTH를 써넣음)
    // --- Read Port ---
    input  wire [ADDR_WIDTH-1:0]    raddr,      // read address (동기식으로 캡처)
    input  wire                     rden,       // read enable: 이 신호가 1로 전이(0→1)될 때마다 한 워드를 읽음
    output wire [BITWIDTH-1:0]      rdata,      // 최종 슬라이싱된 한 워드 (BITWIDTH 폭)
    // --- layer num ---
    input  wire [2:0]               layer_num
);

    //======================================================================
    // 1) 내부 메모리 선언
    //======================================================================
    localparam DEPTH = (1 << ADDR_WIDTH);
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    integer i;

    //----------------------------------------------------------------------
    // 2) 동기식 쓰기 (Write)
    //----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1) begin
                mem[i] <= {DATA_WIDTH{1'b0}};
            end
        end else if (we) begin
            mem[waddr] <= wdata;
        end
    end

    //----------------------------------------------------------------------
    // 3) 읽기 주소 레지스터 캡처 (Read Address Register)
    //----------------------------------------------------------------------
    reg [ADDR_WIDTH-1:0] raddr_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            raddr_reg <= {ADDR_WIDTH{1'b0}};
        end else begin
            raddr_reg <= raddr;
        end
    end

    //----------------------------------------------------------------------
    // 4) rden 엣지 감지용: 이전 클록의 rden 값을 저장
    //----------------------------------------------------------------------
    reg prev_rden;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prev_rden <= 1'b0;
        end else begin
            prev_rden <= rden;
        end
    end

    //----------------------------------------------------------------------
    // 5) "rden이 0→1으로 전이된 순간"에만 read_buf에 메모리 로드
    //----------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] read_buf;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_buf <= {DATA_WIDTH{1'b0}};
        end 
        // rden이 0→1로 바뀌는 순간: (rden=1 & prev_rden=0)
        else if (rden && !prev_rden) begin
            read_buf <= mem[raddr_reg];
        end
        // 그 외(= rden 계속 1이거나, 혹은 rden=0인 상태)에는 버퍼 유지를 함
    end

    //----------------------------------------------------------------------
    // 6) "rden이 0→1 전이된 순간"마다 cnt를 0→1→…→max-1→0 순으로 증가시키기
    //----------------------------------------------------------------------
    reg [3:0] cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
        if (layer_num == 1) begin
            cnt <= LAYER_1_NUM_WORD - 1;
            end
        else begin
        cnt <= NUM_WORD - 1;
        end
        end 
        // rden이 0→1로 바뀔 때마다 cnt를 +1 혹은 wrap-around
        else if (rden && !prev_rden) begin
            if (layer_num == 1) begin
                // layer_num==1일 때는 LAYER_1_NUM_WORD 개수만큼만 순환
                if (cnt == (LAYER_1_NUM_WORD - 1))
                    cnt <= 4'd0;
                else
                    cnt <= cnt + 4'd1;
            end else begin
                // layer_num!=1일 때는 NUM_WORD 개수만큼만 순환
                if (cnt == (NUM_WORD - 1))
                    cnt <= 4'd0;
                else
                    cnt <= cnt + 4'd1;
            end
        end
        // rden이 1이 아니거나, rden이 1인데 바로 전 클록엔 prev_rden도 1인 경우(cnt 증분 조건 아님) → cnt 유지
    end

    //----------------------------------------------------------------------
    // 7) 슬라이싱된 rdata 출력
    //    read_buf에 저장된 DATA_WIDTH 폭 전체에서
    //    cnt*BITWIDTH 위치부터 BITWIDTH 폭만큼 잘라서 내보냄
    //----------------------------------------------------------------------
    assign rdata = read_buf[ cnt * BITWIDTH +: BITWIDTH ];

endmodule
