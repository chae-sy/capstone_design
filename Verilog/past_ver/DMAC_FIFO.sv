module DMAC_FIFO #(
    parameter       DEPTH_LG2           = 4,      // FIFO 깊이를 2^DEPTH_LG2로 설정 (기본값: 16)
    parameter       DATA_WIDTH          = 32      // 데이터 폭 (기본값: 32비트)
    )
(
    input   wire                        clk,      // 클럭 입력
    input   wire                        rst_n,    // 비동기 리셋 (active-low)

    output  wire                        full_o,   // FIFO가 가득 찼을 때 1로 올라감
    input   wire                        wren_i,   // 쓰기 enable 신호 (1이면 wdata_i를 FIFO에 쓰려 함)
    input   wire    [DATA_WIDTH-1:0]    wdata_i,  // FIFO에 쓸 데이터 입력

    output  wire                        empty_o,  // FIFO가 비었을 때 1로 올라감
    input   wire                        rden_i,   // 읽기 enable 신호 (1이면 rdata_o로 데이터 읽기를 시도)
    output  wire    [DATA_WIDTH-1:0]    rdata_o   // FIFO에서 읽어온 데이터 출력
);

    // 실제 FIFO 깊이 = 2^DEPTH_LG2
    localparam  FIFO_DEPTH              = (1 << DEPTH_LG2);

    // FIFO 내부 메모리: DATA_WIDTH 폭을 가지는 2^DEPTH_LG2 개의 레지스터
    reg     [DATA_WIDTH-1:0]            data[FIFO_DEPTH-1:0];

    // 현재 상태(full, empty)와 다음 상태(full_n, empty_n) 플래그
    reg                                 full,       full_n,
                                        empty,      empty_n;
    // 읽기/쓰기 포인터: (DEPTH_LG2+1) 비트
    // MSB가 넘어가면 full/empty 판단에 사용
    reg     [DEPTH_LG2:0]               wrptr,      wrptr_n,  // 쓰기 포인터
                                        rdptr,      rdptr_n;  // 읽기 포인터

    // -------------------------------------------------------------
    // 1) 동기식 리셋 및 상태(포인터, 플래그) 업데이트 블록
    //    - posedge clk 에서 rst_n이 0이면 모든 것을 초기화
    //    - 그렇지 않으면 다음 상태 값(full_n 등)을 현재 상태에 할당
    //    - wren_i가 1이면 data 메모리에 wdata_i를 저장
    // -------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            // 리셋 시: FIFO는 비어있는 상태
            full                        <= 1'b0;  // 가득 차지 않음
            empty                       <= 1'b1;  // 비어 있음

            // 포인터도 0으로 초기화
            wrptr                       <= {(DEPTH_LG2+1){1'b0}};
            rdptr                       <= {(DEPTH_LG2+1){1'b0}};

            // 데이터 메모리도 모두 0으로 초기화 (선택 사항이지만, 시뮬레이션 안정성을 위해)
            for (int i = 0; i < FIFO_DEPTH; i++) begin
                data[i]                     <= {DATA_WIDTH{1'b0}};
            end
        end
        else begin
            // 리셋이 아닐 때는 다음 상태 값을 현재 상태로 업데이트
            full                        <= full_n;
            empty                       <= empty_n;
            wrptr                       <= wrptr_n;
            rdptr                       <= rdptr_n;

            // 쓰기 요청이 들어오고 FIFO가 가득 차 있지 않으면 실제로 메모리에 저장
            if (wren_i && !full) begin
                // ***************************************************
                // 이 부분이 핵심:
                // wrptr[DEPTH_LG2-1:0] 는 “쓰기 포인터의 하위 DEPTH_LG2 비트”만
                // 주소로 사용한다는 의미.
                // wrptr는 DEPTH_LG2+1 비트이므로 MSB(가장 상위 비트)는
                // full/empty 판단용 순환 표시 비트로 쓰이고,
                // 나머지 하위 비트들만 실제 메모리 인덱스로 사용된다.
                // -> 예: DEPTH_LG2=4라면 wrptr는 5비트. wrptr[3:0]은 실제
                //    16개 깊이 메모리의 주소(0~15)로 사용.
                // ***************************************************
                data[wrptr[DEPTH_LG2-1:0]]  <= wdata_i;
            end
        end
    end

    // -------------------------------------------------------------
    // 2) 조합 논리로 이루어진 “다음 상태 연산” 블록
    //    - wrptr_n, rdptr_n, full_n, empty_n 계산
    //    - wren_i, rden_i, 현재 full/empty 플래그를 기준으로 결정
    // -------------------------------------------------------------
    always_comb begin
        // 기본적으로 “변경 전 상태”를 복사해 둠
        wrptr_n                     = wrptr;
        rdptr_n                     = rdptr;

        // 쓰기 동작: wren_i=1 이고 FIFO가 가득 차 있지 않으면
        if (wren_i && !full) begin
            // 다음 쓰기 포인터 = 현재 쓰기 포인터 + 1
            wrptr_n                     = wrptr + 1;
        end

        // 읽기 동작: rden_i=1 이고 FIFO가 비어 있지 않으면
        if (rden_i && !empty) begin
            // 다음 읽기 포인터 = 현재 읽기 포인터 + 1
            rdptr_n                     = rdptr + 1;
        end

        // empty_n: 다음 쓰기 포인터와 다음 읽기 포인터가 같으면 비어있는 상태
        // (읽기 포인터가 쓰기 포인터를 따라잡았을 때)
        empty_n                     = (wrptr_n == rdptr_n);

        // full_n: 쓰기 포인터의 MSB가 읽기 포인터의 MSB와 다르고,
        //         하위 DEPTH_LG2 비트가 같으면 가득 찬 상태
        //  - MSB: 순환 여부를 나타내므로, 한 바퀴를 돌았다는 의미
        //  - 하위 비트: 실제 위치가 같다는 의미 (write와 read가 같은 인덱스에서 만나면)
        full_n                      = (wrptr_n[DEPTH_LG2] != rdptr_n[DEPTH_LG2]) &&
                                     (wrptr_n[DEPTH_LG2-1:0] == rdptr_n[DEPTH_LG2-1:0]);
    end

    // 출력 포트 연결
    assign  full_o                      = full;
    assign  empty_o                     = empty;
    // 읽기 데이터는 항상 “현재 rdptr 하위 DEPTH_LG2 비트” 위치에서 가져옴
    assign  rdata_o                     = data[rdptr[DEPTH_LG2-1:0]];

endmodule
