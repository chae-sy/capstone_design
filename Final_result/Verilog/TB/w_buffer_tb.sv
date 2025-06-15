`timescale 1ns / 1ps

module w_buffer_tb;

    //================================================================
    // 1) 파라미터 (f_buffer_v1 모듈과 동일하게 맞춰야 함)
    //================================================================
    parameter WIDTH_FSRAM_WL  = 128;
    parameter DATA_WIDTH      = 8;
    parameter NUM_CHNL        = 16;
    parameter SIZE_BUFFER_H   = 3;
    parameter SIZE_BUFFER_W   = 3;
    parameter SIZE_KERNEL_H   = 3;
    parameter SIZE_KERNEL_W   = 3;

    //================================================================
    // 2) 테스트벤치 신호 선언
    //================================================================
    reg                                 clk;
    reg                                 rst_n;
    reg                                 wren;
    reg                                 rden;
    reg  [WIDTH_FSRAM_WL-1:0]           data_in;
    wire [DATA_WIDTH*NUM_CHNL-1:0]      data_out;
    wire                                w_buffer_done;

    //================================================================
    // 3) DUT 인스턴스화
    //================================================================
    w_buffer #(
        .WIDTH_FSRAM_WL  (WIDTH_FSRAM_WL),
        .DATA_WIDTH      (DATA_WIDTH),
        .NUM_CHNL        (NUM_CHNL),
        .SIZE_BUFFER_H   (SIZE_BUFFER_H),
        .SIZE_BUFFER_W   (SIZE_BUFFER_W),
        .SIZE_KERNEL_H   (SIZE_KERNEL_H),
        .SIZE_KERNEL_W   (SIZE_KERNEL_W)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .wren           (wren),
        .rden           (rden),
        .data_in        (data_in),
        .data_out       (data_out),
        .w_buffer_done  (w_buffer_done)
    );

    //================================================================
    // 4) 클럭 생성: 10ns 주기 (100MHz)에 맞춰 토글
    //================================================================
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 반주기 5ns → 주기 10ns
    end

    //================================================================
    // 5) 테스트 시나리오
    //================================================================
    initial begin
        integer idx;
        reg [DATA_WIDTH-1:0] sample_data [0:15]; // 최대 16개 채널용 샘플

        // (1) 초기화
        rst_n       = 1'b0;
        wren        = 1'b0;
        rden        = 1'b0;
        data_in     = {WIDTH_FSRAM_WL{1'b0}};
        #20;                // 2클럭 대기 (20ns)
        rst_n       = 1'b1; // 리셋 해제
        #5;
        // (2) 초기 로딩(initial load)
        // wren = 1, 9 사이클 동안 sample_data를 채워넣음
        wren        = 1'b1;
        for (idx = 0; idx < SIZE_KERNEL_H*SIZE_KERNEL_W; idx = idx + 1) begin
            // 각 사이클마다 채울 16채널 샘플 데이터를 준비 (예: idx 값 반복)
            // sample_data[a] = idx + a  (임의 패턴)
             @(posedge clk);
            for (int a = 0; a < NUM_CHNL; a = a + 1) begin
                sample_data[a] = idx + a;
            end
            // 16채널을 128bit로 패킹
            data_in = { 
                sample_data[15], sample_data[14], sample_data[13], sample_data[12],
                sample_data[11], sample_data[10], sample_data[9],  sample_data[8],
                sample_data[7],  sample_data[6],  sample_data[5],  sample_data[4],
                sample_data[3],  sample_data[2],  sample_data[1],  sample_data[0]
            };

            // 매 클럭마다 데이터 입력
            // w_buffer_done이 1로 올라오는 시점을 관찰
            if (w_buffer_done) begin
                $display("[%0t] Initial load done at idx=%0d", $time, idx);
            end
        end
        
        // 초기 로드 종료 후 한 클럭 더 유지
        @(posedge clk);
        wren = 1'b0;
        

        // (5) 후속 3×3 출력(read-out) 테스트
        rden = 1'b1;
        for (idx = 0; idx < SIZE_KERNEL_H * SIZE_KERNEL_W; idx = idx + 1) begin
            @(posedge clk);
            $display("[%0t] Post-shift Read-out [%0d]: data_out = %h, w_buffer_done = %b", 
                      $time, idx, data_out, w_buffer_done);
        end
        rden = 1'b0;
        @(posedge clk);
        
        // (5-2) rden, wden 동시에 해보기
        rden = 1'b1;
        wren=1'b1;
        for (idx = 0; idx < SIZE_KERNEL_H*SIZE_KERNEL_W; idx = idx + 1) begin
            // 각 사이클마다 채울 16채널 샘플 데이터를 준비 (예: idx 값 반복)
            // sample_data[a] = idx + a  (임의 패턴) 
            @(posedge clk);
            for (int a = 0; a < NUM_CHNL; a = a + 1) begin
                sample_data[a] = 100+ idx + a;
            end
            // 16채널을 128bit로 패킹
            data_in = { 
                sample_data[15], sample_data[14], sample_data[13], sample_data[12],
                sample_data[11], sample_data[10], sample_data[9],  sample_data[8],
                sample_data[7],  sample_data[6],  sample_data[5],  sample_data[4],
                sample_data[3],  sample_data[2],  sample_data[1],  sample_data[0]
            };

            // 매 클럭마다 데이터 입력
           
            // w_buffer_done이 1로 올라오는 시점을 관찰
            if (w_buffer_done) begin
                $display("[%0t] Initial load done at idx=%0d", $time, idx);
            end
        end
        rden=1'b0;
        wren=1'b0;
        // (6) 시뮬레이션 종료
        #20;
        $display("[%0t] Testbench finished.", $time);
        $stop;
    end

endmodule
