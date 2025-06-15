`timescale 1ns / 1ps

module tb_regfile_sync;

    // Parameter 정의 (DUT와 동일하게 맞춰야 함)
    localparam BITWIDTH   = 32;
    localparam NUM_WORD   = 16;
    localparam DATA_WIDTH = NUM_WORD * BITWIDTH;
    localparam ADDR_WIDTH = 4;  // 2^4 = 16

    // Testbench 신호 선언
    reg                     clk;
    reg                     rst_n;
    reg                     we;
    reg  [ADDR_WIDTH-1:0]   waddr;
    reg  [DATA_WIDTH-1:0]   wdata;
    reg  [ADDR_WIDTH-1:0]   raddr;
    reg                     rden;
    wire [BITWIDTH-1:0]     rdata;
    reg [2:0] layer_num;
    // DUT 인스턴스화
    regfile_sync #(
        .BITWIDTH   (BITWIDTH),
        .NUM_WORD   (NUM_WORD),
        .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) DUT (
        .clk     (clk),
        .rst_n   (rst_n),
        .we      (we),
        .waddr   (waddr),
        .wdata   (wdata),
        .raddr   (raddr),
        .rden    (rden),
        .rdata   (rdata),
        .layer_num (layer_num)
    );

    // 1) 클록 생성: 10ns 주기 (5ns마다 토글)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // 2) 초기 블록: 리셋, 쓰기, 읽기 순서 제어
    initial begin
        // (1) 초기값 설정
        rst_n = 1'b0;   // 리셋 활성화 (low-active)
        we    = 1'b0;
        waddr = {ADDR_WIDTH{1'b0}};
        wdata = {DATA_WIDTH{1'b0}};
        raddr = {ADDR_WIDTH{1'b0}};
        rden  = 1'b0;
        layer_num = 3'd0;
        // (2) 리셋 유지: 20ns 동안 (2클록) 리셋
        #20;
        rst_n = 1'b1;   // 리셋 해제

        // (3) 잠시 대기 후 쓰기 동작
        #10;
        //   waddr = 0번 블록에 쓰기
        waddr = 4'd0;
        we    = 1'b1;
        layer_num = 3'd2;
        //   wdata: {word15, word14, …, word1, word0}
        //   각 워드는 32비트, 값은 0x0, 0x1, 0x2, …, 0xF
        wdata = {
            32'h0000000F,  // word15
            32'h0000000E,  // word14
            32'h0000000D,  // word13
            32'h0000000C,  // word12
            32'h0000000B,  // word11
            32'h0000000A,  // word10
            32'h00000009,  // word9
            32'h00000008,  // word8
            32'h00000007,  // word7
            32'h00000006,  // word6
            32'h00000005,  // word5
            32'h00000004,  // word4
            32'h00000003,  // word3
            32'h00000002,  // word2
            32'h00000001,  // word1
            32'h00000000   // word0
        };
        #10;            // 한 클록 주기 동안 write
        we = 1'b0;      // 쓰기 완료

        // (4) 잠시 대기 후 읽기 준비
        #20;
        raddr = 4'd0;   // 읽을 블록 주소 = 0

        // 16번의 rden 펄스를 통해 cnt = 0~15번 워드 차례대로 읽기
        repeat (16) begin
            #10;
            rden = 1'b1;
            #10;
            rden = 1'b0;
        end

        // (5) 추가로 rden=0인 상태로 몇 클록 기다려본 뒤 시뮬레이션 종료
        #20;
        
        
        // layer 1 일 때 simulation
        
        //   waddr = 1번 블록에 쓰기
        waddr = 4'd1;
        we    = 1'b1;
        layer_num = 3'd1;
        //   wdata: {word15, word14, …, word1, word0}
        //   각 워드는 32비트, 값은 0x0, 0x1, 0x2, …, 0xF
        wdata = {
            32'h00000000,  // word15
            32'h00000000,  // word14
            32'h00000000,  // word13
            32'h00000000,  // word12
            32'h00000000,  // word11
            32'h00000000,  // word10
            32'h00000000,  // word9
            32'h00000000,  // word8
            32'h00000000,  // word7
            32'h00000000,  // word6
            32'h00000000,  // word5
            32'h00000000,  // word4
            32'h00000000,  // word3
            32'h00000000,  // word2
            32'h00000012,  // word1
            32'h00000011   // word0
        };
        #10;            // 한 클록 주기 동안 write
        we = 1'b0;      // 쓰기 완료

        // (4) 잠시 대기 후 읽기 준비
        #20;
        raddr = 4'd1;   // 읽을 블록 주소 = 1
        // 16번의 rden 펄스를 통해 cnt = 0~15번 워드 차례대로 읽기
        repeat (2) begin
            #10;
            rden = 1'b1;
            #10;
            rden = 1'b0;
        end

        // (5) 추가로 rden=0인 상태로 몇 클록 기다려본 뒤 시뮬레이션 종료
        #20;
        
        $finish;
    end

    // 3) 모니터링: 시뮬레이션 시간, cnt(내부), rdata 값 출력
    //    cnt를 직접 보려면 DUT 내부 신호를 가져와야 함
    //    ("DUT.cnt" 대신, 실제 인스턴스 이름과 신호 이름을 맞춰서 사용)
    initial begin
        $display("Time\t cnt \t rdata");
        $monitor("%0dns:\t %0d \t 0x%0h", $time, DUT.cnt, rdata);
    end

endmodule
