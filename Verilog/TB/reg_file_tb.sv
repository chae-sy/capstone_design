`timescale 1ns / 1ps

module tb_regfile_sync;

    // 파라미터
    localparam DATA_WIDTH = 32;
    localparam ADDR_WIDTH = 5;
    localparam CLK_PERIOD = 10;

    // 신호 선언
    reg                         clk;
    reg                         rst_n;
    reg                         we;
    reg  [ADDR_WIDTH-1:0]       waddr;
    reg  [DATA_WIDTH-1:0]       wdata;
    reg  [ADDR_WIDTH-1:0]       raddr;
    wire [DATA_WIDTH-1:0]       rdata;

    // DUT 인스턴스
    regfile_sync #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) uut (
        .clk    (clk),
        .rst_n  (rst_n),
        .we     (we),
        .waddr  (waddr),
        .wdata  (wdata),
        .raddr  (raddr),
        .rdata  (rdata)
    );

    // 클록 생성: 10ns 주기
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        // 초기화
        rst_n = 1'b0; we = 1'b0;
        waddr = 0;   wdata = 0;
        raddr = 0;
        #20;
        rst_n = 1'b1;

        // 1) addr=3에 쓰기
        @(posedge clk);
        we    = 1'b1;
        waddr = 5'd3;
        wdata = 32'hDEAD_BEEF;
        @(posedge clk);
        we = 1'b0;

        // 2) 같은 클럭 싸이클에 읽기
        //   -> 다음 상승 전 사이클에 raddr 세팅
        raddr = 5'd3;
        @(posedge clk);
        
        $display("Cycle %0t: Read @3 -> rdata = 0x%08X (expect DEAD_BEEF)", $time, rdata);

        // 3) addr=5에 쓰기
        @(posedge clk);
        we    = 1'b1;
        waddr = 5'd5;
        wdata = 32'h1234_5678;
        @(posedge clk);
        we = 1'b0;

        // 4) 같은 싸이클에 읽기
        raddr = 5'd5;
        @(posedge clk);
        
        $display("Cycle %0t: Read @5 -> rdata = 0x%08X (expect 12345678)", $time, rdata);

        // 5) 다시 addr=3 읽기
        raddr = 5'd3;
        @(posedge clk);
        
        $display("Cycle %0t: Read @3 -> rdata = 0x%08X (expect DEAD_BEEF)", $time, rdata);

        #10;
        $finish;
    end

endmodule
