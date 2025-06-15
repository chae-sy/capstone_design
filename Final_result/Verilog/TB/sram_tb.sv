`timescale 1ns / 1ps

module tb_memory_w_v0;

    // Parameters
    localparam ADDR_WIDTH = 2;
    localparam DATA_WIDTH = 128;
    localparam WR_DELAY   = 8;

    // Testbench signals
    reg                        CLK;
    reg                        CEB;
    reg                        WEB;
    reg  [ADDR_WIDTH-1:0]      A;
    reg  [DATA_WIDTH-1:0]      D;
    wire [DATA_WIDTH-1:0]      Q;

    // Instantiate DUT
    memory_w_v0 #(
        .addr_width (ADDR_WIDTH),
        .data_width (DATA_WIDTH),
        .wr_delay   (WR_DELAY)
    ) dut (
        .CLK (CLK),
        .CEB (CEB),
        .WEB (WEB),
        .A   (A),
        .D   (D),
        .Q   (Q)
    );

    // Clock generation: 10 ns period
    initial begin
        CLK = 0;
        forever #5 CLK = ~CLK;
    end

    // Test procedure
    integer i;
    reg [DATA_WIDTH-1:0] golden_mem [0:2**ADDR_WIDTH-1];

    initial begin
        // 초기화
        CEB = 1;
        WEB = 1;
        A   = 0;
        D   = {DATA_WIDTH{1'b0}};
        // Wait for reset
        #20;

        // 1) 모든 주소에 랜덤 데이터 쓰기
        for (i = 0; i < 4; i = i + 1) begin
            @(posedge CLK);
            CEB = 0;        // chip enable active
            WEB = 0;        // write enable active
            A   = i;
            D   = i;
            golden_mem[i] = D;
        end

        // 2) 읽기 모드로 전환 후, 데이터 검증
        for (i = 0; i < 4; i = i + 1) begin
            @(posedge CLK);
            CEB = 0;        // chip enable active
            WEB = 1;        // write disable -> read
            A   = i;
            D   = {DATA_WIDTH{1'bx}};  // don't care
            
            if (Q !== golden_mem[i]) begin
                $display("ERROR: addr=%0d, expected=%h, got=%h", i, golden_mem[i], Q);
            end else begin
                $display("PASS : addr=%0d, data=%h", i, Q);
            end
        end

        // 3) 비활성화 모드에서 Q가 x인지 확인
        @(posedge CLK);
        CEB = 1;    // chip disabled
        WEB = 1;
        A   = 0;
        #10;
        if (Q === {DATA_WIDTH{1'bx}}) begin
            $display("PASS : CEB=1, Q is undefined (x)");
        end else begin
            $display("ERROR: CEB=1, Q should be x but got %h", Q);
        end

        $display("===== TEST COMPLETE =====");
        $finish;
    end

endmodule
