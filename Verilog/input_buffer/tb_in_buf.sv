`timescale 1ns/1ps

module tb_in_buf;
    reg clk = 0, rst = 0;
    reg wren;
    reg [7:0] wr_data;
    reg [6:0] wr_row, wr_col;
    reg [3:0] wr_chn;
    reg [6:0] start_row, start_col;
    reg rden;

    wire [7:0] rd_patch [0:2][0:4][0:15];

    // 테스트할 input_buffer 인스턴스
    in_buf uut (
        .clk(clk), .rst(rst),
        .wren(wren), .wr_data(wr_data),
        .wr_row(wr_row), .wr_col(wr_col), .wr_chn(wr_chn),
        .start_row(start_row), .start_col(start_col),
        .rden(rden),
        .rd_patch(rd_patch)
    );

    // Clock
    always #5 clk = ~clk;

    initial begin
        clk = 0; rst = 1; wren = 0; rden = 0;
        #10 rst = 0;
        
        // Write 동작 : (0,0,0) 위치에 8'hAA 저장장
        wren = 1;
        wr_data = 8'hAA;
        wr_row = 0; wr_col = 0; wr_chn = 0;
        #10;
        wren = 0;

        // Read 동작 : (0,0) 위치에서 3x5 패치 읽기
        start_row = 0;
        start_col = 0;
        rden = 1;
        #10;
        rden = 0;

        #50 $finish;
    end

    // icarus verilog용 덤프파일 설정
    initial begin
        $dumpfile("input_buffer_tb.vcd");
        $dumpvars(0, uut);
    end


endmodule
