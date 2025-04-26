`timescale 1ns/1ps

module tb_w_buf;
    reg clk = 0, rst = 0;
    reg wren;
    reg [7:0] wr_data;
    reg [4:0] wr_row, wr_col;
    reg [3:0] wr_chn;
    reg rden;
    wire [7:0] rd_weight [0:2][0:2][0:15];

    w_buf uut (
        .clk(clk), .rst(rst),
        .wren(wren), .wr_data(wr_data),
        .wr_row(wr_row), .wr_col(wr_col), .wr_chn(wr_chn),
        .rden(rden),
        .rd_weight(rd_weight)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0; rst = 1; wren = 0; rden = 0;
        #10 rst = 0;

        // Write : (0,0,0) 위치에 8'hBB 저장장
        wren = 1;
        wr_data = 8'hBB;
        wr_row = 0; wr_col = 0; wr_chn = 0;
        #10;
        wren = 0;

        // read weight
        rden = 1;
        #10;
        rden = 0;

        #50 $finish;
    end

    initial begin
        $dumpfile("weight_buffer_tb.vcd");
        $dumpvars(0, uut);
    end

endmodule
