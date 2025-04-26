`timescale 1ns/1ps

module tb_out_buf;
    reg clk = 0, rst = 0;
    reg wren;
    reg [7:0] wr_data;
    reg [6:0] wr_row, wr_col;
    reg [3:0] wr_chn;

    reg rden;
    reg [6:0] rd_row, rd_col;
    reg [3:0] rd_chn;
    wire [7:0] rd_data;

    out_buf uut (
        .clk(clk), .rst(rst),
        .wren(wren), .wr_data(wr_data),
        .wr_row(wr_row), .wr_col(wr_col), .wr_chn(wr_chn),
        .rden(rden),
        .rd_row(rd_row), .rd_col(rd_col), .rd_chn(rd_chn),
        .rd_data(rd_data)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0; rst = 1; wren = 0; rd_en = 0;
        #10 rst = 0;

        // Write : (5,5,0) 위치에 8'hCC 저장장
        wren = 1;
        wr_data = 8'hCC;
        wr_row = 5; wr_col = 5; wr_chn = 0;
        #10;
        wren = 0;

        // Read
        rden = 1;
        rrd_row = 5; rd_col = 5; rd_chn = 0;
        #10;
        rden = 0;

        #50 $finish;
    end

    initial begin
        $dumpfile("tb_out_buf.vcd");
        $dumpvars(0, uut);
    end

endmodule
