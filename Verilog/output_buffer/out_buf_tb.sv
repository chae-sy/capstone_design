module out_buf_tb;
    reg clk = 0, rst = 0;
    reg wr_en;
    reg [7:0] wr_data;
    reg [6:0] wr_row, wr_col;
    reg [3:0] wr_chn;

    reg rd_en;
    reg [6:0] rd_row, rd_col;
    reg [3:0] rd_chn;
    wire [7:0] rd_data;

    output_buffer dut (
        .clk(clk), .rst(rst),
        .wr_en(wr_en), .wr_data(wr_data),
        .wr_row(wr_row), .wr_col(wr_col), .wr_chn(wr_chn),
        .rd_en(rd_en),
        .rd_row(rd_row), .rd_col(rd_col), .rd_chn(rd_chn),
        .rd_data(rd_data)
    );

    always #5 clk = ~clk;

    initial begin
        rst = 1; #10; rst = 0;

        // Write
        wr_en = 1;
        wr_data = 8'h3C;
        wr_row = 1; wr_col = 2; wr_chn = 3;
        #10;
        wr_en = 0;

        // Read
        rd_en = 1;
        rd_row = 1; rd_col = 2; rd_chn = 3;
        #10;
        rd_en = 0;

        #20 $finish;
    end
endmodule
