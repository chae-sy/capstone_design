module w_buf_tb;
    reg clk = 0, rst = 0;
    reg wr_en;
    reg [7:0] wr_data;
    reg [4:0] wr_row, wr_col;
    reg [3:0] wr_chn;

    reg rd_en;
    wire [7:0] rd_weight [0:2][0:2][0:15];

    weight_buffer dut (
        .clk(clk), .rst(rst),
        .wr_en(wr_en), .wr_data(wr_data),
        .wr_row(wr_row), .wr_col(wr_col), .wr_chn(wr_chn),
        .rd_en(rd_en),
        .rd_weight(rd_weight)
    );

    always #5 clk = ~clk;

    initial begin
        rst = 1; #10; rst = 0;

        // Write weight
        wr_en = 1;
        wr_data = 8'h55;
        wr_row = 0; wr_col = 0; wr_chn = 0;
        #10;
        wr_en = 0;

        // Read all weights
        rd_en = 1;
        #10;
        rd_en = 0;

        #20 $finish;
    end
endmodule
