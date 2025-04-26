module in_buf_tb;
    reg clk = 0, rst = 0;
    reg wr_en;
    reg [7:0] wr_data;
    reg [6:0] wr_row, wr_col;
    reg [3:0] wr_chn;

    reg [6:0] start_row, start_col;
    reg rd_en;

    wire [7:0] rd_patch [0:2][0:4][0:15];

    // DUT
    input_buffer dut (
        .clk(clk), .rst(rst),
        .wr_en(wr_en), .wr_data(wr_data),
        .wr_row(wr_row), .wr_col(wr_col), .wr_chn(wr_chn),
        .start_row(start_row), .start_col(start_col),
        .rd_en(rd_en),
        .rd_patch(rd_patch)
    );

    // Clock
    always #5 clk = ~clk;

    initial begin
        rst = 1; #10; rst = 0;
        
        // Write sample data
        wr_en = 1;
        wr_data = 8'hAA;
        wr_row = 0; wr_col = 0; wr_chn = 0;
        #10;
        wr_en = 0;

        // Read patch
        start_row = 0;
        start_col = 0;
        rd_en = 1;
        #10;
        rd_en = 0;

        #20 $finish;
    end
endmodule
