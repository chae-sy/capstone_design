`timescale 1ns/1ps

module SPRNN_Top_tb;

    // Clock and reset
    reg clk;
    reg rst_n;

    // Inputs
    reg start;
    reg [15:0] memA_addr_i;
    reg [15:0] memB_addr_i;
    reg [9:0]  wmem_addr_i;
    reg [127:0] memA_d_i;
    reg [127:0] memB_d_i;
    reg [127:0] wmem_d_i;
    reg wren_bias_i;
    reg [2:0] write_addr_bias_i;
    reg [511:0] write_data_bias_i;
    reg initial_SRAMw_done;
    reg initial_weight_done;

    // Outputs
    wire fin;
    wire total_done_o;

    // DUT 인스턴스
    SPRNN_Top #(
        .DATA_WIDTH(8),
        .NUM_COLOR(3),
        .NUM_CHNL(16)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .memA_addr_i(memA_addr_i),
        .memB_addr_i(memB_addr_i),
        .wmem_addr_i(wmem_addr_i),
        .memA_d_i(memA_d_i),
        .memB_d_i(memB_d_i),
        .wmem_d_i(wmem_d_i),
        .wren_bias_i(wren_bias_i),
        .write_addr_bias_i(write_addr_bias_i),
        .write_data_bias_i(write_data_bias_i),
        .initial_SRAMw_done(initial_SRAMw_done),
        .initial_weight_done(initial_weight_done),
        .total_done_o(total_done_o)
    );

    // Clock 생성
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz
    end

    integer i;

    initial begin
        // 초기화
        rst_n = 0; start = 0; initial_SRAMw_done = 0; initial_weight_done = 0;
        memA_addr_i = 0; memB_addr_i = 0; wmem_addr_i = 0;
        memA_d_i = 0; memB_d_i = 0; wmem_d_i = 0;
        wren_bias_i = 0; write_addr_bias_i = 0; write_data_bias_i = 0;

        #20 rst_n = 1;
        #5; start = 1;
        // Memory A, B, W 초기화
        for (i = 0; i < 31004; i = i + 1) begin
            memA_addr_i = i;
            memA_d_i = 128'h0000_0000_0000_0000_0000_0000_0000_0000 + i;
            memB_addr_i = i;
            memB_d_i = 0;
            #10;
        end
        
        // Memory weight 초기화
        for (i = 0; i < 582; i = i + 1) begin
            wmem_addr_i = i;
            wmem_d_i = 128'h0000_0000_0000_0000_0000_0000_0000_0002;
            #10;
        end
        
        // Bias regfile 초기화
        for (i = 0; i < 8; i = i + 1) begin
            wren_bias_i = 1;
            write_addr_bias_i = i[2:0];
            write_data_bias_i = {64{i}}; // 각 주소마다 값 변화
            #10;
        end
        wren_bias_i = 0;

        // 초기화 완료 시그널
        initial_SRAMw_done = 1;
        initial_weight_done = 1;

        // 동작 관찰
        #200000000 $finish;
    end

    // 출력 모니터링
    initial begin
        $monitor("Time=%0t | rst_n=%b | start=%b | total_done_o=%b",
                 $time, rst_n, start, total_done_o);
    end

endmodule

