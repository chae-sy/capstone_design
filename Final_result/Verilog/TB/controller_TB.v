`timescale 1ns / 1ps

module Controller_tb;

    // Parameters
    localparam MA_BIT_LEN = $clog2(480000);
    localparam MB_BIT_LEN = $clog2(480000);
    localparam WMEM_BIT_LEN = $clog2(2304);
    localparam NUM_CHANNEL = 16;

    // DUT Inputs
    reg rst_n;
    reg clk;
    reg initial_SRAMw_done;
    reg initial_weight_done;
    reg data_map_done;
    reg layer_done;    // 수정: layer_done_o 아니고 layer_done (Controller 입력 맞춤)

    // DUT Outputs
    wire [WMEM_BIT_LEN-1:0] wmem_addr_o;
    wire wmem_wenb_o;
    wire wmem_enb_o;
    wire [MA_BIT_LEN-1:0] memA_addr_o;
    wire memA_wenb_o;
    wire memA_cenb_o;
    wire [MB_BIT_LEN-1:0] memB_addr_o;
    wire memB_wenb_o;
    wire memB_cenb_o;
    wire in_buf_en_o;
    wire in_buf_sel_o;
    wire in_buf_rst_o;
    wire wei_buff_en_o;
    wire pe_en_o;
    wire pe_rst_o;
    wire relu_en_o;
    wire [31:0] out_buf_en_o;
    wire out_buf_sel_o;
    wire out_buf_rst_o;
    wire pool_sel_o;
    wire [2:0] layer_state;
    wire done_o;
    wire layer_done_o;
    wire data_map_enb;

    // Instantiate DUT
    Controller #(
        .MA_BIT_LEN(MA_BIT_LEN),
        .MB_BIT_LEN(MB_BIT_LEN),
        .WMEM_BIT_LEN(WMEM_BIT_LEN),
        .NUM_CHANNEL(NUM_CHANNEL)
    ) dut (
        .rst_n(rst_n),
        .clk(clk),
        .initial_SRAMw_done(initial_SRAMw_done),
        .initial_weight_done(initial_weight_done),
        .data_map_done(data_map_done),
        .layer_done(layer_done),
        .wmem_addr_o(wmem_addr_o),
        .wmem_wenb_o(wmem_wenb_o),
        .wmem_enb_o(wmem_enb_o),
        .memA_addr_o(memA_addr_o),
        .memA_wenb_o(memA_wenb_o),
        .memA_cenb_o(memA_cenb_o),
        .memB_addr_o(memB_addr_o),
        .memB_wenb_o(memB_wenb_o),
        .memB_cenb_o(memB_cenb_o),
        .in_buf_en_o(in_buf_en_o),
        .in_buf_sel_o(in_buf_sel_o),
        .in_buf_rst_o(in_buf_rst_o),
        .wei_buff_en_o(wei_buff_en_o),
        .pe_en_o(pe_en_o),
        .pe_rst_o(pe_rst_o),
        .relu_en_o(relu_en_o),
        .out_buf_en_o(out_buf_en_o),
        .out_buf_sel_o(out_buf_sel_o),
        .out_buf_rst_o(out_buf_rst_o),
        .pool_sel_o(pool_sel_o),
        .layer_state(layer_state),
        .done_o(done_o),
        .layer_done_o(layer_done_o),
        .data_map_enb(data_map_enb)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk; // 100MHz clock

    // Test sequence
    initial begin
        // Initialize
        rst_n = 0;
        initial_SRAMw_done = 0;
        initial_weight_done = 0;
        data_map_done = 0;
        layer_done = 0;  // 수정된 입력 이름
        #20;

        // Reset Release
        rst_n = 1;
        $display("[%0t ns] Reset deasserted", $time);

        // Initial SRAM Write and Weight Load Done
        #30;
        initial_SRAMw_done = 1;
        initial_weight_done = 1;
        $display("[%0t ns] Initial SRAM and Weight Load Done", $time);
        #10;
        initial_SRAMw_done = 0;
        initial_weight_done = 0;

        // Layer1 ~ Layer5 완료 시나리오
        repeat (5) begin
            #100;
            layer_done = 1;
            #10;
            layer_done = 0;
            $display("[%0t ns] One Layer Done Signal Asserted", $time);
        end

        // 데이터 매핑 시퀀스
        #100;
        data_map_done = 1;
        $display("[%0t ns] Data Mapping Done Signal Asserted", $time);
        #10;
        data_map_done = 0;

        // 시뮬레이션 종료
        #100;
        $stop;
    end

    // Waveform Dump
    initial begin
        $dumpfile("controller_tb.vcd");
        $dumpvars(0, Controller_tb);
    end

endmodule
