`timescale 1ns / 1ps

module tb_controller_top;

    // Parameters
    parameter IM_BIT_LEN = $clog2(100);
    parameter MA_BIT_LEN = $clog2(16);
    parameter MB_BIT_LEN = $clog2(20);
    parameter WM_BIT_LEN = $clog2(144);
    parameter NUM_CHANNEL = 32;

    // Inputs (reg because they are controlled by the testbench)
    reg rstb;
    reg clk;
    reg en;
    reg weight_en;
    reg calc_en;

    // Outputs (wire because they are driven by the DUT)
    reg weight_buf_done_i;
    wire weight_buf_rst_o;
    wire [NUM_CHANNEL-1:0] weight_buf_en_o;
    wire [WM_BIT_LEN-1:0] wm_addr_o;
    wire wm_web_o;
    wire wm_ceb_o;
    wire [IM_BIT_LEN-1:0] im_addr_o;
    wire im_web_o;
    wire im_ceb_o;
    wire [MA_BIT_LEN-1:0] ma_addr_o;
    wire ma_web_o;
    wire ma_ceb_o;
    wire [MB_BIT_LEN-1:0] mb_addr_o;
    wire mb_web_o;
    wire mb_ceb_o;
    wire pe_en_o;
    wire pe_rst_o;
    wire [31:0] out_buf_en_o;
    wire out_buf_sel_o;
    wire out_buf_rst_o;
    wire pool_sel_o;
    wire comp_start_o;
    wire [2:0] layer;
    wire done_o;

    // Instantiate the DUT (Device Under Test)
    controller_top #(
        .IM_BIT_LEN(IM_BIT_LEN),
        .MA_BIT_LEN(MA_BIT_LEN),
        .MB_BIT_LEN(MB_BIT_LEN),
        .WM_BIT_LEN(WM_BIT_LEN)
    ) dut (
        .rstb(rstb),
        .clk(clk),
        .en(en),
        .weight_en(weight_en),
        .calc_en(calc_en),
        //.weight_buf_done_i(weight_buf_done_i),
        .weight_buf_rst_o(weight_buf_rst_o),
        .weight_buf_en_o(weight_buf_en_o),
        .wm_addr_o(wm_addr_o),
        .wm_web_o(wm_web_o),
        .wm_ceb_o(wm_ceb_o),
        .im_addr_o(im_addr_o),
        .im_web_o(im_web_o),
        .im_ceb_o(im_ceb_o),
        .ma_addr_o(ma_addr_o),
        .ma_web_o(ma_web_o),
        .ma_ceb_o(ma_ceb_o),
        .mb_addr_o(mb_addr_o),
        .mb_web_o(mb_web_o),
        .mb_ceb_o(mb_ceb_o),
        .pe_en_o(pe_en_o),
        .pe_rst_o(pe_rst_o),
        .out_buf_en_o(out_buf_en_o),
        .out_buf_sel_o(out_buf_sel_o),
        .out_buf_rst_o(out_buf_rst_o),
        .pool_sel_o(pool_sel_o),
        .comp_start_o(comp_start_o),
        .layer(layer),
        .done_o(done_o)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns clock period
    end

    integer i, j;

    always @(*) begin
        if (done_o) 
            # 50 calc_en = 1;

        if (calc_en)
            #10 calc_en = 0;
    end

    // Initial block to set up inputs and run the test sequence
    initial begin
        // Initialize inputs

        $vcdplusfile("tb_controller_top.vpd");
        $vcdpluson(0, tb_controller_top);
        $vcdplusmemon();


        rstb = 0;
        en = 0;
        weight_en = 0;
        calc_en = 0;

        #10 rstb = 0;
        // Apply reset
        #10 rstb = 1;  // Release reset

        // Test sequence
        #10 en = 1;        // Enable module
        #20 en = 0;
        

        for(i=0; i<144; i=i+1) begin
            #10 weight_en = 1; // Trigger weight initialization
            #320 weight_en = 0;
            #10 weight_buf_done_i = 1;
            #10 weight_buf_done_i = 0;
        end

        for(j=0; j<29; j=j+1) begin
            #100 calc_en = 1;
            #10 calc_en = 0;
            #4000;
        end


        // Finish simulation
        #100000 $finish;
    end

endmodule
