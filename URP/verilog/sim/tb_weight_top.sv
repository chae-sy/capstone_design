`timescale 1ns/1ps

module tb_weight_top;

    // Parameters
    parameter NUM_CHANNEL = 32;
    parameter WEIGHT_BIT = 8;
    parameter MEM_LENGTH = 144;

    // Inputs
    reg en;
    reg data_en;
    reg rstb;
    reg clk;
    reg [7:0] data_i;

    // Outputs

    wire done;


    reg [31:0] count;

    // Instantiate the Unit Under Test (UUT)
    weight_top #(
        .NUM_CHANNEL(NUM_CHANNEL),
        .WEIGHT_BIT(WEIGHT_BIT),
        .MEM_LENGTH(MEM_LENGTH)
    ) uut (
        .clk(clk),
        .rstb(rstb),
        .en(en),
        .data_en(data_en),
        .data_i(data_i),
        .done(done)
    );

    // Clock generation
    always begin 
        #5 clk = ~clk;
    end
    always @ (posedge clk) begin
        count <= count + 1;
    end
    // Test sequence
    initial begin
        $vcdplusfile("tb_weight_top.vpd");
	    $vcdpluson(0, tb_weight_top);
	    $vcdplusmemon();
        // Initialize inputs
        clk = 0;
        rstb = 0;
        en = 0;
        data_en = 0;
        data_i = 0;
        count = 0;

        // Apply reset
        #10 rstb = 1;  // Release reset
        #10 en = 1;    // Enable the module
        #20 en = 0;

        // Test Case 1: Initial IDLE to WAIT state transition
        for (integer j=0; j<MEM_LENGTH; j=j+1) begin
            #10 data_en = 1;
            for (integer i = 0; i < NUM_CHANNEL; i = i + 1) begin
                data_i = i;
                #10;
            end
            data_en = 0;
            #10;
        end
        //#10 $display("After MEM_LENGTH: done = %b, wm_addr = %d", done, wm_addr);

        // Check reset state
        //#10 $display("After reset: buf_rst = %b, buf_en = %b, wm_addr = %d, wm_wen = %b, done = %b", buf_rst, buf_en, wm_addr, wm_wen, done);

        // End simulation
        #30 $finish;
    end

endmodule
