`timescale 1ns/1ps

module tb_ib_ctrl;

    // Testbench signals
    reg rstb;
    reg clk;
    reg wr_input_on;

    wire [1:0] state;
    wire [3:0] ib_addr;

    // Instantiate the DUT (Device Under Test)
    ib_ctrl dut (
        .rstb(rstb),
        .clk(clk),
        .wr_input_on(wr_input_on),
        .state(state),
        .ib_addr(ib_addr)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns clock period
    end

    // Initial block for simulation
    initial begin
        $vcdplusfile("tb_ib_ctrl.vpd");
	    $vcdpluson(0, tb_ib_ctrl);
	    $vcdplusmemon();
        // Initialize inputs
        rstb = 0;
        wr_input_on = 0;

        // Apply reset
        #10 rstb = 1;

        // Test sequence
        #10 wr_input_on = 1;   // Start write operation
        #50 wr_input_on = 0;   // Switch to read operation
        #200 wr_input_on = 1;  // Enable write again
        #20 wr_input_on = 0;   // Back to read

        // End simulation
        #200 $finish;
    end

    // Monitor outputs
    initial begin
        $monitor("Time=%0t, rstb=%b, clk=%b, wr_input_on=%b, state=%b, ib_addr=%d", 
                 $time, rstb, clk, wr_input_on, state, ib_addr);
    end

endmodule
