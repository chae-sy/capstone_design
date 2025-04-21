`timescale 1ns/1ps

module tb_weight_buf;

    // Parameters
    parameter NUM_CHANNEL = 32;
    parameter WEIGHT_BIT = 4;

    // Inputs
    reg [WEIGHT_BIT-1:0] data_i;
    reg [NUM_CHANNEL-1:0] en;
    reg clk;
    reg rstb;
    reg rst_local;

    // Outputs
    wire done;
    wire [WEIGHT_BIT*NUM_CHANNEL-1:0] data_o;

    // Instantiate the Unit Under Test (UUT)
    weight_buf #(
        .NUM_CHANNEL(NUM_CHANNEL),
        .WEIGHT_BIT(WEIGHT_BIT)
    ) uut (
        .data_i(data_i),
        .en(en),
        .clk(clk),
        .rstb(rstb),
        .rst_local(rst_local),
        .done(done),
        .data_o(data_o)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Initialize signals and apply test cases
    initial begin
     	$vcdplusfile("tb_weight_buf.vpd");
	    $vcdpluson(0, tb_weight_buf);
	    $vcdplusmemon();
        // Initialize inputs
        clk = 0;
        rstb = 0;
        rst_local = 0;
        data_i = 0;
        en = 0;

        // Apply reset
        #10 rstb = 1;  // Release reset
        #10 rst_local = 1;  // Local reset
        #10 rst_local = 0;

        // Test Case 1: Load data into channels
        for (int i = 0; i < NUM_CHANNEL; i = i + 1) begin
            data_i = i % (1 << WEIGHT_BIT); // Test data pattern
            en = (1 << i);                  // Enable one channel at a time
            #10;                            // Wait for clock edge
        end

        // Test Case 2: Check for completion
        en = 0;           // Disable all channels
        #10;
        //if (done) $display("All data loaded: done signal asserted");

        // Display the contents of data_o
        //$display("data_o: %h", data_o);

        // Test Case 3: Apply local reset and verify buffer reset
        rst_local = 1;
        #10;
        rst_local = 0;
        #10;

        //$display("After local reset, data_o: %h", data_o);

        // Finish simulation
        $finish;
    end

endmodule
