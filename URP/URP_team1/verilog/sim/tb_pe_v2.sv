`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Kang Research Group
// Engineer: Chanhong Jeon
// 
// Create Date: 2024/09/30
// Design Name: Processing Element Array
// Module Name: pe_arr
// Project Name: KWS Chip Tape-out
// Target Devices: Samsung 28nm
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 

module tb_pe_v2;

  // Parameters
  parameter DATA_WIDTH = 4;
  parameter WEIGHT_WIDTH = 4;
  parameter INT_EXTEND = 9;

  // Inputs
  reg clk;
  reg rstb;
  reg en;
  reg rst_local;
  reg signed [DATA_WIDTH-1:0] data_i;
  reg signed [WEIGHT_WIDTH-1:0] weight_i;

  // Outputs
  wire signed [DATA_WIDTH+WEIGHT_WIDTH+INT_EXTEND-1:0] data_o;

  // Instantiate the Unit Under Test (UUT)
  pe #(.DATA_WIDTH(DATA_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH), .INT_EXTEND(INT_EXTEND)) uut (
    .clk(clk),
    .rstb(rstb),
    .en(en),
    .rst_local(rst_local),
    .data_i(data_i),
    .weight_i(weight_i),
    .data_o(data_o)
  );

  // Clock generation
  always #5 clk = ~clk;  // 10 time units clock period

  // Testbench logic
  initial begin
 	$vcdplusfile("tb_pe_v2.vpd");
	$vcdpluson(0, tb_pe_v2);
	$vcdplusmemon();

    // Initialize Inputs
    clk = 0;
    rstb = 0;
    en = 0;
    rst_local = 0;
    data_i = 0;
    weight_i = 0;

    // Reset sequence
    #10 rstb = 1;  // Release global reset
    #10 rstb = 0;  // Assert global reset
    #10 rstb = 1;  // Release global reset again

    // Apply inputs
    @(posedge clk);
    rst_local = 1;  // Assert local reset
    @(posedge clk);
    rst_local = 0;  // Release local reset
	#5
    // Test Case 1: Apply data and weight
    en = 1;
    data_i = 12'h005;  // Example data
    weight_i = 8'h03;  // Example weight
    #20;

    // Test Case 2: Change input values
    data_i = -12'h004;
    weight_i = 8'h02;
    #20;

    // Test Case 3: Disable the enable signal
    en = 0;
    #20;

    // Test Case 4: Enable again with new values
    en = 1;
    data_i = 12'h010;
    weight_i = 8'hFF;  // Negative weight example
    #20;

    // Complete simulation
    #20 rst_local = 1;
    #10
    $stop;
  end

endmodule

