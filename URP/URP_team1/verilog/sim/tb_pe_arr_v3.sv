`timescale 1ns/1ps

module tb_pe_arr_v3;

  // Parameters
  parameter DATA_WIDTH = 8;
  parameter WEIGHT_WIDTH = 8;
  parameter INT_EXTEND = 9;
  parameter PE_NUM = 32;

  // Inputs
  reg clk;
  reg rstb;
  reg en;
  reg rst_local;
  reg sel;
  reg [DATA_WIDTH-1:0] data_i [0:PE_NUM-1];  // 2D array for data input
  reg [WEIGHT_WIDTH-1:0] weight_i [0:PE_NUM-1];  // 2D array for weight input

  // Outputs
  wire [(DATA_WIDTH+WEIGHT_WIDTH+INT_EXTEND)-1:0] data_o [0:PE_NUM-1];

  // Instantiate the Unit Under Test (UUT)
  pe_arr #(.DATA_WIDTH(DATA_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH), .INT_EXTEND(INT_EXTEND), .PE_NUM(PE_NUM)) uut (
    .clk(clk),
    .rstb(rstb),
    .en(en),
    .rst_local(rst_local),
    .sel(sel),
    .data_i(data_i),
    .weight_i(weight_i),
    .data_o(data_o)
  );

  // Clock generation
  always #5 clk = ~clk;  // 10 time unit clock period

  // Testbench logic
  initial begin
    $vcdplusfile("tb_pe_arr_v3.vpd");
    $vcdpluson(0, tb_pe_arr_v3);
    $vcdplusmemon();
    
    // Initialize Inputs
    clk = 0;
    rstb = 1;
    en = 0;
    rst_local = 0;
    sel = 1;
    // Reset sequence
    #10 rstb = 0;  // Release global reset
    #10 rstb = 1;  // Assert global reset

    // Apply inputs and test various cases
    @(posedge clk);
    rst_local = 1;  // Assert local reset for all PEs
    @(posedge clk);
    rst_local = 0;  // Release local reset
    #5
    
    // Test Case 1: Apply data and weights to all PEs
    en = 1;
    for (int i = 0; i < PE_NUM; i++) begin
      data_i[i] = 4'h5;  // Example data (all elements set to 5)
      weight_i[i] = 4'h3;   // Example weight (all elements set to 3)
    end
    #20;

    // Test Case 2: Change input data and weights for all PEs
    for (int i = 0; i < PE_NUM; i++) begin
      data_i[i] = 4'h4;  // Example data (all elements set to 4)
      weight_i[i] = 4'h2;   // Example weight (all elements set to 2)
    end
    #20;

    // Test Case 3: Apply different data and weight patterns
    for (int i = 0; i < PE_NUM; i++) begin
      data_i[i] = 4'h1;  // Example pattern
      weight_i[i] = 4'h1;   // Example weight pattern
    end
    #20;

    // Test Case 4: Disable enable signal (PEs should hold previous values)
    en = 0;
    #20;

    // Test Case 5: Re-enable and apply new inputs
    en = 1;
    sel = 0;
    for (int i = 0; i < PE_NUM; i++) begin
      data_i[i] = 4'hA;  // Example data (all elements set to 16)
      weight_i[i] = 4'hF;   // Example weight (all elements set to -1)
    end
    #20;

    // Assert local reset again
    rst_local = 1;
    #10;
    rst_local = 0;
    #10;

    // Complete simulation
    $stop;
  end

endmodule
