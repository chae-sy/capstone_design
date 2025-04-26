`timescale 1ns/1ps

module tb_maxpool_2x4_16ch;

  // Testbench parameters must match DUT
  localparam DATA_WIDTH = 8;
  localparam H_IN       = 100;
  localparam W_IN       = 100;
  localparam CH         = 16;

  // Clock, reset, valid in
  reg                         clk;
  reg                         rstb;
  reg                         valid_in;
  // Input feature‐map stream (signed DATA_WIDTH bits × CH channels)
  reg  signed [DATA_WIDTH-1:0] in_data [0:CH-1];
  // Outputs from DUT
  wire                        valid_out;
  wire signed [DATA_WIDTH-1:0] out_data [0:CH-1];

  // Instantiate the Device Under Test
  maxpool_2x4_16ch #(
    .DATA_WIDTH(DATA_WIDTH),
    .H_IN      (H_IN),
    .W_IN      (W_IN),
    .CH        (CH)
  ) dut (
    .clk       (clk),
    .rstb      (rstb),
    .valid_in  (valid_in),
    .in_data   (in_data),
    .valid_out (valid_out),
    .out_data  (out_data)
  );

  // Clock generation: 100 MHz
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  integer row, col, c;

  // Apply reset, then drive a full 100×100 frame of test data
  initial begin
    // Initialize
    rstb      = 0;
    valid_in  = 0;
    for (c = 0; c < CH; c = c + 1)
      in_data[c] = 0;

    // Hold reset low for a couple of cycles
    #20 rstb = 1;  

    // small delay after reset release
    #10;

    // Feed in a full frame
    for (row = 0; row < H_IN; row = row + 1) begin
      for (col = 0; col < W_IN; col = col + 1) begin
        // Example pattern: each channel gets (row*W_IN + col) + channel_index
        for (c = 0; c < CH; c = c + 1)
          in_data[c] = (row * W_IN + col) + c;
        valid_in = 1;
        #10;
      end
    end

    // Finish input
    valid_in = 0;
    #200;
    $finish;
  end

  // Monitor the pooling outputs when valid_out is asserted
  initial begin
    $display("=== MaxPool 2×4×16ch Testbench ===");
    $display("Time    valid_in valid_out   out_data[0:15]");
  end

  always @(posedge clk) begin
    if (valid_out) begin
      $write("%0t      %b        %b    ", $time, valid_in, valid_out);
      // print all 16 channels
      for (c = 0; c < CH; c = c + 1) begin
        $write("%0d%s", out_data[c], (c==CH-1) ? "\n" : ",");
      end
    end
  end

endmodule

testbench