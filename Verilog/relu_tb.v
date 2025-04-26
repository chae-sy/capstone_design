`timescale 1ns/1ps

module tb_relu_stream_with_last;

  // Parameters must match the DUT
  localparam DATA_WIDTH = 8;
  localparam CH         = 16;

  // Clock & reset
  reg                         clk;
  reg                         rstb;
  // Handshake signals
  reg                         valid_in;
  reg                         last_in;
  // Input data stream
  reg signed [DATA_WIDTH-1:0] in_data [0:CH-1];
  // Output from DUT
  wire                        valid_out;
  wire                        last_out;
  wire signed [DATA_WIDTH-1:0] out_data [0:CH-1];

  // Instantiate the DUT
  relu_stream_with_last #(
    .DATA_WIDTH(DATA_WIDTH),
    .CH        (CH)
  ) dut (
    .clk       (clk),
    .rstb      (rstb),
    .valid_in  (valid_in),
    .last_in   (last_in),
    .in_data   (in_data),
    .valid_out (valid_out),
    .last_out  (last_out),
    .out_data  (out_data)
  );

  // Clock generation: 100 MHz
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  integer i, idx;

  // Apply reset, then drive test vectors
  initial begin
    // Initialize
    rstb     = 0;
    valid_in = 0;
    last_in  = 0;
    for (i = 0; i < CH; i = i + 1)
      in_data[i] = 0;

    // Hold reset low for 20 ns
    #20 rstb = 1;

    // short pause
    #10;

    // Send 10 words of data as one "frame"
    for (idx = 0; idx < 10; idx = idx + 1) begin
      valid_in = 1;
      last_in  = (idx == 9);  // assert last on final word
      // Create a pattern: even channels negative, odd channels positive
      for (i = 0; i < CH; i = i + 1) begin
        if (i % 2 == 0)
          in_data[i] = -i;       // negative value
        else
          in_data[i] =  i;       // positive value
      end
      #10;
    end

    // Deassert valid & last
    valid_in = 0;
    last_in  = 0;

    // Wait a bit and finish
    #50;
    $finish;
  end

  // Monitor outputs
  initial begin
    $display("=== Testbench: relu_stream_with_last ===");
    $display("time | valid_in last_in | valid_out last_out | out_data[0] ... out_data[15]");
  end

  always @(posedge clk) begin
    if (valid_out) begin
      $write("%4t |    %b       %b   |     %b        %b   | ",
              $time, valid_in, last_in, valid_out, last_out);
      // Print all channels
      for (i = 0; i < CH; i = i + 1) begin
        $write("%4d%s", out_data[i], (i == CH-1) ? "\n" : ",");
      end
    end
  end

endmodule
