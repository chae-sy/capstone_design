`timescale 1ns / 1ps
/*
 * Single-Port Register File with NUM_LANE Parallel Reads
 * - One write port (wen, addr, wdata)
 * - NUM_LANE asynchronous read ports via flat vector
 */
module register_file_single #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 5,        // up to 2^ADDR_WIDTH registers
    parameter NUM_LANE   = 4         // number of parallel read ports
)(
    input  wire                         clk,
    input  wire                         rst_n,
    // write port
    input  wire                         wen,
    input  wire [ADDR_WIDTH-1:0]        addr,    // write address
    input  wire [DATA_WIDTH-1:0]        wdata,   // write data

    // read ports (flat vectors)
    input  wire [NUM_LANE*ADDR_WIDTH-1:0] addr_flat,   // read addresses
    output wire [NUM_LANE*DATA_WIDTH-1:0] rdata_flat   // read data outputs
);

    // Internal register bank
    localparam DEPTH = (1<<ADDR_WIDTH);
    reg [DATA_WIDTH-1:0] regs [0:DEPTH-1];
    integer i;

    // Write and reset
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1)
                regs[i] <= {DATA_WIDTH{1'b0}};
        end else if (wen) begin
            regs[addr] <= wdata;
        end
    end

    // NUM_LANE parallel asynchronous reads
    genvar lane;
    generate
      for (lane = 0; lane < NUM_LANE; lane = lane + 1) begin : GEN_READ
        // slice out this lane's address
        wire [ADDR_WIDTH-1:0] rd_addr = addr_flat[lane*ADDR_WIDTH +: ADDR_WIDTH];
        // assign the corresponding data slice
        assign rdata_flat[lane*DATA_WIDTH +: DATA_WIDTH] = regs[rd_addr];
      end
    endgenerate

endmodule
