/*
 * Single-Port Register File
 * Parameterizable by data width and number of registers
 * - One read/write port
 * - Synchronous write on rising clk edge, asynchronous read
 */
module register_file_single #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 5  // up to 32 registers
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   wen,                // write enable
    input  wire [ADDR_WIDTH-1:0]  addr,               // read/write address
    input  wire [DATA_WIDTH-1:0]  wdata,              // write data
    output wire [DATA_WIDTH-1:0]  rdata               // read data
);

    // Internal register array
    reg [DATA_WIDTH-1:0] regs [(1<<ADDR_WIDTH)-1:0];
    integer i;

    // Write and reset logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < (1<<ADDR_WIDTH); i = i + 1) begin
                regs[i] <= {DATA_WIDTH{1'b0}};
            end
        end else if (wen) begin
            regs[addr] <= wdata;
        end
    end

    // Asynchronous read from same port address
    assign rdata = regs[addr];

endmodule
