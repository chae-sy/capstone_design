//top
module SPRNN_Top
(
    input   wire                clk,
    input   wire                rst_n; 
);


    mem       u_mem
    (
        .clk                        (clk),
        .rst_n                      (rst_n),
    );
    SRAM_W32_A64 u_memA(  // Data Storage
    	.CLK                        (clk),
    	input		CEB,
    	input		WEB,
    	input	[5:0]	A,
    	input	[31:0]	D,
    	output	[31:0]	Q
    );
    in_buf      u_in_buf
    (
        .clk                        (clk),
        .rst_n                      (rst_n),
    );
    out_buf     u_out_buf
    (
        .clk                        (clk),
        .rst_n                      (rst_n),
    );
    PE_array    u_PE_array
    (
        .clk                        (clk),
        .rst_n                      (rst_n),
    );
    ReLU        u_ReLU
    (
        .clk                        (clk),
        .rst_n                      (rst_n),
    );
    controller  u_controller
    (
        .clk                        (clk),
        .rst_n                      (rst_n),
    );


endmodule
