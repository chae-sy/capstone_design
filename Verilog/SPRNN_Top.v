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
    SRAM_W32_A64 u_memA(  // Data Storage A
    	.CLK                        (clk),
        .CEB                        (),
        .WEB                        (),
        .A                          (),
    	.D                          (),
    	.Q                          (),
    );
    
    SRAM_W32_A64 u_memB(  // Data Storage B
    	.CLK                        (clk),
        .CEB                        (),
        .WEB                        (),
        .A                          (),
    	.D                          (),
    	.Q                          (),
    );
    
    SRAM_W32_A64 u_memW(  // Data Storage weight
    	.CLK                        (clk),
        .CEB                        (),
        .WEB                        (),
        .A                          (),
    	.D                          (),
    	.Q                          (),
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
