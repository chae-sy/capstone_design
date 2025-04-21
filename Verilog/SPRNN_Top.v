//top
module SPRNN_Top
(
    input   wire                clk,
    input   wire                rst_n; 
);


    mem_A       u_mem_A
    (
        .clk                        (clk),
        .rst_n                      (rst_n),
    );
    mem_B       u_mem_B
    (
        .clk                        (clk),
        .rst_n                      (rst_n),
    );
    in_buff     u_in_buff
    (
        .clk                        (clk),
        .rst_n                      (rst_n),
    );
    out_buff    u_out_buff
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
