`timescale 1ns / 1ps

module tb_output_buf_v3;

    // Parameters
    localparam INPUT_WIDTH = 4;
    localparam WEIGHT_WIDTH = 4;
    localparam INT_EXTEND = 9;
    localparam PE_NUM = 32;
    localparam INPUT_INT_WIDTH = 1;
    localparam WEIGHT_INT_WIDTH = 1;
    
    // Inputs
    reg clk;
    reg rstb;
    reg rst_local;
    reg sel;
    reg [PE_NUM-1:0] en_in;
    reg signed [(INPUT_WIDTH + WEIGHT_WIDTH + INT_EXTEND)-1:0] data_i [0:PE_NUM-1];
    reg signed [(INPUT_WIDTH + WEIGHT_WIDTH + INT_EXTEND)-1:0] adder_i;

    // Outputs
    wire [INPUT_WIDTH-1:0] data_o [0:PE_NUM-1];

    // Instantiate the output_buf module
    output_buf #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .INT_EXTEND(INT_EXTEND),
        .PE_NUM(PE_NUM),
        .INPUT_INT_WIDTH(INPUT_INT_WIDTH),
        .WEIGHT_INT_WIDTH(WEIGHT_INT_WIDTH)
    ) uut (
        .clk(clk),
        .rstb(rstb),
        .rst_local(rst_local),
        .sel(sel),
        .en_in(en_in),
        .data_i(data_i),
        .adder_i(adder_i),
        .data_o(data_o)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns clock period
    end

    // Test procedure
    initial begin
        $vcdplusfile("tb_output_buf_v3.vpd");
        $vcdpluson(0, tb_output_buf_v3);
        $vcdplusmemon();
        // Initialize signals
        rstb = 0;
        rst_local = 0;
        sel = 0;
        en_in = 0;
        adder_i = 0;

        // Apply reset
        #10;
        rstb = 1; // Release reset
        
        // Test scenario 1: Enable PE input with some data
        #10;
        en_in = 32'hFFFFFFFF; // Enable all PE
        sel = 1; // Select PE input
        for (int i = 0; i < PE_NUM; i++) begin
            data_i[i] = $signed(i); // Assign signed input data
        end

        // Check outputs after some time
        #10;
        for (int i = 0; i < PE_NUM; i++) begin
            $display("data_o[%0d]: %d", i, data_o[i]);
        end

        // Test scenario 2: Switch to adder input
        #10;
        sel = 0; // Select adder input
        adder_i = $signed(50); // Set adder input to a constant value

        // Check outputs after some time
        #10;
        for (int i = 0; i < PE_NUM; i++) begin
            $display("data_o[%0d]: %d", i, data_o[i]);
        end
        
        // Test scenario 3: Reset local
        #10;
        rst_local = 1; // Set local reset
        #10;
        rst_local = 0; // Release local reset

        // Check outputs after reset
        #10;
        for (int i = 0; i < PE_NUM; i++) begin
            $display("data_o[%0d]: %d (After Local Reset)", i, data_o[i]);
        end

        // End simulation
        #20;
        $finish;
    end

endmodule
