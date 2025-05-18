`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Testbench for updated f_buffer_v1
//////////////////////////////////////////////////////////////////////////////////

module tb_f_buffer_v1_updated;
    // Match DUT parameters
    parameter WIDTH_F_DATA    = 8;
    parameter NUM_CHNL        = 16;
    parameter SIZE_BUFFER_H   = 3;
    parameter SIZE_BUFFER_W   = 4;
    parameter SIZE_KERNEL_H   = 3;
    parameter SIZE_KERNEL_W   = 3;

    // Clock / reset
    reg clk;
    reg rst_n;

    // DUT inputs
    reg [4:0] buffer_mode;
    reg       buffer_load_f;
    reg [$clog2(SIZE_BUFFER_H)-1:0] buffer_ptr_h_f;
    reg [$clog2(SIZE_BUFFER_W)-1:0] buffer_ptr_w_f;
    reg       buffer_start;
    reg       shift;
    reg       pad_en;
    reg [WIDTH_F_DATA-1:0] f_data [0:NUM_CHNL-1];

    // DUT output
    wire [WIDTH_F_DATA*NUM_CHNL-1:0] f_buffer_out;

    // DUT instance
    f_buffer_v1 #(
        .WIDTH_F_DATA(WIDTH_F_DATA),
        .NUM_CHNL(NUM_CHNL),
        .SIZE_BUFFER_H(SIZE_BUFFER_H),
        .SIZE_BUFFER_W(SIZE_BUFFER_W),
        .SIZE_KERNEL_H(SIZE_KERNEL_H),
        .SIZE_KERNEL_W(SIZE_KERNEL_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .buffer_mode(buffer_mode),
        .buffer_load_f(buffer_load_f),
        .buffer_ptr_h_f(buffer_ptr_h_f),
        .buffer_ptr_w_f(buffer_ptr_w_f),
        .buffer_start(buffer_start),
        .shift(shift),
        .pad_en(pad_en),
        .f_data(f_data),
        .f_buffer_out(f_buffer_out)
    );

    // 100 MHz clock
    initial clk = 0;
    always #5 clk = ~clk;

    integer i;

    initial begin
        // INITIALIZE
        rst_n           = 0;
        buffer_mode     = 0;
        buffer_load_f   = 0;
        buffer_ptr_h_f  = 0;
        buffer_ptr_w_f  = 0;
        buffer_start    = 0;
        shift           = 0;
        pad_en          = 0;
        for (i = 0; i < NUM_CHNL; i = i + 1) f_data[i] = 0;

        // DEASSERT RESET
        #20;
        rst_n = 1;
        #10;

        // === MODE 1: 3×3 patch + padding + stream ===
        buffer_mode = 1;

        // Load row 0, col 0 with values 0-15
        buffer_ptr_h_f = 0;
        buffer_ptr_w_f = 0;
        for (i = 0; i < NUM_CHNL; i = i + 1) f_data[i] = i+1;
        buffer_load_f = 1;
        #10; buffer_load_f = 0;
        
        buffer_ptr_h_f = 0;
        buffer_ptr_w_f = 1;
        for (i = 0; i < NUM_CHNL; i = i + 1) f_data[i] = i+22;
        buffer_load_f = 1;
        #10; buffer_load_f = 0;
        
        buffer_ptr_h_f = 0;
        buffer_ptr_w_f = 2;
        for (i = 0; i < NUM_CHNL; i = i + 1) f_data[i] = i+3;
        buffer_load_f = 1;
        #10; buffer_load_f = 0;
        
        buffer_ptr_h_f = 0;
        buffer_ptr_w_f = 3;
        for (i = 0; i < NUM_CHNL; i = i + 1) f_data[i] = i+44;
        buffer_load_f = 1;
        #10; buffer_load_f = 0;
        
        // Load row 1, col 1 with values 16-31
        buffer_ptr_h_f = 1;
        buffer_ptr_w_f = 1;
        for (i = 0; i < NUM_CHNL; i = i + 1) f_data[i] = i + 16;
        buffer_load_f = 1;
        #10; buffer_load_f = 0;

        // Inject a padded column at col=2
        buffer_ptr_w_f = 0;
        pad_en = 1;
        #10; pad_en = 0;

        // Start streaming 3×3 taps
        buffer_start = 1;
        // 3×3 = 9 cycles
        #90;
        buffer_start = 0;
        #20;

        // === MODE 2: full-width row + shift + stream ===
        buffer_mode = 2;

        // Load row 0, col 0 with values 100-115
        buffer_ptr_h_f = 0;
        buffer_ptr_w_f = 0;
        for (i = 0; i < NUM_CHNL; i = i + 1) f_data[i] = 100 + i;
        buffer_load_f = 1;
        #10; buffer_load_f = 0;

        // Shift the window right by one, inserting values 200-215
        shift = 1;
        for (i = 0; i < NUM_CHNL; i = i + 1) f_data[i] = 100 + i;
        #10; 
        for (i = 0; i < NUM_CHNL; i = i + 1) f_data[i] = 200 + i;
        #10;
        for (i = 0; i < NUM_CHNL; i = i + 1) f_data[i] = 300 + i;
        #10;
        shift = 0;

        // Start streaming SIZE_BUFFER_W positions (4 cycles)
        buffer_start = 1;
        #40;
        buffer_start = 0;
        #20;

        // SIMULATION COMPLETE
        #50;
        $finish;
    end

endmodule
