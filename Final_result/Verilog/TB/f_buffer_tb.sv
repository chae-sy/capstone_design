`timescale 1ns / 1ps

module f_buffer_tb;

    parameter DATA_WIDTH      = 8;
    parameter NUM_CHNL        = 16;
    parameter SIZE_BUFFER_H   = 3;
    parameter SIZE_BUFFER_W   = 4;
    parameter SIZE_KERNEL_H   = 3;
    parameter SIZE_KERNEL_W   = 3;
    parameter TOTAL_WIDTH     = NUM_CHNL * DATA_WIDTH;

    reg                          clk;
    reg                          rst_n;
    reg                          is_initial;
    reg                          wren;
    reg                          rden;
    reg  [TOTAL_WIDTH-1:0]       data_in;
    wire [TOTAL_WIDTH-1:0]       data_out;
    wire                         f_buffer_done;

    f_buffer #(
        .DATA_WIDTH     (DATA_WIDTH),
        .NUM_CHNL       (NUM_CHNL),
        .SIZE_BUFFER_H  (SIZE_BUFFER_H),
        .SIZE_BUFFER_W  (SIZE_BUFFER_W),
        .SIZE_KERNEL_H  (SIZE_KERNEL_H),
        .SIZE_KERNEL_W  (SIZE_KERNEL_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .is_initial(is_initial),
        .wren(wren),
        .rden(rden),
        .data_in(data_in),
        .data_out(data_out),
        .f_buffer_done(f_buffer_done)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        integer i;

        rst_n = 0; is_initial = 0; wren = 0; rden = 0; data_in = 0;
        #20; rst_n = 1;
        #5;
        // ========== CASE 1: 초기 로딩 ==========
        $display("\n=== CASE 1: 초기 로딩 (9사이클 wren=1, is_initial=1) ===");
        
        for (i = 0; i < 9; i = i + 1) begin
            @(posedge clk);
            data_in = TOTAL_WIDTH'(i);
            is_initial = 1; wren = 1;
            $display("[%0t] CASE1 [%0d] data_in=%h data_out=%h done=%b", $time, i, data_in, data_out, f_buffer_done);
        end
        @(posedge clk);
        wren = 0; is_initial = 0;

        $display("\n=== CASE 1: 바로 다음 9사이클 rden=1 ===");
        rden = 1;
        for (i = 0; i < 9; i = i + 1) begin
            @(posedge clk);
            $display("[%0t] CASE1-rden [%0d] data_out=%h done=%b", $time, i, data_out, f_buffer_done);
        end
        rden = 0;

        // ========== CASE 2: 3사이클 wren=1 + 9사이클 rden=1 ==========
        $display("\n=== CASE 2: 3사이클 wren=1 + 9사이클 rden=1 ===");
        rden = 1; wren = 1;
        for (i = 0; i < 3; i = i + 1) begin
            data_in = TOTAL_WIDTH'(100 + i);
            @(posedge clk);
            $display("[%0t] CASE2-wr [%0d] data_in=%h data_out=%h done=%b", $time, i, data_in, data_out, f_buffer_done);
            $display("[%0t] CASE2-rd [%0d] data_out=%h done=%b", $time, i, data_out, f_buffer_done);
        end
        wren = 0;
        for (; i < 9; i = i + 1) begin
            @(posedge clk);
            $display("[%0t] CASE2-rd [%0d] data_out=%h done=%b", $time, i, data_out, f_buffer_done);
        end
        rden = 0;
        
        $display("\n=== CASE 2: 바로 다음 9사이클 rden=1 ===");
        rden = 1;
        for (i = 0; i < 9; i = i + 1) begin
            @(posedge clk);
            $display("[%0t] CASE2-rden [%0d] data_out=%h done=%b", $time, i, data_out, f_buffer_done);
        end
        rden = 0;

        // ========== CASE 3: rden=1(9사이클) + wren=1(4사이클 후 3사이클) ==========
        $display("\n=== CASE 3: rden=1(9사이클) + wren=1(4사이클 후 3사이클) ===");
        rden = 1;
        for (i = 0; i < 3; i = i + 1) begin
            @(posedge clk);
            $display("[%0t] CASE3-rd-pre [%0d] data_out=%h done=%b", $time, i, data_out, f_buffer_done);
        end
        wren = 1;
        for (; i < 6; i = i + 1) begin
            data_in = TOTAL_WIDTH'(200 + i);
            @(posedge clk);
            $display("[%0t] CASE3-wr [%0d] data_in=%h data_out=%h done=%b", $time, i, data_in, data_out, f_buffer_done);
            $display("[%0t] CASE3-rd [%0d] data_out=%h done=%b", $time, i, data_out, f_buffer_done);
        end
        wren = 0;
        for (; i < 9; i = i + 1) begin
            @(posedge clk);
            $display("[%0t] CASE3-rd-post [%0d] data_out=%h done=%b", $time, i, data_out, f_buffer_done);
        end
        rden = 0;
        
        $display("\n=== CASE 3: 바로 다음 9사이클 rden=1 ===");
        rden = 1;
        for (i = 0; i < 9; i = i + 1) begin
            @(posedge clk);
            $display("[%0t] CASE3-rden [%0d] data_out=%h done=%b", $time, i, data_out, f_buffer_done);
        end
        rden = 0;

        // ========== CASE 4: rden=1(9사이클) + wren=1(7사이클 후 3사이클) ==========
        $display("\n=== CASE 4: rden=1(9사이클) + wren=1(7사이클 후 3사이클) ===");
        rden = 1;
        for (i = 0; i < 6; i = i + 1) begin
            @(posedge clk);
            $display("[%0t] CASE4-rd-pre [%0d] data_out=%h done=%b", $time, i, data_out, f_buffer_done);
        end
        wren = 1;
        for (; i < 9; i = i + 1) begin
            data_in = TOTAL_WIDTH'(300 + i);
            @(posedge clk);
            $display("[%0t] CASE4-wr [%0d] data_in=%h data_out=%h done=%b", $time, i, data_in, data_out, f_buffer_done);
            $display("[%0t] CASE4-rd [%0d] data_out=%h done=%b", $time, i, data_out, f_buffer_done);
        end
        wren = 0;
        rden = 0;
        
        $display("\n=== CASE 4: 바로 다음 9사이클 rden=1 ===");
        rden = 1;
        for (i = 0; i < 9; i = i + 1) begin
            @(posedge clk);
            $display("[%0t] CASE4-rden [%0d] data_out=%h done=%b", $time, i, data_out, f_buffer_done);
        end
        rden = 0;

        #20; $display("[%0t] Testbench finished.", $time); $stop;
    end

endmodule
