`timescale 1ns / 1ps

module tb_mac_pipeline_superscalar;

    // Parameters
    parameter DATA_WIDTH = 8;
    parameter NUM_STAGE  = 9;
    parameter LANE_NUM   = 3;

    // Signals
    reg clk;
    reg rst_n;
    reg pe_en;
    reg [DATA_WIDTH-1:0] data_in [0:LANE_NUM-1];
    reg [DATA_WIDTH-1:0] weight_in;

    wire pe_done;
    wire [19:0] result_out_flat[0:LANE_NUM-1];

    // Instantiate DUT
    mac_pipeline_superscalar #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_STAGE(NUM_STAGE),
        .LANE_NUM(LANE_NUM)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .pe_en(pe_en),
        .data_in(data_in),
        .weight_in(weight_in),
        .pe_done(pe_done),
        .result_out_flat(result_out_flat)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Task to drive data inputs on posedge clk
    task send_data(input int i, input [7:0] w);
        begin
            @(posedge clk);
            pe_en <= 1;
            weight_in <= w;
            data_in[0] <= 2 * i;
            data_in[1] <= 3 * i;
            data_in[2] <= 4 * i;
        end
    endtask

    // Test sequence
    initial begin
        // 초기화
        rst_n = 0;
        pe_en = 0;
        weight_in = 0;
        data_in[0] = 0;
        data_in[1] = 0;
        data_in[2] = 0;
        @(posedge clk);
        @(posedge clk);
        rst_n = 1;

        // 첫 번째 패스: weight=1
        for (int i = 1; i <= NUM_STAGE; i++) begin
            send_data(i, 8'd1);
        end

        // 두 번째 패스: weight=2
        for (int i = 1; i <= NUM_STAGE; i++) begin
            send_data(i, 8'd2);
        end
         @(posedge clk);
        pe_en = 0;
        // 파이프라인 비우기
        repeat (NUM_STAGE + 3) @(posedge clk);

        $display("Simulation completed at %0t ns", $time);
        $finish;
    end

    // 출력 모니터링
    initial begin
        $monitor("%0t ns | pe_done=%b | result=[%0d,%0d,%0d]", 
                 $time, 
                 pe_done,
                 result_out_flat[0],
                 result_out_flat[1],
                 result_out_flat[2]);
    end

endmodule
