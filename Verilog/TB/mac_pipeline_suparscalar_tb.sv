`timescale 1ns / 1ps

module tb_mac_pipeline_superscalar;
    // Parameters
    parameter DATA_WIDTH = 8;
    parameter NUM_STAGE  = 9;
    parameter LANE_NUM   = 3;
    
    integer i;
    // Clock and reset
    reg clk;
    reg rst_n;

    // Drive signals
    reg pe_en;
    reg [DATA_WIDTH-1:0] data_in [0:LANE_NUM-1];
    reg [DATA_WIDTH-1:0] weight_in;

    // Outputs
    wire pe_done;
    wire [LANE_NUM*2*DATA_WIDTH-1:0] result_out_flat;

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

    // Clock generation: 10ns period
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test sequence
    initial begin
        // Initialize inputs
        rst_n = 0;
        pe_en = 0;
        weight_in = 0;
        data_in[0] = 0;
        data_in[1] = 0;
        data_in[2] = 0;
        #20;

        // Release reset
        rst_n = 1;
        #10;
        
        #5;
        for (i=1; i<(NUM_STAGE+1); i=i+1) begin
            weight_in = 1;
            data_in[0] = 2*i;
            data_in[1] = 3*i;
            data_in[2] = 4*i;
            pe_en=1;
            #10;
        end
       
        pe_en = 0;  // disable

        // Wait for pipeline to flush
        #((NUM_STAGE+2)*10);

        $display("Simulation completed at %0t ns", $time);
        $finish;
    end

    // Monitor outputs
    initial begin
        $monitor("%0t ns | pe_done=%b | result=[%0d,%0d,%0d]", 
                 $time, 
                 pe_done,
                 result_out_flat[ 2*DATA_WIDTH*0 +: 2*DATA_WIDTH],
                 result_out_flat[ 2*DATA_WIDTH*1 +: 2*DATA_WIDTH],
                 result_out_flat[ 2*DATA_WIDTH*2 +: 2*DATA_WIDTH]);
    end

endmodule
