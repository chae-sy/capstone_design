`timescale 1ns / 1ps
`define CLK_SPI_PERIOD  10.1
`define CLK_DATA_PERIOD 7.8125 

module tb_top_dscnn_wrapper;

    parameter WEIGHT_BIT = 8;
    parameter INPUT_BIT = 8;
    parameter NUM_CHANNEL = 32;

    reg rstb;
    reg clk;

    reg ib_ss;
    reg ib_sdata;
    reg wr_input_on;

    reg rf_sclk;
    reg rf_ss;
    reg rf_sdata;
    reg wr_weight_on;

    wire valid_o;
    wire[3:0] decision_o;

    top_dscnn_wrapper #(
        .WEIGHT_BIT(8),
        .INPUT_BIT(8),
        .NUM_CHANNEL(32)
    ) uut (
        .rstb(rstb),
        .clk(clk),
        
        .ib_ss(ib_ss),
        .ib_sdata(ib_sdata),
        .wr_input_on(wr_input_on),
        
        .rf_sclk(rf_sclk),
        .rf_ss(rf_ss),
        .rf_sdata(rf_sdata),
        .wr_weight_on(wr_weight_on),
        .valid_o(valid_o),
        .decision_o(decision_o)
    );



    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns clock period
    end

    integer i, j;

    always @(*) begin
        if (valid_o) begin
            #10 wr_input_on = 1;
            for (integer addr = 0; addr < 10; addr = addr + 1) begin
                spi_input_write(addr, {8'b1});  // Write data equal to the address
                #20;  // Add some delay between each write operation
            end
            #20 wr_input_on = 0;  // Add some delay between each write operation
        end
        
    end

    // Initial block to set up inputs and run the test sequence
    initial begin
        // Initialize inputs

        $vcdplusfile("tb_top_dscnn_wrapper.vpd");
        $vcdpluson(0, tb_top_dscnn_wrapper);
        $vcdplusmemon();
        wr_input_on = 0;
        wr_weight_on = 0;

        rstb = 0;
        //calc_en_i = 0;
        //data_i = 1;

        #30 rstb = 0;
        // Apply reset
        #10 rstb = 1;  // Release reset

        // Test sequence
        wr_weight_on = 1;
        #(`CLK_DATA_PERIOD*1);
        #(`CLK_DATA_PERIOD*10);
        rf_sclk = 0;
        #(`CLK_SPI_PERIOD/2);
        rf_sclk = 1;
        #(`CLK_SPI_PERIOD/2);
        rf_ss = 1;
		//integrate_counte_spi
		for (integer addr = 0; addr < 144; addr = addr + 1) begin
            spi_weight_write(addr, {32{8'b1}});  // Write data equal to the address
            #20;  // Add some delay between each write operation
        end
        wr_weight_on = 0;

/*
        for(i=0; i<144; i=i+1) begin
            #20 weight_en_i = 1; // Trigger weight initialization
            #320 weight_en_i = 0;
        end
*/

        for(j=0; j<29; j=j+1) begin
            #10 wr_input_on = 1;
            #(`CLK_DATA_PERIOD*1);
            #(`CLK_DATA_PERIOD*10);
            rf_sclk = 0;
            #(`CLK_SPI_PERIOD/2);
            rf_sclk = 1;
            #(`CLK_SPI_PERIOD/2);
            ib_ss = 1;
            for (integer addr = 0; addr < 10; addr = addr + 1) begin
                spi_input_write(addr, {8'b1});  // Write data equal to the address
                #20;  // Add some delay between each write operation
            end
            #20 wr_input_on = 0;  // Add some delay between each write operation
            #6000;
        end


        // Finish simulation
        #100000 $finish;
    end

    task spi_weight_write(
        input [7:0] addr,
        input [255:0] data
        );
        integer i;
        begin
            #(`CLK_SPI_PERIOD/2);
            rf_ss = 0;
            #(`CLK_SPI_PERIOD/2);
            rf_sclk = 1;
            #(`CLK_SPI_PERIOD/2);
            rf_sclk = 0;
            //send ADDR
            for (i=0;i<8;i=i+1) begin
                #(`CLK_SPI_PERIOD/2);
                rf_sclk = 1;
                rf_sdata = addr[7-i];
                #(`CLK_SPI_PERIOD/2);
                rf_sclk = 0;
            end
    
            //send DATA
            for (i=0;i<256;i=i+1) begin
                #(`CLK_SPI_PERIOD/2);
                rf_sclk = 1;
                rf_sdata = data[255-i];
                #(`CLK_SPI_PERIOD/2);
                rf_sclk = 0;
            end
            #(`CLK_SPI_PERIOD/2);
            rf_sclk = 0;
            #(`CLK_SPI_PERIOD/2);
            rf_sclk = 1;
            #(`CLK_SPI_PERIOD/2);
            rf_ss = 1;
        end
    endtask

    task spi_input_write(
        input [3:0] addr,
        input [7:0] data
        );
        integer i;
        begin
            #(`CLK_SPI_PERIOD/2);
            ib_ss = 0;
            #(`CLK_SPI_PERIOD/2);
            rf_sclk = 1;
            #(`CLK_SPI_PERIOD/2);
            rf_sclk = 0;
            //send ADDR
            for (i=0;i<4;i=i+1) begin
                #(`CLK_SPI_PERIOD/2);
                rf_sclk = 1;
                ib_sdata = addr[3-i];
                #(`CLK_SPI_PERIOD/2);
                rf_sclk = 0;
            end
    
            //send DATA
            for (i=0;i<8;i=i+1) begin
                #(`CLK_SPI_PERIOD/2);
                rf_sclk = 1;
                ib_sdata = data[7-i];
                #(`CLK_SPI_PERIOD/2);
                rf_sclk = 0;
            end
            #(`CLK_SPI_PERIOD/2);
            rf_sclk = 0;
            #(`CLK_SPI_PERIOD/2);
            rf_sclk = 1;
            #(`CLK_SPI_PERIOD/2);
            ib_ss = 1;
        end
    endtask
    

endmodule
