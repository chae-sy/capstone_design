`timescale 1ns / 1ps

module tb_top_BCRes_inc;
    // Parameters
    parameter WIDTH_WSRAM_WL = 128;
    parameter WIDTH_FSRAM_WL = 128;
    parameter WIDTH_W_DATA = 8;
    parameter WIDTH_F_DATA = 8;
    parameter WIDTH_FSRAM_ADDR = 10;
    parameter WIDTH_WSRAM_ADDR = 10;

    parameter WIDTH_PE_O_DATA = 20; //18; 
    parameter WIDTH_NORM_O_DATA = 21; //19;
    parameter WIDTH_O_DATA = 8;


    parameter SIZE_KERNEL_H = 3;
    parameter SIZE_KERNEL_W = 3;

    parameter NUM_PE = 16;
    parameter WR_DELAY = 3;

    parameter RELU_MAX_VAL = 6;

    parameter NUM_POOL = 22;

    // Inputs
    reg clk;
    reg rstb;
    reg start;

    reg wr_wsram_sclk;
    reg wr_wsram_ss;
    reg wr_wsram_sdata;
    reg wr_weight_on;

    reg wr_fsram_clk;
    reg [WIDTH_FSRAM_WL-1:0] wr_fsram_data;
    reg [WIDTH_FSRAM_ADDR-1:0] wr_fsram_addr;
    reg wr_fsram_ceb;
    reg wr_fsram_web;
    reg wr_fsram_mux;

    // Outputs
    wire [3:0] max_index;

    // DUT Instance
    top_BCRes #(
    ) uut (
        .rstb(rstb),
        .clk(clk),
        .start(start),
        
        .wr_wsram_sclk(wr_wsram_sclk),
        .wr_wsram_ss(wr_wsram_ss),
        .wr_wsram_sdata(wr_wsram_sdata),
        .wr_weight_on(wr_weight_on),
        
        .wr_fsram_clk(wr_fsram_clk),
        .wr_fsram_data(wr_fsram_data),
        .wr_fsram_addr(wr_fsram_addr),
        .wr_fsram_ceb(wr_fsram_ceb),
        .wr_fsram_web(wr_fsram_web),
        .wr_fsram_mux(wr_fsram_mux),
        
        .max_index(max_index)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100 MHz clock
    end

    // Clock generation
    initial begin
        wr_fsram_clk = 0;
        forever #5 wr_fsram_clk = ~wr_fsram_clk; // 100 MHz clock
    end

    // Testbench stimul
    // Task to load data from file and write to fsram
    integer data_file;
    integer scan_file;
    reg [WIDTH_FSRAM_WL-1:0] data_buffer;
    task load_data_to_fsram;
        input integer start_address;
        integer i;
        begin
            for (i = 0; i < 7; i = i + 1) begin
                wr_fsram_addr = start_address + i;
                scan_file = $fscanf(data_file, "%h\n", data_buffer);
                if (scan_file != 1) begin
                    $display("Error reading data from file!");
                    $finish;
                end
                wr_fsram_data = data_buffer;
                wr_fsram_ceb = 0;
                wr_fsram_web = 0;
                #10;
                wr_fsram_ceb = 1;
                wr_fsram_web = 1;
            end
        end
    endtask

    // Testbench stimulus
    integer iteration;
    initial begin
        // Open data file
//        data_file = $fopen("C:/vivado/CNN_HEAD_MODULE/CNN_HEAD_v2/CNN_HEAD_v2.sim/sim_1/behav/xsim/feature_data_hexa.txt", "r");
        data_file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_hexa.txt", "r");
//        data_file = $fopen("feature_data_hexa.txt", "r");
        if (data_file == 0) begin
            $display("Error opening feature_data_hexa.txt!");
            $finish;
        end

        // Initialize signals
        rstb = 0;
        start = 0;
        wr_wsram_sclk = 0;
        wr_wsram_ss = 0;
        wr_wsram_sdata = 0;
        wr_weight_on = 0;
        wr_fsram_mux = 0;
        wr_fsram_data = 0;
        wr_fsram_addr = 0;
        wr_fsram_ceb = 1;
        wr_fsram_web = 1;
        
        // Reset sequence
        #10 rstb = 1; // Release reset

        // Test scenario
        for (iteration = 0; iteration < 31; iteration = iteration + 1) begin // it=30 fsram write
        
            // Step 1: Write data to fsram addresses 0~6
            wr_fsram_mux = 1;
            load_data_to_fsram((iteration % 3) * 7);
            wr_fsram_mux = 0;

            // Step 2: Trigger start signal
            #10 start = 1;
            #10 start = 0;
                
            
            // Step 3: Wait for 1900 clk
            #19000;
            save_mem_to_file(iteration);
        end
//        save_mem_to_file();
        $fclose(data_file);
        $finish;
    end
    
    // Task to save memory contents in signed decimal format, 8 bits per entry
    task save_mem_to_file;
        input integer iteration;
        integer i, j;
        integer file;
        reg signed [7:0] mem_byte;
        reg [7:0] i_start, i_end;
           
        begin
            if (iteration == 0) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_1.txt", "w");
            else if (iteration == 1) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_2.txt", "w");
            else if (iteration == 2) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_3.txt", "w");
            else if (iteration == 3) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_4.txt", "w");
            else if (iteration == 4) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_5.txt", "w");
            else if (iteration == 5) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_6.txt", "w");
            else if (iteration == 6) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_7.txt", "w");
            else if (iteration == 7) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_8.txt", "w");
            else if (iteration == 8) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_9.txt", "w");
            else if (iteration == 9) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_10.txt", "w");
            else if (iteration == 10) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_11.txt", "w");
            else if (iteration == 11) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_12.txt", "w");
            else if (iteration == 12) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_13.txt", "w");
            else if (iteration == 13) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_14.txt", "w");
            else if (iteration == 14) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_15.txt", "w");
            else if (iteration == 15) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_16.txt", "w");
            else if (iteration == 16) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_17.txt", "w");
            else if (iteration == 17) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_18.txt", "w");
            else if (iteration == 18) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_19.txt", "w");
            else if (iteration == 19) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_20.txt", "w");
            else if (iteration == 20) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_21.txt", "w");
            else if (iteration == 21) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_22.txt", "w");
            else if (iteration == 22) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_23.txt", "w");
            else if (iteration == 23) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_24.txt", "w");
            else if (iteration == 24) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_25.txt", "w");
            else if (iteration == 25) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_26.txt", "w");
            else if (iteration == 26) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_27.txt", "w");
            else if (iteration == 27) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_28.txt", "w");
            else if (iteration == 28) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_29.txt", "w");
            else if (iteration == 29) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_30.txt", "w");
            else if (iteration == 30) file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_31.txt", "w");
            else file = $fopen("C:/Users/tjwws/RES_OUT/feature_data_out_00.txt", "w");
            if (file == 0) begin
                $display("Failed to open file!");
                $finish;
            end
//            $fwrite(" iteration",iteration,"\n");
            // Access F_SRAM_W32_A64 instance directly to retrieve memory content
            
//            if (iteration < 2) begin // 0 1 - only load 
//                i_start = 0;
//                i_end = 20; 
//            end
//            else if ( iteration == 2 ) begin // 2 - load + L1~4
//                i_start = 0;
//                i_end = 62;
//            end
//            else begin
//                i_start = 21;
//                if (iteration < 4) i_end = 62 ; // 3 - L1~4
//                else if (iteration < 6) i_end = 92; // 4 5 - L1~11
//                else if (iteration < 8) i_end = 106; // 6 7 - L1~17
//                else i_end = 143; // 8~30 - L1~22 / 23
//            end    
            i_start = 0;
            i_end = 143;    
            for (i = i_start; i <= i_end; i = i + 1) begin
                // Split 128-bit data into 8 signed 8-bit segments
                $fwrite(file,"%0d : ",  i);
                for (j = 15; j >= 0; j = j - 1) begin
                    mem_byte = uut.xF_SRAM.mem[i][j*8 +: 8];
                    if (j > 0 ) $fwrite(file, "%0d ", mem_byte);
                    else $fwrite(file, "%0d", mem_byte);
                end
                $fwrite(file, "\n");
            end

            $fclose(file);
            $display("Memory values for iteration: ", iteration);
        end
    endtask
    
    // Monitor outputs
    initial begin
        $monitor("Time: %0d, max_index: %b", $time, max_index);
    end

endmodule
