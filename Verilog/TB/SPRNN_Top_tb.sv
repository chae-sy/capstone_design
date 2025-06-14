`timescale 1ns/1ps

module SPRNN_Top_tb;

    // Clock and reset
    reg clk;
    reg rst_n;

    // Inputs
    reg start;
    reg [15:0] memA_addr_i;
    reg [15:0] memB_addr_i;
    reg [9:0]  wmem_addr_i;
    reg [127:0] memA_d_i;
    reg [127:0] memB_d_i;
    reg [127:0] wmem_d_i;
    reg wren_bias_i;
    reg [2:0] write_addr_bias_i;
    reg [511:0] write_data_bias_i;
    reg initial_SRAMw_done;
    reg initial_weight_done;
    reg [2:0] layer_num;
    reg layer_done;
    reg total_done_o;

    // DUT ?씤?뒪?꽩?뒪
    SPRNN_Top #(
        .DATA_WIDTH(8),
        .NUM_COLOR(3),
        .NUM_CHNL(16)
    ) dut (
        .clk                (clk),
        .rst_n              (rst_n),
        .start              (start),
        .memA_addr_i        (memA_addr_i),
        .memB_addr_i        (memB_addr_i),
        .wmem_addr_i        (wmem_addr_i),
        .memA_d_i           (memA_d_i),
        .memB_d_i           (memB_d_i),
        .wmem_d_i           (wmem_d_i),
        .wren_bias_i        (wren_bias_i),
        .write_addr_bias_i  (write_addr_bias_i),
        .write_data_bias_i  (write_data_bias_i),
        .initial_SRAMw_done (initial_SRAMw_done),
        .initial_weight_done(initial_weight_done),
        .layer_num_o        (layer_num),
        .layer_done_o       (layer_done),
        .total_done_o       (total_done_o)
    );

    // Clock ?깮?꽦
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz
    end

    // Task to load data from file and write to sram
    integer data_file_in, data_file_w, data_file_b;
    integer scan_file_in, scan_file_w, scan_file_b;
    reg [127:0] data_buffer_in, data_buffer_w;
    reg [511:0] data_buffer_b;

    task load_data_to_sram;
        input integer start_address;
        integer i;
        begin
            // Bias regfile 
            for (i = 0; i < 5; i = i + 1) begin
                wren_bias_i = 1;
                write_addr_bias_i = i;
                scan_file_b = $fscanf(data_file_b, "%h\n", data_buffer_b);
                if (scan_file_b != 1) begin
                    $display("Error reading data from file!");
                    $finish;
                end
                write_data_bias_i = data_buffer_b;
                #10;
            end
            wren_bias_i = 0;

            // weight 
            for (i = 0; i < 585; i = i + 1) begin
                wmem_addr_i = i;
                scan_file_w = $fscanf(data_file_w, "%h\n", data_buffer_w);
                if (scan_file_w != 1) begin
                    $display("Error reading data from file!");
                    $finish;
                end
                wmem_d_i = data_buffer_w;
                #10;
            end

            // memA, memB 
            for (i = 0; i < 31008; i = i + 1) begin
                memA_addr_i = i;
                memB_addr_i = i;
                scan_file_in = $fscanf(data_file_in, "%h\n", data_buffer_in);
                if (scan_file_in != 1) begin
                    $display("Error reading data from file!");
                    $finish;
                end
                memA_d_i = data_buffer_in;
                memB_d_i = 128'd0;
                #10;
            end
            initial_SRAMw_done = 1;
            initial_weight_done = 1;
        end
    endtask

    integer it, i;
    initial begin

        // Open data file
        data_file_in = $fopen("C:/Users/LG/OneDrive/문서/GitHub/capstone_design/Verification/text/input.txt", "r");
        data_file_w = $fopen("C:/Users/LG/OneDrive/문서/GitHub/capstone_design/Verification/text/mem_w.txt", "r");
        data_file_b = $fopen("C:/Users/LG/OneDrive/문서/GitHub/capstone_design/Verification/text/reg_b.txt", "r");
        if (data_file_in == 0) begin
            $display("Error opening feature_data_hexa.txt!");
            $finish;
        end
        if (data_file_w == 0) begin
            $display("Error opening feature_data_hexa.txt!");
            $finish;
        end
        if (data_file_b == 0) begin
            $display("Error opening feature_data_hexa.txt!");
            $finish;
        end

        rst_n = 0; start = 0; initial_SRAMw_done = 0; initial_weight_done = 0;
        memA_addr_i = 0; memB_addr_i = 0; wmem_addr_i = 0;
        memA_d_i = 0; memB_d_i = 0; wmem_d_i = 0;
        wren_bias_i = 0; write_addr_bias_i = 0; write_data_bias_i = 0;

        #20 rst_n = 1;
        #5  start = 1;

        load_data_to_sram((i % 3) * 7);
        
        @(negedge layer_done);
        
        // Test scenario
        for (it = 0; it < 6; it = it + 1) begin // 
            @(negedge layer_done);
            i = layer_num-2;
            save_mem_to_file(i);
            # 20;
//            @(posedge layer_done);
        end
        
        @(negedge layer_done);
        i = 5;
        save_mem_to_file(i);

        $fclose(data_file_in);
        $fclose(data_file_w);
        $fclose(data_file_b);

        #100;
        $finish;
    end
    

        // Task to save memory contents in signed decimal format, 8 bits per entry
    task save_mem_to_file;
        input integer i;
        integer j, k;
        integer file;
        reg signed [7:0] mem_byte;
        reg [7:0] i_start, i_end;
           
        begin
            if (i == 0) file = $fopen("C:/Users/LG/OneDrive/문서/GitHub/capstone_design/Verification/text/feature_data_out_1.txt", "w");
            else if (i == 1) file = $fopen("C:/Users/LG/OneDrive/문서/GitHub/capstone_design/Verification/text/feature_data_out_2.txt", "w");
            else if (i == 2) file = $fopen("C:/Users/LG/OneDrive/문서/GitHub/capstone_design/Verification/text/feature_data_out_3.txt", "w");
            else if (i == 3) file = $fopen("C:/Users/LG/OneDrive/문서/GitHub/capstone_design/Verification/text/feature_data_out_4.txt", "w");
            else if (i == 4) file = $fopen("C:/Users/LG/OneDrive/문서/GitHub/capstone_design/Verification/text/feature_data_out_5.txt", "w");
            else if (i == 5) file = $fopen("C:/Users/LG/OneDrive/문서/GitHub/capstone_design/Verification/text/feature_data_out_6.txt", "w");
            else file = $fopen("C:\Users\LG\OneDrive\문서\GitHub\capstone_design\Verification\text\feature_data_out_00.txt", "w");
            if (file == 0) begin
                $display("Failed to open file!");
                $finish;
            end
 
            if (i % 2 == 0) begin // layer 0, 2, 4
                 for (j = 0; j < 31008; j = j + 1) begin
                    // Split 128-bit data into 8 signed 8-bit segments
                    $fwrite(file,"%0d : ",  j);
                    for (k = 15; k >= 0; k = k - 1) begin
                        mem_byte = dut.u_memB.mem_W[j][k * 8 +: 8];
                        if (k > 0 ) $fwrite(file, "%0d ", mem_byte);
                        else $fwrite(file, "%0d", mem_byte);
                    end
                    $fwrite(file, "\n");
                end
            end 
            else begin // layer 1, 3, 5
                for (j = 0; j < 31008; j = j + 1) begin
                    // Split 128-bit data into 8 signed 8-bit segments
                    $fwrite(file,"%0d : ",  j);
                    for (k = 15; k >= 0; k = k - 1) begin
                        mem_byte = dut.u_memA.mem_W[j][k * 8 +: 8];
                        if (k > 0 ) $fwrite(file, "%0d ", mem_byte);
                        else $fwrite(file, "%0d", mem_byte);
                    end
                    $fwrite(file, "\n");
                end
            end

            $fclose(file);
            $display("Memory values for iteration: ", i);
        end
    endtask

    initial begin
        $monitor("Time=%0t | rst_n=%b | start=%b | layer_done_o=%b | total_done_o=%b",
                 $time, rst_n, start, layer_done, total_done_o);
    end

endmodule

