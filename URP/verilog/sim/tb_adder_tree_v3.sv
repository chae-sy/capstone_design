`timescale 1ns / 1ps

module tb_adder_tree_v3;

    // Parameters
    parameter INPUT_WIDTH = 8;
    parameter WEIGHT_WIDTH = 8;
    parameter INT_EXTEND = 9;
    parameter DATA_WIDTH = INPUT_WIDTH + WEIGHT_WIDTH + INT_EXTEND;

    // Inputs
    reg signed [(DATA_WIDTH)-1:0] data_i [0:31];
    reg sel;
    // Output
    wire signed [(DATA_WIDTH)-1:0] data_o;

    // Instantiate the adder_tree
    adder_tree #(INPUT_WIDTH, WEIGHT_WIDTH, INT_EXTEND) uut (
        .sel(sel),
        .data_i(data_i),
        .data_o(data_o)
    );

    // Initialize inputs
    initial begin
        $vcdplusfile("tb_adder_tree_v3.vpd");
        $vcdpluson(0, tb_adder_tree_v3);
        $vcdplusmemon();
        // Test case 1
        sel = 0;

        data_i[0] = 1;   // Input data (example values)
        data_i[1] = 2;
        data_i[2] = 3;
        data_i[3] = 4;
        data_i[4] = 5;
        data_i[5] = 6;
        data_i[6] = 7;
        data_i[7] = 8;
        data_i[8] = 9;
        data_i[9] = 10;
        data_i[10] = 11;
        data_i[11] = 12;
        data_i[12] = 13;
        data_i[13] = 14;
        data_i[14] = 15;
        data_i[15] = 16;
        data_i[16] = 17;
        data_i[17] = 18;
        data_i[18] = 19;
        data_i[19] = 20;
        data_i[20] = 21;
        data_i[21] = 22;
        data_i[22] = 23;
        data_i[23] = 24;
        data_i[24] = 25;
        data_i[25] = 26;
        data_i[26] = 27;
        data_i[27] = 28;
        data_i[28] = 29;
        data_i[29] = 30;
        data_i[30] = 31;
        data_i[31] = 32;

        // Wait for the calculations to complete
        #10;

        // Display the output
        $display("Output (data_o): %d", data_o);

        // Add more test cases as needed
        data_i[0] = 10;   // +10
        data_i[1] = 20;   // +20
        data_i[2] = -5;   // -5
        data_i[3] = -15;  // -15
        data_i[4] = 30;   // +30
        data_i[5] = -10;  // -10
        data_i[6] = 25;   // +25
        data_i[7] = -5;   // -5
        data_i[8] = 0;    // 0
        data_i[9] = -20;  // -20
        data_i[10] = 15;  // +15
        data_i[11] = -30; // -30
        data_i[12] = 10;  // +10
        data_i[13] = 5;   // +5
        data_i[14] = -25; // -25
        data_i[15] = 35;  // +35
        data_i[16] = -10; // -10
        data_i[17] = 10;  // +10
        data_i[18] = -5;  // -5
        data_i[19] = 0;   // 0
        data_i[20] = -10; // -10
        data_i[21] = 20;  // +20
        data_i[22] = -15; // -15
        data_i[23] = 25;  // +25
        data_i[24] = -30; // -30
        data_i[25] = 15;  // +15
        data_i[26] = 20;  // +20
        data_i[27] = -10; // -10
        data_i[28] = 5;   // +5
        data_i[29] = -20; // -20
        data_i[30] = 0;   // 0
        data_i[31] = 10;  // +10


        #10
        sel = 1;
        // End simulation
        #10;
        $finish;
    end

endmodule
