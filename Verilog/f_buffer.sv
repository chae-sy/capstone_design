`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
//
// Create Date: 2025/05/18
// Design Name: 
// Module Name: f_buffer_v1 (simplified with for-loops)
//
// Description: 
//   Refactored INPUT_BUFFER (formerly f_buffer_v0) using loops and packed arrays
//////////////////////////////////////////////////////////////////////////////////

module f_buffer_v1 #(
    parameter WIDTH_FSRAM_WL = 128,
    parameter DATA_WIDTH    = 8,
    parameter NUM_CHNL          = 16,
    parameter SIZE_BUFFER_H   = 3, 
    parameter SIZE_BUFFER_W   = 4,
    parameter SIZE_KERNEL_H = 3,
    parameter SIZE_KERNEL_W = 3
)(
    input                                 clk,
    input                                 rst_n,
    input                                 buffer_load_f, // load feature
    input        [$clog2(SIZE_BUFFER_H*SIZE_BUFFER_W)-1:0] buffer_ptr_f, // pointer 
    input                                 buffer_start, // output start
    input                                 shift,
    // pack f_data ports into an array
    input  [WIDTH_FSRAM_WL-1:0]           f_data_in, // from SRAM
    output reg [DATA_WIDTH*NUM_CHNL-1:0]  f_buffer_out
);

    // buffer storage: [row][pe]
    reg [DATA_WIDTH-1:0] buffer_data [0:SIZE_BUFFER_H*SIZE_BUFFER_W-1][0:NUM_CHNL-1];
    reg [5:0] counter;
    integer i, k, r;

    // 8?��꾪듃吏쒕?�� 16媛� �슂�냼 諛곗�? �꽑�뼵 (SystemVerilog �뒪���씪)
    wire [DATA_WIDTH-1:0] f_data [0:NUM_CHNL-1];

    genvar a;
    generate
    for (a = 0; a < NUM_CHNL; a = a + 1) begin
        // 128 -> 8bits x 16 channel
        assign f_data[a] = f_data_in[DATA_WIDTH*a +: DATA_WIDTH];
    end
    endgenerate


    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // reset all buffer_data elements
            for (r = 0; r < SIZE_BUFFER_W*SIZE_BUFFER_H; r = r + 1) begin
                for (i = 0; i < NUM_CHNL; i = i + 1) begin
                        buffer_data[r][i] <= {DATA_WIDTH{1'b0}};
                    end
                end
            
            counter <= 0;
            f_buffer_out <= 0;

        end else begin
            
                // load new features into the pointer-specified row, column
                if (buffer_load_f) begin
                    for (i = 0; i < NUM_CHNL; i = i + 1) begin
                        buffer_data[buffer_ptr_f][i] <= f_data[i];
                    end
                end

                // shift window right by one
                if (shift) begin
                counter <= 0;
                if (counter == 0) begin
                    
                    for (r = 0; r < SIZE_BUFFER_W-1; r = r + 1) begin
                        for (i = 0; i < NUM_CHNL; i = i + 1) begin
                            for (k = 0; k < SIZE_BUFFER_H; k = k + 1) begin
                                buffer_data[k][r][i] <= buffer_data[k][r+1][i]; // shift
                            end
                        end
                    end
                   
                    end
                    if (counter < SIZE_BUFFER_H) begin
                        for (i = 0; i < NUM_CHNL; i = i + 1) begin
                            buffer_data[counter][SIZE_BUFFER_W-1][i] <= f_data[i]; //loading new data 
                        end
                        counter <= counter + 1;
                     end else begin
                        
                        counter <= 0;
                    end
                end
                
                
                // start output
                if (buffer_start) begin
                            // example for 3x3 head
                            if (counter < SIZE_KERNEL_W*SIZE_KERNEL_H-1) begin
                                // broadcast one tapped value to all PEs
                                for (i = 0; i < NUM_CHNL; i = i + 1) begin
                                    f_buffer_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] 
                                    <= buffer_data[counter % SIZE_KERNEL_H][counter / SIZE_KERNEL_H][i];
                                end
                                counter <= counter + 1;
                          
                        
                end else begin
                    f_buffer_out <= 0;
                end
            end
        end//59
    end
endmodule
