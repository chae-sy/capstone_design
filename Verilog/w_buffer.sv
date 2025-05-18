`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
//
// Create Date: 2025/05/18
// Design Name: 
// Module Name: w_buffer_v1 (simplified with for-loops)
//
// Description: 
//   Refactored INPUT_BUFFER (formerly f_buffer_v0) using loops and packed arrays
//////////////////////////////////////////////////////////////////////////////////

module w_buffer_v1 #(
    parameter WIDTH_WSRAM_WL = 128,
    parameter DATA_WIDTH    = 8,
    parameter NUM_CHNL          = 16,
    parameter SIZE_BUFFER_H   = 3, 
    parameter SIZE_BUFFER_W   = 3,
    parameter SIZE_KERNEL_H = 3,
    parameter SIZE_KERNEL_W = 3
)(
    input                                 clk,
    input                                 rst_n,
    input        [4:0]                   buffer_mode,
    input                                 buffer_load_w, // load feature
    input        [$clog2(SIZE_BUFFER_H)-1:0] buffer_ptr_h_w, // pointer for height
    input        [$clog2(SIZE_BUFFER_W)-1:0] buffer_ptr_w_w, // pointer for width
    input                                 buffer_start, // output start
    // pack w_data ports into an array
    input  [WIDTH_WSRAM_WL-1:0]           w_data_in, // from SRAM
    output reg [DATA_WIDTH*NUM_CHNL-1:0]  w_buffer_out
);

    // buffer storage: [row][pe]
    reg [DATA_WIDTH-1:0] buffer_data [0:SIZE_BUFFER_H-1][0:SIZE_BUFFER_W-1][0:NUM_CHNL-1];
    reg [5:0] counter;
    integer i, k, r;

    // 8비트짜리 16?? ?????? 배열 ?????? (SystemVerilog ????????)
    wire [DATA_WIDTH-1:0] w_data [0:NUM_CHNL-1];

    genvar a;
    generate
    for (a = 0; a < 16; a = a + 1) begin
        // data_in[8*i +: 8] ?? data_in[8*i +7 : 8*i] ?? ??????
        assign w_data[a] = w_data_in[8*a +: 8];
    end
    endgenerate


    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // reset all buffer_data elements
            for (r = 0; r < SIZE_BUFFER_H; r = r + 1) begin
                for (i = 0; i < NUM_CHNL; i = i + 1) begin
                    for (k = 0; k < SIZE_BUFFER_W; k = k + 1) begin
                        buffer_data[r][k][i] <= {DATA_WIDTH{1'b0}};
                    end
                end
            end
            counter <= 0;
            w_buffer_out <= 0;

        end else begin
            if (buffer_mode == 0) begin
                counter <= 0;
                w_buffer_out <= 0;
            end else begin
                // load new features into the pointer-specified row, column
                if (buffer_load_w) begin
                    for (i = 0; i < NUM_CHNL; i = i + 1) begin
                        buffer_data[buffer_ptr_h_w][buffer_ptr_w_w][i] <= w_data[i];
                    end
                end

                
                   
                    end
                    if (counter < SIZE_BUFFER_H-1) begin
                        for (i = 0; i < NUM_CHNL; i = i + 1) begin
                            buffer_data[counter][SIZE_BUFFER_W-1][i] <= w_data[i]; //loading new data 
                        end
                        counter <= counter + 1;
                     end else begin
                      // counter == SIZE_BUFFER_H-1
                      for (i = 0; i < NUM_CHNL; i = i + 1) begin
                            buffer_data[SIZE_BUFFER_H-1][SIZE_BUFFER_W-1][i] <= w_data[i]; //loading new data
                        end
                        counter <= 0;
                    end
                end
                
              
                // output based on mode
                if (buffer_start) begin
                    case (buffer_mode)
                        1: begin
                            // example for 3x3 head
                            if (counter < SIZE_KERNEL_W*SIZE_KERNEL_H-1) begin
                                // broadcast one tapped value to all PEs
                                for (i = 0; i < NUM_CHNL; i = i + 1) begin
                                    w_buffer_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] 
                                    <= buffer_data[counter % SIZE_KERNEL_H][counter / SIZE_KERNEL_H][i];
                                end
                                counter <= counter + 1;
                            end else begin
                            // counter = SIZE_KERNEL_H * SIZE_KERNEL_H -1
                            for (i = 0; i < NUM_CHNL; i = i + 1) begin
                                 w_buffer_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] 
                                    <= buffer_data[(SIZE_KERNEL_W*SIZE_KERNEL_H-1) % SIZE_KERNEL_H]
                                        [(SIZE_KERNEL_W*SIZE_KERNEL_H-1) / SIZE_KERNEL_H]
                                        [i];
                            end
                                counter <= 0;
                                      
                                
                            end //88
                        end // 86

                       2: begin
                        // stream across the W-wide buffer row, then wrap
                        if (counter < SIZE_BUFFER_W-1) begin
                            for (i = 0; i < NUM_CHNL; i = i + 1) begin
                            // broadcast the same tap to all PEs:
                            w_buffer_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] 
                                <= buffer_data[0][counter][/* pick your channel, e.g. 0 */0];
                            end
                            counter <= counter + 1;
                        end else begin
                            // last tap at counter == W-1, then wrap
                            for (i = 0; i < NUM_CHNL; i = i + 1) begin
                            w_buffer_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] 
                                <= buffer_data[0][SIZE_BUFFER_W-1][/* same channel */0];
                            end
                            counter <= 0;
                        end
                        end

                        default: begin
                            // other modes: user can extend patterns similarly
                            w_buffer_out <= 0;
                        end
                    endcase
                end else begin
                    w_buffer_out <= 0;
                end
          
        end//59
    
endmodule
