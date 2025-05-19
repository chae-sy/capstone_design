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
    input        [4:0]                   buffer_mode,
    input                                 buffer_load_f, // load feature
    input        [$clog2(SIZE_BUFFER_H)-1:0] buffer_ptr_h_f, // pointer for height
    input        [$clog2(SIZE_BUFFER_W)-1:0] buffer_ptr_w_f, // pointer for width
    input                                 buffer_start, // output start
    input                                 shift,
    //input                                 pad_en, // padding enable
    // pack f_data ports into an array
    input  [WIDTH_FSRAM_WL-1:0]           f_data_in, // from SRAM
    output reg [DATA_WIDTH*NUM_CHNL-1:0]  f_buffer_out
);

    // buffer storage: [row][pe]
    reg [DATA_WIDTH-1:0] buffer_data [0:SIZE_BUFFER_H-1][0:SIZE_BUFFER_W-1][0:NUM_CHNL-1];
    reg [5:0] counter;
    integer i, k, r;

    // 8鍮꾪듃吏쒕━ 16媛� �슂�냼 諛곗뿴 �꽑�뼵 (SystemVerilog �뒪���씪)
    wire [DATA_WIDTH-1:0] f_data [0:NUM_CHNL-1];

    genvar a;
    generate
    for (a = 0; a < 16; a = a + 1) begin
        // data_in[8*i +: 8] �� data_in[8*i +7 : 8*i] �� �룞�씪
        assign f_data[a] = f_data_in[8*a +: 8];
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
            f_buffer_out <= 0;

        end else begin
            if (buffer_mode == 0) begin
                counter <= 0;
                f_buffer_out <= 0;
            end else begin
                // load new features into the pointer-specified row, column
                if (buffer_load_f) begin
                    for (i = 0; i < NUM_CHNL; i = i + 1) begin
                        buffer_data[buffer_ptr_h_f][buffer_ptr_w_f][i] <= f_data[i];
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
                    if (counter < SIZE_BUFFER_H-1) begin
                        for (i = 0; i < NUM_CHNL; i = i + 1) begin
                            buffer_data[counter][SIZE_BUFFER_W-1][i] <= f_data[i]; //loading new data 
                        end
                        counter <= counter + 1;
                     end else begin
                      // counter == SIZE_BUFFER_H-1
                      for (i = 0; i < NUM_CHNL; i = i + 1) begin
                            buffer_data[SIZE_BUFFER_H-1][SIZE_BUFFER_W-1][i] <= f_data[i]; //loading new data
                        end
                        counter <= 0;
                    end
                end
                
                //if (pad_en) begin // pad 3*1*16
                //   for (k = 0; k < SIZE_BUFFER_H; k = k + 1) begin
                //        for (i = 0; i < NUM_CHNL; i = i + 1) begin
                //            buffer_data[k][buffer_ptr_w_f][i] <= {DATA_WIDTH{1'b0}};
                //        end
                //    end
                //end
                // output based on mode
                if (buffer_start) begin
                    case (buffer_mode)
                        1: begin
                            // example for 3x3 head
                            if (counter < SIZE_KERNEL_W*SIZE_KERNEL_H-1) begin
                                // broadcast one tapped value to all PEs
                                for (i = 0; i < NUM_CHNL; i = i + 1) begin
                                    f_buffer_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] 
                                    <= buffer_data[counter % SIZE_KERNEL_H][counter / SIZE_KERNEL_H][i];
                                end
                                counter <= counter + 1;
                            end else begin
                            // counter = SIZE_KERNEL_H * SIZE_KERNEL_H -1
                            for (i = 0; i < NUM_CHNL; i = i + 1) begin
                                 f_buffer_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] 
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
                            f_buffer_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] 
                                <= buffer_data[0][counter][/* pick your channel, e.g. 0 */0];
                            end
                            counter <= counter + 1;
                        end else begin
                            // last tap at counter == W-1, then wrap
                            for (i = 0; i < NUM_CHNL; i = i + 1) begin
                            f_buffer_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] 
                                <= buffer_data[0][SIZE_BUFFER_W-1][/* same channel */0];
                            end
                            counter <= 0;
                        end
                        end

                        default: begin
                            // other modes: user can extend patterns similarly
                            f_buffer_out <= 0;
                        end
                    endcase
                end else begin
                    f_buffer_out <= 0;
                end
            end
        end//59
    end
endmodule
