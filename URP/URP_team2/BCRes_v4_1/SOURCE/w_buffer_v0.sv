`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Donghwan So
// 
// Create Date: 2024/09/30 14:08:56
// Design Name: 
// Module Name: WEIGHT_BUFFER
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
// 1st FIN - only this TB

module w_buffer_v0
#(
    parameter WIDTH_WSRAM_WL = 128,
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_W_DATA = 8,
    parameter NUM_PE = 16,
    parameter SIZE_KERNEL_W = 3,
    parameter SIZE_KERNEL_H = 3
)
(
    input                                           clk,
    input                                           rstb,
    
    input           [4:0]                           buffer_mode,
    input                                           buffer_load_w,
    input           [3:0]                           buffer_loc_w,
    input                                           buffer_start,
    
    input           [WIDTH_WSRAM_WL-1:0]            w_data,             // WL = 36, so we can get 6 weights at once 
    
    output  reg    [WIDTH_W_DATA*NUM_PE-1:0]        w_buffer_out
);
    // to store the data # of buffer_data means the row of the buffer size(= kernel's height) 
    reg [WIDTH_W_DATA-1:0] buffer_data [0:SIZE_KERNEL_H*SIZE_KERNEL_W-1][0:NUM_PE-1];
    reg [5:0] counter;
    
    always @(posedge clk or negedge rstb) begin
        if(!rstb) begin
            buffer_data[0][0] <= 0; buffer_data[0][1] <= 0; buffer_data[0][2] <= 0; buffer_data[0][3] <= 0;
            buffer_data[0][4] <= 0; buffer_data[0][5] <= 0; buffer_data[0][6] <= 0; buffer_data[0][7] <= 0;
            buffer_data[0][8] <= 0; buffer_data[0][9] <= 0; buffer_data[0][10] <= 0; buffer_data[0][11] <= 0;
            buffer_data[0][12] <= 0; buffer_data[0][13] <= 0; buffer_data[0][14] <= 0; buffer_data[0][15] <= 0;

            buffer_data[1][0] <= 0; buffer_data[1][1] <= 0; buffer_data[1][2] <= 0; buffer_data[1][3] <= 0;
            buffer_data[1][4] <= 0; buffer_data[1][5] <= 0; buffer_data[1][6] <= 0; buffer_data[1][7] <= 0;
            buffer_data[1][8] <= 0; buffer_data[1][9] <= 0; buffer_data[1][10] <= 0; buffer_data[1][11] <= 0;
            buffer_data[1][12] <= 0; buffer_data[1][13] <= 0; buffer_data[1][14] <= 0; buffer_data[1][15] <= 0;

            buffer_data[2][0] <= 0; buffer_data[2][1] <= 0; buffer_data[2][2] <= 0; buffer_data[2][3] <= 0;
            buffer_data[2][4] <= 0; buffer_data[2][5] <= 0; buffer_data[2][6] <= 0; buffer_data[2][7] <= 0;
            buffer_data[2][8] <= 0; buffer_data[2][9] <= 0; buffer_data[2][10] <= 0; buffer_data[2][11] <= 0;
            buffer_data[2][12] <= 0; buffer_data[2][13] <= 0; buffer_data[2][14] <= 0; buffer_data[2][15] <= 0;

            buffer_data[3][0] <= 0; buffer_data[3][1] <= 0; buffer_data[3][2] <= 0; buffer_data[3][3] <= 0;
            buffer_data[3][4] <= 0; buffer_data[3][5] <= 0; buffer_data[3][6] <= 0; buffer_data[3][7] <= 0;
            buffer_data[3][8] <= 0; buffer_data[3][9] <= 0; buffer_data[3][10] <= 0; buffer_data[3][11] <= 0;
            buffer_data[3][12] <= 0; buffer_data[3][13] <= 0; buffer_data[3][14] <= 0; buffer_data[3][15] <= 0;

            buffer_data[4][0] <= 0; buffer_data[4][1] <= 0; buffer_data[4][2] <= 0; buffer_data[4][3] <= 0;
            buffer_data[4][4] <= 0; buffer_data[4][5] <= 0; buffer_data[4][6] <= 0; buffer_data[4][7] <= 0;
            buffer_data[4][8] <= 0; buffer_data[4][9] <= 0; buffer_data[4][10] <= 0; buffer_data[4][11] <= 0;
            buffer_data[4][12] <= 0; buffer_data[4][13] <= 0; buffer_data[4][14] <= 0; buffer_data[4][15] <= 0;

            buffer_data[5][0] <= 0; buffer_data[5][1] <= 0; buffer_data[5][2] <= 0; buffer_data[5][3] <= 0;
            buffer_data[5][4] <= 0; buffer_data[5][5] <= 0; buffer_data[5][6] <= 0; buffer_data[5][7] <= 0;
            buffer_data[5][8] <= 0; buffer_data[5][9] <= 0; buffer_data[5][10] <= 0; buffer_data[5][11] <= 0;
            buffer_data[5][12] <= 0; buffer_data[5][13] <= 0; buffer_data[5][14] <= 0; buffer_data[5][15] <= 0;

            buffer_data[6][0] <= 0; buffer_data[6][1] <= 0; buffer_data[6][2] <= 0; buffer_data[6][3] <= 0;
            buffer_data[6][4] <= 0; buffer_data[6][5] <= 0; buffer_data[6][6] <= 0; buffer_data[6][7] <= 0;
            buffer_data[6][8] <= 0; buffer_data[6][9] <= 0; buffer_data[6][10] <= 0; buffer_data[6][11] <= 0;
            buffer_data[6][12] <= 0; buffer_data[6][13] <= 0; buffer_data[6][14] <= 0; buffer_data[6][15] <= 0;

            buffer_data[7][0] <= 0; buffer_data[7][1] <= 0; buffer_data[7][2] <= 0; buffer_data[7][3] <= 0;
            buffer_data[7][4] <= 0; buffer_data[7][5] <= 0; buffer_data[7][6] <= 0; buffer_data[7][7] <= 0;
            buffer_data[7][8] <= 0; buffer_data[7][9] <= 0; buffer_data[7][10] <= 0; buffer_data[7][11] <= 0;
            buffer_data[7][12] <= 0; buffer_data[7][13] <= 0; buffer_data[7][14] <= 0; buffer_data[7][15] <= 0;

            buffer_data[8][0] <= 0; buffer_data[8][1] <= 0; buffer_data[8][2] <= 0; buffer_data[8][3] <= 0;
            buffer_data[8][4] <= 0; buffer_data[8][5] <= 0; buffer_data[8][6] <= 0; buffer_data[8][7] <= 0;
            buffer_data[8][8] <= 0; buffer_data[8][9] <= 0; buffer_data[8][10] <= 0; buffer_data[8][11] <= 0;
            buffer_data[8][12] <= 0; buffer_data[8][13] <= 0; buffer_data[8][14] <= 0; buffer_data[8][15] <= 0;
            counter <= 0;
        end
        else begin
            if(buffer_load_w) begin
                buffer_data[buffer_loc_w][0] <= w_data[WIDTH_WSRAM_WL-1:WIDTH_WSRAM_WL-WIDTH_W_DATA];
                buffer_data[buffer_loc_w][1] <= w_data[WIDTH_WSRAM_WL-WIDTH_W_DATA-1:WIDTH_WSRAM_WL-2*WIDTH_W_DATA];
                buffer_data[buffer_loc_w][2] <= w_data[WIDTH_WSRAM_WL-2*WIDTH_W_DATA-1:WIDTH_WSRAM_WL-3*WIDTH_W_DATA];
                buffer_data[buffer_loc_w][3] <= w_data[WIDTH_WSRAM_WL-3*WIDTH_W_DATA-1:WIDTH_WSRAM_WL-4*WIDTH_W_DATA];                    
                buffer_data[buffer_loc_w][4] <= w_data[WIDTH_WSRAM_WL-4*WIDTH_W_DATA-1:WIDTH_WSRAM_WL-5*WIDTH_W_DATA];
                buffer_data[buffer_loc_w][5] <= w_data[WIDTH_WSRAM_WL-5*WIDTH_W_DATA-1:WIDTH_WSRAM_WL-6*WIDTH_W_DATA];
                buffer_data[buffer_loc_w][6] <= w_data[WIDTH_WSRAM_WL-6*WIDTH_W_DATA-1:WIDTH_WSRAM_WL-7*WIDTH_W_DATA];
                buffer_data[buffer_loc_w][7] <= w_data[WIDTH_WSRAM_WL-7*WIDTH_W_DATA-1:WIDTH_WSRAM_WL-8*WIDTH_W_DATA];
                buffer_data[buffer_loc_w][8] <= w_data[WIDTH_WSRAM_WL-8*WIDTH_W_DATA-1:WIDTH_WSRAM_WL-9*WIDTH_W_DATA];
                buffer_data[buffer_loc_w][9] <= w_data[WIDTH_WSRAM_WL-9*WIDTH_W_DATA-1:WIDTH_WSRAM_WL-10*WIDTH_W_DATA];
                buffer_data[buffer_loc_w][10] <= w_data[WIDTH_WSRAM_WL-10*WIDTH_W_DATA-1:WIDTH_WSRAM_WL-11*WIDTH_W_DATA];
                buffer_data[buffer_loc_w][11] <= w_data[WIDTH_WSRAM_WL-11*WIDTH_W_DATA-1:WIDTH_WSRAM_WL-12*WIDTH_W_DATA];
                buffer_data[buffer_loc_w][12] <= w_data[WIDTH_WSRAM_WL-12*WIDTH_W_DATA-1:WIDTH_WSRAM_WL-13*WIDTH_W_DATA];
                buffer_data[buffer_loc_w][13] <= w_data[WIDTH_WSRAM_WL-13*WIDTH_W_DATA-1:WIDTH_WSRAM_WL-14*WIDTH_W_DATA];
                buffer_data[buffer_loc_w][14] <= w_data[WIDTH_WSRAM_WL-14*WIDTH_W_DATA-1:WIDTH_WSRAM_WL-15*WIDTH_W_DATA];
                buffer_data[buffer_loc_w][15] <= w_data[WIDTH_WSRAM_WL-15*WIDTH_W_DATA-1:0];
            end
            if(buffer_mode == 0) begin
                counter <= 0;
            end
            
            //layer 1
            else if(buffer_mode == 1) begin
                if(buffer_start) begin
                    if(counter < 9) begin 
                        w_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                     buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7],
                                     buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11],
                                     buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                        counter <= counter + 1;
                    end
                    else begin 
                        counter <= 0;
                        w_buffer_out <= 0;
                    end
                end 
            end
            
            //layer 2
            else if(buffer_mode == 2) begin
                if(buffer_start) begin
                    if(counter < 16) begin
                        counter <= counter + 1;
                        w_buffer_out <= {buffer_data[0][counter], buffer_data[1][counter], buffer_data[2][counter], buffer_data[3][counter],
                                         buffer_data[4][counter], buffer_data[5][counter], buffer_data[6][counter], buffer_data[7][counter], 
                                         buffer_data[0][counter], buffer_data[1][counter], buffer_data[2][counter], buffer_data[3][counter], 
                                         buffer_data[4][counter], buffer_data[5][counter], buffer_data[6][counter], buffer_data[7][counter]};
                    end
                    else begin 
                        counter <= 0;
                        w_buffer_out <= 0;
                    end
                end
            end
            
            //layer 3
            else if(buffer_mode == 3) begin
                if(buffer_start) begin
                    if(counter < 3) begin 
                        w_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                        buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7], 
                                        buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3], 
                                        buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7]};
                        counter <= counter + 1;
                    end
                    else begin 
                        counter <= 0;
                        w_buffer_out <= 0;
                    end
                end
            end
            
            //layer 4
            else if (buffer_mode == 4) begin
				if (buffer_start) begin
				    if (counter <2) begin
				        counter <= counter +1;
				        w_buffer_out <= {buffer_data[0][0], buffer_data[0][1], buffer_data[0][2], buffer_data[0][3], 
				        buffer_data[0][4], buffer_data[0][5], buffer_data[0][6], buffer_data[0][7],
				        8'b0,8'b0,8'b0,8'b0,8'b0,8'b0,8'b0,8'b0};
				    end
				    else begin
				        counter <=0;
				        w_buffer_out <=0;
				    end
				end
			end
			
            //layer 5
            else if(buffer_mode == 5) begin
                if(buffer_start) begin
                    if(counter < 3) begin 
                        w_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                        buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7], 
                                        8'b000000, 8'b000000, 8'b000000, 8'b000000,
                                        8'b000000, 8'b000000, 8'b000000, 8'b000000  };
                        counter <= counter + 1;
                    end
                    else begin 
                        counter <= 0;
                        w_buffer_out <= 0;
                    end
                end
            end
            
            //layer 6
            else if(buffer_mode == 6) begin
                if(buffer_start) begin
                    if(counter < 8) begin 
                        w_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                        buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7], 
                                        8'b000000, 8'b000000, 8'b000000, 8'b000000,
                                        8'b000000, 8'b000000, 8'b000000, 8'b000000};
                        counter <= counter + 1;
                    end
                    else begin 
                        counter <= 0;
                        w_buffer_out <= 0;
                    end
                end
            end
            
            //layer 7
            else if(buffer_mode == 7) begin
                if(buffer_start) begin
                    if(counter < 3) begin 
                        w_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                        buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7], 
                                        buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                        buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7]};
                        counter <= counter + 1;
                    end
                    else begin 
                        counter <= 0;
                        w_buffer_out <= 0;
                    end
                end
            end
            
            //layer 8
            else if(buffer_mode == 8) begin
                if(buffer_start) begin
                    if(counter < 2) begin 
                        w_buffer_out <= {buffer_data[0][0], buffer_data[0][1], buffer_data[0][2], buffer_data[0][3],
                                        buffer_data[0][4], buffer_data[0][5], buffer_data[0][6], buffer_data[0][7], 
                                        buffer_data[0][8], buffer_data[0][9], buffer_data[0][10], buffer_data[0][11],
                                        buffer_data[0][12], buffer_data[0][13], buffer_data[0][14], buffer_data[0][15]};
                        counter <= counter + 1;
                    end
                    else begin 
                        counter <= 0;
                        w_buffer_out <= 0;
                    end
                end
            end
            
            //layer 9
            else if(buffer_mode == 9) begin
                if(buffer_start) begin
                    if(counter < 16) begin 
                        w_buffer_out <= {buffer_data[counter%8][0], buffer_data[counter%8][1], buffer_data[counter%8][2], buffer_data[counter%8][3],
                                        buffer_data[counter%8][4], buffer_data[counter%8][5], buffer_data[counter%8][6], buffer_data[counter%8][7], 
                                        buffer_data[counter%8][8], buffer_data[counter%8][9], buffer_data[counter%8][10], buffer_data[counter%8][11],
                                        buffer_data[counter%8][12], buffer_data[counter%8][13], buffer_data[counter%8][14], buffer_data[counter%8][15]};
                        counter <= counter + 1;
                    end
                    else begin 
                        counter <= 0;
                        w_buffer_out <= 0;
                    end
                end
            end
            
            //layer 10
            else if(buffer_mode == 10) begin
                if(buffer_start) begin
                    if(counter < 3) begin 
                        w_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                         buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7],
                                         buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11],
                                         buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                        counter <= counter + 1;
                    end
                    else begin 
                        counter <= 0;
                        w_buffer_out <= 0;
                    end
                end
            end
            
            //layer 11
            else if (buffer_mode == 11) begin
				if (buffer_start) begin
				    if (counter <3) begin
				        counter <= counter +1;
				        w_buffer_out <= {buffer_data[0][0], buffer_data[0][1], buffer_data[0][2], buffer_data[0][3], 
				        buffer_data[0][4], buffer_data[0][5], buffer_data[0][6], buffer_data[0][7], 
				        buffer_data[0][8], buffer_data[0][9], buffer_data[0][10], buffer_data[0][11], 
				        buffer_data[0][12], buffer_data[0][13], buffer_data[0][14], buffer_data[0][15]};
				    end
				    else begin
				        counter <=0;
				        w_buffer_out <=0;
				    end
				end
			end
			
            //layer 12
            else if(buffer_mode == 12) begin
                if(buffer_start) begin
                    if(counter < 3) begin 
                        w_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                         buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7],
                                         buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11],
                                         buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                        counter <= counter + 1;
                    end
                    else begin 
                        counter <= 0;
                        w_buffer_out <= 0;
                    end
                end
            end
            
            //layer 13
            else if(buffer_mode == 13) begin
                if(buffer_start) begin
                    if(counter < 16) begin 
                        w_buffer_out <= {buffer_data[counter%8][0], buffer_data[counter%8][1], buffer_data[counter%8][2], buffer_data[counter%8][3],
                                         buffer_data[counter%8][4], buffer_data[counter%8][5], buffer_data[counter%8][6], buffer_data[counter%8][7],
                                         buffer_data[counter%8][8], buffer_data[counter%8][9], buffer_data[counter%8][10], buffer_data[counter%8][11],
                                         buffer_data[counter%8][12], buffer_data[counter%8][13], buffer_data[counter%8][14], buffer_data[counter%8][15]};
                        counter <= counter + 1;
                    end
                    else begin 
                        counter <= 0;
                        w_buffer_out <= 0;
                    end
                end
            end
            
            //layer 14
            else if (buffer_mode == 14) begin
               if (buffer_start) begin
                    if (counter < 9 ) begin
                        counter <= counter + 1;
                        if( counter < 3 ) begin
                            w_buffer_out <= {buffer_data[0][0], buffer_data[0][1], buffer_data[0][2], buffer_data[0][3],
                                             buffer_data[0][4], buffer_data[0][5], buffer_data[0][6], buffer_data[0][7], 
                                             buffer_data[0][8], buffer_data[0][9], buffer_data[0][10], buffer_data[0][11], 
                                             buffer_data[0][12], buffer_data[0][13], buffer_data[0][14], buffer_data[0][15]}; 
                        end
                        else if ( counter < 6 ) begin
                            w_buffer_out <= {buffer_data[1][0], buffer_data[1][1], buffer_data[1][2], buffer_data[1][3],
                                             buffer_data[1][4], buffer_data[1][5], buffer_data[1][6], buffer_data[1][7], 
                                             buffer_data[1][8], buffer_data[1][9], buffer_data[1][10], buffer_data[1][11], 
                                             buffer_data[1][12], buffer_data[1][13], buffer_data[1][14], buffer_data[1][15]};
                        end
                        else begin
                            w_buffer_out <= {buffer_data[2][0], buffer_data[2][1], buffer_data[2][2], buffer_data[2][3],
                                             buffer_data[2][4], buffer_data[2][5], buffer_data[2][6], buffer_data[2][7], 
                                             buffer_data[2][8], buffer_data[2][9], buffer_data[2][10], buffer_data[2][11], 
                                             buffer_data[2][12], buffer_data[2][13], buffer_data[2][14], buffer_data[2][15]};
                        end
                    end // end for c < 9 
                end
                else begin // buf_start = 0
                    w_buffer_out <= 0; 
                    if (counter == 9) counter <=0;
                end
            end
            
            //layer 15
            else if(buffer_mode == 15) begin
                if(buffer_start) begin
                    if(counter < 2) begin 
                        w_buffer_out <= {buffer_data[0][0], buffer_data[0][1], buffer_data[0][2], buffer_data[0][3],
                                        buffer_data[0][4], buffer_data[0][5], buffer_data[0][6], buffer_data[0][7], 
                                        buffer_data[0][8], buffer_data[0][9], buffer_data[0][10], buffer_data[0][11],
                                        buffer_data[0][12], buffer_data[0][13], buffer_data[0][14], buffer_data[0][15]};
                        counter <= counter + 1;
                    end
                    else begin 
                        counter <= 0;
                        w_buffer_out <= 0;
                    end
                end
            end
            
            //layer 16
            else if(buffer_mode == 16) begin
                if(buffer_start) begin
                    if(counter < 16) begin 
                        w_buffer_out <= {buffer_data[counter%8][0], buffer_data[counter%8][1], buffer_data[counter%8][2], buffer_data[counter%8][3],
                                         buffer_data[counter%8][4], buffer_data[counter%8][5], buffer_data[counter%8][6], buffer_data[counter%8][7],
                                         buffer_data[counter%8][8], buffer_data[counter%8][9], buffer_data[counter%8][10], buffer_data[counter%8][11],
                                         buffer_data[counter%8][12], buffer_data[counter%8][13], buffer_data[counter%8][14], buffer_data[counter%8][15]};
                        counter <= counter + 1;
                    end
                    else begin 
                        counter <= 0;
                        w_buffer_out <= 0;
                    end
                end
            end
            
            else if (buffer_mode == 17) begin 
                if (buffer_start) begin
                    if (counter == 5 ) counter <= 0;
                    else counter <= counter + 1;
                    if(counter < 6) w_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                                     buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7], 
                                                     buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11], 
                                                     buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]}; 
                end
                else begin
                    w_buffer_out <= 0; 
                    if (counter == 6) counter <=0;
                end
			end


			else if (buffer_mode == 18) begin
                if (buffer_start) begin
                    if (counter == 5 ) counter <= 0;
                    else counter <= counter + 1;
                    if(counter < 6) w_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                                     buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7], 
                                                     buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11], 
                                                     buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                end
                else begin
                    w_buffer_out <= 0; 
                    if (counter == 6) counter <=0;
                end
			end

            
			else if (buffer_mode == 19) begin
                if (buffer_start) begin
                    if (counter == 7 ) counter <= 0;
                    else counter <= counter + 1;
                    if(counter < 8) w_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                                     buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7], 
                                                     buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11], 
                                                     buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                end
                else begin
                    w_buffer_out <= 0; 
                    counter <=0;
                end
			end
            
            
            else if (buffer_mode == 20) begin //
                if (buffer_start) begin
                    if (counter == 5 ) counter <= 0;
                    else counter <= counter + 1;
                    if(counter < 6) w_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                                     buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7], 
                                                     buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11], 
                                                     buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                end
                else begin
                    w_buffer_out <= 0; 
                    if (counter == 6) counter <=0;
                end
            end                

			else if (buffer_mode == 21) begin
				if (buffer_start) begin
                    w_buffer_out <= {buffer_data[0][0], buffer_data[0][1], buffer_data[0][2], buffer_data[0][3],
                                     buffer_data[0][4], buffer_data[0][5], buffer_data[0][6], buffer_data[0][7], 
                                     buffer_data[0][8], buffer_data[0][9], buffer_data[0][10], buffer_data[0][11], 
                                     buffer_data[0][12], buffer_data[0][13], buffer_data[0][14], buffer_data[0][15]};
				end
                else begin
                    w_buffer_out <= 0; 
                    counter <=0;
                end				    
			end
			           
            // clf - pw(L19) -> 22
            else if (buffer_mode == 22) begin
                if (buffer_start) begin
                    if (counter == 7 ) counter <= 0;
                    else counter <= counter + 1;
                    if(counter < 8) w_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                                     buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7], 
                                                     buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11], 
                                                     buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                end
                else begin
                    w_buffer_out <= 0; 
                    counter <=0;
                end
            end
        end
    end
    
    
endmodule
