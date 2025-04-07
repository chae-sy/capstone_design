`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Donghwan So
// 
// Create Date: 2024/09/30 14:08:56
// Design Name: 
// Module Name: INPUT_BUFFER

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

module f_buffer_v0
#(
    parameter WIDTH_FSRAM_WL = 128,
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
    input                                           buffer_load_f,
    input           [3:0]                           buffer_loc_f,
    input                                           buffer_start,
    
    input                                           shift,
    
    input           [WIDTH_F_DATA-1:0]              f_data1,             // # of f_data means that how many size needs to be convolution 
    input           [WIDTH_F_DATA-1:0]              f_data2,
    input           [WIDTH_F_DATA-1:0]              f_data3,
    input           [WIDTH_F_DATA-1:0]              f_data4,
    input           [WIDTH_F_DATA-1:0]              f_data5,
    input           [WIDTH_F_DATA-1:0]              f_data6,
    input           [WIDTH_F_DATA-1:0]              f_data7,
    input           [WIDTH_F_DATA-1:0]              f_data8,
    input           [WIDTH_F_DATA-1:0]              f_data9,
    input           [WIDTH_F_DATA-1:0]              f_data10,
    input           [WIDTH_F_DATA-1:0]              f_data11,
    input           [WIDTH_F_DATA-1:0]              f_data12,
    input           [WIDTH_F_DATA-1:0]              f_data13,
    input           [WIDTH_F_DATA-1:0]              f_data14,
    input           [WIDTH_F_DATA-1:0]              f_data15,
    input           [WIDTH_F_DATA-1:0]              f_data16,
       
    output  reg     [WIDTH_F_DATA*NUM_PE-1:0]       f_buffer_out
);
    // to store the data # of buffer_data means the row of the buffer size(= kernel's height) 
    reg [WIDTH_F_DATA-1:0] buffer_data [0:SIZE_KERNEL_H-1][0:NUM_PE-1];
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
        end
        else begin
        
        
            // layer 0
            if(buffer_mode == 0) begin 
                counter <= 0;
            end
            
            //layer 1
            else if(buffer_mode == 1) begin
                if(buffer_load_f) begin
                    if(buffer_loc_f < 3) begin          // can implement with %3 !! please check which way is better way to select
                         buffer_data[buffer_loc_f][0] <= f_data1;
                         buffer_data[buffer_loc_f][1] <= f_data2;
                         buffer_data[buffer_loc_f][2] <= f_data3; 
                    end
                    else begin
                        buffer_data[buffer_loc_f - 3][3] <= f_data1;
                        buffer_data[buffer_loc_f - 3][4] <= f_data2;
                        buffer_data[buffer_loc_f - 3][5] <= f_data3; 
                    end
                end
                if(buffer_start) begin
                    if(counter < 9) begin 
                        f_buffer_out <= {NUM_PE{buffer_data[counter%3][counter/3]}};
                        counter <= counter + 1;
                    end
                    else begin 
                        counter <= 0;
                        f_buffer_out <= 0;
                    end
                end
                if(shift) begin
                    buffer_data[0][0] <= buffer_data[0][1];
                    buffer_data[0][1] <= buffer_data[0][2];
                    buffer_data[0][2] <= buffer_data[0][3];
                    buffer_data[0][3] <= buffer_data[0][4];
                    buffer_data[0][4] <= buffer_data[0][5];
                    buffer_data[0][6] <= 0;

                    buffer_data[1][0] <= buffer_data[1][1];
                    buffer_data[1][1] <= buffer_data[1][2];
                    buffer_data[1][2] <= buffer_data[1][3];
                    buffer_data[1][3] <= buffer_data[1][4];
                    buffer_data[1][4] <= buffer_data[1][5];
                    buffer_data[1][6] <= 0;

                    buffer_data[2][0] <= buffer_data[2][1];
                    buffer_data[2][1] <= buffer_data[2][2];
                    buffer_data[2][2] <= buffer_data[2][3];
                    buffer_data[2][3] <= buffer_data[2][4];
                    buffer_data[2][4] <= buffer_data[2][5];
                    buffer_data[2][6] <= 0;
                end
            end
            
            //layer > 1
            else begin
                if(buffer_load_f) begin
                     buffer_data[buffer_loc_f][0] <= f_data1;
                     buffer_data[buffer_loc_f][1] <= f_data2;
                     buffer_data[buffer_loc_f][2] <= f_data3; 
                     buffer_data[buffer_loc_f][3] <= f_data4; 
                     buffer_data[buffer_loc_f][4] <= f_data5; 
                     buffer_data[buffer_loc_f][5] <= f_data6; 
                     buffer_data[buffer_loc_f][6] <= f_data7; 
                     buffer_data[buffer_loc_f][7] <= f_data8; 
                     buffer_data[buffer_loc_f][8] <= f_data9; 
                     buffer_data[buffer_loc_f][9] <= f_data10; 
                     buffer_data[buffer_loc_f][10] <= f_data11; 
                     buffer_data[buffer_loc_f][11] <= f_data12; 
                     buffer_data[buffer_loc_f][12] <= f_data13; 
                     buffer_data[buffer_loc_f][13] <= f_data14; 
                     buffer_data[buffer_loc_f][14] <= f_data15; 
                     buffer_data[buffer_loc_f][15] <= f_data16; 
                end
                
                
                //layer 2
                if(buffer_mode == 2) begin
                    if(buffer_start) begin
                        if(counter < 16) begin
                            counter <= counter + 1;
                            f_buffer_out <= {buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter],
                                             buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter],
                                             buffer_data[1][counter], buffer_data[1][counter], buffer_data[1][counter], buffer_data[1][counter],
                                             buffer_data[1][counter], buffer_data[1][counter], buffer_data[1][counter], buffer_data[1][counter]};
                        end
                        else begin 
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                end
                
                
                //layer 3
                else if(buffer_mode == 3) begin
                    if(buffer_start) begin
                        if(counter < 3) begin
                            counter <= counter + 1;
                            f_buffer_out <= {buffer_data[counter/2][8*(counter%2)+0], buffer_data[counter/2][8*(counter%2)+1], buffer_data[counter/2][8*(counter%2)+2], buffer_data[counter/2][8*(counter%2)+3],
                                            buffer_data[counter/2][8*(counter%2)+4], buffer_data[counter/2][8*(counter%2)+5], buffer_data[counter/2][8*(counter%2)+6], buffer_data[counter/2][8*(counter%2)+7],
                                            buffer_data[counter/2+1][8*(counter%2)+0], buffer_data[counter/2+1][8*(counter%2)+1], buffer_data[counter/2+1][8*(counter%2)+2], buffer_data[counter/2+1][8*(counter%2)+3],
                                            buffer_data[counter/2+1][8*(counter%2)+4], buffer_data[counter/2+1][8*(counter%2)+5], buffer_data[counter/2+1][8*(counter%2)+6], buffer_data[counter/2+1][8*(counter%2)+7]};
                        end
                        else begin 
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                end
                
                //layer 4
                else if (buffer_mode == 4) begin
				    if (buffer_start) begin
                        if(counter < 2) begin
                            counter <= counter + 1;
                            f_buffer_out <= {buffer_data[0][8*counter+0], buffer_data[0][8*counter+1], buffer_data[0][8*counter+2], buffer_data[0][8*counter+3], 
                            buffer_data[0][8*counter+4], buffer_data[0][8*counter+5], buffer_data[0][8*counter+6], buffer_data[0][8*counter+7],
                            8'b0,8'b0,8'b0,8'b0,8'b0,8'b0,8'b0,8'b0 };
                        end
                        else begin 
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                end
				
                //layer 5
                else if(buffer_mode == 5) begin
                    if(buffer_start) begin
                        if(counter < 3) begin
                            counter <= counter + 1;
                            f_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                             buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7],
                                             8'b00000000, 8'b00000000, 8'b00000000, 8'b00000000, 
                                             8'b00000000, 8'b00000000, 8'b00000000, 8'b00000000};
                        end
                        else begin 
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                end
                
                // layer 6
                else if(buffer_mode == 6) begin
                    if(buffer_start) begin
                        if(counter < 8) begin
                            counter <= counter + 1;
                            f_buffer_out <= {buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter],
                                            buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter],
                                            buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter],
                                            buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter]};                  // Chack this one more time because we don't use 9~~16 PE in layer 6)
                        end
                        else begin 
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                end
                
                //layer 7
                else if(buffer_mode == 7) begin
                    if(buffer_start) begin
                        if(counter < 3) begin
                            counter <= counter + 1;
                            f_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                             buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7],
                                             buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11],
                                             buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                        end
                        else begin 
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                end
                
                //layer 8
                else if(buffer_mode == 8) begin
                    if(buffer_start) begin
                        if(counter < 2) begin
                            counter <= counter + 1;
                            if(counter == 0) f_buffer_out <= {buffer_data[0][0], buffer_data[0][1], buffer_data[0][2], buffer_data[0][3],
                                                             buffer_data[0][4], buffer_data[0][5], buffer_data[0][6], buffer_data[0][7],
                                                             buffer_data[0][0], buffer_data[0][1], buffer_data[0][2], buffer_data[0][3],
                                                             buffer_data[0][4], buffer_data[0][5], buffer_data[0][6], buffer_data[0][7]};
                            else f_buffer_out <= {buffer_data[1][0], buffer_data[1][1], buffer_data[1][2], buffer_data[1][3],
                                                             buffer_data[1][4], buffer_data[1][5], buffer_data[1][6], buffer_data[1][7],
                                                             buffer_data[1][8], buffer_data[1][9], buffer_data[1][10], buffer_data[1][11],
                                                             buffer_data[1][12], buffer_data[1][13], buffer_data[1][14], buffer_data[1][15]};
                        end
                        else begin 
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                end
                
                //layer 9
                else if(buffer_mode == 9) begin
                    if(buffer_start) begin
                        if(counter < 16) begin
                            counter <= counter + 1;
                            f_buffer_out <= {16{buffer_data[0][counter]}}; 
                        end
                        else begin 
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                end
                
                //layer 10
                else if(buffer_mode == 10) begin
                    if(buffer_start) begin
                        if(counter < 3) begin
                            counter <= counter + 1;
                            f_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                             buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7],
                                             buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11],
                                             buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                        end
                        else begin 
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                end
                
                //layer 11
                else if (buffer_mode == 11) begin
                    if (buffer_start) begin
                        if (counter<3) begin
                            counter<=counter+1;
                            f_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3], 
                            buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7], 
                            buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11], 
                            buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                        end
                        else begin
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                end
			
                //layer 12
                else if(buffer_mode == 12) begin
                    if(buffer_start) begin
                        if(counter < 3) begin
                            counter <= counter + 1;
                            f_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                             buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7],
                                             buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11],
                                             buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                        end
                        else begin 
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                end
                
                
                //layer 13
                else if(buffer_mode == 13) begin
                    if(buffer_start) begin
                        if(counter < 16) begin
                            counter <= counter + 1;
                            f_buffer_out <= {16{buffer_data[0][counter]}};                  
                        end
                        else begin 
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                end
                
                //layer 14
                else if (buffer_mode == 14) begin
                    if(buffer_start) begin
                        if( counter < 3 ) begin
                            counter <= counter + 1;
                            f_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                             buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7],
                                             buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11],
                                             buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                        end
                        else begin
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                    else begin // start = 0 
                        f_buffer_out <= 0;
                        if ( counter ==  3 ) counter <= 0;
                    end
                end
                
                //layer 15
                else if(buffer_mode == 15) begin
                    if(buffer_start) begin
                        if(counter < 2) begin
                            counter <= counter + 1;
                            f_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                             buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7],
                                             buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11],
                                             buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                        end
                        else begin 
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                end
                
                //layer 16
                else if(buffer_mode == 16) begin
                    if(buffer_start) begin
                        if(counter < 16) begin
                            counter <= counter + 1;
                            f_buffer_out <= {16{buffer_data[0][counter]}};                  
                        end
                        else begin 
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                end
                
                //layer 17
                else if (buffer_mode == 17) begin
                    if(buffer_start) begin
                        if( counter < 3 ) begin
                            counter <= counter + 1;
                            f_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                             buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7],
                                             buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11],
                                             buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                        end
                        else begin
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                    else begin // start = 0 
                        f_buffer_out <= 0;
                        if ( counter ==  3 ) counter <= 0;
                    end
                end
                
                
                //layer 18
                else if (buffer_mode == 18) begin
                    if(buffer_start) begin
                        if( counter < 3 ) begin
                            counter <= counter + 1;
                            f_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                             buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7],
                                             buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11],
                                             buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                        end
                        else begin
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                    else begin // start = 0 
                        f_buffer_out <= 0;
                        if ( counter ==  3 ) counter <= 0;
                    end
                end
                
                
                //layer 19
                else if (buffer_mode == 19) begin
                    if(buffer_start) begin
                        if( counter < 32 ) begin
                            counter <= counter + 1;
                            if ( counter < 16 ) begin
                                f_buffer_out <= {buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter],
                                                 buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter],
                                                 buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter],
                                                 buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter]};
                            end 
                            else begin
                                f_buffer_out <= {buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16],
                                                 buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16],
                                                 buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16],
                                                 buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16]};
                            end 
                        end
                        else begin 
                            f_buffer_out <= 0;               
                        end
                    end
                    else f_buffer_out <= 0; // when start = 0  
                    if (counter == 32) counter <= 0;
                end 
                
                
                //layer 20
                else if (buffer_mode == 20) begin 
                    if(buffer_start) begin
                        if( counter < 3 ) begin
                            counter <= counter + 1;
                            f_buffer_out <= {buffer_data[counter][0], buffer_data[counter][1], buffer_data[counter][2], buffer_data[counter][3],
                                             buffer_data[counter][4], buffer_data[counter][5], buffer_data[counter][6], buffer_data[counter][7],
                                             buffer_data[counter][8], buffer_data[counter][9], buffer_data[counter][10], buffer_data[counter][11],
                                             buffer_data[counter][12], buffer_data[counter][13], buffer_data[counter][14], buffer_data[counter][15]};
                        end
                        else begin
                            counter <= 0;
                            f_buffer_out <= 0;
                        end
                    end
                    else begin // start = 0 
                        f_buffer_out <= 0;
                        if ( counter ==  3 ) counter <= 0;
                    end
                end		
                
                
                //layer 21
                else if (buffer_mode == 21) begin
                    if(buffer_start) begin
                        if( counter < 2 ) begin
                            counter <= counter + 1;
                            if ( counter < 1 ) begin // 0
                                f_buffer_out <= {buffer_data[0][0], buffer_data[0][1], buffer_data[0][2], buffer_data[0][3],
                                                 buffer_data[0][4], buffer_data[0][5], buffer_data[0][6], buffer_data[0][7],
                                                 buffer_data[0][8], buffer_data[0][9], buffer_data[0][10], buffer_data[0][11],
                                                 buffer_data[0][12], buffer_data[0][13], buffer_data[0][14], buffer_data[0][15]};
                            end
                            else begin // 1
                                f_buffer_out <= {buffer_data[1][0], buffer_data[1][1], buffer_data[1][2], buffer_data[1][3],
                                                 buffer_data[1][4], buffer_data[1][4], buffer_data[1][6], buffer_data[1][7],
                                                 buffer_data[1][8], buffer_data[1][9], buffer_data[1][10], buffer_data[1][11],
                                                 buffer_data[1][12], buffer_data[1][13], buffer_data[1][14], buffer_data[1][15]};
                            end
                        end
                        else begin 
                            f_buffer_out <= 0;               
                        end
                    end
                    else begin
                        f_buffer_out <= 0; // when start = 0  
                        counter <= 0;
                    end
                end      
                
                
                //layer 22
                else if(buffer_mode == 22) begin
                    if(buffer_start) begin
                        if( counter < 32 ) begin
                            counter <= counter + 1;
                            if ( counter < 16 ) begin
                                f_buffer_out <= {buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter],
                                                 buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter],
                                                 buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter],
                                                 buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter], buffer_data[0][counter]};
                            end
                            else begin
                                f_buffer_out <= {buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16],
                                                 buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16],
                                                 buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16],
                                                 buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16], buffer_data[1][counter-16]};
                            end
                        end
                        else begin 
                            counter <= 0;
                            f_buffer_out <= 0;               
                        end
                    end
                    else f_buffer_out <= 0; // when start = 0  
                end
                
                
                //layer 23
                
            end
        end
    end
    
    // for CNN HEAD We use buffer like this for paste same feature for each PEs.
endmodule