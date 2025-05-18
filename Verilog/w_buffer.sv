`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Donghwan So
//
// Create Date: 2024/09/30 14:08:56
// Design Name: 
// Module Name: WEIGHT_BUFFER (refactored)
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: Simplified with generate and for-loops
//////////////////////////////////////////////////////////////////////////////////
module w_buffer_v1
#(
    parameter WIDTH_WSRAM_WL = 128,
    parameter WIDTH_F_DATA  = 8,
    parameter WIDTH_W_DATA  = 8,
    parameter NUM_PE        = 16,
    parameter SIZE_KERNEL_W = 3,
    parameter SIZE_KERNEL_H = 3
)
(
    input                              clk,
    input                              rstb,
    input       [4:0]                  buffer_mode,
    input                              buffer_load_w,
    input       [3:0]                  buffer_loc_w,
    input                              buffer_start,
    input       [WIDTH_WSRAM_WL-1:0]   w_data,
    output reg  [WIDTH_W_DATA*NUM_PE-1:0] w_buffer_out
);

    // 2D buffer: (rows = K_H*K_W) x (cols = NUM_PE)
    reg [WIDTH_W_DATA-1:0] buffer_data [0:SIZE_KERNEL_H*SIZE_KERNEL_W-1][0:NUM_PE-1];
    reg [5:0] counter;
    integer i, j;

    // Sequential logic
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            counter <= 0;
            // reset all buffer_data entries
            for (i = 0; i < SIZE_KERNEL_H*SIZE_KERNEL_W; i = i + 1) begin
                for (j = 0; j < NUM_PE; j = j + 1) begin
                    buffer_data[i][j] <= {WIDTH_W_DATA{1'b0}};
                end
            end
            w_buffer_out <= {WIDTH_W_DATA*NUM_PE{1'b0}};
        end else begin
            // Load weights into one row at position buffer_loc_w
            if (buffer_load_w) begin
                for (j = 0; j < NUM_PE; j = j + 1) begin
                    buffer_data[buffer_loc_w][j] <= 
                        w_data[WIDTH_WSRAM_WL - j*WIDTH_W_DATA - 1 -: WIDTH_W_DATA];
                end
            end
            
            // Default for mode 0
            if (buffer_mode == 0) begin
                counter <= 0;
                w_buffer_out <= {WIDTH_W_DATA*NUM_PE{1'b0}};
            end
            else if (buffer_start) begin
                // example: layer 1 (mode == 1), stream rows
                if (buffer_mode == 1) begin
                    if (counter < SIZE_KERNEL_H*SIZE_KERNEL_W) begin
                        for (j = 0; j < NUM_PE; j = j + 1) begin
                            w_buffer_out[j*WIDTH_W_DATA +: WIDTH_W_DATA] <= buffer_data[counter][j];
                        end
                        counter <= counter + 1;
                    end else begin
                        counter <= 0;
                        w_buffer_out <= {WIDTH_W_DATA*NUM_PE{1'b0}};
                    end
                end
                // example: layer 2 (mode == 2), stream columns
                else if (buffer_mode == 2) begin
                    if (counter < NUM_PE) begin
                        for (i = 0; i < SIZE_KERNEL_H*SIZE_KERNEL_W; i = i + 1) begin
                            w_buffer_out[i*WIDTH_W_DATA +: WIDTH_W_DATA] <= buffer_data[i][counter];
                        end
                        counter <= counter + 1;
                    end else begin
                        counter <= 0;
                        w_buffer_out <= {WIDTH_W_DATA*NUM_PE{1'b0}};
                    end
                end
                // other modes: similar pattern using loops
                else begin
                    // Add further mode cases here, utilizing nested loops
                    counter <= counter;
                    w_buffer_out <= w_buffer_out;
                end
            end
        end
    end
endmodule
