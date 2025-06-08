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

module output_buffer #(
    parameter t_WIDTH = 128,
    parameter DATA_WIDTH = 8,
    parameter NUM_CHNL = 16,
    parameter NUM_COLOR = 3
    
)(
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             wren[0:2],
    input  wire [DATA_WIDTH-1:0]            data_in_r,
    input  wire [DATA_WIDTH-1:0]            data_in_g,
    input  wire [DATA_WIDTH-1:0]            data_in_b,
    input  wire                             rden[0:2],
    input  wire [2:0]                       layer_num,
    input  wire                             layer_start,
    output reg                              o_buffer_done,
    output reg  [t_WIDTH-1:0]               data_out
);

    // buffer storage
    reg [DATA_WIDTH-1:0] buffer_data_r_array [0:NUM_CHNL-1];
    reg [DATA_WIDTH-1:0] buffer_data_g_array [0:NUM_CHNL-1];
    reg [DATA_WIDTH-1:0] buffer_data_b_array [0:NUM_CHNL-1];
    reg [t_WIDTH-1:0] buffer_data_r;
    reg [t_WIDTH-1:0] buffer_data_g;
    reg [t_WIDTH-1:0] buffer_data_b;
    reg [4:0] cnt[0:NUM_COLOR-1];
    reg [4:0] cnt_n[0:NUM_COLOR-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            for (int i = 0; i < NUM_COLOR; i = i + 1) begin
                cnt[i]          <= 0;
            end
        end else begin
            for (int i = 0; i < NUM_COLOR; i = i + 1) begin
                cnt[i]          <= cnt_n[i];
            end
            if (wren[0]) buffer_data_r_array[cnt[0]] <= data_in_r;
            if (wren[1]) buffer_data_g_array[cnt[1]] <= data_in_g;
            if (wren[2]) buffer_data_b_array[cnt[2]] <= data_in_b;
        end
    end
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            buffer_data_r = 'b0;
            buffer_data_g = 'b0;
            buffer_data_b = 'b0;
        end
    end
    
    generate
        for (genvar i=0; i<NUM_CHNL; i=i+1) begin
            always_comb begin
                buffer_data_r[(i+1)*DATA_WIDTH-1:i*DATA_WIDTH] = buffer_data_r_array[i];
                buffer_data_g[(i+1)*DATA_WIDTH-1:i*DATA_WIDTH] = buffer_data_g_array[i];
                buffer_data_b[(i+1)*DATA_WIDTH-1:i*DATA_WIDTH] = buffer_data_b_array[i];
            end
        end
    endgenerate


    always_comb begin
        o_buffer_done = 0;
        for (int i = 0; i < NUM_COLOR; i = i + 1 ) begin
            cnt_n[i] = cnt[i]; 
        end
        case (layer_num)
            3'd6: begin
                for (int i = 0; i < NUM_COLOR; i = i + 1 ) begin
                    if (wren[i]) o_buffer_done = 1;
                end

                if (rden[0]) begin
                    data_out = {{(t_WIDTH-DATA_WIDTH){1'b0}}, buffer_data_r[DATA_WIDTH-1:0]};
                end
                else if (rden[1]) begin
                    data_out = {{(t_WIDTH-DATA_WIDTH){1'b0}}, buffer_data_g[DATA_WIDTH-1:0]};
                end
                else if (rden[2]) begin    
                    data_out = {{(t_WIDTH-DATA_WIDTH){1'b0}}, buffer_data_b[DATA_WIDTH-1:0]};
                end
            end
            default: begin
                for (int i = 0; i < NUM_COLOR; i = i + 1 ) begin
                    if (wren[i]) cnt_n[i] = cnt[i] + 1;
                end
                if ((cnt[0] == 15) & wren[0]) begin
                    o_buffer_done = 1;
                    cnt_n[0] = 0;
                    cnt_n[1] = 0;
                    cnt_n[2] = 0;
                end
                if (rden[0]) begin
                    data_out = buffer_data_r;
                end
                else if (rden[1]) begin
                    data_out = buffer_data_g;
                end
                else if (rden[2]) begin
                    data_out = buffer_data_b;
                end
            end

        endcase
    end
     
endmodule