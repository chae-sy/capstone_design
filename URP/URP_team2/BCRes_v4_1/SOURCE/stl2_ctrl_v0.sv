`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Donghwan So
// 
// Create Date: 2024/10/21 14:31:10
// Design Name: Layer 2 FSM controller 
// Module Name: stl2_ctrl_v0
// Project Name: KWS BC ResNet
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


module stl2_ctrl_v0
#(
    parameter WIDTH_WSRAM_WL = 128,
    parameter WIDTH_FSRAM_WL = 128,
    parameter WIDTH_W_DATA = 8,
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_FSRAM_ADDR = 10,
    parameter WIDTH_WSRAM_ADDR = 10,
    parameter ADDR_START_L2_W = 10,
    parameter ADDR_START_L2_F = 21,
    parameter NUM_L1_OUT_CHANNEL = 16,
    parameter NUM_L2_OUT_CHANNEL = 8

)
(
    input                                       clk,
    input                                       rstb,
    input       [4:0]                           layer_state,
    input                                       layer_start,

    output  reg     [WIDTH_FSRAM_ADDR-1:0]      fsram_addr,
    output  reg     [WIDTH_WSRAM_ADDR-1:0]      wsram_addr,
    output  reg                                 c_w_ceb,
    output  reg                                 c_w_web,
    output  reg                                 c_f_ceb,
    output  reg                                 c_f_web,
    
    output  reg                                 buffer_start,
    output  reg     [4:0]                       buffer_mode_w,
    output  reg     [4:0]                       buffer_mode_f,
    output  reg     [3:0]                       buffer_loc_w,
    output  reg     [3:0]                       buffer_loc_f,
    output  reg                                 buffer_load_w,
    output  reg                                 buffer_load_f,

    
    output  reg                                 pe_clear,
    output  reg                                 pe_en,
    
    output  reg                                 shift,
    
    output  reg                                 norm_on,
    output  reg                                 relu_on,
    
    output  reg                                 l2_done
);
    localparam                          ST_IDLE             = 0,
                                        ST_L2_WLOAD             = 1,
                                        ST_L2_FLOAD             = 2,
                                        ST_L2_CAL               = 3,
                                        ST_L2_DONE              = 4;
                                       

    localparam                        NUM_W_ADDR              = 8,    // Total # of weight address (16*8/16 = 8)
                                        NUM_FLOAD               = 9,    // $ceil(18/2)  
                                        SIZE_TIME_DOMAIN        = 1,
                                        SIZE_FREQ_DOMAIN        = 18;   // 20(Layer1 FREQ) - 2
                                        
    reg [6:0] cnt;
    reg [5:0] cnt_out;
    reg [2:0] state;
    reg [WIDTH_FSRAM_ADDR-1:0] fsram_addr_out;
    reg [WIDTH_FSRAM_ADDR-1:0] fsram_addr_in;
                                       
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            state <= ST_IDLE;
            fsram_addr <= 0;
            wsram_addr <= 0;
            fsram_addr_out <= 0;
            fsram_addr_in <= 0;
            cnt <= 0;
            cnt_out <= 0;
            buffer_start <= 0;
            buffer_mode_w <= 0;
            buffer_mode_f <= 0;
            buffer_loc_w <= 0;
            buffer_loc_f <= 0;
            buffer_load_w <= 0;
            buffer_load_f <= 0;
            pe_clear <= 0;
            pe_en <= 0;
            norm_on <= 0;
            relu_on <= 0;
            l2_done <= 0;
            c_w_web <= 0;
            c_w_ceb <= 0;
            c_f_web <= 0;
            c_f_ceb <= 0;
            shift <= 0;
        end
        else begin
            case (state)
                ST_IDLE: begin
                    cnt <= 0;
                    cnt_out <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    pe_clear <= 1;
                    pe_en <= 0;
                    fsram_addr <= ADDR_START_L2_F;
                    wsram_addr <= ADDR_START_L2_W;
                    fsram_addr_out <= ADDR_START_L2_F + SIZE_FREQ_DOMAIN;
                    fsram_addr_in <= ADDR_START_L2_F;
                    l2_done <= 0;
                    norm_on <= 0;
                    relu_on <= 0;
                    c_w_web <= 1;
                    c_w_ceb <= 1;
                    c_f_web <= 1;
                    c_f_ceb <= 1;
                    if (layer_state == 2 && layer_start) begin
                        state <= ST_L2_WLOAD;
                        c_w_web <= 1;
                        c_w_ceb <= 0;
                        c_f_web <= 1;
                        c_f_ceb <= 0;
                    end
                end   

                ST_L2_WLOAD: begin
                    cnt <= cnt + 1;
                    buffer_mode_w <= 2;
                    buffer_load_w <= 1;
                    buffer_loc_w <= cnt;
                    wsram_addr <= wsram_addr + 1;                   
                    if (cnt == NUM_W_ADDR - 1) begin
                        state <= ST_L2_FLOAD;
                        cnt <= 0;
                        pe_en <= 0;
                        pe_clear <= 1;
                    end 
                end    

                ST_L2_FLOAD: begin 
                    cnt <= cnt + 1;
                    buffer_mode_f <= 2;
                    buffer_load_w <= 0;
                    if(cnt == 0) begin
                        fsram_addr <= fsram_addr + 1;
                        buffer_load_f <= 1;
                        buffer_loc_f <= 0;
                    end
                    if(cnt == 1) begin
                        fsram_addr_in <= fsram_addr + 1;
                        buffer_loc_f <= 1;
                    end
                    if(cnt == 2) begin 
                        buffer_load_f <= 0;
                        buffer_start <= 1;
                    end
                    if(cnt == 3) begin
                        state <= ST_L2_CAL;
                        cnt <= 0;
                        pe_en <= 1;
                        pe_clear <= 0;
                    end
                end
                
                ST_L2_CAL: begin 
                    cnt <= cnt + 1;
                    if(cnt == NUM_L1_OUT_CHANNEL - 1) begin 
                        buffer_start <= 0;
                        pe_clear <= 1;
                        norm_on <= 1;
                        
                    end
                    else if(cnt == NUM_L1_OUT_CHANNEL) begin 
                        fsram_addr <= fsram_addr_out;
                        fsram_addr_out <= fsram_addr_out + 1;
                        norm_on <= 0;                               // If we don't need to change 
                        relu_on <= 1;
                        c_f_web <= 0;
                    end
                    else if(cnt == NUM_L1_OUT_CHANNEL + 1) begin
                        if(cnt_out == NUM_FLOAD - 1) begin
                            state <= ST_L2_DONE;
                            l2_done <= 1;
                        end
                        else begin
                            state <= ST_L2_FLOAD;
                        end
                        cnt <= 0;
                        fsram_addr <= fsram_addr_in;
                        relu_on <= 0;
                        c_f_web <= 1;
                        cnt_out <= cnt_out + 1;
                    end
                end
                
                ST_L2_DONE: begin
                    cnt <= 0;
                    cnt_out <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    buffer_start <= 0;
                    pe_clear <= 1;
                    pe_en <= 0;
                    fsram_addr <= ADDR_START_L2_F;
                    wsram_addr <= ADDR_START_L2_W;
                    fsram_addr_out <= ADDR_START_L2_F + SIZE_FREQ_DOMAIN;
                    fsram_addr_in <= ADDR_START_L2_F;
                    l2_done <= 0;
                    if(layer_state == 2 && layer_start) begin
                        state <= ST_L2_WLOAD;
                        c_w_web <= 1;
                        c_w_ceb <= 0;
                        c_f_web <= 1;
                        c_f_ceb <= 0;
                    end
                end
            endcase
        end
    end
endmodule