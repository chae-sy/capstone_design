`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Donghwan So
// 
// Create Date: 2024/11/19 15:54:47
// Design Name: 
// Module Name: stl12_ctrl_v0
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


module stl13_ctrl_v0
#(
    parameter WIDTH_WSRAM_WL = 128,
    parameter WIDTH_FSRAM_WL = 128,
    parameter WIDTH_W_DATA = 8,
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_FSRAM_ADDR = 10,
    parameter WIDTH_WSRAM_ADDR = 10,
    parameter ADDR_START_L13_W = 78,
    parameter ADDR_START_L13_F = 93

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
    
    output  reg                                 l13_done
);
    localparam                          ST_IDLE             = 0,
                                        ST_L13_FLOAD             = 1,
                                        ST_L13_WLOAD1            = 2,
                                        ST_L13_CAL1              = 3,
                                        ST_L13_WLOAD2            = 4,
                                        ST_L13_CAL2              = 5,
                                        ST_L13_DONE              = 6;
                                       

    localparam                        NUM_W_ADDR              = 8;    // Total # of weight address (16*8/16 = 8)

    reg [6:0] cnt;
    reg [2:0] state;
                                       
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            state <= ST_IDLE;
            fsram_addr <= 0;
            wsram_addr <= 0;
            cnt <= 0;
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
            l13_done <= 0;
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
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    buffer_start <= 0;
                    fsram_addr <= ADDR_START_L13_F;
                    wsram_addr <= ADDR_START_L13_W;
                    l13_done <= 0;
                    norm_on <= 0;
                    relu_on <= 0;
                    c_w_web <= 1;
                    c_w_ceb <= 1;
                    c_f_web <= 1;
                    c_f_ceb <= 1;
                    pe_clear <= 1;
                    pe_en <= 0;
                    shift <= 0;
                    if (layer_state == 13 && layer_start) begin
                        state <= ST_L13_FLOAD;
                        c_w_web <= 1;
                        c_w_ceb <= 0;
                        c_f_web <= 1;
                        c_f_ceb <= 0;
                    end
                end   
                
                ST_L13_FLOAD: begin 
                    buffer_mode_f <= 13;
                    buffer_loc_f <= 0;
                    buffer_load_f <= 1;
                    fsram_addr <= fsram_addr + 1;
                    pe_clear <= 1;
                    pe_en <= 0;     
                    state <= ST_L13_WLOAD1;
                    cnt <= 0;
                end
                
                ST_L13_WLOAD1: begin
                    cnt <= cnt + 1;                         
                    if (cnt == NUM_W_ADDR) begin
                        buffer_load_w <= 0;
                        buffer_start <= 1;
                        state <= ST_L13_CAL1;
                        cnt <= 0;
                    end 
                    else begin
                        buffer_loc_w <= cnt;
                        wsram_addr <= wsram_addr + 1;  
                        if(cnt == 0) begin 
                            buffer_load_f <= 0;
                            buffer_load_w <= 1;
                            buffer_mode_w <= 13;
                        end
                    end
                end    

                
                
                ST_L13_CAL1: begin 
                    cnt <= cnt + 1;
                    if(cnt == 0) begin 
                        pe_clear <= 0;
                        pe_en <= 1;
                    end
                    else if(cnt == NUM_W_ADDR - 1) begin 
                        buffer_start <= 0;
                    end
                    else if(cnt == NUM_W_ADDR) begin 
                        pe_en <= 0;
                        norm_on <= 0;
                        relu_on <= 0;
                        state <= ST_L13_WLOAD2;
                        cnt <= 0;
                    end
                end
                
                ST_L13_WLOAD2: begin
                    cnt <= cnt + 1;                         
                    if (cnt == NUM_W_ADDR) begin
                        buffer_load_w <= 0;
                        buffer_start <= 1;
                        state <= ST_L13_CAL2;
                        cnt <= 0;
                    end 
                    else begin
                        buffer_loc_w <= cnt;
                        wsram_addr <= wsram_addr + 1;  
                        if(cnt == 0) begin 
                            buffer_load_w <= 1;
                        end
                    end
                end  
                
                ST_L13_CAL2: begin 
                    cnt <= cnt + 1;
                    if(cnt == 0) begin 
                        pe_clear <= 0;
                        pe_en <= 1;
                    end
                    else if(cnt == NUM_W_ADDR) begin 
                        buffer_start <= 0;
                        pe_clear <= 1;
                        pe_en <= 0;
                        norm_on <= 0;
                        relu_on <= 0;
                    end
                    else if(cnt == NUM_W_ADDR + 1) begin 
                        c_f_web <= 0;
                    end
                    else if(cnt == NUM_W_ADDR + 2) begin 
                        c_f_web <= 1;
                        l13_done <= 1;
                        state <= ST_L13_DONE;
                        cnt <= 0;
                    end
                end
                
                ST_L13_DONE: begin
                    cnt <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    buffer_start <= 0;
                    fsram_addr <= ADDR_START_L13_F;
                    wsram_addr <= ADDR_START_L13_W;
                    l13_done <= 0;
                    if(layer_state == 13 && layer_start) begin
                        state <= ST_L13_FLOAD;
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