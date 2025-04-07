`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Donghwan So
// 
// Create Date: 2024/11/19 13:55:01
// Design Name: 
// Module Name: stl11_ctrl_v0
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

module stl12_ctrl_v0
#(
    parameter WIDTH_WSRAM_WL = 128,
    parameter WIDTH_FSRAM_WL = 128,
    parameter WIDTH_W_DATA = 8,
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_FSRAM_ADDR = 10,
    parameter WIDTH_WSRAM_ADDR = 10,
    parameter ADDR_START_L12_W = 74,
    parameter ADDR_START_L12_F = 90

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
    
    output  reg                                 l12_done
);
    localparam                          ST_IDLE             = 0,
                                        ST_L12_WLOAD             = 1,
                                        ST_L12_FLOAD             = 2,
                                        ST_L12_CAL               = 3,
                                        ST_L12_DONE              = 4;
                                       

    localparam                        NUM_W_ADDR              = 3,    // Total # of weight address (16*8/16 = 8)
                                       L12_OUT_FSRAM_ADDR        = 93;
    reg [6:0] cnt;
    reg [2:0] state;
    reg [WIDTH_FSRAM_ADDR-1:0] l12_start_fsram_addr_in;
                                       
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            state <= ST_IDLE;
            fsram_addr <= 0;
            wsram_addr <= 0;
            l12_start_fsram_addr_in <= 0;
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
            l12_done <= 0;
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
                    fsram_addr <= ADDR_START_L12_F;
                    wsram_addr <= ADDR_START_L12_W;
                    l12_start_fsram_addr_in <= ADDR_START_L12_F;
                    l12_done <= 0;
                    norm_on <= 0;
                    relu_on <= 0;
                    c_w_web <= 1;
                    c_w_ceb <= 1;
                    c_f_web <= 1;
                    c_f_ceb <= 1;
                    pe_clear <= 1;
                    pe_en <= 0;
                    shift <= 0;
                    if (layer_state == 12 && layer_start) begin
                        state <= ST_L12_WLOAD;
                        c_w_web <= 1;
                        c_w_ceb <= 0;
                        c_f_web <= 1;
                        c_f_ceb <= 0;
                    end
                end   

                ST_L12_WLOAD: begin
                    pe_clear <= 1;
                    pe_en <= 0;
                    cnt <= cnt + 1;
                    buffer_mode_w <= 12;
                    buffer_load_w <= 1;
                    buffer_loc_w <= cnt;
                    wsram_addr <= wsram_addr + 1;                   
                    if (cnt == NUM_W_ADDR - 1) begin
                        state <= ST_L12_FLOAD;
                        cnt <= 0;
                    end 
                end    

                ST_L12_FLOAD: begin 
                    cnt <= cnt + 1;
                    buffer_load_w <= 0;
                    buffer_mode_f <= 12;
                    buffer_load_f <= 1;
                    buffer_loc_f <= cnt;
                    
                    
                    if(cnt == 2) begin 
                        state <= ST_L12_CAL;
                        cnt <= 0;
                        buffer_start <= 1;
                    end
                    else begin 
                        if(fsram_addr == ADDR_START_L12_F || fsram_addr == ADDR_START_L12_F + 1) begin
                            fsram_addr <= fsram_addr + 1;
                        end
                        else begin
                            fsram_addr <= ADDR_START_L12_F; 
                        end
                    end
                end
                
                ST_L12_CAL: begin 
                    cnt <= cnt + 1;
                    buffer_load_f <= 0;
                    if(cnt == 0) begin 
                        pe_clear <= 0;
                        pe_en <= 1;
                    end
                    else if(cnt == 3) begin 
                        buffer_start <= 0;
                        pe_clear <= 1;
                        pe_en <= 0;
                        norm_on <= 1;
                    end
                    else if(cnt == 4) begin 
                        norm_on <= 0;
                        relu_on <= 1;
                        c_f_web <= 0;
                        fsram_addr <= L12_OUT_FSRAM_ADDR;
                    end
                    else if(cnt == 5) begin 
                        state <= ST_L12_DONE;
                        if(l12_start_fsram_addr_in == ADDR_START_L12_F || l12_start_fsram_addr_in == ADDR_START_L12_F + 1) begin
                            l12_start_fsram_addr_in <= l12_start_fsram_addr_in + 1;
                        end
                        else begin
                            l12_start_fsram_addr_in <= ADDR_START_L12_F; 
                        end
                        cnt <= 0;
                        relu_on <= 0;
                        c_f_web <= 1;
                        l12_done <= 1;
                    end
                end
                
                ST_L12_DONE: begin
                    cnt <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    buffer_start <= 0;
                    fsram_addr <= l12_start_fsram_addr_in;
                    wsram_addr <= ADDR_START_L12_W;
                    l12_done <= 0;
                    if(layer_state == 12 && layer_start) begin
                        state <= ST_L12_WLOAD;
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