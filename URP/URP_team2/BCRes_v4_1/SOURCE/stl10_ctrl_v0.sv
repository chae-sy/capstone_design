`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Donghwan So
// 
// Create Date: 2024/11/19 12:11:49
// Design Name: 
// Module Name: stl9_ctrl_v0
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


module stl10_ctrl_v0
#(
    parameter WIDTH_WSRAM_WL = 128,
    parameter WIDTH_FSRAM_WL = 128,
    parameter WIDTH_W_DATA = 8,
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_FSRAM_ADDR = 10,
    parameter WIDTH_WSRAM_ADDR = 10,
    parameter ADDR_START_L10_W = 61,
    parameter ADDR_START_L10_F = 73

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
    
    output  reg                                 l10_done
);
    localparam                          ST_IDLE             = 0,
                                        ST_L10_WLOAD             = 1,
                                        ST_L10_FLOAD             = 2,
                                        ST_L10_CAL               = 3,
                                        ST_L10_DONE              = 4;
                                       

    localparam                        NUM_W_ADDR              = 3,    // Total # of weight address (16*8/16 = 8)
                                        L10_OUT_ADDR            = 81;
    reg [6:0] cnt;
    reg [2:0] cnt_out;
    reg [2:0] state;
    reg [WIDTH_FSRAM_ADDR-1:0] fsram_addr_in;
    reg [WIDTH_FSRAM_ADDR-1:0] l10_fsram_addr_out;

                                       
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            state <= ST_IDLE;
            fsram_addr <= 0;
            wsram_addr <= 0;
            fsram_addr_in <= 0;
            l10_fsram_addr_out <= 0;
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
            l10_done <= 0;
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
                    fsram_addr <= ADDR_START_L10_F;
                    wsram_addr <= ADDR_START_L10_W;
                    fsram_addr_in <=  ADDR_START_L10_F;
                    l10_fsram_addr_out <= L10_OUT_ADDR;
                    l10_done <= 0;
                    norm_on <= 0;
                    relu_on <= 0;
                    c_w_web <= 1;
                    c_w_ceb <= 1;
                    c_f_web <= 1;
                    c_f_ceb <= 1;
                    pe_clear <= 1;
                    pe_en <= 0;
                    shift <= 0;
                    if (layer_state == 10 && layer_start) begin
                        state <= ST_L10_WLOAD;
                        c_w_web <= 1;
                        c_w_ceb <= 0;
                        c_f_web <= 1;
                        c_f_ceb <= 0;
                    end
                end   

                ST_L10_WLOAD: begin
                    cnt <= cnt + 1;
                    buffer_mode_w <= 10;
                    buffer_load_w <= 1;
                    buffer_loc_w <= cnt; 
                    pe_clear <= 1;
                    pe_en <= 0;                 
                    if (cnt == NUM_W_ADDR) begin
                        state <= ST_L10_FLOAD;
                        cnt <= 0;
                        buffer_load_w <= 0;
                    end 
                    else begin 
                        wsram_addr <= wsram_addr + 1; 
                    end
                end    

                ST_L10_FLOAD: begin 
                    cnt <= cnt + 1;
                    buffer_mode_f <= 10;
                    buffer_load_f <= 1;
                    buffer_loc_f <= cnt;                  
                    if(cnt == 2) begin 
                        state <= ST_L10_CAL;
                        cnt <= 0;
                        fsram_addr <= l10_fsram_addr_out;
                        l10_fsram_addr_out <= l10_fsram_addr_out + 1;
                    end
                    else begin  
                        fsram_addr <= fsram_addr + 1;
                        fsram_addr_in <= fsram_addr_in + 1;
                    end
                end
                
                ST_L10_CAL: begin 
                    cnt <= cnt + 1;
                    buffer_load_f <= 0;
                    if(cnt == 0) begin 
                        buffer_start <= 1;
                    end
                    else if(cnt == 1) begin 
                        pe_clear <= 0;
                        pe_en <= 1;
                    end
                    else if(cnt == 1 + NUM_W_ADDR) begin 
                        buffer_start <= 0;
                        pe_clear <= 1;
                        pe_en <= 0;
                        norm_on <= 1;
                    end
                    else if(cnt == 2 + NUM_W_ADDR) begin 
                        norm_on <= 0;
                        c_f_web <= 0;
                        cnt_out <= cnt_out + 1;
                    end
                    else if(cnt == 3 + NUM_W_ADDR) begin 
                        c_f_web <= 1;
                        if(cnt_out == 3) begin 
                            state <= ST_L10_DONE;
                            if(l10_fsram_addr_out > 89) begin 
                                l10_fsram_addr_out <= L10_OUT_ADDR;
                            end
                            else begin
                                l10_fsram_addr_out <= l10_fsram_addr_out;
                            end
                            l10_done <= 1;
                            cnt <= 0;
                        end
                        else begin 
                            state <= ST_L10_WLOAD;
                            cnt <= 0;
                            fsram_addr <= fsram_addr_in;
                            wsram_addr <= wsram_addr + 1;
                        end
                    end
                end
                
                ST_L10_DONE: begin
                    l10_done <= 0;
                    cnt <= 0;
                    cnt_out <= 0;
                    norm_on <= 0;
                    relu_on <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    buffer_start <= 0;
                    fsram_addr <= ADDR_START_L10_F;
                    wsram_addr <= ADDR_START_L10_W;
                    fsram_addr_in <= ADDR_START_L10_F;
                    if(layer_state == 10 && layer_start) begin
                        state <= ST_L10_WLOAD;
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

