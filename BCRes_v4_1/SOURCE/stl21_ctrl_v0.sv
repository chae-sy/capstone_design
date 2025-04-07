`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: sjkim
// 
// Create Date: 2024/11/16 18:01:17
// Design Name: 
// Module Name: stl18_ctrl_v0 -> L21
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


module stl21_ctrl_v0
#(
    parameter WIDTH_WSRAM_WL = 128,
    parameter WIDTH_FSRAM_WL = 128,
    parameter WIDTH_W_DATA = 8,
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_FSRAM_ADDR = 10,
    parameter WIDTH_WSRAM_ADDR = 10,
    parameter ADDR_START_L21_W = 218, // 
    parameter ADDR_START_L21_F = 115 // 
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

    output  reg                                 l21_done
    );
    
    localparam                          ST_IDLE             = 0,
                                        ST_L21_WLOAD             = 1,
                                        ST_L21_FLOAD             = 2,
                                        ST_L21_CAL               = 3,
                                        ST_L21_DONE              = 4;    
 
    localparam                          L21_F_h     = 2,
                                        ADDR_F2 = ADDR_START_L21_F + 2; // L20 out temp??;
                                        
    reg [2:0] cnt;
    reg [1:0] cnt_f;
    reg [1:0] cnt_inf; // for L16 in
    reg [2:0] state_l21;
    
    
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            state_l21 <= ST_IDLE;
            fsram_addr <= 0;
            wsram_addr <= 0;
            cnt <= 0;
            cnt_f <= 0;
            cnt_inf <= 0;
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
            l21_done <= 0;
            c_w_web <= 0;
            c_w_ceb <= 0;
            c_f_web <= 0;
            c_f_ceb <= 0;
            shift <= 0;
        end
        else begin
            case (state_l21)
                ST_IDLE: begin //0
                    cnt <= 0;
                    cnt_f <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    fsram_addr <= ADDR_START_L21_F;
                    wsram_addr <= ADDR_START_L21_W;
                    l21_done <= 0;
                    norm_on <= 0;
                    relu_on <= 0;
                    c_w_web <= 1;
                    c_w_ceb <= 0;
                    c_f_web <= 1;
                    c_f_ceb <= 0;
                    shift <= 0;
                    if (layer_state == 21 && layer_start) begin
                        state_l21 <= ST_L21_WLOAD;
                        pe_clear <= 1; 
                        buffer_load_w <= 1;
                    end
                end 
                
                ST_L21_WLOAD: begin //1
                    pe_clear <= 0;
                    buffer_mode_w <= 21;
                    buffer_load_w <= 0;
                    buffer_loc_w <= 0;
                    state_l21 <= ST_L21_FLOAD;
                    end
                
                ST_L21_FLOAD: begin //2
                    cnt <= cnt+1;
                    buffer_mode_f <= 21;
                    buffer_load_f <= 1;
//                    if (cnt == 0 ) begin
//                        fsram_addr <= ADDR_START_L21_F + cnt_f;
//                    end
//                    else if (cnt == 1) begin
//                        fsram_addr <= ADDR_START_L21_F + L21_F_h + cnt_f;
//                        buffer_loc_f <= 1;
//                        end
//                    else if (cnt == 2) begin
//                        state_l21 <= ST_L21_CAL;
//                        fsram_addr <= ADDR_START_L21_F + 2*(L21_F_h) + cnt_f;
//                        cnt_f <= cnt_f +1;
//                        buffer_load_f <= 0;
//                        buffer_start <= 1;
//                        cnt <= 0;
//                        pe_en <= 1;
//                        end
                    if (cnt == 0 ) begin
                        fsram_addr <= ADDR_START_L21_F + L21_F_h + cnt_f;
                    end
                    else 
                    if (cnt == 1) begin
                        buffer_loc_f <= 1;
                        end
                    else if (cnt == 2) begin
                        state_l21 <= ST_L21_CAL;
                        fsram_addr <= ADDR_START_L21_F + 2*(L21_F_h) + cnt_f;
                        cnt_f <= cnt_f +1;
                        buffer_load_f <= 0;
                        buffer_start <= 1;
                        cnt <= 0;
                        pe_en <= 1;
                        end
                end
                
                ST_L21_CAL: begin //3
                    cnt <= cnt + 1;                                        
                    if ( cnt == 1 ) begin
                        buffer_start <=0;
                    end                           
                    else if ( cnt == 2) begin
                        pe_en <= 0;
                    end
                    else if (cnt == 3 ) begin
                        relu_on <=1;
                        c_f_web <= 0;
                        pe_clear <= 1;
                    end
                    else if (cnt == 4) begin
                        cnt <= 0;
                        c_f_web <= 1;
                        buffer_loc_f <= 0;
                        relu_on <= 0;
//                        fsram_addr <= ADDR_START_L21_F + cnt_f + 2*L21_F_h - 1; // store addr
                        pe_clear <= 0;
                        if ( cnt_f == 1 ) begin
                            state_l21 <= ST_L21_FLOAD;
                            fsram_addr <= ADDR_START_L21_F + cnt_f;
                        end
                        else if (cnt_f == 2) begin
                            state_l21 <= ST_L21_DONE;               
                            l21_done <= 1;
//                            if ( cnt_inf < 2 ) cnt_inf <= cnt_inf + 1;
//                            else cnt_inf <= 0;
                            cnt_inf <= 0;
                        end 
                    end
               end
              
                ST_L21_DONE: begin //4
                    cnt <= 0;
                    cnt_f <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    fsram_addr <= ADDR_START_L21_F;// load addr
                    wsram_addr <= ADDR_START_L21_W;
                    l21_done <= 0;
                    norm_on <= 0;
                    relu_on <= 0;
                    c_w_web <= 1;
                    c_w_ceb <= 0;
                    c_f_web <= 1;
                    c_f_ceb <= 0;
                    shift <= 0;
                    if (layer_state == 21 && layer_start) begin
                        state_l21 <= ST_L21_WLOAD;
                        pe_clear <= 1; 
                        buffer_load_w <= 1;
                    end
                end
            endcase
        end
    end
    

endmodule