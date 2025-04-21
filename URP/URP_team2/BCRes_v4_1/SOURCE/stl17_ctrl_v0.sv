`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: sjkim
// 
// Create Date: 2024/11/20 12:44:03
// Design Name: 
// Module Name: stl15_ctrl_v0 -> L17
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


module stl17_ctrl_v0
#(
    parameter WIDTH_WSRAM_WL = 128,
    parameter WIDTH_FSRAM_WL = 128,
    parameter WIDTH_W_DATA = 8,
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_FSRAM_ADDR = 10,
    parameter WIDTH_WSRAM_ADDR = 10,
    parameter ADDR_START_L17_W = 132,
    parameter ADDR_START_L17_F = 101
     

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

    output  reg                                 l17_done
    );
    
    localparam                          ST_IDLE             = 0,
                                        ST_L17_WLOAD             = 1,
                                        ST_L17_FLOAD             = 2,
                                        ST_L17_CAL               = 3,
                                        ST_L17_DONE              = 4;    
                                        
    localparam                          L17_W_h                 = 6,
                                        L17_F_h                 = 6,
                                        L17_OUT_FSRAM_ADDR      = ADDR_START_L17_F + L17_F_h;
 
    reg [3:0] cnt;
    reg [2:0] cnt_f;
    reg [WIDTH_FSRAM_ADDR-1:0] l17_store_fsram_addr_in; 
    reg [2:0] state_17;
    
     
    
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            state_17 <= ST_IDLE;
            fsram_addr <= 0;
            wsram_addr <= 0;
            cnt <= 0;
            cnt_f <= 0;
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
            l17_done <= 0;
            c_w_web <= 0;
            c_w_ceb <= 0;
            c_f_web <= 0;
            c_f_ceb <= 0;
            shift <= 0;
        end
        else begin
            case (state_17)
                ST_IDLE: begin //0
                    cnt <= 0;
                    cnt_f <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    fsram_addr <= ADDR_START_L17_F;
                    wsram_addr <= ADDR_START_L17_W;
                    l17_store_fsram_addr_in <= L17_OUT_FSRAM_ADDR;
                    l17_done <= 0;
                    norm_on <= 0;
                    relu_on <= 0;
                    c_w_web <= 1;
                    c_w_ceb <= 0;
                    c_f_web <= 1;
                    c_f_ceb <= 0;
                    shift <= 0;
                    if (layer_state == 17 && layer_start) begin
                        state_17 <= ST_L17_WLOAD;
                        pe_clear <= 1;  
                    end
                end   

                ST_L17_WLOAD: begin //1
                    pe_clear <= 0;
                    cnt <= cnt + 1;
                    buffer_mode_w <= 17;
                    buffer_load_w <= 1;
                    buffer_loc_w <= cnt;
                    wsram_addr <= wsram_addr + 2;
                    if(cnt == 2) begin
                        wsram_addr <= ADDR_START_L17_W + 1; 
                    end
                    else if(cnt == 5) begin
                        wsram_addr <= ADDR_START_L17_W + L17_W_h; // for bn
                        cnt <= 0;
                        state_17 <= ST_L17_FLOAD;
                        buffer_mode_f <= 17;
                    end
                end
                
                ST_L17_FLOAD: begin //2
                    cnt <= cnt + 1;
                    c_f_web <= 1;
                    pe_en <= 0;
                    pe_clear <= 0;
                    buffer_load_w <= 0;
                    buffer_load_f <= 1;  
                    buffer_loc_f <= cnt;
                    if(cnt == 2) begin
                        fsram_addr <= l17_store_fsram_addr_in + cnt_f; // for store
                        state_17 <= ST_L17_CAL;
                        cnt <= 0;
                    end
                    else begin 
                        fsram_addr <= fsram_addr + 2;
                    end
                end    
                
                ST_L17_CAL: begin //3  
                    cnt <= cnt + 1; 
                    buffer_load_f <= 0;
                    if ( cnt ==  0) begin // 
                        pe_en <= 1;
                        buffer_start <= 1;
                    end
                    else if ( cnt == 3 ) begin // 
                        buffer_start <= 0;
                    end
                    else if ( cnt == 4 ) begin
                        pe_en <= 0;
                        norm_on <= 1;
                    end
                    else if ( cnt == 5 ) begin
                        norm_on <= 0;
                        c_f_web <=0;
                    end
                    else if (cnt == 6) begin
                        cnt <= 0;
                        c_f_web <= 1;
                        pe_clear <= 1;
                        if ( cnt_f == 0) begin
                            state_17 <= ST_L17_FLOAD;
                            fsram_addr <= ADDR_START_L17_F + 1; //
                            wsram_addr <= wsram_addr + 1;
                            cnt_f <= cnt_f + 1;
                        end
                        else if ( cnt_f == 1) begin
                            state_17 <= ST_L17_DONE;
                            l17_done <= 1;
                            if(l17_store_fsram_addr_in == L17_OUT_FSRAM_ADDR || l17_store_fsram_addr_in == L17_OUT_FSRAM_ADDR + 2) begin
                                l17_store_fsram_addr_in <= l17_store_fsram_addr_in + 2;
                            end
                            else begin
                                l17_store_fsram_addr_in <= L17_OUT_FSRAM_ADDR;
                            end
                        end
                    end
                end
                
                ST_L17_DONE: begin //4
                    cnt <= 0;
                    cnt_f <= 0;
                    pe_en <= 0;
                    norm_on <= 0;
                    c_f_web <= 1;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    buffer_start <= 0;
                    fsram_addr <= ADDR_START_L17_F;
                    wsram_addr <= ADDR_START_L17_W;
                    l17_done <= 0;
                    shift <= 0;
                    if(layer_state == 17 && layer_start) begin
                        state_17 <= ST_L17_WLOAD;
                        pe_clear <= 1;
                    end
                end
            endcase
        end
    end
endmodule