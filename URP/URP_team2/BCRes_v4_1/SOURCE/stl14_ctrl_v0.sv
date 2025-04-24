`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: sjkim
// 
// Create Date: 2024/11/27 15:11:22
// Design Name: 
// Module Name: stl14_ctrl_v0
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


module stl14_ctrl_v0
#(
    parameter WIDTH_WSRAM_WL = 128,
    parameter WIDTH_FSRAM_WL = 128,
    parameter WIDTH_W_DATA = 8,
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_FSRAM_ADDR = 10,
    parameter WIDTH_WSRAM_ADDR = 10,
    parameter ADDR_START_L14_W = 94,
    parameter ADDR_START_L14_F = 81,
    parameter ADDR_START_L15_F = 94 // 
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

    output  reg                                 l14_done
    );
    
    localparam                          ST_IDLE             = 0,
                                        ST_L14_WLOAD             = 1,
                                        ST_L14_FLOAD             = 2,
                                        ST_L14_CAL               = 3,
                                        ST_L14_DONE              = 4;    
                                        
    localparam                          L14_W_h                 = 6,
                                        L14_F_h                 = 2,
                                        L14_OUT_FSRAM_ADDR      = ADDR_START_L15_F + 1;
 
    reg [3:0] cnt;
    reg [3:0] cnt_f;
    reg [WIDTH_FSRAM_ADDR-1:0] l14_start_fsram_addr_in; 
    reg [2:0] state_14;


    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            state_14 <= ST_IDLE;
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
            l14_done <= 0;
            c_w_web <= 0;
            c_w_ceb <= 0;
            c_f_web <= 0;
            c_f_ceb <= 0;
            shift <= 0;
        end
        else begin
            case (state_14)
                ST_IDLE: begin //0
                    cnt <= 0;
                    cnt_f <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    fsram_addr <= ADDR_START_L14_F;
                    wsram_addr <= ADDR_START_L14_W;
                    l14_start_fsram_addr_in <= ADDR_START_L14_F;
                    l14_done <= 0;
                    norm_on <= 0;
                    relu_on <= 0;
                    c_w_web <= 1;
                    c_w_ceb <= 0;
                    c_f_web <= 1;
                    c_f_ceb <= 0;
                    shift <= 0;
                    if (layer_state == 14 && layer_start) begin
                        state_14 <= ST_L14_WLOAD;
                        pe_clear <= 1;  
                    end
                end   

                ST_L14_WLOAD: begin //1
                    pe_clear <= 0;
                    cnt <= cnt + 1;
                    buffer_mode_w <= 14;
                    buffer_load_w <= 1;
                    buffer_loc_w <= cnt;
                    wsram_addr <= wsram_addr + 1;
                    if(cnt == 2) begin
                        cnt <= 0;
                        state_14 <= ST_L14_FLOAD;
                        buffer_mode_f <= 14;
                    end
                end
                
                ST_L14_FLOAD: begin //2
                    cnt <= cnt + 1;
                    c_f_web <= 1;
                    pe_en <= 0;
                    pe_clear <= 0;
                    buffer_load_w <= 0;
                    buffer_load_f <= 1;  
                    buffer_loc_f <= cnt;
                    if ( fsram_addr == ADDR_START_L14_F + 6 ) begin
                        fsram_addr <= ADDR_START_L14_F;
                    end
                    else if ( fsram_addr == ADDR_START_L14_F + 7) begin
                        fsram_addr <= ADDR_START_L14_F + 1;
                    end
                    else if ( fsram_addr == ADDR_START_L14_F + 8) begin
                        fsram_addr <= ADDR_START_L14_F + 2;
                    end
                    else begin
                        fsram_addr <= fsram_addr + 3;
                    end
                    if(cnt == 2) begin
                        fsram_addr <= L14_OUT_FSRAM_ADDR + cnt_f; // for store
                        state_14 <= ST_L14_CAL;
                        cnt <= 0;
                    end
                end
                    
                ST_L14_CAL: begin //3  
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
//                        norm_on <= 1;
                    end
                    else if ( cnt == 5 ) begin
                        norm_on <= 0;
//                        relu_on <= 1;
                        c_f_web <=0;
                    end
                    else if (cnt == 6) begin
                        cnt <= 0;
//                        relu_on <= 0;
                        c_f_web <= 1;
                        pe_clear <= 1;
           
                        if ( cnt_f == 2 ) begin
                            state_14 <= ST_L14_DONE;
                            l14_done <= 1;
                            if(l14_start_fsram_addr_in == ADDR_START_L14_F || l14_start_fsram_addr_in == ADDR_START_L14_F + 3) begin
                                l14_start_fsram_addr_in <= l14_start_fsram_addr_in + 3;
                            end
                            else begin
                                l14_start_fsram_addr_in <= ADDR_START_L14_F;
                            end
                        end                        
                        else begin
                            state_14 <= ST_L14_FLOAD;
                            fsram_addr <= l14_start_fsram_addr_in + 1 + cnt_f; //
                            cnt_f <= cnt_f + 1;
                        end

                    end
                end
                
                ST_L14_DONE: begin //4
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
                    fsram_addr <= l14_start_fsram_addr_in;
                    wsram_addr <= ADDR_START_L14_W;
                    l14_done <= 0;
                    shift <= 0;
                    if(layer_state == 14 && layer_start) begin
                        state_14 <= ST_L14_WLOAD;
                        pe_clear <= 1;
                    end
                end
            endcase
        end
    end
endmodule