`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: sjkim
// 
// Create Date: 2024/11/19 12:21:56
// Design Name: 
// Module Name: stl17_ctrl_v0 -> L19
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


module stl19_ctrl_v0
#(
    parameter WIDTH_WSRAM_WL = 128,
    parameter WIDTH_FSRAM_WL = 128,
    parameter WIDTH_W_DATA = 8,
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_FSRAM_ADDR = 10,
    parameter WIDTH_WSRAM_ADDR = 10,
    parameter ADDR_START_L19_W = 148,
    parameter ADDR_START_L19_F = 113 // temp
     // temp

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

    output  reg                                 l19_done
    );
    
    localparam                          ST_IDLE             = 0,
                                        ST_L19_FLOAD             = 1,
                                        ST_L19_WLOAD             = 2,
                                        ST_L19_CAL               = 3,
                                        ST_L19_DONE              = 4;    
                                        
    localparam                          L19_F_h    = 2,
                                        NUM_W_ADDR = 8;
 
    reg [3:0] cnt;
    reg [3:0] cnt_f;
    reg [2:0] state_19;
    
     
    
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            state_19 <= ST_IDLE;
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
            l19_done <= 0;
            c_w_web <= 0;
            c_w_ceb <= 0;
            c_f_web <= 0;
            c_f_ceb <= 0;
            shift <= 0;
        end
        else begin
            case (state_19)
                ST_IDLE: begin //0
                    cnt <= 0;
                    cnt_f <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    fsram_addr <= ADDR_START_L19_F;
                    wsram_addr <= ADDR_START_L19_W;
                    l19_done <= 0;
                    norm_on <= 0;
                    relu_on <= 0;
                    c_w_web <= 1;
                    c_w_ceb <= 0;
                    c_f_web <= 1;
                    c_f_ceb <= 0;
                    shift <= 0;
                    if (layer_state == 19 && layer_start) begin
                        state_19 <= ST_L19_FLOAD;
                        pe_clear <= 1;  
                    end
                end   

                ST_L19_FLOAD: begin //1
                    buffer_load_f <= 1;
                    pe_clear <= 0;
                    cnt <= cnt + 1;
                    buffer_mode_f <= 19;
                    if(cnt == 0) begin
                        fsram_addr <= fsram_addr + 1;
                        buffer_loc_f <= 0;
                    end
                    if(cnt == 1) begin
                        buffer_loc_f <= 1;
                        cnt <= 0;
                        state_19 <= ST_L19_WLOAD;
                    end
                end
                
                ST_L19_WLOAD: begin //2
                    pe_en <= 0;
                    c_f_web <= 1;
                    buffer_load_w <= 1;
                    pe_clear <= 0;
                    cnt <= cnt + 1;
                    buffer_load_f <= 0;                  
                    buffer_mode_w <= 19;
                    buffer_loc_w <= cnt;
                    if (cnt < NUM_W_ADDR -1 ) wsram_addr <= wsram_addr + 2;            
                    if (cnt == NUM_W_ADDR ) begin
                        state_19 <= ST_L19_CAL;
                        cnt <= 0;
                        cnt_f <= cnt_f + 1;
                        buffer_load_w <= 0;
                        buffer_mode_f <= 19;
                        buffer_start <= 1;
                        pe_en <= 1;
                    end 
                end    
                
                ST_L19_CAL: begin //3     
                    cnt <= cnt + 1; 
                    pe_clear <= 0;
                    if ( cnt ==  NUM_W_ADDR -1) begin // cnt = 7
                        buffer_start <= 0;
                         wsram_addr <= wsram_addr + 2;
                    end
                    else if ( cnt == NUM_W_ADDR ) begin // cnt = 8
                        pe_en <= 0;
                        if (cnt_f == 4) begin
                            pe_clear <= 1;
                            cnt <= cnt +1;
                            fsram_addr <= ADDR_START_L19_F + L19_F_h; // for store
                        end
                        else if (cnt_f == 8) begin
//                            state_17 <= ST_L17_DONE;
                            cnt <= cnt +1;
                            fsram_addr <= ADDR_START_L19_F + L19_F_h +1; // for store
                        end
                        else begin
                            state_19 <= ST_L19_WLOAD;
                            cnt <= 0;
                        end
                    end
                    else if (cnt == NUM_W_ADDR + 1) begin // cnt = 9
                        c_f_web <= 0;
                        if (cnt_f == 4) begin
                            state_19 <= ST_L19_WLOAD;
                            wsram_addr <= ADDR_START_L19_W + 1;
                            cnt <= 0;                
                        end
                    end
                    else if ( cnt == NUM_W_ADDR + 2 )  begin //10
                            c_f_web <= 1;
                            state_19 <= ST_L19_DONE;
                            l19_done <= 1;
                        end 
                end

                
                ST_L19_DONE: begin //4
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
                    fsram_addr <= ADDR_START_L19_F;
                    wsram_addr <= ADDR_START_L19_W;
                    l19_done <= 0;
                    shift <= 0;
                    if(layer_state == 19 && layer_start) begin
                        state_19 <= ST_L19_FLOAD;
                        pe_clear <= 1;
                    end
                end
            endcase
        end
    end
endmodule                                   