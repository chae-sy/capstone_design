`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: sjkim
// 
// Create Date: 2024/11/12 14:51:54
// Design Name: 
// Module Name: stl19_ctrl_v0 -> 22
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


module stl22_ctrl_v0
#(
    parameter WIDTH_WSRAM_WL = 96,
    parameter WIDTH_FSRAM_WL = 128,
    parameter WIDTH_W_DATA = 6,
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_FSRAM_ADDR = 10,
    parameter WIDTH_WSRAM_ADDR = 10,
    parameter ADDR_START_L22_W = 219, // temp
    parameter ADDR_START_L22_F = 119, // temp
    parameter NUM_POOL = 22
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

    output  reg                                 l22_done
    );
    
    localparam                          ST_IDLE             = 0,
                                        ST_L22_FLOAD             = 1,
                                        ST_L22_WLOAD             = 2,
                                        ST_L22_CAL               = 3,
                                        ST_L22_DONE              = 4;    
                                        
    localparam                          NUM_W_ADDR = 8,
                                        NUM_W_LOAD = 4,
                                        classifier_pw_layernum = 22; // temp
 
    reg [6:0] cnt;
    reg [3:0] cnt_w;
    reg [2:0] state_22;
    reg [4:0] cnt_out;
     
    
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            state_22 <= ST_IDLE;
            fsram_addr <= 0;
            wsram_addr <= 0;
            cnt <= 0;
            cnt_w <= 0;
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
            l22_done <= 0;
            c_w_web <= 0;
            c_w_ceb <= 0;
            c_f_web <= 0;
            c_f_ceb <= 0;
            shift <= 0;
            cnt_out <=0;
        end
        else begin
            case (state_22)
                ST_IDLE: begin //0
                    cnt <= 0;
                    cnt_out <= 0;
                    cnt_w <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    fsram_addr <= ADDR_START_L22_F;
                    wsram_addr <= ADDR_START_L22_W;
                    l22_done <= 0;
                    norm_on <= 0;
                    relu_on <= 0;
                    c_w_web <= 1;
                    c_w_ceb <= 0;
                    c_f_web <= 1;
                    c_f_ceb <= 0;
                    shift <= 0;
                    if (layer_state == classifier_pw_layernum && layer_start) begin
                        state_22 <= ST_L22_FLOAD;
                        pe_clear <= 1; 
                    end
                end   

                ST_L22_FLOAD: begin //1
                    cnt <= cnt + 1;
                    pe_clear <= 0;
                    buffer_mode_f <= 22;
                    buffer_load_f <= 1;
                    buffer_load_w <= 0;
                    if(cnt == 0) begin
                        fsram_addr <= fsram_addr + 1;
                        buffer_loc_f <= 0;
                    end
                    if(cnt == 1) begin
                        buffer_loc_f <= 1;
//                        buffer_load_w <= 1;
                        cnt <= 0;
                        state_22 <= ST_L22_WLOAD;
                        cnt_out <= cnt_out+1;
                    end
//                    if(cnt == 2) begin
//                        state <= ST_L19_WLOAD;
//                        cnt <= 0; // ?¡×¡¤??
//                        buffer_load_f <= 0; //¢´¢´
//                        pe_clear <= 1; //
//                    end
                end
                
                ST_L22_WLOAD: begin //2
                    pe_clear <= 0;
                    cnt <= cnt + 1;
                    buffer_load_f <= 0;
                    buffer_load_w <= 1;                    
                    buffer_mode_w <= 22;
                    buffer_loc_w <= cnt;
//                    wsram_addr <= wsram_addr + 1;    
//                    if (cnt == NUM_W_ADDR - 2) begin
//                        buffer_start <=1; end
                    if (cnt < NUM_W_ADDR) wsram_addr <= wsram_addr + 1;            
                    if (cnt == NUM_W_ADDR ) begin
                        state_22 <= ST_L22_CAL;
                        cnt <= 0;
                        cnt_w <= cnt_w + 1;
//                        buffer_start <= 1;
//                        wsram_addr <= wsram_addr-1;
//                        buffer_start <= 1;
                        buffer_load_w <= 0;
//                        wsram_addr <= wsram_addr + 1;    
                    end 
                end    
                
                ST_L22_CAL: begin //3
                    buffer_start <= 1;
                    pe_en <= 1;
                    cnt <= cnt + 1; 
                    pe_clear <= 0;
//                    if(cnt == NUM_W_ADDR ) begin
//                        buffer_start <= 0;
//                        if(cnt_w == NUM_W_LOAD ) begin
//                            state <= ST_L19_DONE;
//                            l19_done <= 1;
//                        end
//                        else begin
//                            state <= ST_L19_WLOAD;
//                        end
//                        cnt <= 0;
//                        relu_on <= 0;
//                        norm_on <= 0;
//                    end
                //
                    if (cnt == NUM_W_ADDR ) begin
                        buffer_start <= 0;
                        if (cnt_w < NUM_W_LOAD) begin
                            state_22 <= ST_L22_WLOAD;
                            cnt <= 0;
                            relu_on <= 0;
                            norm_on <= 0;
                        end
                    end
                    else if (cnt == NUM_W_ADDR+1) begin
                        buffer_start <= 0;
                        relu_on <=0;
                        norm_on <= 1;                        
//                        state <= ST_L19_DONE;
//                        l19_done <= 1;                   
//                        buffer_start <= 0;
//                        cnt <=0;
//                        relu_on <=0;
//                        norm_on <= 1;
                        fsram_addr <= ADDR_START_L22_F + 1 + cnt_out;
                    end
                    else if (cnt == NUM_W_ADDR+2 ) begin
//                        state <= ST_L19_DONE;
//                        l19_done <= 1;                   
//                        cnt <=0;
                        relu_on <=0;
                        norm_on <= 0;
                        c_f_web <= 0;
                        
                    end
                    else if (cnt == NUM_W_ADDR+3) begin
                        c_f_web <= 1;
                        state_22 <= ST_L22_DONE;
                        l22_done <= 1;                           
                        cnt <=0;
                        
                    end 
                    //                    
                end
                
                ST_L22_DONE: begin //4
                    cnt <= 0;
                    cnt_w <= 0;
                    pe_en <= 0;
                    norm_on <= 0;
                    c_f_web <= 1;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    buffer_start <= 0;
                    fsram_addr <= ADDR_START_L22_F;
                    wsram_addr <= ADDR_START_L22_W;
                    l22_done <= 0;
                    shift <= 0;
                    if (cnt_out == NUM_POOL + 1 ) cnt_out <= 0;
                    if(layer_state == classifier_pw_layernum && layer_start) begin
                        state_22 <= ST_L22_FLOAD;
                    end
                end
            endcase
        end
    end
endmodule                                   