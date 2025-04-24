`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Donghwan So
// 
// Create Date: 2024/10/15 14:31:10
// Design Name: 
// Module Name: stl1_ctrl_v0
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


module stl1_ctrl_v0_1
#(
    parameter WIDTH_WSRAM_WL = 128,
    parameter WIDTH_FSRAM_WL = 128,
    parameter WIDTH_W_DATA = 8,
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_FSRAM_ADDR = 10,
    parameter WIDTH_WSRAM_ADDR = 10,
    parameter ADDR_START_L1_W = 0,
    parameter ADDR_START_L1_F = 0,
    parameter SIZE_L1_OUT_CHANNEL = 16,
    parameter SIZE_KERNEL_H = 3,
    parameter SIZE_KERNEL_W = 3

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

    output  reg                                 l1_done
);
    localparam                          ST_IDLE             = 0,
                                        ST_L1_WLOAD             = 1,
                                        ST_L1_FLOAD_INIT        = 2,
                                        ST_L1_CAL1_FLOAD        = 3,
                                        ST_L1_CAL2              = 4,
                                        ST_L1_CAL3              = 5,
                                        ST_L1_DONE              = 6;
                                       

    localparam                          NUM_W_ADDR              = 9,           // $ceil(3 * 3 * 16 / 6)
                                        NUM_F_ADDR_1TIME        = 7,             // $ceil(20 / 3)
                                        SIZE_TIME_DOMAIN        = 3,
                                        SIZE_FREQ_DOMAIN        = 20,
                                        NUM_WEIGHT_KERNEL       = SIZE_KERNEL_H * SIZE_KERNEL_W;  // 9

    reg [6:0] cnt;
    reg [5:0] cnt_out;
    reg [2:0] state;
    reg [WIDTH_FSRAM_ADDR-1:0] fsram_addr_temp;    // temporal address for next feature address
    reg [WIDTH_FSRAM_ADDR-1:0] fsram_addr_out;
    reg [WIDTH_FSRAM_ADDR-1:0] l1_start_fsram_addr;
                                       
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            state <= ST_IDLE;
            fsram_addr <= 0;
            wsram_addr <= 0;
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
            shift <= 0;
            c_w_web <= 0;
            c_w_ceb <= 0;
            c_f_web <= 0;
            c_f_ceb <= 0;
            norm_on <= 0;
            relu_on <= 0;
            l1_done <= 0;
            fsram_addr_temp <= 0;
            l1_start_fsram_addr <= 0;
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
                    fsram_addr <= ADDR_START_L1_F;
                    fsram_addr_temp <= ADDR_START_L1_F;
                    l1_start_fsram_addr <= ADDR_START_L1_F;
                    wsram_addr <= ADDR_START_L1_W;
                    l1_done <= 0;
                    c_w_web <= 1;
                    c_w_ceb <= 1;
                    c_f_web <= 1;
                    c_f_ceb <= 1;
                    pe_en <= 0;
                    pe_clear <= 1;
                    if (layer_state == 1 && layer_start) begin
                        state <= ST_L1_WLOAD;
                        c_w_web <= 1;
                        c_w_ceb <= 0;
                        c_f_web <= 1;
                        c_f_ceb <= 0;
                    end
                end   

                ST_L1_WLOAD: begin
                    cnt <= cnt + 1;
                    buffer_mode_w <= 1;
                    buffer_load_w <= 1;
                    buffer_loc_w <= cnt;
                    wsram_addr <= wsram_addr + 1;                   
                    if (cnt == NUM_W_ADDR - 1) begin
                        state <= ST_L1_FLOAD_INIT;
                        l1_start_fsram_addr <= fsram_addr;
                        cnt <= 0;
                    end 
                end    

                ST_L1_FLOAD_INIT: begin
                    cnt <= cnt +1;
                    buffer_load_w <= 0;
                    buffer_mode_f <= 1;
                    buffer_load_f <= 1;
                    if(fsram_addr + NUM_F_ADDR_1TIME > 20) begin 
                        fsram_addr <= fsram_addr + NUM_F_ADDR_1TIME - 21;
                    end
                    else begin 
                        fsram_addr <= fsram_addr + NUM_F_ADDR_1TIME;
                    end
                    buffer_loc_f <= cnt;
                    if(cnt == 1) begin                        
                        pe_clear <= 1;
                        buffer_start <= 1;
                    end
                    else if (cnt == SIZE_KERNEL_H - 1) begin    // cnt == 2
                        state <= ST_L1_CAL1_FLOAD;
                        fsram_addr_temp <= fsram_addr_temp + 1;
                        fsram_addr_out <= 21;
                        cnt <= 0;                        
                        pe_clear <= 0;
                        pe_en <= 1;
                    end
                end  

                ST_L1_CAL1_FLOAD: begin 
                    cnt <= cnt + 1;
                    if(cnt == SIZE_KERNEL_H - 1) begin      // cnt == 2
                        fsram_addr <= fsram_addr_temp;
                        fsram_addr_temp <= fsram_addr_temp + 1;
                    end
                    else if((cnt > SIZE_KERNEL_H - 1) && (cnt < SIZE_KERNEL_H * 2)) begin 
                        buffer_load_f <= 1;
                        if(fsram_addr + NUM_F_ADDR_1TIME > 20) begin 
                            fsram_addr <= fsram_addr + NUM_F_ADDR_1TIME - 21;
                        end
                        else begin 
                            fsram_addr <= fsram_addr + NUM_F_ADDR_1TIME;
                        end
                        buffer_loc_f <= cnt;
                    end
                    else begin 
                        buffer_load_f <= 0;
                        buffer_loc_f <= 0;
                    end
                    if(cnt == NUM_WEIGHT_KERNEL - 1) begin
                        buffer_start <= 0;
                        pe_clear <= 1;
                        shift <= 1;
                        norm_on <= 1;
                        cnt_out <= cnt_out + 1; 
                    end
                    else if(cnt == NUM_WEIGHT_KERNEL) begin 
                        shift <= 0;
                        buffer_start <= 1;
                        norm_on <= 0;
                        relu_on <= 1;
                        fsram_addr <= fsram_addr_out;
                        fsram_addr_out <= fsram_addr_out + 1;
                        c_f_web <= 0;
                    end 
                    else if(cnt == NUM_WEIGHT_KERNEL+1) begin 
                        state <= ST_L1_CAL2;
                        pe_clear <= 0;
                        relu_on <= 0;
                        c_f_web <= 1;
                        cnt <= 0;
                    end
                end
                    
                ST_L1_CAL2: begin
                    cnt <= cnt + 1;
                    if(cnt == NUM_WEIGHT_KERNEL - 1) begin
                        buffer_start <= 0;
                        pe_clear <= 1;
                        shift <= 1;
                        norm_on <= 1;
                        cnt_out <= cnt_out + 1; 
                    end
                    else if(cnt == NUM_WEIGHT_KERNEL) begin 
                        shift <= 0;
                        buffer_start <= 1;
                        norm_on <= 0;
                        relu_on <= 1;
                        fsram_addr <= fsram_addr_out;
                        fsram_addr_out <= fsram_addr_out + 1;
                        c_f_web <= 0;
                    end 
                    else if(cnt == NUM_WEIGHT_KERNEL+1) begin 
                        state <= ST_L1_CAL3;
                        pe_clear <= 0;
                        relu_on <= 0;
                        c_f_web <= 1;
                        cnt <= 0;
                    end
                end    

                ST_L1_CAL3: begin 
                    cnt <= cnt + 1;
                    if(cnt == NUM_WEIGHT_KERNEL - 1) begin
                        buffer_start <= 0;
                        pe_clear <= 1;
                        shift <= 1;
                        norm_on <= 1;
                        cnt_out <= cnt_out + 1; 
                    end
                    else if(cnt == NUM_WEIGHT_KERNEL) begin 
                        shift <= 0;
                        buffer_start <= 1;
                        norm_on <= 0;
                        relu_on <= 1;
                        fsram_addr <= fsram_addr_out;
                        fsram_addr_out <= fsram_addr_out + 1;
                        c_f_web <= 0;
                    end 
                    else if(cnt == NUM_WEIGHT_KERNEL+1) begin    
                        pe_clear <= 0;
                        relu_on <= 0;
                        c_f_web <= 1;
                        cnt <= 0;
                        if(cnt_out == 18) begin 
                            state <= ST_L1_DONE;
                            l1_done <= 1;
                        end
                        else begin 
                            state <= ST_L1_CAL1_FLOAD;
                        end
                    end
                end
                
                ST_L1_DONE: begin
                    cnt <= 0;
                    cnt_out <= 0;
                    pe_clear <= 1;
                    pe_en <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    buffer_start <= 0;
                    if(l1_start_fsram_addr + NUM_F_ADDR_1TIME > 20) begin 
                        fsram_addr <= l1_start_fsram_addr + NUM_F_ADDR_1TIME - 21;
                    end
                    else begin 
                        fsram_addr <= l1_start_fsram_addr + NUM_F_ADDR_1TIME;
                    end
                    if(l1_start_fsram_addr + NUM_F_ADDR_1TIME > 20) begin 
                        fsram_addr_temp <= l1_start_fsram_addr + NUM_F_ADDR_1TIME - 21;
                    end
                    else begin 
                        fsram_addr_temp <= l1_start_fsram_addr + NUM_F_ADDR_1TIME;
                    end
                    wsram_addr <= ADDR_START_L1_W;
                    fsram_addr_out <= 21;
                    c_f_ceb <= 1;
                    c_w_ceb <= 1;
                    l1_done <= 0;
                    if(layer_state == 1 && layer_start) begin
                        state <= ST_L1_WLOAD;
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