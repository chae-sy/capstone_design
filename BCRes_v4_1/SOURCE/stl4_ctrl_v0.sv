`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/11/12 01:51:02
// Design Name: 
// Module Name: stl4_ctrl_v0
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


module stl4_ctrl_v0 #(
    parameter WIDTH_WSRAM_WL = 128,
    parameter WIDTH_FSRAM_WL = 128,
    parameter WIDTH_W_DATA = 8,
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_FSRAM_ADDR = 10,
    parameter WIDTH_WSRAM_ADDR = 10,
    parameter ADDR_START_L4_W = 35,
    parameter ADDR_START_L4_F = 48,
    parameter SIZE_L3_OUT_CHANNEL = 8,
    parameter NUM_L4_OUT_CHANNEL = 8

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

    output  reg                                 l4_done
);
    localparam                          ST_IDLE             = 0,
                                        ST_L4_WLOAD         = 1,
                                        ST_L4_FLOAD         = 2,
                                        ST_L4_CAL           = 3,
                                        ST_L4_OUT           = 4,
                                        ST_L4_DONE          = 5;
                                       

    localparam                         NUM_F_SRAM_ADDR         = 16,        // number of features that are saved in one addr in F_SRAM
                                        SIZE_KERNEL_W           =3,                                                                    
                                        SIZE_TIME_DOMAIN        = 1,
                                        SIZE_FREQ_DOMAIN        = 8,
                                        ADDR_START_L4_F_OUT = ADDR_START_L4_F + SIZE_KERNEL_W*SIZE_L3_OUT_CHANNEL/(NUM_F_SRAM_ADDR/SIZE_FREQ_DOMAIN); //48+ 3*8/(16/8) =60

    reg [6:0] cnt;
    reg [5:0] cnt_out;
    reg [2:0] state;
    reg [WIDTH_FSRAM_ADDR-1:0] fsram_addr_in;   
    reg [WIDTH_FSRAM_ADDR-1:0] fsram_addr_out;
    reg [WIDTH_FSRAM_ADDR-1:0] fsram_addr_in_next;
    reg [WIDTH_FSRAM_ADDR-1:0] fsram_addr_out_next;

    always @(posedge clk or negedge rstb) begin
       if (!rstb) begin
            state <= ST_IDLE;
            fsram_addr <= 0;
            wsram_addr <= 0;
            fsram_addr_in <= 0;
            fsram_addr_out <= 0;
            fsram_addr_in_next<=0;
            fsram_addr_out_next<=0;
            cnt <= 0;
            cnt_out <= 0;
            buffer_start <= 0;
            buffer_mode_w <= 0;
            buffer_mode_f <= 0;
            buffer_loc_w <= 0;
            buffer_loc_f <= 0;
            buffer_load_w <= 0;
            buffer_load_f <= 0;
            pe_en <= 0;
            pe_clear <= 0;
            norm_on <= 0;
            relu_on <= 0;
            l4_done <= 0;
            c_w_web <= 0;
            c_w_ceb <= 0;
            c_f_web <= 0;
            c_f_ceb <= 0;
            shift <= 0;
        end // rstb
        else begin
            case (state)
                ST_IDLE: begin
                    cnt <= 0;
                    cnt_out <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    fsram_addr <= ADDR_START_L4_F;
                    wsram_addr <= ADDR_START_L4_W;
                    fsram_addr_out <= ADDR_START_L4_F_OUT;
                    fsram_addr_out_next <= ADDR_START_L4_F_OUT;
                    fsram_addr_in <= ADDR_START_L4_F;
                    fsram_addr_in_next<=ADDR_START_L4_F;
                    pe_en <= 0;
                    pe_clear <= 0;
                    l4_done <= 0;
                    norm_on <= 0;
                    relu_on <= 0;
                    c_w_web <= 1;
                    c_w_ceb <= 0;
                    c_f_web <= 1;
                    c_f_ceb <= 0;
                    
                    if (layer_state == 4 && layer_start) begin
                        state <= ST_L4_WLOAD;
                    end
                end // ST_IDLE

                ST_L4_WLOAD: begin
                    cnt <= cnt + 1;
                    buffer_mode_w <= 4;
                    buffer_load_w <= 1;
                    buffer_loc_w <= 0; // 0
                               
                    if (cnt == 1) begin
                        state <= ST_L4_FLOAD;
                        buffer_load_w <=0;
                        cnt <= 0;
                    end
                end // Wload
                
                ST_L4_FLOAD: begin
                    cnt <= cnt + 1;
                    
                   if (cnt==0) begin                   
                       fsram_addr <= fsram_addr_in;
                   end
                   else if (cnt==1) begin
                       buffer_mode_f <= 4;
                       buffer_load_f <= 1;
                       buffer_loc_f <=0;
                   end
                   else if (cnt==2) begin
                        buffer_load_f<=0;
                        buffer_start <=1;  
                        end
                  else if (cnt==3) begin
                        fsram_addr_in <= fsram_addr_in+1;
                        state <= ST_L4_CAL;
                        pe_en <=1;
                        cnt<=0;
                        end
                end//fload
                
                ST_L4_CAL: begin
                    cnt <= cnt+1;
                    if (cnt == (NUM_F_SRAM_ADDR/SIZE_FREQ_DOMAIN)-1) begin // cnt==1
                        pe_en<=0; 
                        buffer_start <= 0;
                        cnt_out <= cnt_out+1;
                        cnt <= cnt+1;
                    end    
                    else if (cnt == (NUM_F_SRAM_ADDR/SIZE_FREQ_DOMAIN)) begin  //cnt=2   
                        cnt <= 0;
                        if (cnt_out == SIZE_L3_OUT_CHANNEL/(NUM_F_SRAM_ADDR/SIZE_FREQ_DOMAIN)) begin // cnt_out==4
                            state <= ST_L4_OUT;
                            norm_on <= 0;
                            relu_on <= 0;
                            end
                        else begin
                            state <= ST_L4_FLOAD;
                        end
                    end
                end // cal
                
                ST_L4_OUT: begin
                cnt<=cnt+1;
                if (cnt==0) begin
                    fsram_addr <= fsram_addr_out;
                    if (fsram_addr_out_next == (ADDR_START_L4_F_OUT + SIZE_KERNEL_W -1) ) begin //62
                        fsram_addr_out_next <= ADDR_START_L4_F_OUT; //60
                    end else begin
                    fsram_addr_out_next <= fsram_addr_out +1;
                    end
                    c_f_web<=0;
                    if (fsram_addr_in_next == (ADDR_START_L4_F+(SIZE_KERNEL_W-1)*SIZE_L3_OUT_CHANNEL/(NUM_F_SRAM_ADDR/SIZE_FREQ_DOMAIN))) begin // 56 (=48+2*4)
                        fsram_addr_in_next <= ADDR_START_L4_F; //48
                    end else begin
                    fsram_addr_in_next<=fsram_addr_in_next+SIZE_L3_OUT_CHANNEL/(NUM_F_SRAM_ADDR/SIZE_FREQ_DOMAIN); // 48 + 4 = 52 or 52 +4=56
                    end
                end
                else if (cnt==1) begin
                    pe_clear<=1;
                    c_f_web<=1;
                    l4_done<=1;
                    state<=ST_L4_DONE;
                end
                end// out
                
                ST_L4_DONE: begin
                    cnt <= 0;
                    cnt_out <= 0;
                    pe_clear<=0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_f <= 0;
                    buffer_loc_w <= 0;
                    buffer_start <= 0;
                    fsram_addr <= ADDR_START_L4_F;
                    wsram_addr <= ADDR_START_L4_W;
                    fsram_addr_out <= fsram_addr_out_next;
                    fsram_addr_in <= fsram_addr_in_next;
                    l4_done <= 0;
                    if(layer_state == 4 && layer_start) begin
                        state <= ST_L4_WLOAD;
                        end
                end//done
                
            endcase              
            end // else
        end //always
    
endmodule
