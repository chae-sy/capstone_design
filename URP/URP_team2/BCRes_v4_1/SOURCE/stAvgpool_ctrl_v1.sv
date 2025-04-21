`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: sjkim
// 
// Create Date: 2024/11/26 21:22:41
// Design Name: 
// Module Name: stAvgpool_ctrl_v1
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


module stAvgpool_ctrl_v1
#(
    parameter WIDTH_FSRAM_WL = 128,
    parameter WIDTH_F_DATA = 8,
    parameter WIDTH_FSRAM_ADDR = 10,
    parameter WIDTH_WSRAM_ADDR = 10,
    parameter ADDR_START_LAVGPOOL = 121,
    parameter NUM_POOL = 22
)
(
    input clk,
    input rstb,
    input [4:0] layer_state,
    input layer_start,
    
    output reg layer_done,
    output reg clf_mode,
    output reg en_avgpool,

    
    output reg c_f_ceb,
    output reg c_f_web,
    
    output reg clear,
    output reg [WIDTH_FSRAM_ADDR-1:0] ADDR, 
    
 
    output  reg     [WIDTH_WSRAM_ADDR-1:0]      wsram_addr,
    output  reg                                 c_w_ceb,
    output  reg                                 c_w_web,

    
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
    output  reg                                 relu_on 
);

    localparam  ST_IDLE = 0,
                 ST_AVGPOOL_SUM1 = 1,
                 ST_AVGPOOL_SUM2 = 2,     
                 ST_AVGPOOL_SUB = 3,
                 ST_AVGPOOL_DONE = 4;
                
    localparam  AVGPOOL_LAYER_NUM = 23;
                
    reg [4:0] cnt;
    reg [1:0] cnt_wait;
    reg [2:0] state_avg;
    
    reg [WIDTH_FSRAM_ADDR-1:0] fsram_addr1; //for sum
    reg [WIDTH_FSRAM_ADDR-1:0] fsram_addr2; //for sub
                        
                
    always @(posedge clk or negedge rstb) begin
        if (!rstb) begin
            cnt <= 0;
            layer_done <= 0;
            en_avgpool <= 0;
            clear <= 0;
            fsram_addr1 <= 0;
            fsram_addr2 <= 0;            
            clf_mode <= 0;
            layer_done <= 0;
            //ADDR <= 0;
            c_f_ceb <= 0;
            c_f_web <= 0;                
            clear <= 0;
            wsram_addr <= 0;
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
            norm_on <= 0;
            relu_on <= 0;
            state_avg <= ST_IDLE;
            cnt_wait <= 0;               
        end
        else begin
            case (state_avg)
                ST_IDLE: begin //0
                    cnt <= 0;
                    layer_done <= 0;
                    fsram_addr1 <= ADDR_START_LAVGPOOL;
                    fsram_addr2 <= ADDR_START_LAVGPOOL + 1; // can't read mem[0] 
                    en_avgpool <= 0;
                    clf_mode <= 0;
                    cnt_wait <= 0;
                    clear <= 0;
                    wsram_addr <= 0;
                    buffer_start <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_w <= 0;
                    buffer_loc_f <= 0;
                    buffer_load_w <= 0;
                    buffer_load_f <= 0;
                    pe_clear <= 0;
                    shift <= 0;
                    c_w_web <= 1;
                    c_w_ceb <= 0;
                    norm_on <= 0;
                    relu_on <= 0;
                    c_f_web <= 1;
                    c_f_ceb <= 0; 
                    shift <= 0;                                       
                    if (layer_state == AVGPOOL_LAYER_NUM && layer_start) begin
                        state_avg <= ST_AVGPOOL_SUM1;
                        en_avgpool <= 1;                      
                    end
                end            

                ST_AVGPOOL_SUM1 : begin //1
                    cnt <= cnt + 1; 
                    fsram_addr1 <= fsram_addr1 + 1;
                    if ( cnt == NUM_POOL - 1) begin // 21
//                        state_avg <= ST_AVGPOOL_DONE;
//                        layer_done <= 1;
                        en_avgpool <= 0;
//                        cnt <= 0;
//                        fsram_addr1 <= ADDR_START_LAVGPOOL + NUM_POOL;
//                        fsram_addr2 <= ADDR_START_LAVGPOOL;
                    end
                    else if (cnt == NUM_POOL) begin
                        state_avg <= ST_AVGPOOL_DONE;
                        layer_done <= 1;
//                        en_avgpool <= 0;
                        cnt <= 0;
                        fsram_addr1 <= ADDR_START_LAVGPOOL + NUM_POOL;
                        fsram_addr2 <= ADDR_START_LAVGPOOL;
                    end                    
                end
                
                ST_AVGPOOL_SUM2 : begin //2
//                    pe_en <= 0;
                    clf_mode <= 1;
//                    en_avgpool <= 1;
                    state_avg <= ST_AVGPOOL_SUB;
//                    layer_done <= 0;
                end
                
                ST_AVGPOOL_SUB : begin //3
                    cnt <= cnt + 1;   
                    if ( cnt == 0 ) begin
                        if (fsram_addr1 == ADDR_START_LAVGPOOL + NUM_POOL)
                            fsram_addr1 <= ADDR_START_LAVGPOOL;
                        else fsram_addr1 <= fsram_addr1 + 1;
                        if (fsram_addr2 == ADDR_START_LAVGPOOL + NUM_POOL) 
                            fsram_addr2 <= ADDR_START_LAVGPOOL;
                        else fsram_addr2 <= fsram_addr2 + 1;
                                  
                        clf_mode <= 0;
                        en_avgpool <= 0;
                    end
                    else if ( cnt == 1 ) begin
                        state_avg <= ST_AVGPOOL_DONE;
                        layer_done <= 1;
                    end     
                end
                
                ST_AVGPOOL_DONE : begin
                    cnt <= 0;
                    layer_done <= 0;
                    en_avgpool <= 0;
                    clf_mode <= 0;
                    clear <= 0;
                    wsram_addr <= 0;
                    buffer_start <= 0;
                    buffer_mode_w <= 0;
                    buffer_mode_f <= 0;
                    buffer_loc_w <= 0;
                    buffer_loc_f <= 0;
                    buffer_load_w <= 0;
                    buffer_load_f <= 0;
                    pe_clear <= 0;
                    shift <= 0;
                    c_w_web <= 1;
                    c_w_ceb <= 0;
                    norm_on <= 0;
                    relu_on <= 0;
                    c_f_web <= 1;
                    c_f_ceb <= 0; 
                    shift <= 0;  
                    if ( layer_state == AVGPOOL_LAYER_NUM && layer_start ) begin
                        state_avg <= ST_AVGPOOL_SUM2;
                        clf_mode <= 0;
                        en_avgpool <= 1;
                    end
 
                end

            endcase
        end
    end                 

    always @(*) begin
        ADDR = (clf_mode)? fsram_addr2: fsram_addr1;
    end
endmodule
