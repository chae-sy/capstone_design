`timescale 1ns/1ps

module layer3#(
    //parameters
    parameter INPUT_HORIZ = 2,
    parameter INPUT_VERT = 1,
    parameter WEIGHT_HORIZ = 1,
    parameter WEIGHT_VERT = 1,
    parameter STRIDE_HORIZ = 1,
    parameter STRIDE_VERT = 1,
    parameter INPUT_BIT_LEN = 8, // Yet to be used
    parameter WEIGHT_BIT_LEN = 32,
    parameter WEIGHT_NUM = 32, 
    parameter NEXT_WEIGHT_VERT = 2, 
    parameter NEXT_INPUT_HORIZ = (INPUT_HORIZ - WEIGHT_HORIZ)/STRIDE_HORIZ +1, 
    parameter MB_BIT_LEN = $clog2(20),
    parameter MA_BIT_LEN = $clog2(16),
    parameter L3_WM_BIT_LEN = $clog2(144)
)(
    input                                        rstb,      
    input                                        layer_en,                                
    input                                        clk,

    output reg  [MA_BIT_LEN - 1:0]               ma_wr_addr, 
    output reg  [MB_BIT_LEN - 1:0]               mb_r_addr,
    output reg  [L3_WM_BIT_LEN - 1:0]            l3_wm_r_addr,
    output reg                                   ma_ceb,
    output reg                                   ma_web,
    output reg                                   mb_ceb,
    output reg                                   mb_web,
    output reg                                   wm_ceb,
    output reg                                   wm_web, 
    output reg  [31:0]                           output_buf_en,
    output reg                                   pe_en,
    output reg                                   pe_rst,
    output reg                                   PA_sel, //PE or Adder tree Select
    output reg                                   output_buf_rst,
    output reg                                   done

);  
    //Bit Length
    localparam W_CNT_BIT_LEN = $clog2(INPUT_HORIZ*INPUT_VERT);
    localparam CAL_CNT_BIT_LEN = $clog2(WEIGHT_NUM);
    localparam MA_W_CNT_BIT_LEN = $clog2(NEXT_INPUT_HORIZ*NEXT_WEIGHT_VERT);

    localparam WEIGHT_VERT_BIT_LEN = WEIGHT_VERT;
    localparam WEIGHT_HORIZ_BIT_LEN = WEIGHT_HORIZ;
    localparam NEXT_INPUT_HORIZ_BIT_LEN = $clog2(NEXT_INPUT_HORIZ);


    //State Names
    localparam IDLE = 0;
    localparam MB_READ = 1;
    localparam WAIT = 2;
    localparam PE_RESET = 3;
    localparam MA_WRITE = 4;
    localparam LAYER_3 = 5;

    //Parameter Setup               
    reg [3:0]                            state, state_n;
    //reg                                  init;
    //reg                                  init_n;
    //reg                                  l3_init;
    //reg                                  l3_init_n;

    reg wait_cnt, wait_cnt_n;

    reg [CAL_CNT_BIT_LEN - 1: 0]           cal_cnt, cal_cnt_n;
    reg [MA_W_CNT_BIT_LEN - 1:0]         ma_w_cnt, ma_w_cnt_n;

    reg [MB_BIT_LEN - 1:0]               mb_r_addr_n;
    reg [MA_BIT_LEN - 1:0]               ma_wr_addr_n;
    reg [L3_WM_BIT_LEN - 1:0]            l3_wm_r_addr_n;
    // reg                                  mb_ren_n;
    // reg                                  ma_wen_n;
    // reg                                  wm_ren_n;
    reg                                  pe_rst_n;
    reg                                  PA_sel_n;

    reg                                   ma_ceb_n;
    reg                                   ma_web_n;
    reg                                   mb_ceb_n;
    reg                                   mb_web_n;
    reg                                   wm_ceb_n;
    reg                                   wm_web_n;

    reg [31:0]                           output_buf_en_n;
    reg                                  output_buf_rst_n;
    reg                                  pe_en_n;

    reg                                  done_n;


    always_comb begin : STATE
        state_n = state;

        //init_n  = init;
        //l3_init_n = l3_init;
        output_buf_en_n = output_buf_en;
        output_buf_rst_n = output_buf_rst;
        pe_en_n = pe_en;
        
        wait_cnt_n = wait_cnt;
        cal_cnt_n = cal_cnt;
        ma_w_cnt_n = ma_w_cnt;
        l3_wm_r_addr_n = l3_wm_r_addr-49;
        mb_r_addr_n = mb_r_addr;
        ma_wr_addr_n = ma_wr_addr-12;
            
        ma_ceb_n = ma_ceb;
        ma_web_n = ma_web;
        mb_ceb_n = mb_ceb;
        mb_web_n = mb_web;
        wm_ceb_n = wm_ceb;
        wm_web_n = wm_web;

        PA_sel_n = PA_sel;
        pe_rst_n = pe_rst;
        done_n   = done;
        case (state)
            IDLE: begin
                // mb_ren_n = 0;
                mb_ceb_n = 1;
                mb_web_n = 1;
                // wm_ren_n = 0;
                wm_ceb_n = 1;
                wm_web_n = 1;

                ma_w_cnt_n = 0;

                pe_rst_n = 0;  
                done_n = 0;
                if (layer_en) begin
                    state_n = MB_READ;
                    cal_cnt_n = 0;
                    //pe_en_n = 1; 
                    // mb_ren_n = 1;
                    mb_ceb_n = 0;
                    mb_web_n = 1;
                    // wm_ren_n = 1;
                    wm_ceb_n = 0;
                    wm_web_n = 1;
                    output_buf_rst_n = 1;

                end
                else begin
                    state_n = IDLE;
                    pe_en_n = 0; 
                    // mb_ren_n = 0;
                    mb_ceb_n = 1;
                    mb_web_n = 1;
                    // wm_ren_n = 0;
                    wm_ceb_n = 1;
                    wm_web_n = 1;
                end
            end

            MB_READ: begin
                state_n = WAIT;
                pe_en_n = 1;
                output_buf_rst_n = 0;
                //cal_cnt_n = cal_cnt + 1;  // weight memory read count
                
                //pe_en_n = 0;
                    if( cal_cnt < WEIGHT_NUM - 1) begin
                        l3_wm_r_addr_n = l3_wm_r_addr-49 + 1;
                        mb_r_addr_n = mb_r_addr;
                        // mb_ren_n = 1;
                        mb_ceb_n = 1;
                        mb_web_n = 1;
                       // wm_ren_n = 1;
                        wm_ceb_n = 1;
                        wm_web_n = 1;

                    end
                    else  begin
                        l3_wm_r_addr_n = 0;
                        mb_r_addr_n = (mb_r_addr + 1)%2; 
                        // mb_ren_n = 0;
                        mb_ceb_n = 1;
                        mb_web_n = 1;
                        // wm_ren_n = 0;
                        wm_ceb_n = 1;
                        wm_web_n = 1;
                    end                
                end
            
    

            
            WAIT: begin
                pe_en_n = 0;
                if(wait_cnt == 1) begin
                    state_n = PE_RESET;
                    output_buf_en_n = 0;
                    pe_rst_n = 1;
                    wait_cnt_n = 0;
                end
                else begin
                    state_n = WAIT;
                    output_buf_en_n[cal_cnt] = 1;
                    wait_cnt_n = 1; 
                end
                //output_buf_en_n = 0;
                
                //pe_rst_n = 1;
            end

            PE_RESET: begin
                pe_rst_n = 0;
                if(cal_cnt != WEIGHT_NUM-1) begin
                    state_n = MB_READ;
                    //output_buf_rst_n = 1;
                    cal_cnt_n = cal_cnt+1;
                    //pe_en_n = 1;
                    
                    mb_ceb_n = 0;
                    mb_web_n = 1;
                
                    wm_ceb_n = 0;
                    wm_web_n = 1;
                end
                else begin
                    state_n = MA_WRITE;
                    ma_ceb_n = 0;
                    ma_web_n = 0;
                    cal_cnt_n = 0;
                end
            
            end

            MA_WRITE: begin
                ma_ceb_n = 1;
                ma_web_n = 1;
            
                ma_w_cnt_n = ma_w_cnt + 1;
                ma_wr_addr_n = (ma_wr_addr-12+1)%4;
                if(ma_w_cnt == NEXT_INPUT_HORIZ - 1) begin
                    done_n = 1;
                    state_n = LAYER_3;
                    ma_w_cnt_n = 0;
                    // ma_wen_n = 0;    
                end
                else begin
                    state_n = MB_READ;
                    //pe_en_n = 1;
                    output_buf_rst_n = 1;
                    mb_ceb_n = 0;
                    wm_ceb_n = 0;
                end
            end

            LAYER_3: begin
                //done_n    = 0;
                state_n   = IDLE;
            end
    endcase
    end
    always_ff @(posedge clk or negedge rstb) begin: block
        if(!rstb) begin
            state <= IDLE;

            //init             <= 0;
            pe_en            <= 0;
            output_buf_en    <= 0;
            wait_cnt <= 0;
            cal_cnt          <= 0;

            ma_w_cnt         <= 0;
            
            // ma_wen           <= 0;
            ma_ceb           <= 1;
            ma_web           <= 1;

            // mb_ren           <= 0;
            mb_ceb           <= 1;
            mb_web           <= 1;
            
            // wm_ren           <= 0;
            wm_ceb           <= 1;
            wm_web           <= 1;
            
            mb_r_addr        <= 0;
            ma_wr_addr       <= 12;
            l3_wm_r_addr     <= 49;
            PA_sel           <= 0;
            output_buf_rst   <= 0;

            pe_rst           <= 0;
            done             <= 0;

        end
        
        else begin
            state <= state_n;

            //init  <= init_n;
            //l3_init <= l3_init_n;
            output_buf_en <= output_buf_en_n;
            output_buf_rst <= output_buf_rst_n;
            pe_en <= pe_en_n;

            wait_cnt <= wait_cnt_n;
            cal_cnt <= cal_cnt_n;
            ma_w_cnt <= ma_w_cnt_n;

            mb_r_addr <= mb_r_addr_n;
            ma_wr_addr <= ma_wr_addr_n+12;
            l3_wm_r_addr <= l3_wm_r_addr_n+49;
            
            ma_ceb <= ma_ceb_n;
            ma_web <= ma_web_n;
            mb_ceb <= mb_ceb_n;
            mb_web <= mb_web_n;
            wm_ceb <= wm_ceb_n;
            wm_web <= wm_web_n;

            PA_sel <= PA_sel_n;
            pe_rst <= pe_rst_n;
            done   <= done_n;
        end
    end

    /*
    //check clock
    reg [15:0]  cnt;
    
    always@(negedge clk or rstb) begin
        if(!rstb) begin
            cnt <= 0;
        end
        else begin 
            cnt <= cnt+1;
        end
    end
    */

endmodule 
