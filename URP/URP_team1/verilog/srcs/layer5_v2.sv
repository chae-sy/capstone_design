`timescale 1ns/1ps

module layer5#(
    //parameters
    parameter INPUT_HORIZ = 1,
    parameter INPUT_VERT = 9,
    parameter WEIGHT_HORIZ = 7,
    parameter WEIGHT_VERT = 9,
    parameter STRIDE_HORIZ = 1,
    parameter STRIDE_VERT = 1,
    parameter INPUT_BIT_LEN = 8, // Yet to be used
    parameter WEIGHT_BIT_LEN = 32,
    parameter WEIGHT_NUM = WEIGHT_HORIZ*WEIGHT_VERT, 
    parameter MB_BIT_LEN = $clog2(WEIGHT_VERT * INPUT_HORIZ),
    parameter OUTPUT_BUF_LEN = 32,
    parameter L5_WM_BIT_LEN = $clog2(256),
    parameter L3_WEIGHT_CHANNEL = 32,
    parameter L2_WEIGHT_HORIZ = 3,
    parameter L2_WEIGHT_VERT = 3,
    parameter L1_WEIGHT_HORIZ = 4,
    parameter L1_WEIGHT_VERT = 10
)(
    input                                        rstb,      
    input                                        layer_en,                                
    input                                        clk,

    output reg  [MB_BIT_LEN - 1:0]               mb_r_addr,
    output reg  [L5_WM_BIT_LEN - 1:0]            l5_wm_r_addr,
    output reg                                   mb_ceb,
    output reg                                   mb_web,
    output reg                                   wm_ceb,
    output reg                                   wm_web, 
    //  output reg                                   mb_ren,
    //  output reg                                   wm_ren,
    output reg                                   pe_en,
    output reg                                   pe_rst,
    output reg                                   PA_sel, //PE or Adder tree Select 
    
    output reg  [OUTPUT_BUF_LEN-1 :0]            output_buf_en,
    output reg                                   output_buf_rst,


    //output reg  [WEIGHT_HORIZ - 1:0]             result,
    output reg                                   comparator_init,
    output reg                                   done

);  
    //Bit Length
    localparam W_CNT_BIT_LEN = $clog2(INPUT_HORIZ*INPUT_VERT);
    localparam CAL_CNT_BIT_LEN = $clog2(WEIGHT_HORIZ*WEIGHT_VERT);


    localparam WEIGHT_VERT_BIT_LEN = WEIGHT_VERT;
    localparam WEIGHT_HORIZ_BIT_LEN = WEIGHT_HORIZ;



    //State Names
    localparam IDLE = 0;
    localparam MB_READ = 1;
    localparam WAIT = 2;
    localparam PE_RESET = 3;
    localparam LAYER_5 = 4;

    //Parameter Setup               
    reg [3:0]                            state, state_n;
    // reg                                  init;
    // reg                                  init_n;
    // reg                                  l3_init;
    // reg                                  l3_init_n;

    reg wait_cnt, wait_cnt_n;
    reg [3:0] mb_cnt, mb_cnt_n;
    reg [CAL_CNT_BIT_LEN - 1: 0]         cal_cnt, cal_cnt_n;

    reg [MB_BIT_LEN - 1:0]               mb_r_addr_n;
    reg [L5_WM_BIT_LEN - 1:0]            l5_wm_r_addr_n;
    reg                                  l5_flag; //0 for odd, 1 for even
    // reg                                  mb_ren_n;
    // reg                                  wm_ren_n;
    reg                                  pe_rst_n;
    reg                                  PA_sel_n;

    reg                                  mb_ceb_n;
    reg                                  mb_web_n;
    reg                                  wm_ceb_n;
    reg                                  wm_web_n;

    reg  [31:0]                          output_buf_en_n;
    reg                                  output_buf_rst_n;
    reg  [WEIGHT_VERT-1:0]               l5_line_cnt,l5_line_cnt_n;

    reg                                  pe_en_n;
    reg                                  l5_flag_n;
    reg                                  comparator_init_n;
    reg                                  done_n;


    always_comb begin : STATE
        state_n = state;
        mb_cnt_n = mb_cnt;
        cal_cnt_n = cal_cnt;
        wait_cnt_n = wait_cnt;
        mb_r_addr_n = mb_r_addr;
        l5_wm_r_addr_n = l5_wm_r_addr-81;

        done_n = done;
        pe_en_n = pe_en;

        //mb_ren_n = mb_ren;
        mb_ceb_n = mb_ceb;
        mb_web_n = mb_web;

        // wm_ren_n = wb_ren;
        wm_ceb_n = wm_ceb;
        wm_web_n = wm_web;

        PA_sel_n   = 1;
        pe_rst_n = pe_rst;
        l5_flag_n = l5_flag;
        output_buf_en_n = output_buf_en;
        comparator_init_n = comparator_init;
        
        output_buf_rst_n = 0;
        l5_line_cnt_n = l5_line_cnt;
                    
        case (state)
            IDLE: begin
                // mb_ren_n = 0;
                mb_ceb_n = 1;
                mb_web_n = 1;
                // wm_ren_n = 0;
                wm_ceb_n = 1;
                wm_web_n = 1;
                
                pe_rst_n = 0;  
                output_buf_en_n = 0;

                
                done_n  = 0;
                l5_line_cnt_n = 0;
                if (layer_en) begin
                    state_n = MB_READ;
                    //pe_en_n = 1; 
                    output_buf_rst_n = 1;
                    mb_r_addr_n = (l5_flag) ? 11+mb_cnt : 2+mb_cnt;
                    // mb_ren_n = 1;
                    mb_ceb_n = 0;
                    mb_web_n = 1;
                    // wm_ren_n = 1;
                    wm_ceb_n = 0;
                    wm_web_n = 1;

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
                pe_en_n = 1;
                output_buf_rst_n = 0;
                
                //pe_en_n = 0;
                cal_cnt_n = (cal_cnt + 1)%INPUT_VERT;

                if(cal_cnt == INPUT_VERT - 1) begin
                    state_n = WAIT;
                    mb_ceb_n = 0;
                    mb_web_n = 1;
         
                    wm_ceb_n = 0;
                    wm_web_n = 1;
                end
                else begin
                    state_n = MB_READ;
                    mb_ceb_n = 1;
                    mb_web_n = 1;

                    wm_ceb_n = 1;
                    wm_web_n = 1;
                end

                if (l5_flag) begin // For odd
                    if( mb_r_addr != 19) begin
                        l5_wm_r_addr_n = l5_wm_r_addr-81 + 1 ; 
                        mb_r_addr_n = mb_r_addr + 1;
                        // mb_ren_n = 1;
                        //mb_ceb_n = 1;
                        //mb_web_n = 1;
                       // wm_ren_n = 1;
                        //wm_ceb_n = 1;
                        //wm_web_n = 1;
                    end
                    else begin
                        l5_wm_r_addr_n = l5_wm_r_addr-81 + 1 ;
                        mb_r_addr_n = 11; 
                        // mb_ren_n = 0;
                        //mb_ceb_n = 1;
                        //mb_web_n = 1;
                        // wm_ren_n = 0;
                        //wm_ceb_n = 1;
                        //wm_web_n = 1;
                    end       
                end
                else begin  //For even
                    if( mb_r_addr != 10) begin
                        l5_wm_r_addr_n = l5_wm_r_addr-81 + 1 ; 
                        mb_r_addr_n = mb_r_addr + 1;
                        // mb_ren_n = 1;
                        //mb_ceb_n = 1;
                        //mb_web_n = 1;
                       // wm_ren_n = 1;
                        //wm_ceb_n = 1;
                        //wm_web_n = 1;

                        //state_n = WAIT;
                    end
                    else begin 
                        l5_wm_r_addr_n = l5_wm_r_addr-81 + 1;
                        mb_r_addr_n = 2; 
                        // mb_ren_n = 0;
                        //mb_ceb_n = 1;
                        //mb_web_n = 1;
                        // wm_ren_n = 0;
                        //wm_ceb_n = 1;
                        //wm_web_n = 1;

                        //state_n = WAIT;
                    end

                end  
                end
            
    

            
            WAIT: begin
                pe_en_n = 0;
                state_n = PE_RESET;
                l5_line_cnt_n = (l5_line_cnt+1);
                output_buf_en_n[l5_line_cnt] = 1;
                
            end

            PE_RESET: begin
                //pe_rst_n = 0;
                output_buf_en_n = 32'b0;
                if(wait_cnt == 0) begin
                    wait_cnt_n = 1;
                    pe_rst_n = 1;
                    output_buf_en_n = 0;
                    state_n = PE_RESET;
                    
                end
                else if(l5_line_cnt != 7)begin
                    state_n = MB_READ;
                    wait_cnt_n = 0;
                    pe_rst_n = 0;
                    //pe_en_n = 1;

                    mb_ceb_n = 0;
                    mb_web_n = 1; 
                    
                    wm_ceb_n = 0;
                    wm_web_n = 1;
                end

                else begin
                    state_n  = LAYER_5;
                    wait_cnt_n = 0;
                    pe_rst_n = 0;
                    done_n = 1;
                    comparator_init_n = 1;
                    cal_cnt_n = 0;
                    l5_wm_r_addr_n = 0;
                    mb_cnt_n = (!l5_flag) ? ((mb_cnt + 1)%9) : mb_cnt;  
                    l5_flag_n = (l5_flag + 1) % 2;
                end
            end

                        
            LAYER_5: begin

                    comparator_init_n    = 0;
                    //done_n               = 1;
                    //output_buf_rst_n     = 1;
                    state_n              = IDLE;
                    l5_line_cnt_n = 0;
                end

            
    endcase
    end
    always_ff @(posedge clk or negedge rstb) begin: block
        if(!rstb) begin
            state <= IDLE;

            pe_en            <= 0;
            output_buf_en    <= 32'b0;
            output_buf_rst   <= 0;
            mb_cnt <= 0;
            cal_cnt            <= 0;
            wait_cnt <= 0;
            // mb_ren           <= 0;
            mb_ceb           <= 1;
            mb_web           <= 1;
            
            // wm_ren           <= 0;
            wm_ceb           <= 1;
            wm_web           <= 1;
            
            mb_r_addr        <= 2;
            l5_flag          <= 0;
            l5_line_cnt      <= 0;
            l5_wm_r_addr     <= 81;
            PA_sel           <= 0;

            pe_rst           <= 0;
            comparator_init  <= 0;
            done             <= 0;
    

        end
        
        else begin
            state <= state_n;

            output_buf_en <= output_buf_en_n;
            output_buf_rst <= output_buf_rst_n;

            pe_en <= pe_en_n;
            mb_cnt <= mb_cnt_n;
            wait_cnt <= wait_cnt_n;
            cal_cnt <= cal_cnt_n;

            mb_r_addr <= mb_r_addr_n;
            l5_wm_r_addr <= l5_wm_r_addr_n+81;
            l5_flag      <= l5_flag_n;
            l5_line_cnt  <= l5_line_cnt_n;

            mb_ceb <= mb_ceb_n;
            mb_web <= mb_web_n;
            wm_ceb <= wm_ceb_n;
            wm_web <= wm_web_n;

            PA_sel <= PA_sel_n;
            pe_rst <= pe_rst_n;


            comparator_init <=comparator_init_n;
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
