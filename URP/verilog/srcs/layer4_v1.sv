`timescale 1ns/1ps



module layer4 #(
    //parameter,
    parameter INPUT_HORIZ = 2,
    parameter INPUT_VERT = 2,
    parameter WEIGHT_HORIZ = 2,
    parameter WEIGHT_VERT = 2,
    parameter STRIDE_HORIZ = 2,
    parameter STRIDE_VERT = 2,
    parameter INPUT_BIT_LEN = 8,
    parameter WEIGHT_BIT_LEN = 32,
    parameter NEXT_WEIGHT_VERT = 1,
    parameter NEXET_INPUT_HORIZ = (INPUT_HORIZ - WEIGHT_HORIZ)/STRIDE_HORIZ + 1,
    parameter MA2_BIT_LEN = $clog2(16),
    parameter REDUCED_HORIZ = 1,
    parameter REDUCED_VERT = 9,
    parameter MB2_BIT_LEN = $clog2(20)
) (  
    //signal
    input                           rstb,       
    input                           clk,
    input                           layer_en, // remove in all state
    //input                           init_i, // from main control, decide 1 or 5
    output  reg [MA2_BIT_LEN-1:0]   ma_r_addr,
    output  reg [MB2_BIT_LEN-1:0]   mb_wr_addr, 
    output  reg                     ma_cen, 
    output  reg                     ma_wen,
    output  reg                     mb_cen,
    output  reg                     mb_wen,  
    output  reg                     pe_rst,
    output  reg                     pe_en,      
    output  reg [31:0]              output_buf_en,
    output  reg                     output_buf_rst,
    output  reg                     layer_done
    //output  reg                     init // go to main control, decide 1 , 5 
);
    //Bit Length
    localparam W_CNT_BIT_LEN = $clog2(INPUT_HORIZ*INPUT_VERT);
    localparam R_CNT_BIT_LEN = $clog2(WEIGHT_VERT * WEIGHT_HORIZ * NEXET_INPUT_HORIZ);
    localparam MA_W_CNT_BIT_LEN = $clog2(NEXET_INPUT_HORIZ * NEXT_WEIGHT_VERT);

    localparam WEIGHT_VERT_BIT_LEN = $clog2(REDUCED_VERT*2);
    localparam WEIGHT_HORIZ_BIT_LEN = $clog2(WEIGHT_HORIZ);
    localparam NEXET_INPUT_HORIZ_BIT_LEN = $clog2(NEXET_INPUT_HORIZ);

    //State Names
    localparam IDLE = 0;
    localparam MA_READ = 1;
    localparam MB_WRITE = 3;
    localparam WAIT = 2;
    localparam LAYER_DONE = 4;
    

    //Parameter Setup
    reg [2:0] state, state_n;   


    reg [WEIGHT_VERT_BIT_LEN -1:0] l4_flag;
    reg [WEIGHT_VERT_BIT_LEN -1:0] l4_flag_n;
    //reg init_n;

    reg wait_cnt, wait_cnt_n;
    reg [R_CNT_BIT_LEN -1:0] r_cnt, r_cnt_n;
    reg [MA_W_CNT_BIT_LEN -1:0] mb_w_cnt, mb_w_cnt_n;
    reg [WEIGHT_VERT_BIT_LEN -1:0] w_v_cnt, w_v_cnt_n;
    reg [WEIGHT_HORIZ_BIT_LEN -1:0] w_h_cnt, w_h_cnt_n;


    reg [MA2_BIT_LEN-1:0] ma_r_addr_n;
    reg [MB2_BIT_LEN-1:0] mb_wr_addr_n;
    reg ma_cen_n;
    reg ma_wen_n;
    reg mb_cen_n;
    reg mb_wen_n;  
    
    reg pe_rst_n;

    reg pe_en_n;
    reg [31:0] output_buf_en_n;
    reg output_buf_rst_n;
    reg layer_done_n;

    always_comb begin : STATE
            state_n = state;

            l4_flag_n = l4_flag;
            //init_n = init;

            wait_cnt_n = wait_cnt;                        
            r_cnt_n = r_cnt;
            mb_w_cnt_n = mb_w_cnt;
            w_h_cnt_n = w_h_cnt;
            w_v_cnt_n = w_v_cnt;

            
            ma_r_addr_n = ma_r_addr-12;
            mb_wr_addr_n = mb_wr_addr-2;
            ma_cen_n = ma_cen;
            ma_wen_n = ma_wen;
            mb_cen_n = mb_cen;
            mb_wen_n = mb_wen;

            pe_rst_n = pe_rst;

            pe_en_n = pe_en;
            output_buf_en_n = output_buf_en;
            output_buf_rst_n = output_buf_rst;

            layer_done_n = layer_done;
            case (state)
                IDLE : begin
                        layer_done_n = 0;
                        if (layer_en) begin
                            state_n = MA_READ;
                            ma_cen_n = 0;
                            ma_wen_n = 1;
                            output_buf_rst_n = 1;
                            //pe_en_n = 1;

                        if (l4_flag[0] == 0) begin
                            mb_wr_addr_n = (l4_flag>>1) + REDUCED_HORIZ*REDUCED_VERT;
                        end else begin
                            mb_wr_addr_n = (l4_flag>>1);
                        end
                        
                        
                        //if (init_i)
                        //    init_n = 1;


                        end                        
                end
                MA_READ : begin
                        pe_en_n = 1;
                        output_buf_rst_n = 0;
                        r_cnt_n = r_cnt +1;
                        ma_r_addr_n = ma_r_addr+1-12;
                        if (r_cnt == (WEIGHT_HORIZ * WEIGHT_VERT - 1)) begin
                            state_n = WAIT;  
                            r_cnt_n = 0;

                            //pe_en_n = 0;

                            ma_cen_n = 1;
                            ma_wen_n = 0;      

                            ma_r_addr_n =0;

                            //output_buf_en_n ='1; 
                        end
                    end

                WAIT : begin
                    pe_en_n = 0;
                    //output_buf_en_n='0;
                    if(wait_cnt == 1) begin
                        output_buf_en_n= '0;
                        state_n = MB_WRITE;
                        mb_wen_n = 0;
                        mb_cen_n = 0;
                        pe_rst_n = 1;
                        wait_cnt_n = 0;
                    end
                    else begin
                        state_n = WAIT;
                        wait_cnt_n = wait_cnt + 1;
                        output_buf_en_n=32'hFFFFFFFF;
                    end
                    //state_n = MB_WRITE;
                    //mb_cen_n = 0;
                    // mb_wen_n = 0;

                    //pe_rst_n = 1;

                end

                MB_WRITE : begin
                    pe_rst_n = 0;
                    output_buf_rst_n = 1;


                    l4_flag_n = (l4_flag+1)%(2*REDUCED_VERT);

                    //if (l4_flag == 2*REDUCED_VERT - 1)
                    //    init_n = 0; 

                    mb_cen_n = 1;
                    mb_wen_n = 1;

                    state_n =LAYER_DONE;
                    
                    layer_done_n = 1;
                end

                LAYER_DONE : begin
                    state_n = IDLE;
                    output_buf_rst_n = 0;
                    //layer_done_n =0;

                end
        endcase

    end


    always_ff @(posedge clk or negedge rstb) begin : block
        if (!rstb) begin
            state <= IDLE;

            //init <= 0;
            pe_en           <= 0;
            output_buf_en   <= '0;

            wait_cnt <= 0;
            r_cnt <=0;

            mb_w_cnt<=0;

            l4_flag <=0;

            ma_cen          <= 1;
            ma_wen          <= 1;
            mb_cen          <= 1;
            mb_wen          <= 1;
            mb_wr_addr      <= 2;
            ma_r_addr <= 12;

            pe_rst <= 0;
            output_buf_rst <= 0;

            layer_done <= 0;
        end 
        else begin
            state<=state_n;

            //init<= init_n;
            output_buf_en<= output_buf_en_n;
            pe_en<=pe_en_n;

            wait_cnt <= wait_cnt_n;
            r_cnt <= r_cnt_n;
            mb_w_cnt <= mb_w_cnt_n;
            l4_flag<=l4_flag_n;

            ma_r_addr<=ma_r_addr_n+12;
            mb_wr_addr<= mb_wr_addr_n+2;
            ma_cen <= ma_cen_n;
            ma_wen <= ma_wen_n;
            mb_cen<= mb_cen_n;
            mb_wen<= mb_wen_n;


            pe_rst <= pe_rst_n;
            output_buf_rst <= output_buf_rst_n;


            layer_done <= layer_done_n;


        end
    end


    /*
    //check clock
    reg [15:0] cnt;

    always@(negedge clk or rstb) begin
        if(!rstb) begin
            cnt<=0;
        end else begin
            cnt <= cnt +1;
        end
    end
    */


endmodule
