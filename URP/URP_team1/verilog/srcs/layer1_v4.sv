`timescale 1ns/1ps



module layer1 #(
    //parameter,
    parameter INPUT_HORIZ = 10,
    parameter INPUT_VERT = 29,
    parameter WEIGHT_HORIZ = 4,
    parameter WEIGHT_VERT = 10,
    parameter STRIDE_HORIZ = 2,
    parameter STRIDE_VERT = 1,
    parameter INPUT_BIT_LEN = 8,
    parameter WEIGHT_BIT_LEN = 32,
    parameter NEXT_WEIGHT_VERT = 3,
    parameter NEXET_INPUT_HORIZ = (INPUT_HORIZ - WEIGHT_HORIZ)/STRIDE_HORIZ + 1,
    parameter IM_BIT_LEN = $clog2((WEIGHT_VERT)*(INPUT_HORIZ)),
    parameter MA1_BIT_LEN = $clog2(NEXT_WEIGHT_VERT*NEXT_WEIGHT_VERT),
    parameter L1_WM_BIT_LEN = $clog2(WEIGHT_HORIZ * WEIGHT_VERT),
    parameter WEIGHT_VERT_BIT_LEN = $clog2(WEIGHT_VERT),
    parameter INPUT_VERT_BIT_LEN = $clog2(INPUT_VERT)
) ( 
    //signal
    input                           rstb,       
    input                           clk,
    input                           layer_en, // layer start signal
    output  reg [IM_BIT_LEN-1:0]    im_addr, 
    output  reg [MA1_BIT_LEN-1:0]   ma_wr_addr, 
    output  reg [L1_WM_BIT_LEN-1:0] l1_wm_r_addr,
    output  reg                     im_cen,
    output  reg                     im_wen,
    output  reg                     ma_cen,
    output  reg                     ma_wen,
    output  reg                     wm_cen,     
    output  reg                     wm_wen,
    output  reg                     pe_rst, // pe clear signal
    output  reg                     pe_en,  // pe start signal    
    output  reg [31:0]              output_buf_en, // output_buf start signal
    output  reg                     output_buf_rst,
    output  reg                     layer_done,  // layer end signal
    output  reg                     init,
    output  reg [INPUT_VERT_BIT_LEN -1:0]                   l1_line_cnt // layer line cnt 
);
    //Bit Length
    localparam W_CNT_BIT_LEN = $clog2(INPUT_HORIZ*INPUT_VERT);
    localparam R_CNT_BIT_LEN = $clog2(WEIGHT_VERT * WEIGHT_HORIZ * NEXET_INPUT_HORIZ);
    localparam MA_W_CNT_BIT_LEN = $clog2(NEXET_INPUT_HORIZ * NEXT_WEIGHT_VERT);

    localparam WEIGHT_HORIZ_BIT_LEN = $clog2(WEIGHT_HORIZ);
    localparam NEXET_INPUT_HORIZ_BIT_LEN = $clog2(NEXET_INPUT_HORIZ);

    //State Names
    localparam IDLE = 0;
    localparam IM_WRITE = 1;
    localparam IM_READ = 2;
    localparam WAIT = 3;
    localparam MA_WRITE = 4;
    localparam LAYER = 5;
    

    //Parameter Setup
    reg [3:0] state, state_n;   


    reg [WEIGHT_VERT_BIT_LEN -1:0] l1_flag;
    reg [WEIGHT_VERT_BIT_LEN -1:0] l1_flag_n;
    reg init_n;


    reg [W_CNT_BIT_LEN -1:0] w_cnt, w_cnt_n;
    reg [R_CNT_BIT_LEN -1:0] r_cnt, r_cnt_n;
    reg [MA_W_CNT_BIT_LEN -1:0] ma_w_cnt, ma_w_cnt_n;
    reg [NEXET_INPUT_HORIZ_BIT_LEN -1:0] pos_cnt, pos_cnt_n;
    reg [WEIGHT_VERT_BIT_LEN -1:0] w_v_cnt, w_v_cnt_n;
    reg [WEIGHT_HORIZ_BIT_LEN -1:0] w_h_cnt, w_h_cnt_n;
    reg [1:0] wait_cnt, wait_cnt_n;

    reg [INPUT_VERT_BIT_LEN -1:0] l1_line_cnt_n;
    reg [IM_BIT_LEN -1:0] im_addr_n;
    reg [IM_BIT_LEN -1:0] im_r_addr, im_r_addr_n;
    reg [IM_BIT_LEN -1:0] im_wr_addr, im_wr_addr_n;
    reg [MA1_BIT_LEN -1:0] ma_wr_addr_n;
    reg [L1_WM_BIT_LEN -1:0] l1_wm_r_addr_n ;
    reg im_cen_n;
    reg im_wen_n;
    reg ma_wen_n;
    reg ma_cen_n;  
    reg wm_cen_n;
    reg wm_wen_n;
    
    reg pe_rst_n;

    reg pe_en_n;
    reg [31:0] output_buf_en_n;
    reg output_buf_rst_n;
    reg layer_done_n;

    always_comb begin : STATE
            state_n = state;

            l1_flag_n = l1_flag;
            init_n = init;

            wait_cnt_n = wait_cnt;
                        
            w_cnt_n = w_cnt;
            r_cnt_n = r_cnt;
            ma_w_cnt_n = ma_w_cnt;
            w_h_cnt_n = w_h_cnt;
            w_v_cnt_n = w_v_cnt;

            pos_cnt_n = pos_cnt;
            l1_line_cnt_n = l1_line_cnt;
            
            im_wr_addr_n = im_wr_addr;
            im_r_addr_n = im_r_addr;
            im_addr_n = im_addr;
            ma_wr_addr_n = ma_wr_addr;
            l1_wm_r_addr_n = l1_wm_r_addr;

            im_wen_n = im_wen;
            im_cen_n = im_cen;
            ma_wen_n = ma_wen;
            ma_cen_n = ma_cen;
            wm_wen_n = wm_wen;
            wm_cen_n = wm_cen;
            

            pe_rst_n = pe_rst;

            pe_en_n = pe_en;
            output_buf_en_n = output_buf_en;
            output_buf_rst_n = output_buf_rst;
            layer_done_n = layer_done;


            case (state)
                IDLE : begin
                        layer_done_n = 0;
                        if (layer_en == 1) begin
                            state_n = IM_WRITE;
                            im_wen_n =0;
                            im_cen_n =0;
                            im_addr_n = im_wr_addr;
                        end                        
                end
                IM_WRITE : begin
                        w_cnt_n = w_cnt +1;
                        im_wr_addr_n = im_wr_addr +1;
                        im_addr_n = im_wr_addr_n;

                        if (w_cnt == INPUT_HORIZ - 1) begin
                            state_n = IM_READ;
                            output_buf_rst_n = 1;
                            w_cnt_n = 0;

                            // im_cen_n = 1;
                            wm_cen_n = 0;
                            im_wen_n = 1;
                            wm_wen_n = 1;

                            //pe_en_n = 1;

                            im_r_addr_n = l1_flag * INPUT_HORIZ;
                            im_addr_n = im_r_addr_n;
                        end 

                        if (im_wr_addr == INPUT_HORIZ * WEIGHT_VERT - 1) begin
                            im_wr_addr_n =0;
                            im_addr_n = im_wr_addr_n;
                        end
                end
                IM_READ : begin
                        r_cnt_n = r_cnt +1;
                        im_r_addr_n = im_r_addr+1;
                        im_addr_n = im_r_addr_n;
                        l1_wm_r_addr_n = l1_wm_r_addr  +1;
                        w_h_cnt_n = w_h_cnt + 1;
                        output_buf_rst_n = 0;
                        pe_en_n = 1;

                        if (w_h_cnt == WEIGHT_HORIZ - 1) begin
                            w_h_cnt_n = 0;
                            w_v_cnt_n = w_v_cnt +1 ;
                            im_r_addr_n = (im_r_addr - WEIGHT_HORIZ + WEIGHT_VERT +1) % (INPUT_HORIZ * WEIGHT_VERT) ;
                            im_addr_n = im_r_addr_n;
                            if (w_v_cnt == WEIGHT_VERT -1) begin
                                w_v_cnt_n = 0;
                                pos_cnt_n = pos_cnt +1;
                                im_r_addr_n = ((pos_cnt+1) * STRIDE_HORIZ + l1_flag* INPUT_HORIZ) % (INPUT_HORIZ * WEIGHT_VERT);
                                im_addr_n = im_r_addr_n;
                            end
                        end
                  
                        if (r_cnt == (WEIGHT_HORIZ * WEIGHT_VERT - 1)) begin
                            state_n = WAIT;  
                            r_cnt_n = 0;
                            l1_wm_r_addr_n = 0;

                            //pe_en_n = 0;
                            

                            im_wen_n = 0;
                            wm_wen_n = 0;
                            wm_cen_n = 1;
                            im_cen_n = 1;

                            
                        end
                    end

                WAIT : begin
                    
                    
                    if(wait_cnt == 1)begin
                        state_n = MA_WRITE;
                        pe_rst_n = 1;
                        wait_cnt_n = 0;
                        output_buf_en_n=0;
                        ma_wen_n = 0;
                        ma_cen_n = 0;
                    end
                    else begin
                        state_n = WAIT;
                        wait_cnt_n = wait_cnt + 1;
                        output_buf_en_n=32'hFFFFFFFF;
                    end
                    

                    pe_en_n = 0;
                    
                    
           
                end

                MA_WRITE : begin
                    pe_rst_n = 0;
                    output_buf_rst_n = 1;

                    ma_wen_n = 1;
                    ma_cen_n = 1;

                    ma_w_cnt_n = ma_w_cnt +1;
                    ma_wr_addr_n = (ma_wr_addr +1) % (NEXET_INPUT_HORIZ* NEXT_WEIGHT_VERT); 
                    if (l1_line_cnt + 1 == INPUT_VERT) 
                        init_n = 0;

                    if(ma_w_cnt == NEXET_INPUT_HORIZ - 1) begin
                        state_n = LAYER;
                        ma_w_cnt_n = 0;
                        l1_line_cnt_n = (l1_line_cnt + 1) % (INPUT_VERT);
                        if (!init || (l1_line_cnt > WEIGHT_VERT - 2)) begin
                            l1_flag_n = (l1_flag +1) % (WEIGHT_VERT);
                        end
                        layer_done_n = 1;

                        // ma_cen_n =1;
                    end else begin
                        state_n = IM_READ;
                        im_cen_n = 0;
                        im_wen_n = 1;
                        wm_cen_n = 0;
                        wm_wen_n = 1;
                        //pe_en_n = 1;


                    end

                    ma_cen_n = 1;
                end

                LAYER : begin
                    state_n = IDLE;
                    output_buf_rst_n = 0;

                    //layer_done_n = 0;
                    

                end
        endcase

    end


    always_ff @(posedge clk or negedge rstb) begin : block
        if (!rstb) begin
            state <= IDLE;

            init <= 1;
            pe_en           <= 0;
            output_buf_en   <= '0;

            w_cnt <=0;
            r_cnt <=0;

            wait_cnt <= 0;

            ma_w_cnt<=0;
            w_v_cnt <=0;
            w_h_cnt <= 0;
            pos_cnt <=0;
            l1_line_cnt<=0;

            l1_flag <=0;

            im_wen          <= 1;
            im_cen          <= 1;
            ma_wen          <= 1;
            ma_cen          <= 1;
            wm_wen          <= 1;
            wm_cen          <= 1;


            im_wr_addr      <= 0;
            ma_wr_addr      <= 0;
            im_r_addr <= 0;
            l1_wm_r_addr<=0;
            im_addr <= 0;

            pe_rst <= 0;
            output_buf_rst <= 0;

            layer_done <=0;

        end 
        else begin
            state<=state_n;

            init<= init_n;
            output_buf_en<= output_buf_en_n;
            pe_en<=pe_en_n;

            wait_cnt <= wait_cnt_n;

            w_cnt <=w_cnt_n;
            r_cnt <= r_cnt_n;
            w_v_cnt<=w_v_cnt_n;
            w_h_cnt <= w_h_cnt_n;
            ma_w_cnt <= ma_w_cnt_n;
            pos_cnt<=pos_cnt_n;

            l1_line_cnt<=l1_line_cnt_n;
            l1_flag<=l1_flag_n;

            im_wr_addr<=im_wr_addr_n;
            im_r_addr<=im_r_addr_n;
            im_addr<= im_addr_n;
            ma_wr_addr<= ma_wr_addr_n;
            l1_wm_r_addr<= l1_wm_r_addr_n;

            im_cen <= im_cen_n;
            im_wen <= im_wen_n;
            ma_cen <= ma_cen_n;
            ma_wen <= ma_wen_n;
            wm_cen <= wm_cen_n;
            wm_wen <= wm_wen_n;

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
