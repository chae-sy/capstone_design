`timescale 1ns / 1ps
module layer_pipeline #(
    parameter           NUM_COLOR = 3
)(
    input               clk,
    input               rst_n,

    //Weight Memory
    output  wire [9:0]  wmem_addr_o,
    output  wire        wmem_wenb_o,
    output  wire        wmem_cenb_o,

    //Memory A 
    output  wire [15:0] memA_addr_o,
    output  wire        memA_wenb_o,
    output  wire        memA_cenb_o,
    //Memory B
    output  wire [15:0] memB_addr_o,
    output  wire        memB_wenb_o,
    output  wire        memB_cenb_o,

    //Weight Buffer
    output  wire        wei_buff_wren_o,
    output  wire        wei_buff_rden_o,
    
    // input buffer
    output  wire        in_buf_wren_r,
    output  wire        in_buf_wren_g,
    output  wire        in_buf_wren_b,
    output  wire        in_buf_rden_r,
    output  wire        in_buf_rden_g,
    output  wire        in_buf_rden_b,
    output  wire        is_initial_o,
    
    // output buffer
    output  wire        out_buf_wren_r,
    output  wire        out_buf_wren_g,
    output  wire        out_buf_wren_b,
    output  wire        out_buf_rden_r,
    output  wire        out_buf_rden_g,
    output  wire        out_buf_rden_b,
    input   wire        out_buf_done_i,
    
    //PE array
    output  wire        pe_en_o,
    input   wire        pe_done_i,
    
    //add tree
    output  wire        addtree_en_o,
    input   wire        addtree_done_i,

    // ReLU
    output  wire        relu_en_o,
    input   wire        relu_done_i,

    //maxpool
    output  wire        maxpool_en_o,
    input   wire        maxpool_done_i,
    output  wire [1:0]  color_o,
    
    input   wire [2:0]  layer_num,
    input   wire [8:0]  weight_num,
    input   wire [4:0]  channel,
    input   wire        layer_start,
    output  wire        layer_done_o
);

    localparam              FIRST = 0,
                            weight_change = 1,
                            inputandweight_change = 2,
                            Second = 3;
    
    localparam              IDLE = 0,
                            enable1 = 1,
                            enable2 = 2,
                            enable3 = 3;

    localparam              red = 0,
                            green = 1,
                            blue = 2;

    reg     [15:0]          mem_rd_addr, mem_rd_addr_n;
    reg     [9:0]           wmem_addr, wmem_addr_n;
    reg     [7:0]           cnt1_row, cnt1_row_n;
    reg     [7:0]           cnt1_column, cnt1_column_n;
    reg     [1:0]           state, state_n;
    reg     [7:0]           num1, num1_n;
    reg     [7:0]           num_w, num_w_n;
    reg     [7:0]           num_w2, num_w2_n;
    reg     [1:0]           cnt2, cnt2_n;
    reg     [4:0]           cnt_ch, cnt_ch_n;
    reg                     row_last, row_last_n,
                            final_data, final_data_n;
    reg     [1:0]           stage2_state, stage2_state_n;
    reg     [1:0]           stage3_state, stage3_state_n;
    reg     [1:0]           color, color_n;
    reg     [1:0]           color_wb, color_wb_n;
    
    reg                     mem_rd_cenb, mem_rd_wenb,
                            wmem_cenb, wmem_wenb,
                            mem_wr_cenb, mem_wr_wenb;
                            
    reg                     in_buf_wren[0:NUM_COLOR-1], in_buf_rden[0:NUM_COLOR-1],
                            out_buf_wren[0:NUM_COLOR-1], out_buf_rden[0:NUM_COLOR-1]; 
    
    reg                     wei_buff_wren, wei_buff_rden;
    
    reg                     pe_en,
                            addtree_en,
                            relu_en,
                            maxpool_en;                   
    
    reg                     stage2_en, stage2_en_n;
    reg                     stage3_en, stage3_en_n;
    reg                     stage4_en, stage4_en_n;
    reg                     stage5_en, stage5_en_n;
    reg                     stage6_en, stage6_en_n;
                                              
    reg                     stage1_done,
                            stage2_done,
                            stage3_done,
                            stage4_done,
                            stage5_done,
                            stage6_done;
                                             
    reg     [15:0]          mem_wr_addr, mem_wr_addr_n;
    reg     [15:0]          mem_wr_pad_addr, mem_wr_pad_addr_n;
    reg     [1:0]           output_state, output_state_n;
    
    reg     [7:0]           cnt2_row, cnt2_row_n;
    reg     [7:0]           cnt2_column, cnt2_column_n;
    reg     [7:0]           num2, num2_n;
    reg     [7:0]           num_p, num_p_n;
    reg     [127:0]         out_data;
    reg                     output_mem_en;
    reg     [6:0]           row_num;
    reg     [13:0]          color_num1, color_num2;
    reg     [15:0]          start_point; 
    reg     [15:0]          start_weight;                    
    
    reg                     stage2_en_reg;
    reg                     wei_buff_wren_reg ;
    reg                     in_buf_wren_reg[0:NUM_COLOR-1];
    reg                     maxpool_en_reg;
    reg                     is_initial_reg, is_initial;
    reg     [1:0]           color_reg;        
    
    integer i;
    
    // 1. mem data fetch (fetch)
    always @( posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            if (layer_num == 4) begin
                mem_rd_addr <= 'd103;
            end
            else begin
                mem_rd_addr <= 'b0;
            end
            case(layer_num)
                1: begin
                    wmem_addr <= 'b0; 
                    start_weight <= 'b0;
                end
                2: begin
                    wmem_addr <= 'd144; 
                    start_weight <= 'd144;
                end
                3: begin
                    wmem_addr <= 'd288; 
                    start_weight <= 'd288;
                end
                5: begin
                    wmem_addr <= 'd432; 
                    start_weight <= 'd432;
                end
                6: begin 
                    wmem_addr <= 'd576; 
                    start_weight <= 'd576;
                end
                default: begin
                    wmem_addr <= 'b0; 
                    start_weight <= 'b0;
                end
            endcase
            cnt1_row <= 'd0;
            cnt1_column <= 'd0;
            state <= FIRST;
            num1 <= 0;
            num_w <= weight_num;
            stage2_en <= 0;
            row_last <= 0;
            final_data <= 0;
            color <= red;
        end else begin
            mem_rd_addr <= mem_rd_addr_n;
            wmem_addr <= wmem_addr_n;
            cnt1_row <= cnt1_row_n;
            cnt1_column <= cnt1_column_n;
            state <= state_n;
            num1 <= num1_n;
            num_w <= num_w_n;
            stage2_en <= stage2_en_n;
            row_last <= row_last_n;
            final_data <= final_data_n;
            color <= color_n;
        end
    end
  
    //channel 동시, (input data 전체) * weight 16개
    always @(*) begin
        mem_rd_addr_n = mem_rd_addr;
        wmem_addr_n = wmem_addr; 
        cnt1_row_n = cnt1_row;
        cnt1_column_n = cnt1_column;
        state_n = state;
        num1_n = num1;
        num_w_n = num_w;
        stage1_done = 0;
        stage2_en_n = stage2_en;
        color_n = color;
    
        mem_rd_cenb         = 1;
        mem_rd_wenb         = 1;
        wmem_wenb           = 1;
        wmem_cenb            = 1;
        wei_buff_wren         = 0;

        maxpool_en = 0;
        is_initial = 1'b0;

        for (i = 0; i < NUM_COLOR; i = i + 1 ) begin
            in_buf_wren[i] = 0;
        end
        row_last_n = row_last;
        final_data_n = final_data;
        case(layer_num)
            4: begin
                case(color)
                    red: begin
                        if (cnt1_column <= 'd96) begin
                            if (cnt1_row <= 98) begin
                                maxpool_en = 1;
                                mem_rd_cenb = 0;
                                num1_n = num1 + 1;
                                if (num1 == 'd7) begin // 4x2 마지막
                                    mem_rd_addr_n = mem_rd_addr - 'd305;
                                    cnt1_row_n = cnt1_row + 2;
                                    num1_n = 0;
                                    if (cnt1_row == 98) begin// 마지막 row
                                        cnt1_row_n = 0;
                                        cnt1_column_n = cnt1_column + 4;
                                        mem_rd_addr_n = mem_rd_addr + 'd3;
                                    end
                                    if ((cnt1_column == 96) & (cnt1_row == 98)) begin // 마지막 column & row
                                        color_n = green;
                                        cnt1_column_n = 0;
                                        mem_rd_addr_n = 'd103 + 'd10302;
                                    end
                                end
                                else if (num1 % 2 == 'd1) begin // 줄 바꾸기
                                    mem_rd_addr_n = mem_rd_addr + 'd101;
                                end
                                else begin
                                    mem_rd_addr_n = mem_rd_addr + 1;
                                end
                            end    
                        end
                    end
                    green: begin
                        if (cnt1_column <= 'd98) begin
                            if (cnt1_row <= 98) begin
                                maxpool_en = 1;
                                mem_rd_cenb = 0;
                                num1_n = num1 + 1;
                                if (num1 == 'd3) begin // 2x2 마지막
                                    mem_rd_addr_n = mem_rd_addr - 'd101;
                                    cnt1_row_n = cnt1_row + 2;
                                    num1_n = 0;
                                    if (cnt1_row == 98) begin// 마지막 row
                                        cnt1_row_n = 0;
                                        cnt1_column_n = cnt1_column + 2;
                                        mem_rd_addr_n = mem_rd_addr + 'd3;
                                    end
                                    if ((cnt1_column == 98) & (cnt1_row == 98)) begin // 마지막 column & row
                                        color_n = blue;
                                        cnt1_column_n = 0;
                                        mem_rd_addr_n = 'd103 + 'd20604;
                                    end
                                end
                                else if (num1 % 2 == 'd1) begin // 줄 바꾸기
                                    mem_rd_addr_n = mem_rd_addr + 'd101;
                                end
                                else begin
                                    mem_rd_addr_n = mem_rd_addr + 1;
                                end
                            end    
                        end
                    end
                    blue: begin
                        if (cnt1_column <= 'd96) begin
                            if (cnt1_row <= 98) begin
                                maxpool_en = 1;
                                mem_rd_cenb = 0;
                                num1_n = num1 + 1;
                                if (num1 == 'd7) begin // 4x2 마지막
                                    mem_rd_addr_n = mem_rd_addr - 'd305;
                                    cnt1_row_n = cnt1_row + 2;
                                    num1_n = 0;
                                    if (cnt1_row == 98) begin// 마지막 row
                                        cnt1_row_n = 0;
                                        cnt1_column_n = cnt1_column + 4;
                                        mem_rd_addr_n = mem_rd_addr + 'd3;
                                    end
                                    if ((cnt1_column == 96) & (cnt1_row == 98)) begin // 마지막 column & row
                                        mem_rd_addr_n = 0;
                                    end
                                end
                                else if (num1 % 2 == 'd1) begin // 줄 바꾸기
                                    mem_rd_addr_n = mem_rd_addr + 'd101;
                                end
                                else begin
                                    mem_rd_addr_n = mem_rd_addr + 1;
                                end
                            end    
                        end
                    end
                endcase
                
            end
            1,2,3,5: begin //1,2,3,5,6 layer
                case(state)
                    FIRST: begin // 처음 input_data:27개, weight:9개 불러오기
                        stage2_en_n = 'b0;      
                        // input data 처리
                        if (num_w != 0) begin // weight 개수
                            is_initial = 1'b1;
                            mem_rd_cenb = 0;
                            num1_n = num1 + 1;
          
                            if (num1 == 'd8) begin // red -> green
                                mem_rd_addr_n = (((layer_num == 1)|(layer_num == 2)|(layer_num == 3)) ? 'd10302 : 'd2652) + 'd102 * cnt1_column;
                            end
                            else if (num1 == 'd17) begin// green -> blue
                                mem_rd_addr_n = (((layer_num == 1)|(layer_num == 2)|(layer_num == 3)) ? 'd20604 : 'd7854) + 'd102 * cnt1_column;
                            end
                            else if (num1 == 'd26) begin // 맨 마지막. 27개 불러옴.
                                stage2_en_n = 'b1;
                                mem_rd_addr_n = cnt1_column * 102 + 3;
                                cnt1_row_n = cnt1_row + 1;
                                state_n = weight_change;
                                num1_n = 0;
                                num_w_n = num_w - 1;
                            end
                            else if ((num1 % 3) == 'd2) begin // 줄 바꾸기
                                mem_rd_addr_n = mem_rd_addr + 100;
                            end
                            else begin // 옆으로 이동
                                mem_rd_addr_n = mem_rd_addr + 1;
                            end
            
                            if (num1 >= 'd9) begin
                                wei_buff_wren = 0;
                            end
                            else begin // 첫번째 weight 불러오기
                                wmem_cenb            = 0;
                                wmem_addr_n = wmem_addr + 1;
                                wei_buff_wren         = 1;
                            end
                            if (num1 <= 'd8) in_buf_wren[0] = 1;
                            else if (num1 <= 'd17) in_buf_wren[1] = 1;
                            else if (num1 <= 'd26) in_buf_wren[2] = 1;                 
            
                        end
                        else begin
                            stage1_done = 1;
                        end
                    end
                    weight_change: begin // weight 개수 따라 불러오기, input_data는 고정
                        stage2_en_n = 'b0;
                        if (num_w != 0) begin // weight 개수
                            wmem_cenb            = 0;
                            wmem_addr_n = wmem_addr + 1;
                            wei_buff_wren         = 1;
                            num1_n = num1 + 1;
                            if (num1 == 'd8) begin
                                num1_n = 0;
                                num_w_n = num_w - 1;
                                stage2_en_n = 'b1;
                            end
                            if (num_w_n == 0) begin // 모든 weight 다 불러옴. addr 초기화 & input/weight 둘 다 불러오는 state로 이동
                                num_w_n = weight_num;
                                num1_n = 0;
                                wmem_addr_n = start_weight;
                                if (row_last) begin
                                    state_n = FIRST;
                                    row_last_n = 0;
                                    if (final_data) begin // 데이터 불러오기 끝
                                        num_w_n = 0;
                                        final_data_n = 0;
                                    end
                                end
                                else begin
                                    state_n = inputandweight_change;
                                end
                            end
                        end
                    end
                    inputandweight_change: begin
                        case (layer_num)
                            1, 2, 3: begin
                                stage2_en_n = 'b0;
                                if (cnt1_column <= 'd99) begin
                                    if (cnt1_row <= 99) begin // 한 줄 개수 세기 (99 = 마지막)
                                        mem_rd_cenb = 0;

                                        // weight 불러오기
                                        wmem_cenb = 0;
                                        wmem_addr_n = wmem_addr + 1;
                                        wei_buff_wren = 1;
                                        num1_n = num1 + 1;

                                        if (num1 == 'd2) begin // red -> green
                                            mem_rd_addr_n = 'd10302 + 'd102 * cnt1_column + cnt1_row + 2;
                                        end
                                        else if (num1 == 'd5) begin// green -> blue
                                            mem_rd_addr_n = 'd20604 + 'd102 * cnt1_column + cnt1_row + 2;
                                        end
                                        else if (num1 == 'd8) begin // 다음 row로 이동 & weight 바꾸기
                                            mem_rd_addr_n = cnt1_column * 102 + cnt1_row + 3;
                                            stage2_en_n = 'b1;
                                            cnt1_row_n = cnt1_row + 1;
                                            num1_n = 0;
                                            num_w_n = num_w - 1;
                                            state_n = weight_change;
                                            if (cnt1_row == 99) begin// 마지막 row
                                                cnt1_row_n = 0;
                                                cnt1_column_n = cnt1_column + 1;
                                                mem_rd_addr_n = cnt1_column_n * 102;
                                                row_last_n = 1;
                                            end
                                            if ((cnt1_column == 99) & (cnt1_row == 99)) begin // 마지막 column & row
                                                final_data_n = 1;
                                                cnt1_column_n = 0;
                                                mem_rd_addr_n = 0;
                                            end
                                        end
                                        else begin
                                            mem_rd_addr_n = mem_rd_addr + 102;
                                        end
                                        if (num1 <= 'd2) in_buf_wren[0] = 1;
                                        else if (num1 <= 'd5) in_buf_wren[1] = 1;
                                        else if (num1 <= 'd8) in_buf_wren[2] = 1;
                                    end
                                end
                                
                            end
                            5, 6: begin // R, B <-> G size different
                                stage2_en_n = 'b0;
                                if (cnt1_column <= 'd24) begin
                                    if (cnt1_row <= 'd49) begin // 한 줄 개수 세기 (R, G, B 다 해당)
                                        mem_rd_cenb = 0;

                                        // weight 불러오기
                                        wmem_cenb = 0;
                                        wmem_addr_n = wmem_addr + 1;
                                        wei_buff_wren = 1;
                                        num1_n = num1 + 1;

                                        if (num1 == 'd2) begin // red -> green
                                            mem_rd_addr_n = 'd2652 + 'd102 * cnt1_column + cnt1_row + 2;
                                        end
                                        else if (num1 == 'd5) begin// green -> blue
                                            mem_rd_addr_n = 'd7854 + 'd102 * cnt1_column + cnt1_row + 2;
                                        end
                                        else if (num1 == 'd8) begin // 다음 row로 이동 & weight 바꾸기
                                            stage2_en_n = 'b1;
                                            cnt1_row_n = cnt1_row + 1;
                                            num1_n = 0;
                                            num_w_n = num_w - 1;
                                            state_n = weight_change;
                                            mem_rd_addr_n = cnt1_column * 102 + cnt1_row + 3;
                                            if (cnt1_row == 49) begin// 마지막 row
                                                cnt1_row_n = 0;
                                                cnt1_column_n = cnt1_column + 1;
                                                row_last_n = 1;
                                                mem_rd_addr_n = cnt1_column_n * 102;
                                            end
                                        end
                                        else begin
                                            mem_rd_addr_n = mem_rd_addr + 102;
                                        end
                                        
                                        if (num1 <= 'd2) in_buf_wren[0] = 1;
                                        else if (num1 <= 'd5) in_buf_wren[1] = 1;
                                        else if (num1 <= 'd8) in_buf_wren[2] = 1;
                                    end
                                end
                                else if (cnt1_column <= 'd49) begin // Green만
                                    if (cnt1_row <= 'd49) begin // 한 줄 개수 세기 (R, G, B 다 해당)
                                        mem_rd_cenb = 0;

                                        // weight 불러오기
                                        wmem_cenb = 0;
                                        wmem_addr_n = wmem_addr + 1;
                                        wei_buff_wren = 1;
                                        num1_n = num1 + 1;

                                        if (num1 == 'd2) begin // red -> green
                                            mem_rd_addr_n = 'd2652 + 'd102 * cnt1_column + cnt1_row + 2;
                                        end
                                        else if (num1 == 'd5) begin// green -> blue
                                            mem_rd_addr_n = 'd7854 + 'd102 * cnt1_column + cnt1_row + 2;
                                        end
                                        else if (num1 == 'd8) begin // 다음 row로 이동 & weight 바꾸기
                                            stage2_en_n = 'b1;
                                            cnt1_row_n = cnt1_row + 1;
                                            num1_n = 0;
                                            num_w_n = num_w - 1;
                                            state_n = weight_change;
                                            mem_rd_addr_n = cnt1_column * 102 + cnt1_row + 3;
                                            if (cnt1_row == 49) begin// 마지막 row
                                                cnt1_row_n = 0;
                                                cnt1_column_n = cnt1_column + 1;
                                                mem_rd_addr_n = cnt1_column_n * 102;
                                                row_last_n = 1;
                                            end
                                            if ((cnt1_column == 49) & (cnt1_row == 49)) begin // 마지막 column & row
                                                final_data_n = 1;
                                                cnt1_column_n = 0;
                                                mem_rd_addr_n = 0;
                                            end
                                            
                                        end
                                        else begin
                                            mem_rd_addr_n = mem_rd_addr + 102;
                                        end
                                        
                                        if (num1 <= 'd2) in_buf_wren[0] = 0;
                                        else if (num1 <= 'd5) in_buf_wren[1] = 1;
                                        else if (num1 <= 'd8) in_buf_wren[2] = 0;
                                    end
                                end
                            end
                        endcase
                    end
                endcase
            end
            6: begin
                case(state)
                    FIRST: begin // 처음 input_data:27개, weight:9개 불러오기
                        stage2_en_n = 'b0;
                        is_initial = 1'b1;
                        // input data 처리
                        if (num_w != 0) begin // weight 개수
                            mem_rd_cenb = 0;
                            num1_n = num1 + 1;
          
                            if (num1 == 'd8) begin // red -> green
                                mem_rd_addr_n = (((layer_num == 1)|(layer_num == 2)|(layer_num == 3)) ? 'd10302 : 'd2652) + 'd102 * cnt1_column;
                            end
                            else if (num1 == 'd17) begin// green -> blue
                                mem_rd_addr_n = (((layer_num == 1)|(layer_num == 2)|(layer_num == 3)) ? 'd20604 : 'd7854) + 'd102 * cnt1_column;
                            end
                            else if (num1 == 'd26) begin // 맨 마지막. 27개 불러옴.
                                stage2_en_n = 'b1;
                                mem_rd_addr_n = cnt1_column * 102 + 3;
                                cnt1_row_n = cnt1_row + 1;
                                state_n = inputandweight_change;
                                num1_n = 0;
                            end
                            else if ((num1 % 3) == 'd2) begin // 줄 바꾸기
                                mem_rd_addr_n = mem_rd_addr + 100;
                            end
                            else begin // 옆으로 이동
                                mem_rd_addr_n = mem_rd_addr + 1;
                            end
            
                            if (num1 >= 'd9) begin
                                wei_buff_wren = 0;
                            end
                            else begin // 첫번째 weight 불러오기
                                wmem_cenb            = 0;
                                wmem_addr_n = wmem_addr + 1;
                                wei_buff_wren         = 1;
                            end
                            if (num1 <= 'd8) in_buf_wren[0] = 1;
                            else if (num1 <= 'd17) in_buf_wren[1] = 1;
                            else if (num1 <= 'd26) in_buf_wren[2] = 1;                 
            
                        end
                        else begin
                            stage1_done = 1;
                        end
                    end
                    Second: begin // input only
                        stage2_en_n = 'b0;
                        is_initial = 1'b1;
                        // input data 처리
                        if (num_w != 0) begin // weight 개수
                            mem_rd_cenb = 0;
                            num1_n = num1 + 1;
          
                            if (num1 == 'd8) begin // red -> green
                                mem_rd_addr_n = (((layer_num == 1)|(layer_num == 2)|(layer_num == 3)) ? 'd10302 : 'd2652) + 'd102 * cnt1_column;
                            end
                            else if (num1 == 'd17) begin// green -> blue
                                mem_rd_addr_n = (((layer_num == 1)|(layer_num == 2)|(layer_num == 3)) ? 'd20604 : 'd7854) + 'd102 * cnt1_column;
                            end
                            else if (num1 == 'd26) begin // 맨 마지막. 27개 불러옴.
                                stage2_en_n = 'b1;
                                mem_rd_addr_n = cnt1_column * 102 + 3;
                                cnt1_row_n = cnt1_row + 1;
                                state_n = inputandweight_change;
                                num1_n = 0;
                            end
                            else if ((num1 % 3) == 'd2) begin // 줄 바꾸기
                                mem_rd_addr_n = mem_rd_addr + 100;
                            end
                            else begin // 옆으로 이동
                                mem_rd_addr_n = mem_rd_addr + 1;
                            end
                            if (num1 <= 'd8) in_buf_wren[0] = 1;
                            else if (num1 <= 'd17) in_buf_wren[1] = 1;
                            else if (num1 <= 'd26) in_buf_wren[2] = 1;                 
            
                        end
                        else begin
                            stage1_done = 1;
                        end
                    end
                    inputandweight_change: begin // input only changed
                        stage2_en_n = 'b0;
                        if (cnt1_column <= 'd24) begin
                            if (cnt1_row <= 'd49) begin // 한 줄 개수 세기 (R, G, B 다 해당)
                                mem_rd_cenb = 0;
                                num1_n = num1 + 1;

                                if (num1 == 'd2) begin // red -> green
                                    mem_rd_addr_n = 'd2652 + 'd102 * cnt1_column + cnt1_row + 2;
                                end
                                else if (num1 == 'd5) begin// green -> blue
                                    mem_rd_addr_n = 'd7854 + 'd102 * cnt1_column + cnt1_row + 2;
                                end
                                else if (num1 == 'd8) begin // 다음 row로 이동 
                                    stage2_en_n = 'b1;
                                    cnt1_row_n = cnt1_row + 1;
                                    num1_n = 0;
                                    mem_rd_addr_n = cnt1_column * 102 + cnt1_row + 3;
                                    if (cnt1_row == 49) begin// 마지막 row
                                        cnt1_row_n = 0;
                                        cnt1_column_n = cnt1_column + 1;
                                        row_last_n = 1;
                                        mem_rd_addr_n = cnt1_column_n * 102;
                                        state_n = Second;
                                    end
                                end
                                else begin
                                    mem_rd_addr_n = mem_rd_addr + 102;
                                end
                                
                                if (num1 <= 'd2) in_buf_wren[0] = 1;
                                else if (num1 <= 'd5) in_buf_wren[1] = 1;
                                else if (num1 <= 'd8) in_buf_wren[2] = 1;
                            end
                        end
                        else if (cnt1_column <= 'd49) begin // Green만
                            if (cnt1_row <= 'd49) begin // 한 줄 개수 세기 (R, G, B 다 해당)
                                mem_rd_cenb = 0;
                                num1_n = num1 + 1;

                                if (num1 == 'd2) begin // red -> green
                                    mem_rd_addr_n = 'd2652 + 'd102 * cnt1_column + cnt1_row + 2;
                                end
                                else if (num1 == 'd5) begin// green -> blue
                                    mem_rd_addr_n = 'd7854 + 'd102 * cnt1_column + cnt1_row + 2;
                                end
                                else if (num1 == 'd8) begin // 다음 row로 이동 & weight 바꾸기
                                    stage2_en_n = 'b1;
                                    cnt1_row_n = cnt1_row + 1;
                                    num1_n = 0;
                                    mem_rd_addr_n = cnt1_column * 102 + cnt1_row + 3;
                                    if (cnt1_row == 49) begin// 마지막 row
                                        cnt1_row_n = 0;
                                        cnt1_column_n = cnt1_column + 1;
                                        mem_rd_addr_n = cnt1_column_n * 102;
                                        row_last_n = 1;
                                        state_n = Second;
                                    end
                                    if ((cnt1_column == 49) & (cnt1_row == 49)) begin // 마지막 column & row
                                        final_data_n = 1;
                                        cnt1_column_n = 0;
                                        mem_rd_addr_n = 0;
                                        num_w_n = num_w - 1;
                                    end
                                    
                                end
                                else begin
                                    mem_rd_addr_n = mem_rd_addr + 102;
                                end
                                
                                if (num1 <= 'd2) in_buf_wren[0] = 0;
                                else if (num1 <= 'd5) in_buf_wren[1] = 1;
                                else if (num1 <= 'd8) in_buf_wren[2] = 0;
                            end
                        end
                    end
                endcase
            end
        endcase
    end

    // 주소가 예정보다 늦게 들어와서 한 사이클씩 뒤로 밀어야함
    always @(posedge clk) begin
        if (!rst_n | layer_start) begin
           stage2_en_reg <= 0;
           is_initial_reg <= 0;
        end
        else begin
            if (layer_num >= 1) begin
                stage2_en_reg <= stage2_en;
                wei_buff_wren_reg <= wei_buff_wren;
                for (i = 0; i < NUM_COLOR; i = i + 1 ) begin
                    in_buf_wren_reg[i] <= in_buf_wren[i];
                end
                maxpool_en_reg <= maxpool_en;
                is_initial_reg <= is_initial;
                color_reg <= color;
            end 
        end  
    end
    // 2. compute (PE_array )

    always @(*) begin
        pe_en = 0;
        wei_buff_rden = 0;
        for (i = 0; i < NUM_COLOR; i = i + 1 ) begin
            in_buf_rden[i] = 0;
        end
        cnt2_n = cnt2;
        stage3_en_n = stage3_en;
        stage2_state_n = stage2_state;
        case(stage2_state)
            IDLE: begin
                stage3_en_n = 0;
                if (stage2_en_reg) begin
                    wei_buff_rden = 1;
                    for (i = 0; i < NUM_COLOR; i = i + 1 ) begin
                        in_buf_rden[i] = 1;
                    end
                    pe_en = 1;
                    stage2_state_n = enable1;
                end
            end
            enable1: begin
                wei_buff_rden = 1;
                for (i = 0; i < NUM_COLOR; i = i + 1 ) begin
                    in_buf_rden[i] = 1;
                end
                pe_en = 1;
                if (pe_done_i) begin
                    stage3_en_n = 1;
                    stage2_state_n = IDLE;
                end
                else begin
                    stage3_en_n = 0;
                end
            end
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            stage2_state <= IDLE;
            stage3_en <= 0;
            stage2_done <= 0;
        end else begin
            stage3_en <= stage3_en_n;
            stage2_state <= stage2_state_n;
        end
        if (pe_done_i) stage2_done <= stage1_done;
    end
    
    // 3. add tree

    always @(*) begin
        addtree_en = 0;
        stage4_en_n = stage4_en;
        stage3_state_n = stage3_state;
        case(stage3_state)
            IDLE: begin
                stage4_en_n = 0;
                if (stage3_en) begin
                    addtree_en = 1;
                    stage3_state_n = enable1;
                end
            end
            enable1: begin
                addtree_en = 0;
                if (addtree_done_i) begin
                    stage4_en_n = 1;
                    stage3_state_n = IDLE;
                end
                else begin
                    stage4_en_n = 0;
                end
            end
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            stage4_en <= 0;
            stage3_done <= 0;
            stage3_state <= IDLE;
        end else begin
            stage4_en <= stage4_en_n;
            stage3_state <= stage3_state_n;
        end
        if (addtree_done_i) stage3_done <= stage2_done;
    end
    


    // 4. activate (ReLU, bias)
    
    //1, 2, 3, 5, 6(relu X) ReLU exist
    
    always @(*) begin
        relu_en   = 0;
        stage5_en_n = stage5_en;
        case (layer_num)
            1, 2, 3, 5, 6: begin
                if (stage4_en) begin
                    relu_en   = 1;
                    if (relu_done_i) begin
                        stage5_en_n = 1;
                    end
                    else begin
                        stage5_en_n = 0;
                    end
                end
                else begin
                    relu_en = 0;
                    stage5_en_n = 0;
                end
            end
            default: begin
                stage5_en_n = 0;
            end
        endcase
    end

    always @( posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            stage5_en <= 0;
            stage4_done <= 0;
        end else begin
            stage5_en <= stage5_en_n;
        end
    end

    // only layer4 exist maxpool
    
    always @(*) begin
        stage6_en_n = 0;
        if (maxpool_done_i) begin
            stage6_en_n = 1;
        end
    end
         
    always @( posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            stage6_en <= 0;
        end else begin
            stage6_en <= stage6_en_n;
        end
    end
    
    // 5. write back (out_buff -> mem) all layer
    always @( posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            mem_wr_addr <= 'd103; // 시작부분
            mem_wr_pad_addr <= 'd153; // 시작부분
            cnt2_row <= 'd0;
            cnt2_column <= 'd0;
            cnt_ch <= 'd0;
            output_state <= IDLE;
            num2 <= 0;
            num_w2 <= weight_num; // 채널 구분
            num_p <= 0;
            color_wb <= red;
        end else begin
            mem_wr_addr <= mem_wr_addr_n;
            mem_wr_pad_addr <= mem_wr_pad_addr_n;
            cnt2_row <= cnt2_row_n;
            cnt2_column <= cnt2_column_n;
            cnt_ch <= cnt_ch_n;
            output_state <= output_state_n;
            num2 <= num2_n;
            num_w2 <= num_w2_n;
            num_p <= num_p;
            color_wb <= color_wb_n;
        end
        
    end
    
    always @(*) begin
        mem_wr_cenb = 1;
        mem_wr_wenb = 1;
        mem_wr_addr_n = mem_wr_addr;
        mem_wr_pad_addr_n = mem_wr_pad_addr;
    
        cnt2_row_n = cnt2_row;
        cnt2_column_n = cnt2_column;
        cnt_ch_n = cnt_ch;
        output_state_n = output_state;
        num2_n = num2;
        num_w2_n = num_w2;
        num_p_n = num_p;
        stage5_done = 0;
        color_wb_n = color_wb;

        for (i = 0; i < NUM_COLOR; i = i + 1 ) begin
            out_buf_wren[i] = 0; 
            out_buf_rden[i] = 0;
        end
        output_mem_en = 0;
        case (layer_num)
            4: begin // R -> G -> B 순서대로 주소(채널 다) 하나씩
                if (stage6_en_n) begin  // maxpool 연산 끝 -> 바로 memory에 저장
                    case(color_wb)
                        red: begin
                            if (cnt2_column <= 'd24) begin
                                if (cnt2_row <= 'd49) begin
                                    cnt2_row_n = cnt2_row + 1;
                                    mem_wr_cenb = 0;
                                    mem_wr_wenb = 0;
                                    if (cnt2_row == 'd49) begin// 마지막 row
                                        cnt2_row_n = 0;
                                        cnt2_column_n = cnt2_column + 1;
                                        mem_wr_addr_n = (cnt2_column_n + 1)*'d102  + 'd1;
                                        if (cnt2_column == 24) begin // 마지막 column & row
                                            color_wb_n = green;
                                            cnt2_column_n = 0;
                                            mem_wr_addr_n = 'd103 + 'd2652;
                                        end
                                    end
                                    else begin
                                        mem_wr_addr_n = mem_wr_addr + 1;
                                    end
                                end    
                            end
                        end
                        green: begin
                            if (cnt2_column <= 'd49) begin
                                if (cnt2_row <= 'd49) begin
                                    cnt2_row_n = cnt2_row + 1;
                                    mem_wr_cenb = 0;
                                    mem_wr_wenb = 0;
                                    if (cnt2_row == 'd49) begin// 마지막 row
                                        cnt2_row_n = 0;
                                        cnt2_column_n = cnt2_column + 1;
                                        mem_wr_addr_n = (cnt2_column_n + 1)*'d102  + 'd1;
                                        if (cnt2_column == 49) begin // 마지막 column & row
                                            color_wb_n = blue;
                                            cnt2_column_n = 0;
                                            mem_wr_addr_n = 'd103 + 'd7854;
                                        end
                                    end
                                    else begin
                                        mem_wr_addr_n = mem_wr_addr + 1;
                                    end
                                end    
                            end
                        end
                        blue: begin
                            if (cnt2_column <= 'd24) begin
                                if (cnt2_row <= 'd49) begin
                                    cnt2_row_n = cnt2_row + 1;
                                    mem_wr_cenb = 0;
                                    mem_wr_wenb = 0;
                                    if (cnt2_row == 'd49) begin// 마지막 row
                                        cnt2_row_n = 0;
                                        cnt2_column_n = cnt2_column + 1;
                                        mem_wr_addr_n = (cnt2_column_n + 1)*'d102  + 'd1;
                                        if (cnt2_column == 24) begin // 마지막 column & row
                                            cnt2_column_n = 0;
                                            mem_wr_addr_n = 'd103 + 'd2652;
                                            stage5_done = 1;
                                        end
                                    end
                                    else begin
                                        mem_wr_addr_n = mem_wr_addr + 1;
                                    end
                                end    
                            end
                        end
                    endcase
                end
                else begin // padding 하기
                    mem_wr_cenb = 0;
                    mem_wr_wenb = 0;
                    out_data = 'd0;
                    num_p_n = num_p + 1;
                    if (num_p == 'd102) begin
                        mem_wr_pad_addr_n = 'd2653;
                    end
                    else if (num_p == 'd153) begin
                        mem_wr_pad_addr_n = 'd7855;
                    end
                    else if (num_p == 'd204) begin
                        mem_wr_pad_addr_n = 'd10507;
                    end
                    else if (num_p >= 'd255) begin // 마지막
                        mem_wr_pad_addr_n = mem_wr_pad_addr;
                        num_p_n = num_p;
                    end
                    else if (num_p > 'd102) begin
                        mem_wr_pad_addr_n = mem_wr_pad_addr + 1;
                    end
                    else begin
                        mem_wr_pad_addr_n = mem_wr_pad_addr + 102;
                    end
                end
            end
            default begin
                case (output_state)
                    IDLE: begin
                        if (relu_done_i) begin // relu 연산 끝 -> output buffer에 저장
                            for (i = 0; i < NUM_COLOR; i = i + 1 ) begin
                                out_buf_wren[i] = 1; 
                            end
                        end

                        if (out_buf_done_i) begin
                            output_state_n = enable1;
                        end
                    end
                    enable1: begin // red
                        out_buf_rden[0] = 1;
                        if (layer_num == 'd6) mem_wr_wenb = 1;
                        else mem_wr_wenb = 0;
                        mem_wr_cenb = 0;
                        output_mem_en = 1;
                        output_state_n = enable2;
                    end
                    enable2: begin // green
                        out_buf_rden[1] = 1;
                        mem_wr_cenb = 0;
                        mem_wr_wenb = 0;
                        output_mem_en = 1;
                        output_state_n = enable3;
                    end
                    enable3: begin // blue
                        out_buf_rden[2] = 1;
                        mem_wr_cenb = 0;
                        if (layer_num == 'd6) mem_wr_wenb = 1;
                        else mem_wr_wenb = 0;
                        output_mem_en = 1;
                        output_state_n = IDLE;
                    end
                endcase
                case(layer_num)
                    1,2,3: begin // padding하는게 추가됨.
                        row_num = 'd100;
                        color_num1 = 'd10302;
                        color_num2 = 'd10302;
                        
                    end
                    5,6: begin
                        row_num = 'd50;
                        color_num1 = 'd2652;
                        color_num2 = 'd5202;
                    end
                    default: begin 
                    end
                endcase
            
                
                if (cnt2_column != row_num) begin // column 수 세기, 모든 column 끝나면 채널 +1
                    if (output_mem_en) begin // output memory에 넣기 (R, G, B 하나씩 나옴)
                        start_point = 'd103 + 'd102*cnt2_column;
                        if (cnt2_row != row_num) begin // row 수 세기, 한 줄 끝나고 줄 바꾸기
                            num2_n = num2 + 1;

                            if (num2 == 'd2) begin
                                num2_n = 0;
                                cnt2_row_n = cnt2_row + 1;
                                mem_wr_addr_n = start_point + cnt2_row + 1; // 옆으로 이동
                            end
                            else if (num2 == 'd0) begin
                                mem_wr_addr_n = mem_wr_addr + color_num1; // r <-> g
                            end
                            else if (num2 == 'd1) begin
                                mem_wr_addr_n = mem_wr_addr + color_num2; // g <-> b
                            end    
                            
                            if (cnt2_row_n == row_num) begin // 다음 column으로 이동
                                cnt2_column_n = cnt2_column + 1;
                                cnt2_row_n = 0;
                                mem_wr_addr_n = start_point + 'd102; 
                            end
                        end       
                    end
                end
                else begin
                    stage5_done = 1;
                    cnt2_column_n = 0;
                end
            end
        endcase
        
    end
   
    assign     wmem_addr_o = wmem_addr;
    assign     wmem_wenb_o = wmem_wenb;
    assign     wmem_cenb_o = wmem_cenb;
    
    assign     memA_addr_o = ((layer_num == 2) | (layer_num == 4) | (layer_num == 6)) ? (((layer_num == 4) & (!stage6_en_n)) ? mem_wr_pad_addr : mem_wr_addr) : mem_rd_addr;
    assign     memA_wenb_o = ((layer_num == 2) | (layer_num == 4) | (layer_num == 6)) ? mem_wr_wenb : mem_rd_wenb;
    assign     memA_cenb_o = ((layer_num == 2) | (layer_num == 4) | (layer_num == 6)) ? mem_wr_cenb : mem_rd_cenb;
    
    assign     memB_addr_o = ((layer_num == 2) | (layer_num == 4) | (layer_num == 6)) ? mem_rd_addr : mem_wr_addr;
    assign     memB_wenb_o = ((layer_num == 2) | (layer_num == 4) | (layer_num == 6)) ? mem_rd_wenb : mem_wr_wenb;
    assign     memB_cenb_o = ((layer_num == 2) | (layer_num == 4) | (layer_num == 6)) ? mem_rd_cenb : mem_wr_cenb;
    
    assign     wei_buff_wren_o = wei_buff_wren_reg;
    assign     wei_buff_rden_o = wei_buff_rden;
  
    assign     in_buf_wren_r = in_buf_wren_reg[0];
    assign     in_buf_wren_g = in_buf_wren_reg[1];
    assign     in_buf_wren_b = in_buf_wren_reg[2];
    assign     in_buf_rden_r = in_buf_rden[0];
    assign     in_buf_rden_g = in_buf_rden[1];
    assign     in_buf_rden_b = in_buf_rden[2];
    
    assign     out_buf_wren_r = out_buf_wren[0];
    assign     out_buf_wren_g = out_buf_wren[1];
    assign     out_buf_wren_b = out_buf_wren[2];
    assign     out_buf_rden_r = out_buf_rden[0];
    assign     out_buf_rden_g = out_buf_rden[1]; 
    assign     out_buf_rden_b = out_buf_rden[2];  

    assign     pe_en_o = pe_en;
    assign     addtree_en_o = addtree_en;
    assign     relu_en_o = relu_en; 
    assign     maxpool_en_o = maxpool_en_reg;
    assign     layer_done_o = stage5_done;
    assign     color_o = color_reg;
    
    assign     is_initial_o = is_initial_reg;
 
endmodule