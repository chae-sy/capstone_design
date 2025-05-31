`timescale 1ns / 1ps
module layer_pipeline(
    input               clk,
    input               rst_n,

    //Weight Memory
    output  wire [8:0]  wmem_addr_o,
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
    output  wire [1:0]  in_buf_wren_o,
    output  wire        in_buf_rden_o,
    output  wire        in_buf_sel_o,
    
    // output buffer
    output  wire [1:0]  out_buf_wren_o,
    output  wire        out_buf_rden_o,
    
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
    
    input   wire [2:0]  layer_num,
    input   wire [5:0]  weight_num,
    input   wire [4:0]  channel,
    input   wire        layer_start,
    output  wire        layer_done_o
);

    localparam              FIRST = 0,
                            weight_change = 1,
                            inputandweight_change = 2;
    
    localparam              IDLE = 0,
                            enable1 = 1,
                            enable2 = 2,
                            enable3 = 3;
                    
    reg     [15:0]          mem_rd_addr, mem_rd_addr_n;
    reg     [15:0]          wmem_addr, wmem_addr_n;
    reg     [7:0]           cnt1_row, cnt1_row_n;
    reg     [7:0]           cnt1_column, cnt1_column_n;
    reg     [1:0]           state, state_n;
    reg     [7:0]           num1, num1_n;
    reg     [7:0]           num_w, num_w_n;
    reg     [1:0]           cnt2, cnt2_n;
    reg     [4:0]           cnt_ch, cnt_ch_n;
    reg                     row_last, row_last_n,
                            final_data, final_data_n;
    reg     [1:0]           stage2_state, stage2_state_n;
    reg     [1:0]           stage3_state, stage3_state_n;
    
    reg                     mem_rd_cenb, mem_rd_wenb,
                            wmem_cenb, wmem_wenb,
                            mem_wr_cenb, mem_wr_wenb;
                            
    reg     [1:0]           in_buf_wren, in_buf_rden,
                            out_buf_wren, out_buf_rden; 
    
    reg                     wei_buff_wren, wei_buff_rden,
    
    reg                     pe_en,
                            addtree_en,
                            relu_en,
                            maxpool_en;                   
    
    reg                     stage2_en, stage2_en_n;
    reg                     stage3_en, stage3_en_n;
    reg                     stage4_en, stage4_en_n;
    reg                     stage5_en, stage5_en_n;
    reg                     stage6_en, stage6_en_n;

    // reg                     stage1_in_valid,
    //                         stage1_weight_valid,
    //                         stage2_valid,
    //                         stage3_valid,
    //                         stage4_valid;
    
    reg                     stage1_in_output,
                            stage1_weight_output,
                            stage2_in_input,
                            stage2_weight_input,
                            stage2_output,
                            stage3_input,
                            stage3_output, 
                            stage4_input,      
                            stage4_output,
                            stage5_input,
                            stage5_output,
                            stage6_input;
                                              
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
    reg                     output_mem_en,
                            row_num,
                            color_num;
    reg     [15:0]          start_point;                                
    
    // 1. mem data fetch (fetch)
    always_ff @( posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            mem_rd_addr <= 'b0;
            wmem_addr <= 'b0;
            cnt1_row <= 'd0;
            cnt1_column <= 'd0;
            state <= FIRST;
            num1 <= 0;
            num_w <= weight_num;
            stage2_en <= 0;
            row_last <= 0;
            final_data <= 0;
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
        end
        
    end
    //channel 동시, (input data 전체) * weight 16개
    always_comb begin
        mem_rd_addr_n = mem_rd_addr;
        wmem_addr_n = wmem_addr; 
        cnt1_row_n = cnt1_row;
        cnt1_column_n = cnt1_column;
        state_n = state;
        num1_n = num1;
        num_w_n = num_w;
        stage1_in_valid = 'b0;
        stage1_weight_valid = 'b0;
        stage1_done = 0;
        stage2_en_n = stage2_en;
    
        mem_rd_cenb         = 1;
        mem_rd_wenb         = 1;
        wmem_wenb           = 1;
        wmem_cenb            = 1;
        wei_buff_wren         = 0;
        in_buf_wren           = 0;
        row_last_n = row_last;
        final_data_n = final_data;
        case(state)
            FIRST: begin // 처음 input_data:27개, weight:9개 불러오기
                stage2_en_n = 'b0;
                // input data 처리
                if (num_w != 0) begin // weight 개수
    
                    mem_rd_cenb = 0;
                    num1_n = num1 + 1;
                    in_buf_wren = 1; 
    
                    if (num1 == 'd8) begin // red -> green
                        mem_rd_addr_n = (((layer_num == 1)|(layer_num == 2)|(layer_num == 3)) ? 'd10302 : 'd5202) + 'd102 * cnt1_column;
                    end
                    else if (num1 == 'd17) begin// green -> blue
                        mem_rd_addr_n = (((layer_num == 1)|(layer_num == 2)|(layer_num == 3)) ? 'd20604 : 'd10404) + 'd102 * cnt1_column;
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
                        num_w_n = num_w - 1;
                        stage2_en_n = 'b1;
                    end
                    if (num_w_n == 0) begin // 모든 weight 다 불러옴. addr 초기화 & input/weight 둘 다 불러오는 state로 이동
                        num_w = weight_num;
                        num1_n = 0;
                        wmem_addr_n = 0;
                        if (row_last) begin
                            state_n = FIRST;
                            row_last_n = 0;
                            if (final_data) begin // 데이터 불러오기 끝
                                num_w = 0;
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
                                wei_buff_wren         = 1;
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
                                        row_last_n = 1;
                                    end
                                    if ((cnt1_column == 99) & (cnt1_row == 99)) begin // 마지막 column & row
                                        final_data_n = 1;
                                        cnt1_column_n = 0;
                                    end
                                end
                                else begin
                                    mem_rd_addr_n = mem_rd_addr + 102;
                                end

                                in_buf_wren     = 1;
                            end
                        end
                        
                    end
                    4, 5: begin // R, B <-> G size different
                        stage2_en_n = 'b0;
                        if (cnt1_column <= 'd49) begin
                            in_buf_wren     = 1;
                            if (cnt1_row <= 'd24) begin // 한 줄 개수 세기 (R, G, B 다 해당)
                                mem_rd_cenb = 0;

                                // weight 불러오기
                                wmem_cenb = 0;
                                wmem_addr_n = wmem_addr + 1;
                                wei_buff_wren         = 1;
                                num1_n = num1 + 1;

                                if (num1 == 'd2) begin // red -> green
                                    mem_rd_addr_n = 'd5202 + 'd102 * cnt1_column + cnt1_row + 2;
                                end
                                else if (num1 == 'd5) begin// green -> blue
                                    mem_rd_addr_n = 'd10404 + 'd102 * cnt1_column + cnt1_row + 2;
                                end
                                else if (num1 == 'd8) begin // 다음 row로 이동 & weight 바꾸기
                                    stage2_en_n = 'b1;
                                    cnt1_row_n = cnt1_row + 1;
                                    if (cnt1_row == 'd24) begin // 이후 row는 Green만 계산
                                        mem_rd_addr_n = 'd5202 + cnt1_column * 102 + cnt1_row + 3;
                                    end
                                    else begin
                                        mem_rd_addr_n = cnt1_column * 102 + cnt1_row + 3;
                                    end
                                    num1_n = 0;
                                    num_w_n = num_w - 1;
                                    state_n = weight_change;
                                    
                                end
                                else begin
                                    mem_rd_addr_n = mem_rd_addr + 102;
                                end
                            end
                            else if (cnt1_row <= 'd49) begin // Green만
                                mem_rd_cenb = 0;

                                // weight 불러오기
                                wmem_cenb = 0;
                                wmem_addr_n = wmem_addr + 1;
                                wei_buff_wren         = 1;
                                num1_n = num1 + 1;
                                if ((num1 > 'd2) | (num1 <= 'd8)) begin // 다음 row로 이동 준비
                                    mem_rd_addr_n = mem_rd_addr;
                                    in_buf_wren     = 0;
                                    mem_rd_cenb = 1;
                                end
                                else begin
                                    mem_rd_addr_n = mem_rd_addr + 102;
                                end
                                
                                if (num1 == 'd2) begin
                                    mem_rd_addr_n = 'd5202 + cnt1_column * 102 + cnt1_row + 3;
                                end
                                else if (num1 == 'd8) begin // 다음 row로 이동 
                                    stage2_en_n = 'b1;
                                    cnt1_row_n = cnt1_row + 1;
                                    num1_n = 0;
                                    num_w_n = num_w - 1;
                                    state_n = weight_change;
                                    if (cnt1_row == 49) begin// 마지막 row
                                        cnt1_row_n = 0;
                                        cnt1_column_n = cnt1_column + 1;
                                        row_last_n = 1;
                                    end
                                    if ((cnt1_column == 49) & (cnt1_row == 49)) begin // 마지막 column & row
                                        final_data_n = 1;
                                        cnt1_column_n = 0;
                                    end
                                end
                            end
                        end
                    end
                endcase
            end
        endcase
    end

    /*
    always_ff @(posedge clk) begin
        if (stage1_in_valid) begin
            stage2_in_input <= stage1_in_output;
        end
        if (stage1_weight_valid) begin
            stage2_weight_input <= stage1_weight_output;
        end
    end                        
    */

    // 2. compute (PE_array )

    always_comb begin
        pe_en = 0;
        stage2_valid = 0;
        wei_buff_rden = 0;
        in_buf_rden = 0;
        cnt2_n = cnt2;
        stage3_en_n = stage3_en;
        case(stage2_state)
            IDLE: begin
                if (stage2_en) begin
                    wei_buff_rden = 1;
                    in_buf_rden = 1;
                    pe_en = 1;
                    stage2_state_n = enable1;
                end
            end
            enable1: begin
                wei_buff_rden = 1;
                in_buf_rden = 1;
                pe_en = 1;
                if (pe_done_i) begin
                    stage3_en_n = 1;
                    if (stage2_en_n) begin
                        stage2_state_n = enable1;
                    end
                    else begin
                        stage2_state_n = IDLE;
                    end
                end
                else begin
                    stage3_en_n = 0;
                end
            end
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            stage3_input <= 0;
            stage3_en <= 0;
            stage2_done <= 0;
        end else begin
            if (stage2_valid) begin
                stage3_input <= stage2_output;
            end
            stage3_en <= stage3_en_n;
        end
        if (pe_done_i) stage2_done <= stage1_done;
    end
    
    // 3. add tree

    always_comb begin
        addtree_en = 0;
        stage4_en_n = stage4_en;
        case(stage3_state)
            IDLE: begin
                if (stage3_en) begin
                    addtree_en = 1;
                    stage3_state_n = enable1;
                end
            end
            enable1: begin
                addtree_en = 1;
                if (addtree_done_i) begin
                    stage4_en_n = 1;
                end
                else begin
                    stage4_en_n = 0;
                end
            end
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            stage4_input <= 0;
            stage4_en <= 0;
            stage3_done <= 0;
        end else begin
            if (stage3_valid) begin
                stage4_input <= stage3_output;
            end
            stage4_en <= stage4_en_n;
        end
        if (addtree_done_i) stage3_done <= stage2_done;
    end
    


    // 4. activate (ReLU, bias)
    
    //1, 2, 3, 4 ReLU exist
    
    always_comb begin
        relu_en   = 0;
        stage5_en_n = stage5_en;
        case (layer_num)
            1, 2, 3, 4: begin
                if (stage4_en) begin
                    relu_en   = 1;
                    if (relu_done_i) begin
                        stage5_en_n = 1;
                    end
                    else begin
                        stage5_en_n = 0;
                    end
                end
            end
            default: begin
                stage5_en_n = 0;
            end
        endcase
    end

    always_ff @( posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            stage5_input <= 0;
            stage5_en <= 0;
            stage4_done <= 0;
        end else begin
            case (layer_num)
                1, 2, 3, 4: begin
                    if (stage4_valid) begin
                        stage5_input <= stage4_output;
                    end
                    if (relu_done_i) stage4_done <= stage3_done;
                end
                default: begin
                   
                end
            endcase
            stage5_en <= stage5_en_n;
        end
    end

    // 5. maxpool exist
    
    always_comb begin
        maxpool_en   = 0;
        stage5_valid = 0;
        stage6_en_n = stage6_en;
        case (layer_num)
            3: begin
                if (stage5_en) begin
                    maxpool_en   = 1;
                    stage5_valid = 1;
                    if (maxpool_done_i) begin
                        stage6_en_n = 1;
                    end
                    else begin
                        stage6_en_n = 0;
                    end
                end
            end
            default: begin
                stage5_valid = 0;
                stage6_en_n = 0;
            end
        endcase
    end
         
    always_ff @( posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            stage6_input <= 0;
            stage6_en <= 0;
            stage5_done <= 0;
        end else begin
            case (layer_num)
                3: begin
                    if (stage5_valid) begin
                        stage6_input <= stage5_output;
                    end
                    if (maxpool_done_i) stage5_done <= stage4_done;
                end
                default: begin
                   
                end
            endcase
            stage6_en <= stage6_en_n;
        end
    end
    
    // 6. write back (out_buff -> mem)
    always_ff @( posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            mem_wr_addr <= 'd103; // 시작부분
            mem_wr_pad_addr <= 'd128; // 시작부분
            cnt2_row <= 'd0;
            cnt2_column <= 'd0;
            cnt_ch <= 'd0;
            output_state <= IDLE;
            num2 <= 0;
            num_w <= weight_num; // 채널 구분
            num_p <= 0;
        end else begin
            mem_wr_addr <= mem_wr_addr_n;
            mem_wr_pad_addr <= mem_wr_pad_addr_n;
            cnt2_row <= cnt2_row_n;
            cnt2_column <= cnt2_column_n;
            cnt_ch <= cnt_ch_n;
            output_state <= output_state_n;
            num2 <= num2_n;
            num_w <= num_w_n;
            num_p <= num_p;
        end
        
    end
    
    always_comb begin
        mem_wr_cenb = 1;
        mem_wr_wenb = 1;
        mem_wr_addr_n = mem_wr_addr;
        mem_wr_pad_addr_n = mem_wr_pad_addr;
    
        cnt2_row_n = cnt2_row;
        cnt2_column_n = cnt2_column;
        cnt_ch_n = cnt_ch;
        output_state_n = output_state;
        num2_n = num2;
        num_w_n = num_w;
        num_p_n = num_p;
        stage6_done = 0;
    
        out_buf_rden = 0;
        out_buf_wren = 0;
        output_mem_en = 0;
        case (output_state)
            IDLE: begin
                if (stage4_en | stage5_en | stage6_en) begin // 연산 끝 -> output buffer에 저장
                    out_buf_wren = 1;
                    if (cnt_ch == (channel-1)) begin //모든 채널 다 모이기까지 기다림
                        output_state_n = enable1;
                        cnt_ch_n = 0;
                    end
                    else begin
                        cnt_ch_n = cnt_ch + 1;
                    end
                end
                if ((layer_num == 3) && (!stage5_en))begin // padding하기
                    mem_wr_cenb = 0;
                    mem_wr_wenb = 0;
                    out_data = 'd0;
                    mem_wr_pad_addr_n = mem_wr_pad_addr + 102;
                    num_p_n = num_p + 1;
                    if (num_p == 49) begin
                        mem_wr_pad_addr_n = 'd5253;
                    end
                    else if (num_p == 101) begin
                        mem_wr_pad_addr_n = 'd10430;
                    end
                    else if (num_p == 153) begin
                        mem_wr_pad_addr_n = 'd5202;
                    end
                    else if (num_p == 205) begin
                        mem_wr_pad_addr_n = 'd10404;
                    end
                    else if (num_p == 257) begin
                        mem_wr_pad_addr_n = 'd15606;
                    end
                    else if (num_p == 284) begin
                        mem_wr_pad_addr_n = mem_wr_pad_addr;
                        num_p_n = num_p;
                    end
                end
            end
            enable1: begin // red
                out_buf_rden = 1;
                mem_wr_cenb = 0;
                mem_wr_wenb = 0;
                output_mem_en = 1;
                output_state_n = enable2;
            end
            enable2: begin // green
                out_buf_rden = 1;
                mem_wr_cenb = 0;
                mem_wr_wenb = 0;
                output_mem_en = 1;
                output_state_n = enable3;
            end
            enable3: begin // blue
                out_buf_rden = 1;
                mem_wr_cenb = 0;
                mem_wr_wenb = 0;
                output_mem_en = 1;
                output_state_n = IDLE;
            end
        endcase
        case(layer_num)
            3: begin // padding하는게 추가됨.
                row_num = 'd50;
                color_num = 'd5202;
                if (stage6_en) begin 
                    out_data = stage6_input;               
                end
            end
            5: begin
                row_num = 'd50;
                color_num = 'd5202;
                if (stage4_en) begin 
                    out_data = stage4_input;
                end
            end
            default: begin 
                row_num = (layer_num == 4) ? 'd50 : 'd100;
                color_num = (layer_num == 4) ? 'd5202 : 'd10302;
                if (stage5_en) begin 
                    out_data = stage5_input;
                end
            end
        endcase
    
        if (output_mem_en) begin // output memory에 넣기 (R, G, B 하나씩 나옴)
            if (cnt2_column != row_num) begin // column 수 세기, 모든 column 끝나면 채널 +1
                start_point = 'd103 + 'd102*cnt2_column;
                if (cnt2_row != row_num) begin // row 수 세기, 한 줄 끝나고 줄 바꾸기
                    num2_n = num2 + 1;

                    if (num2 == 'd2) begin
                        num2_n = 0;
                        cnt2_row_n = cnt2_row + 1;
                        mem_wr_addr_n = start_point + cnt2_row + 1; // 옆으로 이동
                    end
                    else begin
                        mem_wr_addr_n = mem_wr_addr + color_num; // 색깔들끼리 차이
                    end

                    if (cnt2_row_n == row_num) begin // 다음 column으로 이동
                        cnt2_column_n = cnt2_column + 1;
                        cnt2_row_n = 0;
                    end
                end

                if (cnt2_column_n == row_num) begin
                    cnt2_column_n = 0;
                    num_w_n = num_w - 1;
                end           
            end
            else begin
                stage6_done = 1;
            end
        end
    end
   
    assign     wmem_addr_o = wmem_addr;
    assign     wmem_wenb_o = wmem_wenb;
    assign     wmem_wenb_o = wmem_wenb;
    
    assign     memA_addr_o = ((layer_num == 2) | (layer_num == 4)) ? mem_wr_addr : mem_rd_addr;
    assign     memA_wenb_o = ((layer_num == 2) | (layer_num == 4)) ? mem_wr_wenb : mem_rd_wenb;
    assign     memA_cenb_o = ((layer_num == 2) | (layer_num == 4)) ? mem_wr_cenb : mem_rd_cenb;
    
    assign     memB_addr_o = ((layer_num == 2) | (layer_num == 4)) ? mem_rd_addr : (((layer_num == 3) & (!output_mem_en)) ? mem_wr_pad_addr : mem_wr_addr);
    assign     memB_wenb_o = ((layer_num == 2) | (layer_num == 4)) ? mem_rd_wenb : mem_wr_wenb;
    assign     memB_cenb_o = ((layer_num == 2) | (layer_num == 4)) ? mem_rd_cenb : mem_wr_cenb;
    
    assign     wei_buff_wren_o = wei_buff_wren;
    assign     wei_buff_rden_o = wei_buff_rden;
  
    assign     in_buf_wren_o = in_buf_wren;
    assign     in_buf_rden_o = in_buf_rden;
    
    assign     out_buf_wren_o = out_buf_wren;
    assign     out_buf_rden_o = out_buf_rden; 

    assign     pe_en_o = pe_en;
    assign     addtree_en_o = addtree_en;
    assign     relu_en_o = relu_en; 
    assign     maxpool_en_o = maxpool_en;
    assign     layer_done_o = stage6_done;

endmodule
