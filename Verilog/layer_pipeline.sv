`timescale 1ns / 1ps
module layer_pipeline#(

)(
    input               rst_n,
    input               clk,

    input               weight_done,
    input               FE_done,

    //Weight Memory
    output  wire        [WMEM_BIT_LEN-1:0] wmem_addr_o,
    output  wire        wmem_wenb_o,
    output  wire        wmem_enb_o,
    /*//Data Memory
    output  reg         [IM_BIT_LEN-1:0] mem_addr_o,
    output  reg         mem_wenb_o,
    output  reg         mem_enb_o,*/
    //Memory A 
    output  wire [MA_BIT_LEN-1:0] memA_addr_o,
    output  wire        memA_wenb_o,
    output  wire        memA_cenb_o,
    //Memory B
    output  wire [MB_BIT_LEN-1:0] memB_addr_o,
    output  wire        memB_wenb_o,
    output  wire        memB_cenb_o,

    //Weight Buffer
    output  wire        wei_buff_en_o,
    
    //PE array
    output  wire        pe_en_o,
    output  wire        pe_rst_o,

    // ReLU
    output  wire        relu_en_o,

    //Output Buffer
    output  wire [31:0] out_buf_en_o,
    output  wire        out_buf_sel_o,
    output  wire        out_buf_rst_o,
    //maxpool
    output  wire        pool_sel_o, //1 at layer 3

    //output  reg     comp_start_o,
    //output reg rf_sel_o,
    output  reg [2:0]   layer_state,
    output  wire        done_o

)

logic [15:0] wmem_addr, wmem_addr_n;
logic [15:0] memA_addr, memA_addr_n;
logic [2:0]  layer_num;
logic        layer_en;
logic [7:0]  out_buf_data;
logic [7:0]  PE_result_n;



/////////////////////////////
/*
할일:
1. input, output port 정리
2. addr 계산
3. 신호 사이즈 맞추기
4. 문제점: buffer read/write 동시에 일어남 (posedge -> read, negaedge -> write ...)
아니면 data forwarding 방법

<buffer 안에 들어가야함.>
always_ff @(posedge clk) begin
  if (wr_en && rd_en && (wr_addr == rd_addr)) begin
    rd_data_reg <= wr_data;           // forwarding
  end
  else if (rd_en) begin
    rd_data_reg <= mem[rd_addr];      // 일반 읽기
  end

  if (wr_en)
    mem[wr_addr] <= wr_data;          // 쓰기
end

assign rd_data = rd_data_reg;

*/
//////////////////////////////
localparam      INPUT_HORIZ         = 5;
localparam      INPUT_VERT          = 3;
localparam      WEIGHT_HORIZ        = 3;
localparam      WEIGHT_VERT         = 3;
localparam      STRIDE_HORIZ        = 3;
localparam      STRIDE_VERT         = 3;
localparam      INPUT_BIT_LEN       = 8;
localparam      WEIGHT_BIT_LEN      = 8;
localparam      NEXT_WEIGHT_VERT    = 16;

/*controller
if(layer_done) begin
    data_num_n          = 'd10404; // cov2 102*102
    channel             = 'd16;
    weight_num_n        = 'd16;
    state_n             = S_Layer2;
    layer_num_n         = 3'd2;
end
*/
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
    end else begin
        mem_rd_addr <= mem_rd_addr_n;
        wmem_addr <= wmem_addr_n;
        cnt1_row <= cnt1_row_n;
        cnt1_column <= cnt1_column_n;
        state <= state_n;
        num1 <= num1_n;
        num_w <= num_w_n;
        stage2_en <= stage2_en_n;
    end
    
end
//channel 동시, (input data 전체) * weight 16개
always_comb @(*) begin
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
    wei_buff_wren         = 0;
    in_buf_wren           = 0;
    case(state)
        FIRST: begin
            stage2_en_n = 'b0;
            // input data 처리
            if (num_w != 0) begin // weight 개수

                mem_rd_cenb = 0;
                num1_n = num1 + 1;
                stage1_in_valid = 'b1;
                in_buf_wren           = 1; 

                if (num1 == 'd8) begin // green
                    mem_rd_addr_n = (((layer_num == 1)|(layer_num == 2)|(layer_num == 3)) ? 'd10302 : 'd5202) + 'd102 * cnt1_column;
                end
                else if (num1 == 'd17) begin// blue
                    mem_rd_addr_n = (((layer_num == 1)|(layer_num == 2)|(layer_num == 3)) ? 'd20604 : 'd10404) + 'd102 * cnt1_column;
                end
                else if (num1 == 'd26) begin // 맨 마지막. 27개 불러옴.
                    stage2_en_n = 'b1;
                    mem_rd_addr_n = cnt1_column * 102 + 3;
                    cnt1_row_n = cnt1_row + 1;
                    state_n = ELSE;
                    num1_n = 0;
                end
                else if ((num1 % 3) == 'd2) begin // 줄 바꾸기
                    mem_rd_addr_n = mem_rd_addr + 100;
                end
                else begin
                    mem_rd_addr_n = mem_rd_addr + 1;
                end

                if (num1 >= 'd9) begin
                    wei_buff_wren = 0;
                end
                else begin
                    wmem_addr_n = wmem_addr + 1;
                    wei_buff_wren         = 1;
                    stage1_weight_valid = 'b1;
                end                    

            end
            else begin
                stage1_done = 1;
            end
        end
        ELSE: begin
            case (layer_num)
                1, 2, 3: begin
                    stage2_en_n = 'b0;
                    if (cnt1_column <= 'd99) begin
                        if (cnt1_row != 99) begin // 한 줄 개수 세기 (99 = 마지막)
                            mem_rd_cenb = 0;
                        
                            if (num1 == 'd2) begin // green
                                mem_rd_addr_n = 'd10302 + 'd102 * cnt1_column + cnt1_row + 2;
                            end
                            else if (num1 == 'd5) begin// blue
                                mem_rd_addr_n = 'd20604 + 'd102 * cnt1_column + cnt1_row + 2;
                            end
                            else if (num1 == 'd8) begin // 다음 row로 이동
                                mem_rd_addr_n = cnt1_column * 102 + cnt1_row + 3;
                                stage2_en_n = 'b1;
                                cnt1_row_n = cnt1_row + 1;
                                num1_n = 0;
                            end
                            else begin
                                mem_rd_addr_n = mem_rd_addr + 102;
                            end
                            num1_n = num1 + 1;
                        end
                        else begin // 한 줄 마지막번째 cnt1_column ++
                            state_n = FIRST;
                            if (cnt1_column == 'd99) begin // 완전 맨 마지막 (new weight 불러오기)
                                num_w_n = num_w - 1;
                                wmem_addr_n = 0;
                                cnt1_column_n = 0;
                            end
                            else begin
                                cnt1_column_n = cnt1_column + 1;
                            end
                            cnt1_row_n = 0;
                            num1_n = 0;
                        end
                        
                        in_buf_wren     = 1;
                        stage1_in_valid = 'b1;
                    end
                    
                end
                4, 5: begin // R, B <-> G size different
                    stage2_en_n = 'b0;
                    if (cnt1_column <= 'd49) begin
                        in_buf_wren     = 1;
                        stage1_in_valid = 'b1;
                        if (cnt1_row <= 'd24) begin // 한 줄 개수 세기 (R, G, B 다 해당)
                            mem_rd_cenb = 0;
                            num1_n = num1 + 1;
                            if (num1 == 'd2) begin // green
                                mem_rd_addr_n = 'd5202 + 'd102 * cnt1_column + cnt1_row + 2;
                            end
                            else if (num1 == 'd5) begin// blue
                                mem_rd_addr_n = 'd10404 + 'd102 * cnt1_column + cnt1_row + 2;
                            end
                            else if (num1 == 'd8) begin // 다음 row로 이동
                                stage2_en_n = 'b1;
                                cnt1_row_n = cnt1_row + 1;
                                if (cnt1_row == 'd24) begin // 이후 row는 Green만 계산
                                    mem_rd_addr_n = 'd5202 + cnt1_column * 102 + cnt1_row + 3;
                                end
                                else begin
                                    mem_rd_addr_n = cnt1_column * 102 + cnt1_row + 3;
                                end
                                num1_n = 0;
                            end
                            else begin
                                mem_rd_addr_n = mem_rd_addr + 102;
                            end
                        end
                        else if (cnt1_row < 'd49) begin // Green만
                            if ((num1 > 'd2) | (num1 <= 'd8)) begin // 다음 row로 이동 준비
                                mem_rd_addr_n = mem_rd_addr;
                                in_buf_wren     = 0;
                                stage1_in_valid = 'b0;
                            end
                            else begin
                                mem_rd_addr_n = mem_rd_addr + 102;
                            end
                            
                            if (num1 == 'd2) begin
                                mem_rd_addr_n = 'd5202 + cnt1_column * 102 + cnt1_row + 3;
                            end
                            else if (num1 == 'd8) begin // 다음 row로 이동 ( PE 시간이랑 맞추기 위해 )
                                stage2_en_n = 'b1;
                                cnt1_row_n = cnt1_row + 1;
                                num1_n = 0;
                            end
                            else begin
                                num1_n = num1 + 1;
                            end                            
                        end
                        else begin // 한 줄 마지막번째 cnt1_column ++
                            state_n = FIRST;
                            if (cnt1_column == 'd49) begin // 완전 맨 마지막 (new weight 불러오기)
                                num_w_n = num_w - 1;
                                wmem_addr_n = 0;
                                cnt1_column_n = 0;
                            end
                            else begin
                                cnt1_column_n = cnt1_column + 1;
                            end
                            cnt1_row_n = 0;
                            num1_n = 0;
                        end
                    end
                end
            endcase
           
        end
    endcase
end

always_ff @(posedge clk) begin
    if (stage1_in_valid) begin
        stage2_in_input <= stage1_in_output;
    end
    if (stage1_weight_valid) begin
        stage2_weight_input <= stage1_weight_output;
    end
end
  

// 2. compute (PE_array )


always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n | layer_start) begin
        stage3_in_input <= 0;
        stage3_en <= 0;
        stage2_done <= 0;
    end else begin
        if (stage2_valid) begin
            stage3_in_input <= stage2_in_output;
        end
        stage3_en <= stage3_en_n;
    end
    if (PE_done) stage2_done <= stage1_done;
end
always_comb @(*) begin
    pe_en       = 0;
    stage2_valid = 0;
    wei_buff_rden         = 0;
    in_buf_rden           = 0;
    cnt2_n = cnt2;
    stage3_en_n = stage3_en;
    if (stage2_en) begin
        wei_buff_rden         = 1;
        in_buf_rden           = 1;
        pe_en   = 1;
        pe_data_in = rd_patch[];
        pe_weight_in = rd_weight;
        cnt2_n = cnt2 + 1;
        if (PE_done) begin
            stage2_valid = 1;
            stage3_en_n = 1;
            cnt2_n = 0;
        end
        else begin
            stage3_en_n = 0;
        end
    end
end

// 3. activate (ReLU, bias)

//1, 2, 3, 4 ReLU exist

always_ff @( posedge clk or negedge rst_n) begin
    if (!rst_n | layer_start) begin
        stage4_in_input <= 0;
        stage4_en <= 0;
        stage3_done <= 0;
    end else begin
        case (layer_num)
            1, 2, 3, 4: begin
                if (stage3_valid) begin
                    stage4_in_input <= stage3_in_output;
                end
                if (relu_done) stage3_done <= stage2_done;
            end
            default: begin
               
            end
        endcase
        stage4_en <= stage4_en_n;
    end
    
end

always_comb @(*) begin
    relu_en   = 0;
    stage3_valid = 0;
    stage4_en_n = stage4_en;
    case (layer_num)
        1, 2, 3, 4: begin
            if (stage3_en) begin
                relu_en   = 1;
                stage3_valid = 1;
                if (relu_done) begin
                    stage4_en_n = 1;
                end
                else begin
                    stage4_en_n = 0;
                end
            end
        end
        default: begin
            stage3_valid = 0;
            stage4_en_n = 0;
        end
    endcase
end
// 4. maxpool exist

always_ff @( posedge clk or negedge rst_n) begin
    if (!rst_n | layer_start) begin
        stage5_in_input <= 0;
        stage5_en <= 0;
        stage4_done <= 0;
    end else begin
        case (layer_num)
            3: begin
                if (stage4_valid) begin
                    stage5_in_input <= stage4_in_output;
                end
                if (maxpool_done) stage4_done <= stage3_done;
            end
            default: begin
               
            end
        endcase
        stage5_en <= stage5_en_n;
    end
    
end

always_comb @(*) begin
    maxpool_en   = 0;
    stage4_valid = 0;
    stage5_en_n = stage5_en;
    case (layer_num)
        3: begin
            if (stage4_en) begin
                maxpool_en   = 1;
                stage4_valid = 1;
                if (maxpool_done) begin
                    stage5_en_n = 1;
                end
                else begin
                    stage5_en_n = 0;
                end
            end
        end
        default: begin
            stage4_valid = 0;
            stage5_en_n = 0;
        end
    endcase
end


// 5. write back (out_buff -> mem)
always_ff @( posedge clk or negedge rst_n) begin
    if (!rst_n | layer_start) begin
        mem_wr_addr <= 'd103; // 시작부분
        mem_wr_pad_addr <= 'd128; // 시작부분
        cnt2_row <= 'd0;
        cnt2_column <= 'd0;
        output_state <= FIRST;
        num2 <= 0;
        num_w <= weight_num; // 채널 구분
        num_p <= 0;
        stage2_en <= 0;
    end else begin
        mem_wr_addr <= mem_wr_addr_n;
        mem_wr_pad_addr <= mem_wr_pad_addr_n;
        cnt2_row <= cnt2_row_n;
        cnt2_column <= cnt2_column_n;
        output_state <= output_state_n;
        num2 <= num2_n;
        num_w <= num_w_n;
        num_p <= num_p;
        stage2_en <= stage2_en_n;
    end
    
end

always_comb @(*) begin
    mem_wr_cenb = 1;
    mem_wr_wenb = 1;
    mem_wr_addr_n = mem_wr_addr;
    mem_wr_pad_addr_n = mem_wr_pad_addr;

    cnt2_row_n = cnt2_row;
    cnt2_column_n = cnt2_column;
    output_state_n = output_state;
    num2_n = num2;
    num_w_n = num_w;
    num_p_n = num_p;
    stage5_done = 0;

    out_buf_rden           = 0;
    output_mem_en = 0;
    case (output_state)
        IDLE: begin
            if (stage3_en | stage4_en | stage5_en) begin
                mem_wr_cenb = 0;
                mem_wr_wenb = 0;
                output_mem_en = 1;
                output_state_n = enable1;
            end
            if ((layer_num == 3) && (!stage5_en))begin // padding하기
                mem_wr_cenb = 0;
                mem_wr_wenb = 0;
                output = 'd0;
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
        enable1: begin
            mem_wr_cenb = 0;
            mem_wr_wenb = 0;
            output_mem_en = 1;
            output_state_n = enable2;
        end
        enable2: begin
            mem_wr_cenb = 0;
            mem_wr_wenb = 0;
            output_mem_en = 1;
            output_state_n = enable3;
        end
        enable3: begin
            mem_wr_cenb = 0;
            mem_wr_wenb = 0;
            output_mem_en = 0;
            output_state_n = IDLE;
        end
    endcase
    case(layer_num)
        3: begin // padding하는게 추가됨.
            row_num = 'd50;
            color_num = 'd5202;
            if (stage5_en) begin 
                output = stage5_in_input;               
            end
        end
        5: begin
            row_num = 'd50;
            color_num = 'd5202;
            if (stage3_en) begin 
                output = stage3_in_input;
            end
        end
        default: begin 
            row_num = (layer_num == 4) ? 'd50 : 'd100;
            color_num = (layer_num == 4) ? 'd5202 : 'd10302;
            if (stage4_en) begin 
                output = stage4_in_input;
            end
        end
    endcase

    if (output_mem_en) begin // output memory에 넣기 (R, G, B 하나씩 나옴)
        if (num_w != 0) begin // weight 개수 = channel 개수, 모든 채널 끝나면 해당 layer done = 1
            out_buf_rden = 1; 
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

                    if (cnt2_row_n == row_num) begin
                        cnt2_column_n = cnt2_column + 1;
                        cnt2_row_n = 0;
                    end
                end

                if (cnt2_column_n == row_num) begin
                    cnt2_column_n = 0;
                    num_w_n = num_w - 1;
                end           
            end
        end
        else begin
            stage5_done = 1;
        end
    end
end


endmodule
