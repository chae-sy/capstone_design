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
controller 
if(layer_done) begin
    data_num_n          = 'd10404; // cov2 102*102
    channel             = 'd16;
    weight_num_n        = 'd16;
    state_n             = S_Layer2;
    layer_num_n         = 3'd2;
end

// 1. mem data fetch (fetch)
always_ff @( posedge clk or negedge rst_n) begin
    if (!rst_n && layer_start) begin
        mem_rd_addr <= 'b0;
        wmem_addr <= 'b0;
        cnt1 <= data_num;
        state <= FIRST;
        num1 <= 0;
        num_w <= weight_num;
        stage2_en <= 0;
    end else begin
        mem_rd_addr <= mem_rd_addr_n;
        wmem_addr <= wmem_addr_n;
        cnt1 <= cnt1_n;
        state <= state_n;
        num1 <= num1_n;
        num_w <= num_w;
        stage2_en <= stage2_en_n;
    end
    
end
//channel 동시, (input data 전체) * weight 16개
always_comb @(*) begin
    mem_rd_addr_n = mem_rd_addr;
    wmem_addr_n = wmem_addr; 
    cnt1_n = cnt1; 
    state_n = state;
    num1_n = num1;
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
                if (cnt1 != 0) begin // 한 conv 진행할 때 전체 data 수
                    mem_rd_cenb = 0;
                    mem_rd_addr_n = mem_rd_addr + 1;
                    cnt1_n = cnt1 - 1;
                    num1_n = num1 + 1;
                    stage1_in_valid = 'b1;
                    in_buf_wren           = 1;
                    if (num1 >= 'd9) begin
                        wei_buff_wren         = 0;
                    end
                    else begin
                        wmem_addr_n = wmem_addr + 1;
                        wei_buff_wren         = 1;
                        stage1_weight_valid = 'b1;
                    end
                    if (num1 == 'd14) begin // 몇 cycle 후인지에 따라 그 다음 단계 진행
                        stage2_en_n = 'b1;
                        state_n = ELSE;
                        num1_n = 0;
                    end
                    
                end
            end
            else begin
                stage1_done = 1;
            end
        end
        ELSE: begin
            stage2_en_n = 'b0;
            if (cnt1 != 0) begin // 한 conv 진행할 때 전체 data 수
                mem_rd_cenb = 0;
                mem_rd_addr_n = mem_rd_addr + 1;
                cnt1_n = cnt1 - 1;
                num1_n = num1 + 1;
                in_buf_wren     = 1;
                stage1_in_valid = 'b1;
                if (num1 == 'd4) begin // 몇 cycle 후인지에 따라 그 다음 단계 진행
                    stage2_en_n = 'b1;
                    num1_n = 0;
                end
            end
            else begin // 전체 data 다 하면
                state_n = FIRST;
                cnt1_n = 0;
                num1_n = 0;
                num_w_n = num_w - 1;
                wmem_addr_n = 0; 
            end
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
    // if ((layer_num == 2) | (layer_num == 4)) begin
    //     // cal addr -> row, col, channel
    //     for ( int i = 0; i < WEIGHT_HORIZ*WEIGHT_VERT; i = i+1 ) begin // 3X3
    //         for ( int j = 0; j < channel; j = j+1) begin // 3x3xchannel 
    //             mem_rd_addr_n = i;
    //             mem_rd_ch_n = j;
                
    //         end
    //     end
    //     for ( int i = 0; i < INPUT_VERT; i = i+1 ) begin // 3
    //         for ( int j = 0; j < INPUT_HORIZ; j = j+1) begin // 5
    //             for ( int k = 0; k < channel; k = k+1) begin // channel
    //                 mem_rd_addr_n = i*j;
    //                 mem_rd_ch_n = j;
    //             end
    //         end
    //     end
    //end
   


/* mem
module SRAM_W32_A64 (  // Data Storage
	
	input		CLK,
	input		CEB,
	input		WEB,
	input	[5:0]	A,
    input   [4:0]   ch,
	input	[31:0]	D,
	output	[31:0]	Q
);*/ 
/* input_buffer
module in_buf (
    input clk,
    input rst,

    // write interface
    input wren, // write enable (1일때 wr_data 저장)
    input [7:0] wr_data, // mem에서 Q 랑 연결
    input [6:0] wr_row,   // up to 127 controller에서 나온 신호
    input [6:0] wr_col,   // up to 127 controller에서 나온 신호
    input [3:0] wr_chn,   // 0~15 controller에서 나온 신호

    // read interface
    input [6:0] start_row,
    input [6:0] start_col,
    input rden, // read enable (1일때 데이터 읽어옴)
    output reg [7:0] rd_patch [0:2][0:4][0:15]  // 3x5x16 patch
);*/
/* weight buffer
module w_buf (
    input clk,
    input rst,
    input wren, // write enable 신호
    input [7:0] wr_data, // 메모리에서 가져온 weight 데이터 (8비트) // mem에서 Q 랑 연결
    input [4:0] wr_row,  // 쓰기용 행 주소 (0~2) controller에서 나온 신호
    input [4:0] wr_col,  // 쓰기용 열 주소 (0~2) controller에서 나온 신호
    input [3:0] wr_chn,  // 쓰기용 채널 번호 (0~15) controller에서 나온 신호

    input rden, // read enable 신호
    output reg [7:0] rd_weight [0:2][0:2][0:15] // 읽어낸 3x3x16 weight 데이터
);*/

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


/*
module systolic_array #(
    parameter ARRAY_WIDTH  = 3,
    parameter ARRAY_HEIGHT = 3,
    parameter DATA_WIDTH   = 8,
    parameter COLOR_WIDTH  = 3
)(
    input  wire                             clk,
    input  wire                             rstb,
    input  wire                             enable,
    input  wire [DATA_WIDTH-1:0]            data_in    [0:ARRAY_HEIGHT-1][0:COLOR_WIDTH-1],
    input  wire [DATA_WIDTH-1:0]            weight_in  [0:ARRAY_WIDTH-1],
    output wire [2*DATA_WIDTH-1:0]          result_out [0:COLOR_WIDTH-1]
);*/
/* reLU
module relu_stream_with_last #(
  parameter DATA_WIDTH = 8,
  parameter CH         = 16
)(
  input  wire                         clk,
  input  wire                         rstb,
  // hand­shake + 프레임 경계 표시
  input  wire                         valid_in,
  input  wire                         last_in,
  input  wire signed [DATA_WIDTH-1:0] in_data [0:CH-1],

  output reg                          valid_out,
  output reg                          last_out,
  output reg signed [DATA_WIDTH-1:0]  out_data[0:CH-1]
);
*/
// 3. activate (ReLU, bias)

//1, 2, 3, 4 ReLU exist

always_ff @( posedge clk or negedge rst_n) begin
    if (!rst_n | layer_start) begin
        stage4_in_input <= 0;
        stage4_en <= 0;
        stage4_done <= 0;
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
        stage5_done <= 0;
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
                if (relu_done) begin
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
/*
module maxpool#(
    parameter DATA_WIDTH = 8,
    parameter CHANNELS = 16,
    parameter LINEBUF_RED_BLUE_SIZE = 8,
    parameter LINEBUF_GREEN_SIZE = 4
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         maxpool_en,
    input  wire [1:0]                   color, // r=0 (4x2), g=1 (4x1), b=2 (4x2)
    input  wire signed [DATA_WIDTH-1:0] in_data   [0:CHANNELS-1],
    output wire                         maxpool_done_o,
    output wire signed [DATA_WIDTH-1:0] out_data_o[0:CHANNELS-1]
);
*/
/* out buffer
module out_buf (
    input clk,
    input rst,

    // write interface
    input wren, // write enable
    input [7:0] wr_data, // 저장할 데이터
    input [6:0] wr_row,  // 저장할 행 번호 (0~127)
    input [6:0] wr_col,  // 저장할 열 번호 (0~127)
    input [3:0] wr_chn,  // 저장할 채널 번호 (0~15)

    // read interface
    input rden, // read enable
    input [6:0] rd_row,  // 읽을 행 번호
    input [6:0] rd_col,  // 읽을 열 번호
    input [3:0] rd_chn,  // 읽을 채널 번호
    output reg [7:0] rd_data // 읽은 데이터 출력
);*/


// 5. write back (out_buff -> mem)

always_ff @( posedge clk or negedge rst_n) begin
    if (!rst_n | layer_start) begin
        mem_wr_addr <= 'b0;

    end else begin
        mem_wr_addr <= mem_wr_addr_n;
    end
end

always_comb @(*) begin
    mem_wr_cenb = 1;
    mem_wr_wenb = 1;
    mem_wr_addr_n = mem_wr_addr;
    case (layer_num)
        3: begin
            if (stage5_en) begin // if channel 당 따로 memory 존재한다면 병렬처리로 인해 output buffer필요없을듯
                mem_wr_cenb = 0;
                mem_wr_wenb = 0;
                mem_wr_addr_n = mem_wr_addr + 1;
                output = stage5_in_input;
            end
        end
        5: begin
            if (stage3_en) begin // if channel 당 따로 memory 존재한다면 병렬처리로 인해 output buffer필요없을듯
                mem_wr_cenb = 0;
                mem_wr_wenb = 0;
                mem_wr_addr_n = mem_wr_addr + 1;
                output = stage3_in_input;
            end
        end
        default: begin 
            if (stage4_en) begin // if channel 당 따로 memory 존재한다면 병렬처리로 인해 output buffer필요없을듯
                mem_wr_cenb = 0;
                mem_wr_wenb = 0;
                mem_wr_addr_n = mem_wr_addr + 1;
                output = stage4_in_input;
            end
        end
    endcase

end
/* mem
module SRAM_W32_A64 (  // Data Storage
	
	input		CLK,
	input		CEB,
	input		WEB,
	input	[5:0]	A,
	input	[31:0]	D,
	output	[31:0]	Q
);*/
endmodule
