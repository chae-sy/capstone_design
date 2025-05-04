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
*/
//////////////////////////////


// 1. mem data fetch (mem -> input_buffer, weight_buffer)
always_ff @( posedge clk ) begin
    if (!rst_n) begin
      
    end else begin
       
    end
end

always_comb @(*) begin
    if (layer_en) begin
        if ((layer_num == 2) | (layer_num == 4)) begin
            // cal addr -> row, col, channel
            memB_addr_n = memB_addr_n + 1;
        end
        else begin
            memA_addr_n = memA_addr_n + 1;
        end

        wmem_addr_n = wmem_addr + 1;
    end

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

// 2. fetch (input/weight_buffer -> PE_array)
always_ff @( posedge clk ) begin
    if (!rst_n) begin
        input_data <= 0;
        weight_data <= 0;
    end else begin
        input_data <= input_data_n;
        weight_data <= weight_data_n;
    end
    
end

always_comb @(*) begin
    
    input_data_n = rd_patch;
    weight_data_n = rd_weight;
end


/* PE_array
module systolic_array_4x4 (
    input clk,
    input rst,
    input [7:0] data_in [0:3],   // Inputs to the first column (one per row)
    input [7:0] weight_in [0:3], // Inputs to the first row (one per column)
    input enable,
    output [15:0] result_out [0:3][0:3] // Result from each PE
);*/

// 3. compute (PE_array -> ReLU, bias, maxpool)
always_ff @( posedge clk ) begin
    if (!rst_n) begin
        PE_result <= 0;
    end else begin
        PE_result <= PE_result_n;
    end
    
end

always_comb @(*) begin
    PE_result_n = result_out;
end


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
/* maxpool
module maxPooling(
    input clk,
	input [7:0] input1,
	input [7:0] input2,
	input [7:0] input3,
	input [7:0] input4,
	input enable,
    output reg signed [7:0] output1,
	output reg maxPoolingDone
    );*/

// 4. activate ( -> output_buffer)

// 1, 2, 3, 4 ReLU exist
always_ff @( posedge clk ) begin
    if (!rst_n) begin
        ReLU_result <= 0;
    end else begin
        ReLU_result <= ReLU_result_n;
    end
    
end

always_comb @(*) begin
    if (layer_num != 5) begin
        ReLU_result_n = out_data;
    end
    else begin
        ReLU_result_n = 0;
    end
end

// 3 maxpool exist
always_ff @( posedge clk ) begin
    if (!rst_n) begin
        maxpool_result <= 0;
    end else begin
        maxpool_result <= maxpool_result_n;
    end

end

always_comb @(*) begin
    if (layer_num == 3) begin
        maxpool_result_n = output1;
    end
    else begin
        maxpool_result_n = 0;
    end
end

always_ff @(posedge clk) begin
    final_output_data <= selected_result;
end

always_comb begin
    case (layer_num)
        3: selected_result = maxpool_result;
        5: selected_result = PE_result;
        default: selected_result = ReLU_result;
    endcase
end

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

always_ff @( posedge clk ) begin
    if (!rst_n) begin
        PE_result <= 0;
    end else begin
        PE_result <= PE_result_n;
    end
end

always_comb @(*) begin
    
    if ((layer_num == 2) | (layer_num == 4)) begin
        // cal addr -> row, col, channel
        // data
        memA_addr_n = memA_addr_n + 1;
    end
    else begin
        memB_addr_n = memB_addr_n + 1;
    end

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
