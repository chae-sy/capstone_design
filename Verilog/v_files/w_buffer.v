`timescale 1ns / 1ps

module w_buffer #(
    parameter WIDTH_FSRAM_WL  = 128,  
    parameter DATA_WIDTH      = 8,     
    parameter NUM_CHNL        = 16,    // 채널 
    parameter SIZE_BUFFER_H   = 3,     // 버퍼 
    parameter SIZE_BUFFER_W   = 3,     // 버퍼
    parameter SIZE_KERNEL_H   = 3,     // 커널 
    parameter SIZE_KERNEL_W   = 3      // 커널
)(
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire                           wren,           // ??????????? feature 로드 ??????
    input  wire                           rden,           // ????????? 출력 ??????
    input  wire [WIDTH_FSRAM_WL-1:0]      data_in,        // SRAM ?????? ????????? 128bit
    input  wire                           layer_start,
    output reg [DATA_WIDTH*NUM_CHNL-1:0]  data_out,       // (출력) ?? 채널별로 8bit??? 묶음
    output reg                            w_buffer_done   // 리턴: 초기 ?????? ?????? 로드/출력??? ???????????? ?????
);

    reg [NUM_CHNL*DATA_WIDTH-1:0] buffer_data [0:SIZE_BUFFER_H-1][0:SIZE_BUFFER_W-1];
    reg [WIDTH_FSRAM_WL-1:0] data_in_reg;
    reg [5:0] load_cnt; 
    reg [5:0] out_cnt, out_cnt_n;   
    reg wren_d;
    integer r, c;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n ) begin
            for (r = 0; r < SIZE_BUFFER_H; r = r + 1)
                for (c = 0; c < SIZE_BUFFER_W; c = c + 1)
                    buffer_data[r][c] <= {NUM_CHNL*DATA_WIDTH{1'b0}};
            load_cnt      <= 0;
            out_cnt       <= 0;
            w_buffer_done  <= 1'b0;
        end 
        else if (layer_start) begin
            for (r = 0; r < SIZE_BUFFER_H; r = r + 1)
                for (c = 0; c < SIZE_BUFFER_W; c = c + 1)
                    buffer_data[r][c] <= {NUM_CHNL*DATA_WIDTH{1'b0}};
            load_cnt      <= 0;
            out_cnt       <= 0;
            w_buffer_done  <= 1'b0;
        end
        else begin
            w_buffer_done  <= 1'b0;
            out_cnt <= out_cnt_n;
            wren_d <= wren;
            if (wren) begin
                data_in_reg <= data_in;
            end
            // ????? 로직
            if (wren_d) begin
                if (load_cnt < SIZE_KERNEL_H * SIZE_KERNEL_W) begin
                    buffer_data[ load_cnt / SIZE_KERNEL_W ][ load_cnt % SIZE_KERNEL_W ] <= data_in_reg;
                    load_cnt <= load_cnt + 1;
                end
                if (load_cnt == (SIZE_KERNEL_H * SIZE_KERNEL_W - 1)) begin
                    w_buffer_done  <= 1'b1;
                    load_cnt <= 0;
                end
            end else begin
                load_cnt <= 0;
            end
        end
    end
   always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out      = {NUM_CHNL*DATA_WIDTH{1'b0}};
        end
        else if (layer_start) begin
            data_out      = {NUM_CHNL*DATA_WIDTH{1'b0}};
        end
    end
    
    always @(*) begin
        out_cnt_n = out_cnt;
        if (rden) begin
            if (out_cnt < SIZE_KERNEL_H * SIZE_KERNEL_W) begin
                data_out = buffer_data[ out_cnt / SIZE_KERNEL_W ][ out_cnt % SIZE_KERNEL_W ];
                out_cnt_n = out_cnt + 1;
            end
            if (out_cnt == (SIZE_KERNEL_H * SIZE_KERNEL_W - 1)) begin
                out_cnt_n = 0;
            end
        end else begin
            out_cnt_n  = 0;
            data_out = {NUM_CHNL*DATA_WIDTH{1'b0}};
        end
    end

endmodule