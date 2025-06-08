`timescale 1ns / 1ps

module regfile_sync(
    input                     clk,       // 클록
    input                     rst_n,     // 비동기 리셋 (low active)
    input                     we,        // write enable
    input  [ADDR_WIDTH-1:0]   waddr,     // write address
    input  [DATA_WIDTH-1:0]   wdata,     // write data
    input  [ADDR_WIDTH-1:0]   raddr,     // read address
    input                     rden,      // read enable
    input  [2:0]              layer_num, // layer selection
    output [BITWIDTH-1:0]     rdata      // 슬라이싱된 한 워드
);

    //======================================================================
    // 1) 파라미터 선언
    //======================================================================
    parameter BITWIDTH            = 32;
    parameter NUM_WORD            = 16;
    parameter LAYER_1_NUM_WORD    = 2;
    parameter DATA_WIDTH          = NUM_WORD * BITWIDTH;
    parameter ADDR_WIDTH          = 3;
    localparam DEPTH              = (1 << ADDR_WIDTH);

    //======================================================================
    // 2) 내부 신호 선언
    //======================================================================
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    integer i;

    reg [ADDR_WIDTH-1:0] raddr_reg;
    reg                 prev_rden;
    reg [DATA_WIDTH-1:0] read_buf;
    reg [3:0]           cnt;

    //======================================================================
    // 3) 동기식 쓰기
    //======================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1)
                mem[i] <= {DATA_WIDTH{1'b0}};
        end else if (we) begin
            mem[waddr] <= wdata;
        end
    end

    //======================================================================
    // 4) 읽기 주소 캡처
    //======================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            raddr_reg <= {ADDR_WIDTH{1'b0}};
        else
            raddr_reg <= raddr;
    end

    //======================================================================
    // 5) rden 엣지 검출
    //======================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            prev_rden <= 1'b0;
        else
            prev_rden <= rden;
    end

    //======================================================================
    // 6) read_buf 업데이트
    //======================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            read_buf <= {DATA_WIDTH{1'b0}};
        else if (rden && !prev_rden)
            read_buf <= mem[raddr_reg];
    end

    //======================================================================
    // 7) cnt 업데이트
    //======================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            if (layer_num == 3'd1)
                cnt <= LAYER_1_NUM_WORD - 1;
            else
                cnt <= NUM_WORD - 1;
        end else if (rden && !prev_rden) begin
            if (layer_num == 3'd1) begin
                if (cnt == LAYER_1_NUM_WORD - 1)
                    cnt <= 4'd0;
                else
                    cnt <= cnt + 4'd1;
            end else begin
                if (cnt == NUM_WORD - 1)
                    cnt <= 4'd0;
                else
                    cnt <= cnt + 4'd1;
            end
        end
    end

    //======================================================================
    // 8) rdata 슬라이싱
    //======================================================================
    assign rdata = read_buf[cnt * BITWIDTH +: BITWIDTH];

endmodule
