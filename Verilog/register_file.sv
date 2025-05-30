`timescale 1ns / 1ps
module regfile_sync #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 5
)(
    input  wire                     clk,       // 클록
    input  wire                     rst_n,     // 비동기 리셋 (low active)
    // --- Write Port ---
    input  wire                     we,        // write enable
    input  wire [ADDR_WIDTH-1:0]    waddr,     // write address
    input  wire [DATA_WIDTH-1:0]    wdata,     // write data
    // --- Read Port ---
    input  wire [ADDR_WIDTH-1:0]    raddr,     // read address (to be registered)
    output wire [DATA_WIDTH-1:0]    rdata      // read data (combinational from reg'd addr)
);

    localparam DEPTH = (1 << ADDR_WIDTH);
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    integer i;
    // 1) 동기식 쓰기: posedge clk, we=1일 때만 non-blocking으로 저장
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 초기화가 필요하면 여기에…
            for (i=0; i<DEPTH; i=i+1) begin
                mem[i] <= {DATA_WIDTH{1'b0}};
            end
           
        end else if (we) begin
            mem[waddr] <= wdata;
        end
    end

    // 2) 읽기 주소 레지스터: posedge clk에서 캡처
    reg [ADDR_WIDTH-1:0] raddr_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            raddr_reg <= {ADDR_WIDTH{1'b0}};
        else
            raddr_reg <= raddr;
    end

    // 3) 같은 클럭 싸이클 내에 combinational하게 데이터 출력
    assign rdata = mem[raddr_reg];

endmodule
