//------------------------------------------------------------------------------
// Progect : COMPASS
// DATE    : 2024/7/11/Thu
// IP      : BufferPump
//------------------------------------------------------------------------------

// Verilog-2001 버전으로 변환
module memory_w_v0
#(
    parameter ADDR_WIDTH = 10,          // 2^10 = 1024 addresses
    parameter DATA_WIDTH = 128,         // 8bit * 16 words
    parameter WR_DELAY   = 8
)
(
    input  wire                       CLK,
    input  wire                       CEB,
    input  wire                       WEB,
    input  wire [ADDR_WIDTH-1:0]      A,
    input  wire [DATA_WIDTH-1:0]      D,
    output wire [DATA_WIDTH-1:0]      Q
);

    // 메모리 깊이 계산
    localparam DEPTH = (1 << ADDR_WIDTH);

    // Verilog-2001 memory 선언
    reg [DATA_WIDTH-1:0] mem_W [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] mem_d;
    reg [ADDR_WIDTH-1:0] temp_A;

    always @(posedge CLK) begin
        // 주소 레지스터에 동기식으로 저장
        temp_A <= A;

        if (!CEB && !WEB) begin
            // write
            mem_W[temp_A] <= D;
        end
        else if (!CEB && WEB) begin
            // read
            mem_d <= mem_W[temp_A];
        end
        else begin
            // 비활성 시 undefined
            mem_d <= {DATA_WIDTH{1'bx}};
        end
    end

    // 출력
    assign Q = mem_d;

endmodule
