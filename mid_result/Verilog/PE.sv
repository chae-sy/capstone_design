module PE_with_clock_gating (
    input clk,
    input enable,
    input reset,                      // 추가된 reset 입력
    input [7:0] data_in, weight,
    output reg [15:0] result
);
    wire gated_clk;
    assign gated_clk = clk & enable;

    always @(posedge gated_clk or posedge reset) begin
        if (reset)
            result <= 16'd0;          // reset 시 result 초기화
        else
            result <= result + data_in * weight;  // PE 연산 수행
    end
endmodule
