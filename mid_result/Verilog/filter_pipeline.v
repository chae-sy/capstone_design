// 파이프라인 스테이지 구현 예시
module filter_pipeline (
    input clk, reset,
    input [7:0] pixel_in,
    output reg [7:0] pixel_out
);
    localparam [7:0] coeff1 = 8'd1;
    localparam [7:0] coeff2 = 8'd2;
    localparam [7:0] coeff3 = 8'd3;

    // 파이프라인 레지스터
    reg [7:0] stage1_out, stage2_out, stage3_out, stage4_out, stage5_out;

    // 스테이지 1: 입력 버퍼 읽기
    always @(posedge clk) if (!reset) stage1_out <= pixel_in;

    // 스테이지 2-4: 필터 연산 (3단계로 분할)
    always @(posedge clk) if (!reset) stage2_out <= stage1_out * coeff1;
    always @(posedge clk) if (!reset) stage3_out <= stage2_out + stage1_out * coeff2;
    always @(posedge clk) if (!reset) stage4_out <= stage3_out + stage1_out * coeff3;

    // 스테이지 5: 정규화 및 양자화
    always @(posedge clk) if (!reset) stage5_out <= stage4_out/3;

    // 스테이지 6: 출력 버퍼 쓰기
    always @(posedge clk) if (!reset) pixel_out <= stage5_out;
endmodule