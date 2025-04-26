//============================================================
// Module: relu_stream_with_last
//  - Streaming ReLU with valid-last handshake
//  - DATA_WIDTH, CH는 파라미터로, 입력 크기가 바뀌어도 재사용 가능
//  - valid/last는 그대로 파이프라인을 통과시켜 downstream 모듈이
//    언제 프레임이 끝나는지 알 수 있게 함
//============================================================
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

  integer i;
  // 1) combinational ReLU 연산
  reg signed [DATA_WIDTH-1:0] relu_mid [0:CH-1];
  always @(*) begin
    for (i = 0; i < CH; i = i + 1) begin
      relu_mid[i] = (in_data[i] < 0) ? 0 : in_data[i];
    end
  end

  // 2) pipeline: 한 클럭 뒤에 valid/last 와 함께 내보내기
  always @(posedge clk or negedge rstb) begin
    if (!rstb) begin
      valid_out <= 0;
      last_out  <= 0;
      for (i = 0; i < CH; i = i + 1)
        out_data[i] <= 0;
    end else begin
      // valid, last 신호 그대로 전달
      valid_out <= valid_in;
      last_out  <= last_in;
      // 유효할 때만 데이터 등록
      if (valid_in) begin
        for (i = 0; i < CH; i = i + 1)
          out_data[i] <= relu_mid[i];
      end
    end
  end

endmodule