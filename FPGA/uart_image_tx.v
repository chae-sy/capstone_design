module uart_image_tx (
    input wire clk,               // 시스템 클럭
    input wire reset,             // 리셋 신호
    input wire [7:0] image_data,  // 8비트 이미지 데이터 (한 픽셀의 1개의 채널)
    input wire send,              // 송신 시작 신호
    output reg tx,               // UART 송신 핀
    output wire tx_done           // 송신 완료 신호
);

    reg [3:0] bit_index;     // 송신된 비트 수
    reg [9:0] shift_reg;     // 데이터 전송용 shift 레지스터
    reg [15:0] baud_counter; // Baud rate 카운터
    reg [7:0] byte_count;    // 전송된 이미지 데이터의 바이트 수

    // 115200 bps에서 Baud rate 설정 (50MHz 시스템 클럭 기준)
    localparam BAUD_RATE = 868;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            tx <= 1'b1;  // idle 상태
            bit_index <= 0;
            shift_reg <= 10'b1111111111;  // idle 상태 (start bit + 데이터 + stop bit)
            baud_counter <= 0;
            byte_count <= 0;
        end else if (send) begin
            shift_reg <= {1'b1, image_data, 1'b0};  // start bit + 이미지 데이터 + stop bit
            bit_index <= 0;
            baud_counter <= 0;
        end else if (baud_counter == BAUD_RATE - 1) begin
            baud_counter <= 0;
            tx <= shift_reg[bit_index];  // 송신할 비트
            if (bit_index < 9) begin
                bit_index <= bit_index + 1;
            end else begin
                bit_index <= 0;
                byte_count <= byte_count + 1;
            end
        end else begin
            baud_counter <= baud_counter + 1;
        end
    end

    // 송신 완료 신호
    assign tx_done = (byte_count == 30000);  // 100x100 이미지 (30000 byte)
endmodule
