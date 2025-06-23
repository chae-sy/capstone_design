module uart_image_tx (
    input  wire        clk,           // 시스템 클럭 (예: 50 MHz)
    input  wire        rst_n,         // 비동기 리셋
    input  wire        valid,         // 1클럭 동안 HIGH → 데이터 전송 요청
    input  wire [7:0]  image_data,    // 전송할 8비트 이미지 데이터
    input  wire [15:0] total_bytes,   // 총 전송할 바이트 수
    output reg         one_byte,     
    output reg         tx,            // UART 송신 핀
    output reg         busy,          // 전송 중 상태 플래그
    output reg         tx_done        // 전체 전송 완료 플래그 (1클럭 HIGH)
);

    localparam BAUD_RATE = 868;  // (100_000_000 / 115200)
    localparam IDLE = 2'd0, LOAD = 2'd1, SEND = 2'd2;

    reg [1:0]  state, next_state;
    reg [9:0]  shift_reg;
    reg [3:0]  bit_index;
    reg [15:0] baud_counter;
    reg [15:0] byte_counter;

    // FSM 상태 전이
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // FSM next state logic
    always @(*) begin
        case (state)
            IDLE:  next_state = (valid && !busy && (byte_counter < total_bytes)) ? LOAD : IDLE;
            LOAD:  next_state = SEND;
            SEND:  next_state = (bit_index == 9 && baud_counter == BAUD_RATE - 1) ? IDLE : SEND;
            default: next_state = IDLE;
        endcase
    end

    // Main logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx <= 1'b1;
            busy <= 1'b0;
            tx_done <= 1'b0;
            shift_reg <= 10'b1111111111;
            bit_index <= 0;
            one_byte <= 0;
            baud_counter <= 0;
            byte_counter <= 0;
        end else begin
            one_byte <= 0;
            case (state)
                IDLE: begin
                    tx <= 1'b1;
                    busy <= 1'b0;
                    baud_counter <= 0;
                    bit_index <= 0;
                    tx_done <= 0;
                 
                    if (byte_counter == total_bytes) begin
                        tx_done <= 1;  // 전송 완료 플래그 1클럭 출력
                        byte_counter <= 0;
                    end
                end

                LOAD: begin
                    shift_reg <= {1'b1, image_data, 1'b0};  // Stop + Data + Start
                    busy <= 1'b1;
                    tx_done <= 0;
                end

                SEND: begin
                    busy <= 1'b1;
                    if (baud_counter == BAUD_RATE - 1) begin
                        baud_counter <= 0;
                        tx <= shift_reg[bit_index];
                        bit_index <= bit_index + 1;

                        if (bit_index == 9) begin
                            byte_counter <= byte_counter + 1;
                            busy <= 1'b0;
                        end
                        else if (bit_index == 8) begin
                            one_byte <= 1;
                        end 
                    end else begin
                        baud_counter <= baud_counter + 1;
                    end
                end
            endcase
        end
    end
endmodule
