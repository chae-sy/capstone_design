module uart_image_tx (
    input wire clk,
    input wire reset,
    input wire [7:0] image_data,
    input wire send,
    output reg tx,
    output reg tx_done
);
    // State encoding
    typedef enum logic [1:0] {
        IDLE = 2'b00,
        LOAD = 2'b01,
        SEND = 2'b10
    } state_t;

    state_t state, next_state;

    reg [9:0] shift_reg;
    reg [3:0] bit_index;
    reg [15:0] baud_counter;
    reg [15:0] byte_count;

    localparam BAUD_DIV = 868;

    // FSM transition
    always @(posedge clk or posedge reset) begin
        if (reset) state <= IDLE;
        else state <= next_state;
    end

    always @(*) begin
        case (state)
            IDLE:  next_state = (send) ? LOAD : IDLE;
            LOAD:  next_state = SEND;
            SEND:  next_state = (bit_index == 10 && baud_counter == BAUD_DIV-1) ? IDLE : SEND;
            default: next_state = IDLE;
        endcase
    end

    // Main logic
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            tx <= 1'b1;
            tx_done <= 1'b0;
            shift_reg <= 10'b1111111111;
            bit_index <= 0;
            baud_counter <= 0;
            byte_count <= 0;
        end else begin
            case (state)
                IDLE: begin
                    tx <= 1'b1;
                    tx_done <= (byte_count == 30000);
                end

                LOAD: begin
                    shift_reg <= {1'b1, image_data, 1'b0}; // Stop(1) + Data + Start(0)
                    bit_index <= 0;
                    baud_counter <= 0;
                    tx_done <= 1'b0;
                end

                SEND: begin
                    tx <= shift_reg[bit_index];
                    if (baud_counter == BAUD_DIV - 1) begin
                        baud_counter <= 0;
                        bit_index <= bit_index + 1;
                        if (bit_index == 9)
                            byte_count <= byte_count + 1;
                    end else begin
                        baud_counter <= baud_counter + 1;
                    end
                end
            endcase
        end
    end
endmodule
