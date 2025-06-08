`timescale 1ns / 1ps
// f_buffer converted to Verilog-2001 (no unpacked arrays, no always_ff/always_comb)

module f_buffer #(
    parameter DATA_WIDTH      = 8,     // 한 채널당 데이터 폭
    parameter NUM_CHNL        = 16,    // 채널 수
    parameter SIZE_BUFFER_H   = 3,     // 버퍼 세로 크기
    parameter SIZE_BUFFER_W   = 4,     // 버퍼 가로 크기
    parameter SIZE_KERNEL_H   = 3,     // 커널 세로 크기
    parameter SIZE_KERNEL_W   = 3      // 커널 가로 크기
)(
    input  wire                                clk,
    input  wire                                rst_n,
    input  wire                                is_initial,   // 초기 로드 플래그
    input  wire                                wren,         // 쓰기 enable
    input  wire                                rden,         // 읽기 enable
    input  wire [NUM_CHNL*DATA_WIDTH-1:0]      data_in,      // 입력
    output reg  [NUM_CHNL*DATA_WIDTH-1:0]      data_out,     // 출력
    output reg                                 f_buffer_done // 완료 펄스
);

    // 1D memory: H×W entries of wide words
    localparam DEPTH = SIZE_BUFFER_H * SIZE_BUFFER_W;
    reg [NUM_CHNL*DATA_WIDTH-1:0] buffer_mem [0:DEPTH-1];

    reg [5:0] load_cnt;
    reg [5:0] out_cnt, out_cnt_n;
    reg write_done, read_done;

    integer idx;
    
    //----------------------------------------------------------------
    // Sequential write + shift + done flag
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // clear memory
            for (idx = 0; idx < DEPTH; idx = idx + 1)
                buffer_mem[idx] <= {(NUM_CHNL*DATA_WIDTH){1'b0}};
            load_cnt      <= 0;
            out_cnt       <= 0;
            data_out      <= {(NUM_CHNL*DATA_WIDTH){1'b0}};
            f_buffer_done <= 1'b0;
            write_done    <= 1'b0;
        end else begin
            f_buffer_done <= 1'b0;
            out_cnt       <= out_cnt_n;

            // WRITE PHASE
            if (wren) begin
                if (is_initial) begin
                    // 초기 로드: 커널 크기만큼 채움
                    if (load_cnt < SIZE_KERNEL_H * SIZE_KERNEL_W) begin
                        buffer_mem[load_cnt] <= data_in;
                        load_cnt <= load_cnt + 1;
                    end
                    if (load_cnt == SIZE_KERNEL_H * SIZE_KERNEL_W - 1) begin
                        f_buffer_done <= 1'b1;
                        load_cnt      <= 0;
                    end
                end else begin
                    // 스트라이드 로드: 맨 마지막 열만 교체
                    if (load_cnt < SIZE_BUFFER_H) begin
                        buffer_mem[ load_cnt * SIZE_BUFFER_W + (SIZE_BUFFER_W-1) ] <= data_in;
                        load_cnt <= load_cnt + 1;
                    end
                    if (load_cnt == SIZE_BUFFER_H - 1) begin
                        f_buffer_done <= 1'b1;
                        write_done    <= 1'b1;
                        load_cnt      <= 0;
                    end
                end
            end else begin
                load_cnt <= 0;
            end

            // SHIFT LEFT AFTER both write_done & read_done
            if (write_done && read_done) begin
                for (idx = 0; idx < SIZE_BUFFER_H; idx = idx + 1) begin
                    integer c;
                    for (c = 0; c < SIZE_BUFFER_W-1; c = c + 1) begin
                        buffer_mem[idx*SIZE_BUFFER_W + c] <=
                            buffer_mem[idx*SIZE_BUFFER_W + c + 1];
                    end
                end
                write_done <= 1'b0;
            end
        end
    end

    //----------------------------------------------------------------
    // Combinational read + done flag
    always @(*) begin
        out_cnt_n = out_cnt;
        read_done = 1'b0;
        data_out  = {(NUM_CHNL*DATA_WIDTH){1'b0}};

        if (rden) begin
            if (out_cnt < SIZE_KERNEL_H * SIZE_KERNEL_W) begin
                data_out  = buffer_mem[out_cnt];
                out_cnt_n = out_cnt + 1;
            end
            if (out_cnt == SIZE_KERNEL_H * SIZE_KERNEL_W - 1) begin
                read_done = 1'b1;
                out_cnt_n = 0;
            end
        end else begin
            out_cnt_n = 0;
        end
    end

endmodule
