`timescale 1ns / 1ps
module mac_pipeline_superscalar #(
    parameter DATA_WIDTH   = 8,    // data, weight 폭
    parameter CHANNEL_NUM  = 16,   // 파이프라인 스테이지 수 = 누산할 채널 수
    parameter LANE_NUM     = 3     // superscalar lane 수 (예: R/G/B)
)(
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       valid_in,
    // superscalar data 입력: data_in[0]=R, [1]=G, [2]=B
    input  wire [DATA_WIDTH-1:0]      data_in   [0:LANE_NUM-1],
    // weight은 lane마다 동일하게 공유
    input  wire [DATA_WIDTH-1:0]      weight_in,
    output reg                        valid_out,
    // superscalar 결과 출력
    output reg [2*DATA_WIDTH-1:0]     result_out[0:LANE_NUM-1]
);

    // 파이프라인 레지스터: [stage][lane]
    reg [2*DATA_WIDTH-1:0] pipe      [0:CHANNEL_NUM-1][0:LANE_NUM-1];
    reg                    val_pipe  [0:CHANNEL_NUM-1][0:LANE_NUM-1];

    integer stage, lane;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            for (stage = 0; stage < CHANNEL_NUM; stage = stage + 1) begin
                for (lane = 0; lane < LANE_NUM; lane = lane + 1) begin
                    pipe[stage][lane]     <= {2*DATA_WIDTH{1'b0}};
                    val_pipe[stage][lane] <= 1'b0;
                end
            end
        end else begin
            // Stage 0: 각 lane 별 첫 곱셈
            for (lane = 0; lane < LANE_NUM; lane = lane + 1) begin
                if (valid_in) begin
                    pipe[0][lane]     <= data_in[lane] * weight_in;
                    val_pipe[0][lane] <= 1'b1;
                end else begin
                    pipe[0][lane]     <= {2*DATA_WIDTH{1'b0}};
                    val_pipe[0][lane] <= 1'b0;
                end
            end

            // Stage k>0: 이전 stage 누산 + 새로운 곱셈
            for (stage = 1; stage < CHANNEL_NUM; stage = stage + 1) begin
                for (lane = 0; lane < LANE_NUM; lane = lane + 1) begin
                    if (val_pipe[stage-1][lane]) begin
                        pipe[stage][lane]     <= pipe[stage-1][lane] 
                                              + (data_in[lane] * weight_in);
                        val_pipe[stage][lane] <= 1'b1;
                    end else begin
                        pipe[stage][lane]     <= {2*DATA_WIDTH{1'b0}};
                        val_pipe[stage][lane] <= 1'b0;
                    end
                end
            end

            // Output Register: 마지막 스테이지 결과
            valid_out <= &{val_pipe[CHANNEL_NUM-1][0],
                           val_pipe[CHANNEL_NUM-1][1],
                           val_pipe[CHANNEL_NUM-1][2]}; 
            // 모든 lane이 유효할 때만 valid_out 상승

            for (lane = 0; lane < LANE_NUM; lane = lane + 1) begin
                result_out[lane] <= pipe[CHANNEL_NUM-1][lane];
            end
        end
    end

endmodule
