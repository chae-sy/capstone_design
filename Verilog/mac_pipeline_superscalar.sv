`timescale 1ns / 1ps

module mac_pipeline_superscalar #(
    parameter DATA_WIDTH   = 8,    // data, weight ?­
    parameter NUM_STAGE    = 9,   // ?ŒŒ?´?”„?¼?¸ ?Š¤?…Œ?´ì§? ?ˆ˜ = ?ˆ„?‚°?•  ì±„ë„ ?ˆ˜
    parameter LANE_NUM     = 3     // superscalar lane ?ˆ˜ (?˜ˆ: R/G/B)
)(
    input  wire                       clk,
    input  wire                       rst_n,
    // ë§? ?‚¬?´?´ ?•˜?‚˜?”© ?“¤?–´?˜¤?Š” data/weight ?Œ
    input  wire                       valid_in,
    // superscalar data ?…? ¥: data_in[0]=R, [1]=G, [2]=B
    input  wire [DATA_WIDTH-1:0]      data_in   [0:LANE_NUM-1],
    // weight?? laneë§ˆë‹¤ ?™?¼?•˜ê²? ê³µìœ 
    input  wire [DATA_WIDTH-1:0]      weight_in,
    // MAC ?™„ë£? ?‹œ ?•œ ?‚¬?´?´ ?”œ? ˆ?´?œ valid + ê²°ê³¼ ì¶œë ¥
    output reg                        valid_out,
    output reg [LANE_NUM*2*DATA_WIDTH-1:0] result_out_flat
);

    // ?ŒŒ?´?”„?¼?¸ ? ˆì§??Š¤?„°: [stage][lane]
    reg [2*DATA_WIDTH-1:0] pipe      [0:NUM_STAGE-1][0:LANE_NUM-1];
    reg                    val_pipe  [0:NUM_STAGE-1];
    integer stage, lane;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            // ?ŒŒ?´?”„?¼?¸ ë°? valid ?‹ ?˜¸ ì´ˆê¸°?™”
            for (stage = 0; stage < NUM_STAGE; stage = stage + 1) begin
                val_pipe[stage] <= 1'b0;
                for (lane = 0; lane < LANE_NUM; lane = lane + 1) begin
                    pipe[stage][lane] <= {2*DATA_WIDTH{1'b0}};
                end
            end
        end else begin
            // Stage 0: ?…? ¥ valid ?‹ ?˜¸ ê·¸ë?ë¡? ë°?ê¸?, ê³±ì…ˆ ?ˆ˜?–‰
            val_pipe[0] <= valid_in;
            for (lane = 0; lane < LANE_NUM; lane = lane + 1) begin
                if (valid_in)
                    pipe[0][lane] <= data_in[lane] * weight_in;
                else
                    pipe[0][lane] <= {2*DATA_WIDTH{1'b0}};
            end

            // Stage k>0: ?´? „ valid ?‹ ?˜¸ ë°?ê¸?, ?´? „ ê°? + ?ƒˆë¡œìš´ ê³±ì…ˆ ?ˆ„?‚°
            for (stage = 1; stage < NUM_STAGE; stage = stage + 1) begin
                val_pipe[stage] <= val_pipe[stage-1];
                for (lane = 0; lane < LANE_NUM; lane = lane + 1) begin
                    if (val_pipe[stage-1])
                        pipe[stage][lane] <= pipe[stage-1][lane] + (data_in[lane] * weight_in);
                    else
                        pipe[stage][lane] <= {2*DATA_WIDTH{1'b0}};
                end
            end

            // Output register: ë§ˆì?ë§? ?Š¤?…Œ?´ì§??˜ valid?? ê²°ê³¼
            valid_out <= val_pipe[NUM_STAGE-1];
            for (lane = 0; lane < LANE_NUM; lane = lane + 1)
                result_out_flat[
        lane*2*DATA_WIDTH +: 2*DATA_WIDTH
      ] <= pipe[NUM_STAGE-1][lane];
        end
    end

endmodule