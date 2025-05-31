module bias_relu #(
    parameter NUM_CHNL       = 16,
    parameter NUM_ACCUMULATE = 9,
    parameter NUM_ADDER_TREE_INPUTS = 16,
    parameter WIDTH_BITWIDTH = 8,
    parameter WIDTH_IN_DATA = WIDTH_BITWIDTH *2 + $clog2(NUM_ACCUMULATE) + $clog2(NUM_ADDER_TREE_INPUTS), // 16+4+4
    parameter WIDTH_BIAS = 32,
    parameter WIDTH_OUT_DATA = 8,
    
    parameter RELU_MAX_VAL = 6,
    
    parameter WIDTH_L1_IN_IL = 9,
    parameter WIDTH_L1_OUT_IL = 1,
    parameter WIDTH_L1_ZERO_POINT = -1,
    parameter WIDTH_L1_SCALE = 0.2,
    
    parameter WIDTH_L2_IN_IL = 7,
    parameter WIDTH_L2_OUT_IL = 1,
    parameter WIDTH_L2_ZERO_POINT = 0,
    parameter WIDTH_L2_SCALE = 0.2,
    
    parameter WIDTH_L3_IN_IL = 12,
    parameter WIDTH_L3_OUT_IL = 1,
    parameter WIDTH_L3_ZERO_POINT = 0,
    parameter WIDTH_L3_SCALE = 0.2,
    
    parameter WIDTH_L4_IN_IL = 7,
    parameter WIDTH_L4_OUT_IL = 0,
    parameter WIDTH_L4_ZERO_POINT = 0,
    parameter WIDTH_L4_SCALE = 0.2,
    
    parameter WIDTH_L5_IN_IL = 8,
    parameter WIDTH_L5_OUT_IL = 0,
    parameter WIDTH_L5_ZERO_POINT = 0,
    parameter WIDTH_L5_SCALE = 0.2,

    parameter WIDTH_L6_IN_IL = 8,
    parameter WIDTH_L6_OUT_IL = 0,
    parameter WIDTH_L6_ZERO_POINT = 0,
    parameter WIDTH_L6_SCALE = 0.2
    
)
(
    input                                           relu_en,
    output   reg                                    relu_done,
    input   [2:0]                                   layer_state,
    input signed [WIDTH_IN_DATA-1:0]                data_in,
    input  signed [WIDTH_BIAS-1:0]                  bias,
  
    output  reg [WIDTH_OUT_DATA-1:0]                data_out
    );
    
  
  localparam WIDTH_L1_IN_FL = WIDTH_IN_DATA-WIDTH_L1_IN_IL-1;
  localparam WIDTH_L1_OUT_FL = WIDTH_OUT_DATA-WIDTH_L1_OUT_IL-1;

  localparam WIDTH_L2_IN_FL = WIDTH_IN_DATA-WIDTH_L2_IN_IL-1;
  localparam WIDTH_L2_OUT_FL = WIDTH_OUT_DATA-WIDTH_L2_OUT_IL-1;

  localparam WIDTH_L3_IN_FL = WIDTH_IN_DATA-WIDTH_L3_IN_IL-1;
  localparam WIDTH_L3_OUT_FL = WIDTH_OUT_DATA-WIDTH_L3_OUT_IL-1;

  localparam WIDTH_L4_IN_FL = WIDTH_IN_DATA-WIDTH_L4_IN_IL-1;
  localparam WIDTH_L4_OUT_FL = WIDTH_OUT_DATA-WIDTH_L4_OUT_IL-1;

  localparam WIDTH_L5_IN_FL = WIDTH_IN_DATA-WIDTH_L5_IN_IL-1;
  localparam WIDTH_L5_OUT_FL = WIDTH_OUT_DATA-WIDTH_L5_OUT_IL-1;

  localparam WIDTH_L6_IN_FL = WIDTH_IN_DATA-WIDTH_L6_IN_IL-1;
  localparam WIDTH_L6_OUT_FL = WIDTH_OUT_DATA-WIDTH_L6_OUT_IL-1;

  // precompute quantized threshold for RELU_MAX_VAL
  localparam real relu_q_real_l1 = (RELU_MAX_VAL * (2.0**WIDTH_L1_IN_FL)) / WIDTH_L1_SCALE;
  localparam int  relu_q_l1 = $rtoi(relu_q_real_l1 + 0.5) + WIDTH_L1_ZERO_POINT;
  localparam real relu_q_real_l2 = (RELU_MAX_VAL * (2.0**WIDTH_L2_IN_FL)) / WIDTH_L2_SCALE;
  localparam int  relu_q_l2 = $rtoi(relu_q_real_l2 + 0.5) + WIDTH_L2_ZERO_POINT;
  localparam real relu_q_real_l3 = (RELU_MAX_VAL * (2.0**WIDTH_L3_IN_FL)) / WIDTH_L3_SCALE;
  localparam int  relu_q_l3 = $rtoi(relu_q_real_l3 + 0.5) + WIDTH_L3_ZERO_POINT;
  localparam real relu_q_real_l4 = (RELU_MAX_VAL * (2.0**WIDTH_L4_IN_FL)) / WIDTH_L4_SCALE;
  localparam int  relu_q_l4 = $rtoi(relu_q_real_l4 + 0.5) + WIDTH_L4_ZERO_POINT;
  localparam real relu_q_real_l5 = (RELU_MAX_VAL * (2.0**WIDTH_L5_IN_FL)) / WIDTH_L5_SCALE;
  localparam int  relu_q_l5 = $rtoi(relu_q_real_l5 + 0.5) + WIDTH_L5_ZERO_POINT;
  localparam real relu_q_real_l6 = (RELU_MAX_VAL * (2.0**WIDTH_L6_IN_FL)) / WIDTH_L6_SCALE;
  localparam int  relu_q_l6 = $rtoi(relu_q_real_l6 + 0.5) + WIDTH_L6_ZERO_POINT;

  // bias를 더한 후 RELU(또는 no_relu) 처리. 동시에 relu_done 신호를 생성.
  wire signed [WIDTH_BIAS-1:0] data_extended = 
        { { WIDTH_BIAS-WIDTH_IN_DATA { data_in[WIDTH_IN_DATA-1] } },
          data_in
        } << (WIDTH_BIAS - WIDTH_IN_DATA);

  reg signed [WIDTH_BIAS-1:0] bias_added;

  always @(*) begin
    // 기본값: 연산이 없거나 레이어 상태가 해당하지 않으면 data_out = 0, relu_done = 0
    data_out   = {WIDTH_OUT_DATA{1'b0}};
    relu_done  = 1'b0;
    bias_added = {WIDTH_BIAS{1'b0}};

    if (relu_en) begin
      case (layer_state)
        3'b001: begin
          bias_added = data_extended + bias;
          // 음수라면 zero-point 출력
          if (bias_added[WIDTH_BIAS-1])
            data_out = WIDTH_L1_ZERO_POINT;
          // Q값보다 크다면 클램핑
          else if (bias_added > relu_q_l1)
            data_out = relu_q_l1;
          else
            data_out = bias_added[WIDTH_IN_DATA - WIDTH_L1_IN_IL - 1 +: WIDTH_OUT_DATA];
          relu_done = 1'b1;  // 연산 완료 신호
        end

        3'b002: begin
          bias_added = data_extended + bias;
          if (bias_added[WIDTH_BIAS-1])
            data_out = WIDTH_L1_ZERO_POINT;
          else if (bias_added > relu_q_l1)
            data_out = relu_q_l1;
          else
            data_out = bias_added[WIDTH_IN_DATA - WIDTH_L1_IN_IL - 1 +: WIDTH_OUT_DATA];
          relu_done = 1'b1;
        end

        3'b003: begin
          bias_added = data_extended + bias;
          if (bias_added[WIDTH_BIAS-1])
            data_out = WIDTH_L1_ZERO_POINT;
          else if (bias_added > relu_q_l1)
            data_out = relu_q_l1;
          else
            data_out = bias_added[WIDTH_IN_DATA - WIDTH_L1_IN_IL - 1 +: WIDTH_OUT_DATA];
          relu_done = 1'b1;
        end

        3'b004: begin // maxpool 모드: bias 추가 없이 data_extended만 사용
          if (data_extended[WIDTH_BIAS-1])
            data_out = WIDTH_L1_ZERO_POINT;
          else if (data_extended > relu_q_l1)
            data_out = relu_q_l1;
          else
            data_out = data_extended[WIDTH_IN_DATA - WIDTH_L1_IN_IL - 1 +: WIDTH_OUT_DATA];
          relu_done = 1'b1;
        end

        3'b005: begin 
          bias_added = data_extended + bias;
          if (bias_added[WIDTH_BIAS-1])
            data_out = WIDTH_L1_ZERO_POINT;
          else if (bias_added > relu_q_l1)
            data_out = relu_q_l1;
          else
            data_out = bias_added[WIDTH_IN_DATA - WIDTH_L1_IN_IL - 1 +: WIDTH_OUT_DATA];
          relu_done = 1'b1;
        end

        3'b006: begin // no_relu 모드: 양수/음수 그대로 양자화만 수행
          bias_added = data_extended + bias;
          if (bias_added[WIDTH_BIAS-1]) begin // 음수
            if (bias_added < -(2**(WIDTH_OUT_DATA-1) << (WIDTH_BIAS-WIDTH_L1_IN_IL-1-WIDTH_L1_OUT_FL)))
              data_out = WIDTH_L1_ZERO_POINT;
            else
              data_out = {1'b1,
                          bias_added[WIDTH_BIAS-WIDTH_L1_IN_IL-1 + WIDTH_L1_OUT_IL:
                                      WIDTH_BIAS-WIDTH_L1_IN_IL-1],
                          bias_added[WIDTH_BIAS-WIDTH_L1_IN_IL-1:
                                      WIDTH_BIAS-WIDTH_L1_IN_IL-1 - WIDTH_L1_OUT_FL]};
          end else begin          // 양수
            if (bias_added > ((2**(WIDTH_OUT_DATA-1)-1) << (WIDTH_BIAS-WIDTH_L1_IN_IL-1-WIDTH_L1_OUT_FL)))
              data_out = WIDTH_L1_ZERO_POINT + (2**WIDTH_OUT_DATA-1);
            else
              data_out = {1'b0,
                          bias_added[WIDTH_BIAS-WIDTH_L1_IN_IL-1 + WIDTH_L1_OUT_IL:
                                      WIDTH_BIAS-WIDTH_L1_IN_IL-1],
                          bias_added[WIDTH_BIAS-WIDTH_L1_IN_IL-1:
                                      WIDTH_BIAS-WIDTH_L1_IN_IL-1 - WIDTH_L1_OUT_FL]};
          end
          relu_done = 1'b1;
        end

        default: begin
          data_out  = {WIDTH_OUT_DATA{1'b0}};
          relu_done = 1'b0;
        end
      endcase
    end
    // relu_en = 0인 경우, data_out와 relu_done은 모두 0 (위에서 기본값으로 할당됨)
  end

endmodule



