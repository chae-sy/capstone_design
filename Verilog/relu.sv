module bias_relu #(
    parameter NUM_CHNL       = 16,
    parameter NUM_ACCUMULATE = 9,
    parameter NUM_ADDER_TREE_INPUTS = 16,
    parameter WIDTH_BITWIDTH = 8,
    parameter WIDTH_IN_DATA = WIDTH_BITWIDTH *2 + $clog2(NUM_ACCUMULATE) + $clog2(NUM_ADDER_TREE_INPUTS), // 16+4+4
    parameter WIDTH_BIAS = 32,
    parameter WIDTH_BIAS_ADDED = 33,
    parameter WIDTH_OUT_DATA = 8,
    
    parameter RELU_MAX_VAL = 6,
    
    parameter WIDTH_L1_IN_IL = 9,
    parameter WIDTH_L1_OUT_IL = 2,
    parameter WIDTH_L1_B_IL = 1,
    
    parameter WIDTH_L2_IN_IL = 9,
    parameter WIDTH_L2_OUT_IL = 2,
    parameter WIDTH_L2_B_IL = 1,
    
    parameter WIDTH_L3_IN_IL = 9,
    parameter WIDTH_L3_OUT_IL = 2,
    parameter WIDTH_L3_B_IL = 1,
    
    parameter WIDTH_L4_IN_IL = 9,
    parameter WIDTH_L4_OUT_IL = 2,
    parameter WIDTH_L4_B_IL = 1,
     
    parameter WIDTH_L5_IN_IL = 9,
    parameter WIDTH_L5_OUT_IL = 2,
    parameter WIDTH_L5_B_IL = 1,
    
    parameter WIDTH_L6_IN_IL = 9,
    parameter WIDTH_L6_OUT_IL = 2,
    parameter WIDTH_L6_B_IL = 1
      
)
(
    input   wire                                   relu_en,
    output  reg                                    relu_done,
    input   wire [2:0]                             layer_state,
    input   wire  signed [WIDTH_IN_DATA-1:0]       data_in,
    input   wire signed [WIDTH_BIAS-1:0]           bias,
    output  reg signed [WIDTH_OUT_DATA-1:0]               data_out
    );
    
  
  localparam WIDTH_L1_IN_FL = WIDTH_IN_DATA-WIDTH_L1_IN_IL-1;
  localparam WIDTH_L1_OUT_FL = WIDTH_OUT_DATA-WIDTH_L1_OUT_IL-1;
  localparam WIDTH_L1_EXT_FL = WIDTH_BIAS-WIDTH_L1_IN_IL-1;
  localparam WIDTH_L1_EXT_IL = WIDTH_BIAS-WIDTH_L1_EXT_FL -1;
  localparam WIDTH_L1_B_FL = WIDTH_BIAS-WIDTH_L1_B_IL-1;
  
  localparam WIDTH_L2_IN_FL = WIDTH_IN_DATA-WIDTH_L2_IN_IL-1;
  localparam WIDTH_L2_OUT_FL = WIDTH_OUT_DATA-WIDTH_L2_OUT_IL-1;
  localparam WIDTH_L2_EXT_FL = WIDTH_BIAS-WIDTH_L2_IN_IL-1;
  localparam WIDTH_L2_EXT_IL = WIDTH_BIAS-WIDTH_L2_EXT_FL -1;
  localparam WIDTH_L2_B_FL = WIDTH_BIAS-WIDTH_L2_B_IL-1;

  localparam WIDTH_L3_IN_FL = WIDTH_IN_DATA-WIDTH_L3_IN_IL-1;
  localparam WIDTH_L3_OUT_FL = WIDTH_OUT_DATA-WIDTH_L3_OUT_IL-1;
  localparam WIDTH_L3_EXT_FL = WIDTH_BIAS-WIDTH_L3_IN_IL-1;
  localparam WIDTH_L3_EXT_IL = WIDTH_BIAS-WIDTH_L3_EXT_FL -1;
  localparam WIDTH_L3_B_FL = WIDTH_BIAS-WIDTH_L3_B_IL-1;

  localparam WIDTH_L4_IN_FL = WIDTH_IN_DATA-WIDTH_L4_IN_IL-1;
  localparam WIDTH_L4_OUT_FL = WIDTH_OUT_DATA-WIDTH_L4_OUT_IL-1;
  localparam WIDTH_L4_EXT_FL = WIDTH_BIAS-WIDTH_L4_IN_IL-1;
  localparam WIDTH_L4_EXT_IL = WIDTH_BIAS-WIDTH_L4_EXT_FL -1;
  localparam WIDTH_L4_B_FL = WIDTH_BIAS-WIDTH_L4_B_IL-1;

  localparam WIDTH_L5_IN_FL = WIDTH_IN_DATA-WIDTH_L5_IN_IL-1;
  localparam WIDTH_L5_OUT_FL = WIDTH_OUT_DATA-WIDTH_L5_OUT_IL-1;
  localparam WIDTH_L5_EXT_FL = WIDTH_BIAS-WIDTH_L5_IN_IL-1;
  localparam WIDTH_L5_EXT_IL = WIDTH_BIAS-WIDTH_L5_EXT_FL -1;
  localparam WIDTH_L5_B_FL = WIDTH_BIAS-WIDTH_L5_B_IL-1;

  localparam WIDTH_L6_IN_FL = WIDTH_IN_DATA-WIDTH_L6_IN_IL-1;
  localparam WIDTH_L6_OUT_FL = WIDTH_OUT_DATA-WIDTH_L6_OUT_IL-1;
  localparam WIDTH_L6_EXT_FL = WIDTH_BIAS-WIDTH_L6_IN_IL-1;
  localparam WIDTH_L6_EXT_IL = WIDTH_BIAS-WIDTH_L6_EXT_FL -1;
  localparam WIDTH_L6_B_FL = WIDTH_BIAS-WIDTH_L6_B_IL-1;

  // bias를 더한 후 RELU(또는 no_relu) 처리. 동시에 relu_done 신호를 생성.
  wire signed [WIDTH_BIAS-1:0] data_extended = // int bit수 그대로, frac bit 수는 WIDTH_BIAS-WIDTH_IN_DATA 만큼 확장
        { { WIDTH_BIAS-WIDTH_IN_DATA { data_in[WIDTH_IN_DATA-1] } },
          data_in
        } << (WIDTH_BIAS - WIDTH_IN_DATA);

  reg signed [WIDTH_BIAS_ADDED-1:0] bias_added;
  reg signed [WIDTH_BIAS-1:0] bias_ext;
  always @(*) begin
    // 기본값: 연산이 없거나 레이어 상태가 해당하지 않으면 data_out = 0, relu_done = 0
    data_out   = {WIDTH_OUT_DATA{1'b0}};
    relu_done  = 1'b0;
    bias_added = {WIDTH_BIAS_ADDED{1'b0}};
    bias_ext   = {WIDTH_BIAS{1'b0}};

    if (relu_en) begin
      case (layer_state)
        3'b001: begin
          bias_ext = {{WIDTH_BIAS-WIDTH_L1_B_IL-WIDTH_L1_EXT_FL{bias[WIDTH_BIAS-1]}}, bias >> (WIDTH_L1_B_FL-WIDTH_L1_EXT_FL)};
          bias_added = data_extended + bias_ext;
          // 음수라면 zero 출력
          if (bias_added[WIDTH_BIAS-1]) begin
            data_out = {WIDTH_OUT_DATA{1'b0}};
          end
          // 최댓값보다 크다면 클램핑 
          else if (bias_added > (2**(WIDTH_L1_OUT_IL+WIDTH_L1_EXT_FL)-1)) begin
            data_out = 8'b0111_1111;
          end
          else begin 
            data_out = {1'b0, bias_added[WIDTH_L1_EXT_FL+WIDTH_L1_OUT_IL:WIDTH_L1_EXT_FL], bias_added[WIDTH_L1_EXT_FL-1:WIDTH_L1_EXT_FL-WIDTH_L1_OUT_FL]};
          end
          relu_done = 1'b1;  // 연산 완료 신호
        end

        3'd2: begin
          bias_ext = {{WIDTH_BIAS-WIDTH_L2_B_IL-WIDTH_L2_EXT_FL{bias[WIDTH_BIAS-1]}}, bias >> (WIDTH_L2_B_FL-WIDTH_L2_EXT_FL)};
          bias_added = data_extended + bias_ext;
          // 음수라면 zero 출력
          if (bias_added[WIDTH_BIAS-1]) begin
            data_out = {WIDTH_OUT_DATA{1'b0}};
          end
          // 최댓값보다 크다면 클램핑 
          else if (bias_added > (2**(WIDTH_L2_OUT_IL+WIDTH_L2_EXT_FL)-1)) begin
            data_out = 8'b0111_1111;
          end
          else begin 
            data_out = {1'b0, bias_added[WIDTH_L2_EXT_FL+WIDTH_L2_OUT_IL:WIDTH_L2_EXT_FL], bias_added[WIDTH_L2_EXT_FL-1:WIDTH_L2_EXT_FL-WIDTH_L2_OUT_FL]};
          end
          relu_done = 1'b1;  // 연산 완료 신호
        end

        3'd3: begin
          bias_ext = {{WIDTH_BIAS-WIDTH_L3_B_IL-WIDTH_L3_EXT_FL{bias[WIDTH_BIAS-1]}}, bias >> (WIDTH_L3_B_FL-WIDTH_L3_EXT_FL)};
          bias_added = data_extended + bias_ext;
          // 음수라면 zero 출력
          if (bias_added[WIDTH_BIAS-1]) begin
            data_out = {WIDTH_OUT_DATA{1'b0}};
          end
          // 최댓값보다 크다면 클램핑 
          else if (bias_added > (2**(WIDTH_L3_OUT_IL+WIDTH_L3_EXT_FL)-1)) begin
            data_out = 8'b0111_1111;
          end
          else begin 
            data_out = {1'b0, bias_added[WIDTH_L3_EXT_FL+WIDTH_L3_OUT_IL:WIDTH_L3_EXT_FL], bias_added[WIDTH_L3_EXT_FL-1:WIDTH_L3_EXT_FL-WIDTH_L3_OUT_FL]};
          end
          relu_done = 1'b1;  // 연산 완료 신호
        end

        3'd4: begin // maxpool 모드: bias 추가 없이 data_extended만 사용
          // 음수라면 zero 출력
          if (data_extended[WIDTH_BIAS-1]) begin
            data_out = {WIDTH_OUT_DATA{1'b0}};
          end
          // 최댓값보다 크다면 클램핑 
          else if (data_extended > (2**(WIDTH_L4_OUT_IL+WIDTH_L4_EXT_FL)-1)) begin
            data_out = 8'b0111_1111;
          end
          else begin 
            data_out = {1'b0, data_extended[WIDTH_L4_EXT_FL+WIDTH_L4_OUT_IL:WIDTH_L4_EXT_FL], data_extended[WIDTH_L4_EXT_FL-1:WIDTH_L4_EXT_FL-WIDTH_L4_OUT_FL]};
          end
          relu_done = 1'b1;  // 연산 완료 신호
        end

        3'd5: begin 
          bias_ext = {{WIDTH_BIAS-WIDTH_L5_B_IL-WIDTH_L5_EXT_FL{bias[WIDTH_BIAS-1]}}, bias >> (WIDTH_L5_B_FL-WIDTH_L5_EXT_FL)};
          bias_added = data_extended + bias_ext;
          // 음수라면 zero 출력
          if (bias_added[WIDTH_BIAS-1]) begin
            data_out = {WIDTH_OUT_DATA{1'b0}};
          end
          // 최댓값보다 크다면 클램핑 
          else if (bias_added > (2**(WIDTH_L5_OUT_IL+WIDTH_L5_EXT_FL)-1)) begin
            data_out = 8'b0111_1111;
          end
          else begin 
            data_out = {1'b0, bias_added[WIDTH_L5_EXT_FL+WIDTH_L5_OUT_IL:WIDTH_L5_EXT_FL], bias_added[WIDTH_L5_EXT_FL-1:WIDTH_L5_EXT_FL-WIDTH_L5_OUT_FL]};
          end
          relu_done = 1'b1;  // 연산 완료 신호
        end

        3'd6: begin // no_relu 모드: 양수/음수 그대로 양자화만 수행
          bias_ext = {{WIDTH_BIAS-WIDTH_L6_B_IL-WIDTH_L6_EXT_FL{bias[WIDTH_BIAS-1]}}, bias >>> (WIDTH_L6_B_FL-WIDTH_L6_EXT_FL)};
          bias_added = data_extended + bias_ext;
          if(bias_added[WIDTH_BIAS_ADDED-1]) begin // 음수인 경우
                        if(bias_added < -(2**(WIDTH_L6_OUT_IL+WIDTH_L6_EXT_FL))) data_out = 8'b10000000; // 표현 가능한 최솟값
                        else if (WIDTH_L6_OUT_IL == 0) data_out = {1'b0, bias_added[WIDTH_L6_EXT_FL:WIDTH_L6_EXT_FL-WIDTH_L6_OUT_FL]};
                        else data_out = {1'b1, bias_added[WIDTH_L6_EXT_FL+WIDTH_L6_OUT_IL-1:WIDTH_L6_EXT_FL], 
                                      bias_added[WIDTH_L6_EXT_FL-1:WIDTH_L6_EXT_FL-WIDTH_L6_OUT_FL]};
                    end
                    else begin // 양수인 경우
                        if(bias_added > 2**(WIDTH_L6_OUT_IL+WIDTH_L6_EXT_FL)-1) data_out = 8'b01111111; //표현 가능한 최댓값
                        else if (WIDTH_L6_OUT_IL == 0) data_out = {1'b0, bias_added[WIDTH_L6_EXT_FL:WIDTH_L6_EXT_FL-WIDTH_L6_OUT_FL]};
                        else data_out = {1'b0, bias_added[WIDTH_L6_EXT_FL+WIDTH_L6_OUT_IL-1:WIDTH_L6_EXT_FL], 
                                      bias_added[WIDTH_L6_EXT_FL-1:WIDTH_L6_EXT_FL-WIDTH_L6_OUT_FL]};
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