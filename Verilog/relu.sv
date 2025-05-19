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
    parameter WIDTH_L5_SCALE = 0.2
    
)
(
    input                                           relu_on,
    input   [2:0]                                   layer_state,
    input signed [WIDTH_IN_DATA-1:0]     data_in,
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
  
  wire signed [WIDTH_BIAS-1:0] data_extended = 
        { { WIDTH_BIAS-WIDTH_IN_DATA { data_in[WIDTH_IN_DATA-1] } },
          data_in
        } << (WIDTH_BIAS - WIDTH_IN_DATA);

     always @(*) begin
        case (layer_state)
          3'b001: begin  // L1
            if (relu_on) begin
              if (data_extended[WIDTH_BIAS-1])
                data_out = WIDTH_L1_ZERO_POINT;
              else if (data_extended > relu_q_l1)
                data_out = relu_q_l1;
              else
                data_out = data_extended[WIDTH_IN_DATA - WIDTH_L1_IN_IL - 1 +: WIDTH_OUT_DATA];
            end else begin
              // if ReLU off,  saturate + fixed-point slicing
              if (data_extended[WIDTH_BIAS-1]) begin
                if (data_extended < -(2**(WIDTH_OUT_DATA-1) << (WIDTH_BIAS-WIDTH_L1_IN_IL-1-WIDTH_L1_OUT_FL)))
                  data_out = WIDTH_L1_ZERO_POINT;
                else
                  data_out = {1'b1,
                                  data_extended[WIDTH_BIAS-WIDTH_L1_IN_IL-1+WIDTH_L1_OUT_IL:
                                                WIDTH_BIAS-WIDTH_L1_IN_IL-1],
                                  data_extended[WIDTH_BIAS-WIDTH_L1_IN_IL-1:
                                                WIDTH_BIAS-WIDTH_L1_IN_IL-1-WIDTH_L1_OUT_FL]};
              end else begin
                if (data_extended > ((2**(WIDTH_OUT_DATA-1)-1) << (WIDTH_BIAS-WIDTH_L1_IN_IL-1-WIDTH_L1_OUT_FL)))
                  data_out = WIDTH_L1_ZERO_POINT + (2**WIDTH_OUT_DATA-1);
                else
                  data_out = {1'b0,
                                  data_extended[WIDTH_BIAS-WIDTH_L1_IN_IL-1+WIDTH_L1_OUT_IL:
                                                WIDTH_BIAS-WIDTH_L1_IN_IL-1],
                                  data_extended[WIDTH_BIAS-WIDTH_L1_IN_IL-1:
                                                WIDTH_BIAS-WIDTH_L1_IN_IL-1-WIDTH_L1_OUT_FL]};
              end
            end
          end
          3'b010: begin  // L2
            if (relu_on) begin
              if (data_extended[WIDTH_BIAS-1])
                data_out = WIDTH_L2_ZERO_POINT;
              else if (data_extended > relu_q_l2)
                data_out = relu_q_l2;
              else
                data_out = data_extended[WIDTH_IN_DATA - WIDTH_L2_IN_IL - 1 +: WIDTH_OUT_DATA];
            end else begin
              //if ReLU off,  saturate + fixed-point slicing
              if (data_extended[WIDTH_BIAS-1]) begin
                if (data_extended < -(2**(WIDTH_OUT_DATA-1) << (WIDTH_BIAS-WIDTH_L2_IN_IL-1-WIDTH_L2_OUT_FL)))
                  data_out = WIDTH_L2_ZERO_POINT;
                else
                  data_out = {1'b1,
                                  data_extended[WIDTH_BIAS-WIDTH_L2_IN_IL-1+WIDTH_L2_OUT_IL:
                                                WIDTH_BIAS-WIDTH_L2_IN_IL-1],
                                  data_extended[WIDTH_BIAS-WIDTH_L2_IN_IL-1:
                                                WIDTH_BIAS-WIDTH_L2_IN_IL-1-WIDTH_L2_OUT_FL]};
              end else begin
                if (data_extended > ((2**(WIDTH_OUT_DATA-1)-1) << (WIDTH_BIAS-WIDTH_L2_IN_IL-1-WIDTH_L2_OUT_FL)))
                  data_out = WIDTH_L2_ZERO_POINT + (2**WIDTH_OUT_DATA-1);
                else
                  data_out = {1'b0,
                                  data_extended[WIDTH_BIAS-WIDTH_L2_IN_IL-1+WIDTH_L2_OUT_IL:
                                                WIDTH_BIAS-WIDTH_L2_IN_IL-1],
                                  data_extended[WIDTH_BIAS-WIDTH_L2_IN_IL-1:
                                                WIDTH_BIAS-WIDTH_L2_IN_IL-1-WIDTH_L2_OUT_FL]};
              end
            end
          end

          3'b011: begin  // L3
            if (relu_on) begin
              if (data_extended[WIDTH_BIAS-1])
                data_out = WIDTH_L3_ZERO_POINT;
              else if (data_extended > relu_q_l3)
                data_out = relu_q_l3;
              else
                data_out = data_extended[WIDTH_IN_DATA - WIDTH_L3_IN_IL - 1 +: WIDTH_OUT_DATA];
            end else begin
              // if ReLU off,  saturate + fixed-point slicing
              if (data_extended[WIDTH_BIAS-1]) begin
                if (data_extended < -(2**(WIDTH_OUT_DATA-1) << (WIDTH_BIAS-WIDTH_L3_IN_IL-1-WIDTH_L3_OUT_FL)))
                  data_out = WIDTH_L3_ZERO_POINT;
                else
                  data_out = {1'b1,
                                  data_extended[WIDTH_BIAS-WIDTH_L3_IN_IL-1+WIDTH_L3_OUT_IL:
                                                WIDTH_BIAS-WIDTH_L3_IN_IL-1],
                                  data_extended[WIDTH_BIAS-WIDTH_L3_IN_IL-1:
                                                WIDTH_BIAS-WIDTH_L3_IN_IL-1-WIDTH_L3_OUT_FL]};
              end else begin
                if (data_extended > ((2**(WIDTH_OUT_DATA-1)-1) << (WIDTH_BIAS-WIDTH_L3_IN_IL-1-WIDTH_L3_OUT_FL)))
                  data_out = WIDTH_L3_ZERO_POINT + (2**WIDTH_OUT_DATA-1);
                else
                  data_out = {1'b0,
                                  data_extended[WIDTH_BIAS-WIDTH_L3_IN_IL-1+WIDTH_L3_OUT_IL:
                                                WIDTH_BIAS-WIDTH_L3_IN_IL-1],
                                  data_extended[WIDTH_BIAS-WIDTH_L3_IN_IL-1:
                                                WIDTH_BIAS-WIDTH_L3_IN_IL-1-WIDTH_L3_OUT_FL]};
              end
            end
          end

          3'b100: begin  // L4
            if (relu_on) begin
              if (data_extended[WIDTH_BIAS-1])
                data_out = WIDTH_L4_ZERO_POINT;
              else if (data_extended > relu_q_l4)
                data_out = relu_q_l4;
              else
                data_out = data_extended[WIDTH_IN_DATA - WIDTH_L4_IN_IL - 1 +: WIDTH_OUT_DATA];
            end else begin
              // if ReLU off,  saturate + fixed-point slicing
              if (data_extended[WIDTH_BIAS-1]) begin
                if (data_extended < -(2**(WIDTH_OUT_DATA-1) << (WIDTH_BIAS-WIDTH_L4_IN_IL-1-WIDTH_L4_OUT_FL)))
                  data_out = WIDTH_L4_ZERO_POINT;
                else
                  data_out = {1'b1,
                                  data_extended[WIDTH_BIAS-WIDTH_L4_IN_IL-1+WIDTH_L4_OUT_IL:
                                                WIDTH_BIAS-WIDTH_L4_IN_IL-1],
                                  data_extended[WIDTH_BIAS-WIDTH_L4_IN_IL-1:
                                                WIDTH_BIAS-WIDTH_L4_IN_IL-1-WIDTH_L4_OUT_FL]};
              end else begin
                if (data_extended > ((2**(WIDTH_OUT_DATA-1)-1) << (WIDTH_BIAS-WIDTH_L4_IN_IL-1-WIDTH_L4_OUT_FL)))
                  data_out = WIDTH_L4_ZERO_POINT + (2**WIDTH_OUT_DATA-1);
                else
                  data_out = {1'b0,
                                  data_extended[WIDTH_BIAS-WIDTH_L4_IN_IL-1+WIDTH_L4_OUT_IL:
                                                WIDTH_BIAS-WIDTH_L4_IN_IL-1],
                                  data_extended[WIDTH_BIAS-WIDTH_L4_IN_IL-1:
                                                WIDTH_BIAS-WIDTH_L4_IN_IL-1-WIDTH_L4_OUT_FL]};
              end
            end
          end

          3'b101: begin  // L5
            if (relu_on) begin
              if (data_extended[WIDTH_BIAS-1])
                data_out = WIDTH_L5_ZERO_POINT;
              else if (data_extended > relu_q_l5)
                data_out = relu_q_l5;
              else
                data_out = data_extended[WIDTH_IN_DATA - WIDTH_L5_IN_IL - 1 +: WIDTH_OUT_DATA];
            end else begin
              // if ReLU off,  saturate + fixed-point slicing
              if (data_extended[WIDTH_BIAS-1]) begin
                if (data_extended < -(2**(WIDTH_OUT_DATA-1) << (WIDTH_BIAS-WIDTH_L5_IN_IL-1-WIDTH_L5_OUT_FL)))
                  data_out = WIDTH_L5_ZERO_POINT;
                else
                  data_out = {1'b1,
                                  data_extended[WIDTH_BIAS-WIDTH_L5_IN_IL-1+WIDTH_L5_OUT_IL:
                                                WIDTH_BIAS-WIDTH_L5_IN_IL-1],
                                  data_extended[WIDTH_BIAS-WIDTH_L5_IN_IL-1:
                                                WIDTH_BIAS-WIDTH_L5_IN_IL-1-WIDTH_L5_OUT_FL]};
              end else begin
                if (data_extended > ((2**(WIDTH_OUT_DATA-1)-1) << (WIDTH_BIAS-WIDTH_L5_IN_IL-1-WIDTH_L5_OUT_FL)))
                  data_out = WIDTH_L5_ZERO_POINT + (2**WIDTH_OUT_DATA-1);
                else
                  data_out = {1'b0,
                                  data_extended[WIDTH_BIAS-WIDTH_L5_IN_IL-1+WIDTH_L5_OUT_IL:
                                                WIDTH_BIAS-WIDTH_L5_IN_IL-1],
                                  data_extended[WIDTH_BIAS-WIDTH_L5_IN_IL-1:
                                                WIDTH_BIAS-WIDTH_L5_IN_IL-1-WIDTH_L5_OUT_FL]};
              end
            end
          end
          
          default: data_out = {WIDTH_OUT_DATA{1'b0}};
        endcase
      end
    

endmodule


