`timescale 1ns / 1ps

module output_buffer #(
    parameter t_WIDTH = 128,
    parameter DATA_WIDTH = 8,
    parameter NUM_CHNL = 16,
    parameter NUM_COLOR = 3
    
)(
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             wren_r,
    input  wire                             wren_g,
    input  wire                             wren_b,
    input  wire [DATA_WIDTH-1:0]            data_in_r,
    input  wire [DATA_WIDTH-1:0]            data_in_g,
    input  wire [DATA_WIDTH-1:0]            data_in_b,
    input  wire                             rden_r,
    input  wire                             rden_g,
    input  wire                             rden_b,
    input  wire [2:0]                       layer_num,
    input  wire                             layer_start,
    output reg                              o_buffer_done,
    output reg  [t_WIDTH-1:0]               data_out
);

    // buffer storage
    reg [DATA_WIDTH-1:0] buffer_data_r_array [0:NUM_CHNL-1];
    reg [DATA_WIDTH-1:0] buffer_data_g_array [0:NUM_CHNL-1];
    reg [DATA_WIDTH-1:0] buffer_data_b_array [0:NUM_CHNL-1];
    reg [t_WIDTH-1:0] buffer_data_r;
    reg [t_WIDTH-1:0] buffer_data_g;
    reg [t_WIDTH-1:0] buffer_data_b;
    reg [4:0] cnt[0:NUM_COLOR-1];
    reg [4:0] cnt_n[0:NUM_COLOR-1];
    reg wren[0:NUM_COLOR-1];
    reg rden[0:NUM_COLOR-1];
    integer i;
    
    always @(*) begin
        wren[0] = wren_r;
        wren[1] = wren_g;
        wren[2] = wren_b;
        rden[0] = rden_r;
        rden[1] = rden_g;
        rden[2] = rden_b;
    end
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            for (i = 0; i < NUM_COLOR; i = i + 1) begin
                cnt[i]          <= 0;
            end
        end else begin
            for (i = 0; i < NUM_COLOR; i = i + 1) begin
                cnt[i]          <= cnt_n[i];
            end
            if (wren[0]) buffer_data_r_array[cnt[0]] <= data_in_r;
            if (wren[1]) buffer_data_g_array[cnt[1]] <= data_in_g;
            if (wren[2]) buffer_data_b_array[cnt[2]] <= data_in_b;
        end
    end
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            buffer_data_r = 'b0;
            buffer_data_g = 'b0;
            buffer_data_b = 'b0;
        end
    end
    
    generate
        genvar j;
        for (j=0; j<NUM_CHNL; j=j+1) begin
            always @(*) begin
                buffer_data_r[(16-j)*DATA_WIDTH-1:(16-j-1)*DATA_WIDTH] = buffer_data_r_array[j];
                buffer_data_g[(16-j)*DATA_WIDTH-1:(16-j-1)*DATA_WIDTH] = buffer_data_g_array[j];
                buffer_data_b[(16-j)*DATA_WIDTH-1:(16-j-1)*DATA_WIDTH] = buffer_data_b_array[j];
            end
        end
    endgenerate


    always @(*) begin
        o_buffer_done = 0;
        for (i = 0; i < NUM_COLOR; i = i + 1 ) begin
            cnt_n[i] = cnt[i]; 
        end
        case (layer_num)
            3'd6: begin
                for (i = 0; i < NUM_COLOR; i = i + 1 ) begin
                    if (wren[i]) o_buffer_done = 1;
                end

                if (rden[0]) begin
                    data_out = buffer_data_r;
                end
                else if (rden[1]) begin
                    data_out = buffer_data_g;
                end
                else if (rden[2]) begin    
                    data_out = buffer_data_b;
                end
            end
            default: begin
                for (i = 0; i < NUM_COLOR; i = i + 1 ) begin
                    if (wren[i]) cnt_n[i] = cnt[i] + 1;
                end
                if ((cnt[0] == 15) & wren[0]) begin
                    o_buffer_done = 1;
                    cnt_n[0] = 0;
                    cnt_n[1] = 0;
                    cnt_n[2] = 0;
                end
                if (rden[0]) begin
                    data_out = buffer_data_r;
                end
                else if (rden[1]) begin
                    data_out = buffer_data_g;
                end
                else if (rden[2]) begin
                    data_out = buffer_data_b;
                end
            end

        endcase
    end
     
endmodule