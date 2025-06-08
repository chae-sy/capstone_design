`timescale 1ns / 1ps
// output_buffer converted to Verilog-2001: no unpacked arrays or SystemVerilog loops

module output_buffer #(
    parameter T_WIDTH    = 128,
    parameter DATA_WIDTH = 8,
    parameter NUM_CHNL   = 16,
    parameter NUM_COLOR  = 3
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire [NUM_COLOR-1:0]     wren,         // {wren[2],wren[1],wren[0]}
    input  wire [DATA_WIDTH-1:0]    data_in_r,
    input  wire [DATA_WIDTH-1:0]    data_in_g,
    input  wire [DATA_WIDTH-1:0]    data_in_b,
    input  wire [NUM_COLOR-1:0]     rden,         // {rden[2],rden[1],rden[0]}
    input  wire [2:0]               layer_num,
    output reg                      o_buffer_done,
    output reg [T_WIDTH-1:0]        data_out
);

    // per-channel FIFOs stored as packed buses
    reg [DATA_WIDTH-1:0] buffer_r  [0:NUM_CHNL-1];
    reg [DATA_WIDTH-1:0] buffer_g  [0:NUM_CHNL-1];
    reg [DATA_WIDTH-1:0] buffer_b  [0:NUM_CHNL-1];

    reg [4:0] cnt      [0:NUM_COLOR-1];
    reg [4:0] cnt_n    [0:NUM_COLOR-1];

    integer i;

    // sequential block: write pointers and storage
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_buffer_done <= 1'b0;
            for (i = 0; i < NUM_COLOR; i = i + 1)
                cnt[i] <= 5'd0;
        end else begin
            o_buffer_done <= 1'b0;
            for (i = 0; i < NUM_COLOR; i = i + 1)
                cnt[i] <= cnt_n[i];

            // write into the appropriate buffer
            if (wren[0])
                buffer_r[cnt[0]] <= data_in_r;
            if (wren[1])
                buffer_g[cnt[1]] <= data_in_g;
            if (wren[2])
                buffer_b[cnt[2]] <= data_in_b;
        end
    end

    // combinational: next cnt and outputs
    always @(*) begin
        // default
        for (i = 0; i < NUM_COLOR; i = i + 1)
            cnt_n[i] = cnt[i];

        o_buffer_done = 1'b0;
        data_out      = {T_WIDTH{1'b0}};

        case (layer_num)
            3'd6: begin
                // on final layer, done as soon as any write occurs
                if (wren != 3'b000)
                    o_buffer_done = 1'b1;

                // on read enable, output only LSB DATA_WIDTH bits
                if (rden[0])
                    data_out = {{(T_WIDTH-DATA_WIDTH){1'b0}}, buffer_r[0]};
                else if (rden[1])
                    data_out = {{(T_WIDTH-DATA_WIDTH){1'b0}}, buffer_g[0]};
                else if (rden[2])
                    data_out = {{(T_WIDTH-DATA_WIDTH){1'b0}}, buffer_b[0]};
            end

            default: begin
                // increment counters on write
                if (wren[0]) cnt_n[0] = cnt[0] + 1;
                if (wren[1]) cnt_n[1] = cnt[1] + 1;
                if (wren[2]) cnt_n[2] = cnt[2] + 1;

                // when buffers fill, signal done & reset
                if (cnt[0] == NUM_CHNL-1 && wren[0]) begin
                    o_buffer_done = 1'b1;
                    cnt_n[0] = 5'd0;
                    cnt_n[1] = 5'd0;
                    cnt_n[2] = 5'd0;
                end

                // read path: output full concatenated buffer_data_<color>
                if (rden[0]) begin
                    // pack buffer_r into data_out
                    for (i = 0; i < NUM_CHNL; i = i + 1)
                        data_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = buffer_r[i];
                end else if (rden[1]) begin
                    for (i = 0; i < NUM_CHNL; i = i + 1)
                        data_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = buffer_g[i];
                end else if (rden[2]) begin
                    for (i = 0; i < NUM_CHNL; i = i + 1)
                        data_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = buffer_b[i];
                end
            end
        endcase
    end

endmodule
