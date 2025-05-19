`timescale 1ns/1ps

module maxpool_16chnl#(
    parameter DATA_WIDTH = 8,
    parameter CHANNELS = 16,
    parameter LINEBUF_RED_BLUE_SIZE = 8,
    parameter LINEBUF_GREEN_SIZE = 4
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         maxpool_en,
    input  wire [1:0]                   color, // r=0 (4x2), g=1 (4x1), b=2 (4x2)
    input  wire signed [DATA_WIDTH-1:0] in_data   [0:CHANNELS-1],
    output wire                         maxpool_done_o,
    output wire signed [DATA_WIDTH-1:0] out_data_o[0:CHANNELS-1]
);

    localparam signed [DATA_WIDTH-1:0] INIT_MAX = -128;

    reg signed [DATA_WIDTH-1:0] max_val [0:CHANNELS-1];
    reg signed [DATA_WIDTH-1:0] max_val_n [0:CHANNELS-1];

    reg signed [DATA_WIDTH-1:0] linebuf_rb [0:CHANNELS-1][0:LINEBUF_RED_BLUE_SIZE-1];
    reg signed [DATA_WIDTH-1:0] linebuf_g  [0:CHANNELS-1][0:LINEBUF_GREEN_SIZE-1];

    reg [3:0] wr_ptr [0:CHANNELS-1];
    reg [3:0] wr_ptr_n [0:CHANNELS-1];

    reg [3:0] cnt [0:CHANNELS-1];
    reg [3:0] cnt_n [0:CHANNELS-1];

    reg maxpool_done;
    reg maxpool_done_n;
    reg signed [DATA_WIDTH-1:0] out_data [0:CHANNELS-1];
    reg meaningless;

    integer ch, i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int ch = 0; ch < CHANNELS; ch = ch + 1) begin
                wr_ptr[ch]       <= 0;
                cnt[ch]          <= 0;
                max_val[ch]      <= INIT_MAX;
            end
            maxpool_done <= 0;
        end else begin
            for (int ch = 0; ch < CHANNELS; ch = ch + 1) begin
                wr_ptr[ch]       <= wr_ptr_n[ch];
                cnt[ch]          <= cnt_n[ch];
                max_val[ch]      <= max_val_n[ch];
            end
            maxpool_done <= maxpool_done_n;
        end
    end
    
    reg signed [DATA_WIDTH-1:0] in_data_reg [0:CHANNELS-1];
    
    always_ff @(posedge clk) begin
        if (maxpool_en) begin
            for (int ch = 0; ch < CHANNELS; ch++) begin
                in_data_reg[ch] <= in_data[ch];
            end
        end
    end
    
    always_comb begin
        maxpool_done_n = 0;
        for (ch = 0; ch < CHANNELS; ch = ch + 1) begin
            wr_ptr_n[ch] = wr_ptr[ch];
            cnt_n[ch] = cnt[ch];
            max_val_n[ch] = max_val[ch];
        
            if (maxpool_en) begin
                if (color == 2'b01) begin // green (4x1)
                    wr_ptr_n[ch] = wr_ptr[ch] + 1;
                    cnt_n[ch] = cnt[ch] + 1;
                    
                    linebuf_g[ch][wr_ptr[ch]] = in_data_reg[ch]; 
                    if (linebuf_g[ch][wr_ptr[ch]] > max_val[ch])
                        max_val_n[ch] = linebuf_g[ch][wr_ptr[ch]];
                    
                    if (cnt[ch] == LINEBUF_GREEN_SIZE-1) begin
                        out_data[ch] = max_val_n[ch];
                        max_val_n[ch] = INIT_MAX;
                        wr_ptr_n[ch] = 0;
                        cnt_n[ch] = 0;
                        for (i = 0; i < 4; i = i + 1) linebuf_g[ch][i] = INIT_MAX;
                        maxpool_done_n = 1;
                    end
                end else begin // red or blue (4x2)
                    wr_ptr_n[ch] = wr_ptr[ch] + 1;
                    cnt_n[ch] = cnt[ch] + 1;
                    
                    linebuf_rb[ch][wr_ptr[ch]] = in_data_reg[ch]; 
                    if (linebuf_rb[ch][wr_ptr[ch]] > max_val[ch])
                        max_val_n[ch] = linebuf_rb[ch][wr_ptr[ch]];

                    if (cnt[ch] == LINEBUF_RED_BLUE_SIZE-1) begin
                        out_data[ch] = max_val_n[ch];
                        max_val_n[ch] = INIT_MAX;
                        wr_ptr_n[ch] = 0;
                        cnt_n[ch] = 0;
                        for (i = 0; i < 8; i = i + 1) linebuf_rb[ch][i] = INIT_MAX;
                        maxpool_done_n = 1;
                    end
                end
            end
        end
    end

    assign maxpool_done_o = maxpool_done;
    generate
        for (genvar ch = 0; ch < CHANNELS; ch = ch + 1) begin : output_assign
            assign out_data_o[ch] = out_data[ch];
        end
    endgenerate

endmodule
