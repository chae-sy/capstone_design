`timescale 1ns / 1ps
// maxpool_16ch converted to Verilog-2001: no unpacked arrays, no always_ff/always_comb

module maxpool_16ch #(
    parameter DATA_WIDTH           = 8,
    parameter CHANNELS             = 16,
    parameter LINEBUF_RED_BLUE_SIZE= 8,
    parameter LINEBUF_GREEN_SIZE   = 4
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          maxpool_en,
    input  wire [CHANNELS*DATA_WIDTH-1:0] in_data_flat,
    input  wire [1:0]                    color,        // 0=R(4×2),1=G(2×2),2=B(4×2)
    output reg                           maxpool_done_o,
    output reg [CHANNELS*DATA_WIDTH-1:0] out_data_o
);

    localparam signed [DATA_WIDTH-1:0] INIT_MAX = -128;

    // Flattened memories
    reg signed [DATA_WIDTH-1:0] max_val       [0:CHANNELS-1];
    reg signed [DATA_WIDTH-1:0] linebuf_rb    [0:CHANNELS*LINEBUF_RED_BLUE_SIZE-1];
    reg signed [DATA_WIDTH-1:0] linebuf_g     [0:CHANNELS*LINEBUF_GREEN_SIZE-1];

    reg [3:0] wr_ptr    [0:CHANNELS-1];
    reg [3:0] cnt       [0:CHANNELS-1];

    reg signed [DATA_WIDTH-1:0] in_data_reg  [0:CHANNELS-1];
    reg signed [DATA_WIDTH-1:0] out_data_reg [0:CHANNELS-1];

    integer ch, i;

    // Unpack in_data_flat into per‐channel register
    always @(*) begin
        for (ch = 0; ch < CHANNELS; ch = ch + 1) begin
            in_data_reg[ch] = in_data_flat[(ch+1)*DATA_WIDTH-1 -: DATA_WIDTH];
        end
    end

    // Sequential: update pointers & maxima
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (ch = 0; ch < CHANNELS; ch = ch + 1) begin
                wr_ptr[ch]       <= 0;
                cnt[ch]          <= 0;
                max_val[ch]      <= INIT_MAX;
                out_data_reg[ch] <= INIT_MAX;
            end
            maxpool_done_o <= 1'b0;
        end else begin
            maxpool_done_o <= 1'b0;
            for (ch = 0; ch < CHANNELS; ch = ch + 1) begin
                if (maxpool_en) begin
                    // choose line buffer size
                    if (color == 2'b01) begin
                        // green path (2×2 → depth = LINEBUF_GREEN_SIZE)
                        linebuf_g[ch*LINEBUF_GREEN_SIZE + wr_ptr[ch]] <= in_data_reg[ch];
                        if (in_data_reg[ch] > max_val[ch])
                            max_val[ch] <= in_data_reg[ch];
                        wr_ptr[ch] <= wr_ptr[ch] + 1;
                        cnt[ch]    <= cnt[ch] + 1;
                        if (cnt[ch] == LINEBUF_GREEN_SIZE-1) begin
                            out_data_reg[ch] <= max_val[ch];
                            max_val[ch]      <= INIT_MAX;
                            wr_ptr[ch]       <= 0;
                            cnt[ch]          <= 0;
                            // clear buffer entries
                            for (i = 0; i < LINEBUF_GREEN_SIZE; i = i + 1)
                                linebuf_g[ch*LINEBUF_GREEN_SIZE + i] <= INIT_MAX;
                            maxpool_done_o <= 1'b1;
                        end
                    end else begin
                        // red/blue path (4×2 → depth = LINEBUF_RED_BLUE_SIZE)
                        linebuf_rb[ch*LINEBUF_RED_BLUE_SIZE + wr_ptr[ch]] <= in_data_reg[ch];
                        if (in_data_reg[ch] > max_val[ch])
                            max_val[ch] <= in_data_reg[ch];
                        wr_ptr[ch] <= wr_ptr[ch] + 1;
                        cnt[ch]    <= cnt[ch] + 1;
                        if (cnt[ch] == LINEBUF_RED_BLUE_SIZE-1) begin
                            out_data_reg[ch] <= max_val[ch];
                            max_val[ch]      <= INIT_MAX;
                            wr_ptr[ch]       <= 0;
                            cnt[ch]          <= 0;
                            for (i = 0; i < LINEBUF_RED_BLUE_SIZE; i = i + 1)
                                linebuf_rb[ch*LINEBUF_RED_BLUE_SIZE + i] <= INIT_MAX;
                            maxpool_done_o <= 1'b1;
                        end
                    end
                end
            end
        end
    end

    // Pack out_data_reg into out_data_o when done
    always @(posedge clk) begin
        for (ch = 0; ch < CHANNELS; ch = ch + 1) begin
            if (maxpool_done_o)
                out_data_o[(ch+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= out_data_reg[ch];
            else
                out_data_o[(ch+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
        end
    end

endmodule
