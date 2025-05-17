`timescale 1ns/1ps

module maxpool#(
    parameter DATA_WIDTH = 8,
    parameter LINEBUF_RED_BLUE_SIZE = 8,
    parameter LINEBUF_GREEN_SIZE = 4
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        maxpool_en,
    input  wire [1:0]                  color, // r=0 (4x2), g=1 (4x1), b=2 (4x2)
    input  wire signed [DATA_WIDTH-1:0] in_data,
    output wire                         maxpool_done_o,
    output wire signed [DATA_WIDTH-1:0] out_data_o
);

    // initial max value (min value) (-128 for signed 8-bit)
    localparam signed [DATA_WIDTH-1:0] INIT_MAX = -128;
    reg signed [DATA_WIDTH-1:0] out_data;
    reg signed [DATA_WIDTH-1:0] max_val, max_val_n; //max value, next max value
    reg signed [1:0] maxpool_done, maxpool_done_n; //maxpool done, next maxpool done

    reg signed [DATA_WIDTH-1:0] linebuf_rb [0:LINEBUF_GREEN_SIZE-1];
    reg signed [DATA_WIDTH-1:0] linebuf_g [0:LINEBUF_GREEN_SIZE-1];

    // pointer
    reg [3:0] wr_ptr, wr_ptr_n;
    reg [3:0] cnt, cnt_n;


    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr       <= 0;
            cnt          <= 0;
            maxpool_done <= 0;
            max_val      <= INIT_MAX;
        end else begin
            wr_ptr       <= wr_ptr_n;
            cnt          <= cnt_n;
            maxpool_done <= maxpool_done_n;
            max_val      <= max_val_n;
        end
    end

 
    always_comb begin
        wr_ptr_n     = wr_ptr;
        cnt_n        = cnt;
        maxpool_done_n  = 0;
        max_val_n =  max_val;
        
        if (maxpool_en) begin
            if ( color == 'b1) begin // green: (4x1) maxpool
                linebuf_g[wr_ptr] = in_data;
                wr_ptr_n = wr_ptr + 1;
                cnt_n = cnt + 1;
                if (linebuf_g[wr_ptr] > max_val) begin
                    max_val_n = linebuf_g[wr_ptr];
                end
                else begin
                    max_val_n = max_val;
                end
                if (cnt == LINEBUF_GREEN_SIZE-1) begin // maxpool done
                    out_data = max_val_n;
                    // reset everything
                    maxpool_done_n  = 1;
                    wr_ptr_n = 0;
                    cnt_n = 0;
                    max_val_n = INIT_MAX;
                    for (int i = 0; i < 4; i = i + 1) begin
                        linebuf_g[i] = INIT_MAX;
                    end
                end
                else begin
                    maxpool_done_n  = 0;
                end
            end
            else begin // red & blue: (4x2) maxpool
                linebuf_rb[wr_ptr] = in_data;
                wr_ptr_n = wr_ptr + 1;
                cnt_n = cnt + 1;
                if (linebuf_rb[wr_ptr] > max_val) begin
                    max_val_n = linebuf_rb[wr_ptr];
                end
                else begin
                    max_val_n = max_val;
                end
                if (cnt == LINEBUF_RED_BLUE_SIZE-1) begin
                    out_data = max_val_n;
                    maxpool_done_n  = 1;
                    wr_ptr_n = 0;
                    cnt_n = 0;
                    max_val_n = INIT_MAX;
                    for (int i = 0; i < 4; i = i + 1) begin
                        linebuf_rb[i] = INIT_MAX;
                    end
                end
                else begin
                    maxpool_done_n  = 0;
                end
            end
        end
    end
    
    assign maxpool_done_o = maxpool_done;
    assign out_data_o = out_data;

endmodule
