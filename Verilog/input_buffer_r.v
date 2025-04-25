
module input_buffer_r #(
    parameter DATA_WIDTH = 8,
    parameter IMG_WIDTH = 16  // image width in pixels
)(
    input clk,
    input rstn,
    input input_valid,
    input [DATA_WIDTH-1:0] data_in,
    input [1:0] channel_sel,   // 0: R, 1: G, 2: B
    output reg window_valid,
    output reg [DATA_WIDTH-1:0] window_out [0:2][0:4]
);

    // Buffering only R channel
    reg [1:0] rgb_counter;
    wire is_red = (rgb_counter == 2'd0);

    // Three rows, 5 columns each
    reg [DATA_WIDTH-1:0] line_buffer_1 [0:IMG_WIDTH-1];
    reg [DATA_WIDTH-1:0] line_buffer_2 [0:IMG_WIDTH-1];
    reg [DATA_WIDTH-1:0] current_line [0:IMG_WIDTH-1];

    // Coordinates
    integer col;
    integer i, j;

    // Address pointers
    reg [$clog2(IMG_WIDTH)-1:0] col_ptr;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            rgb_counter <= 2'd0;
            col_ptr <= 0;
            window_valid <= 0;
        end else if (input_valid) begin
            rgb_counter <= (rgb_counter == 2) ? 0 : rgb_counter + 1;

            if (is_red) begin
                // Shift data in
                current_line[col_ptr] <= data_in;

                // Advance column pointer
                if (col_ptr == IMG_WIDTH-1)
                    col_ptr <= 0;
                else
                    col_ptr <= col_ptr + 1;
            end
        end
    end

    // Generate output window
    always @(posedge clk) begin
        
    if (is_red && col_ptr >= 4 && col_ptr <= IMG_WIDTH - 1) begin

            for (i = 0; i < 3; i = i + 1) begin
                for (j = 0; j < 5; j = j + 1) begin
                    case (i)
                        0: window_out[i][j] <= line_buffer_2[col_ptr - 4 + j];
                        1: window_out[i][j] <= line_buffer_1[col_ptr - 4 + j];
                        2: window_out[i][j] <= current_line[col_ptr - 4 + j];
                    endcase
                end
            end
            window_valid <= 1;
        end else begin
            window_valid <= 0;
        end
    end

    // Rotate buffers every IMG_WIDTH red pixels
    reg [$clog2(IMG_WIDTH)-1:0] pixel_count;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            pixel_count <= 0;
        end else if (input_valid && is_red) begin
            pixel_count <= pixel_count + 1;
            if (pixel_count == IMG_WIDTH - 1) begin
                pixel_count <= 0;
                for (i = 0; i < IMG_WIDTH; i = i + 1) begin
                    line_buffer_2[i] <= line_buffer_1[i];
                    line_buffer_1[i] <= current_line[i];
                end
            end
        end
    end
endmodule
