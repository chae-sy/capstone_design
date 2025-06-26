// 1) VGA 타이밍 생성기 (1920×1080 @60 Hz)
module vga_sync (
    input  wire clk,        // 148.5 MHz pixel clock for 1080p60
    input  wire resetn,     // active-low reset
    output reg  hsync,
    output reg  vsync,
    output wire video_on,
    output reg  [11:0] pixel_x,  // 0…1919
    output reg  [10:0] pixel_y   // 0…1079
);

    // 화면 크기
    localparam SCR_W      = 1920;
    localparam SCR_H      = 1080;

    // horizontal parameters for 1080p60
    localparam H_FRONT    = 88;
    localparam H_PULSE    = 44;
    localparam H_BACK     = 148;
    localparam H_TOTAL    = SCR_W + H_FRONT + H_PULSE + H_BACK; // 2200

    // vertical parameters for 1080p60
    localparam V_FRONT    = 4;
    localparam V_PULSE    = 5;
    localparam V_BACK     = 36;
    localparam V_TOTAL    = SCR_H + V_FRONT + V_PULSE + V_BACK; // 1125

    reg [11:0] h_count;
    reg [10:0] v_count;

    // horizontal counter
    always @(posedge clk or negedge resetn) begin
        if (!resetn)
            h_count <= 0;
        else if (h_count == H_TOTAL-1)
            h_count <= 0;
        else
            h_count <= h_count + 1;
    end

    // vertical counter
    always @(posedge clk or negedge resetn) begin
        if (!resetn)
            v_count <= 0;
        else if (h_count == H_TOTAL-1) begin
            if (v_count == V_TOTAL-1)
                v_count <= 0;
            else
                v_count <= v_count + 1;
        end
    end

    // hsync, vsync (active low)
    always @(posedge clk) begin
        hsync <= ~(h_count >= (SCR_W + H_FRONT) && h_count < (SCR_W + H_FRONT + H_PULSE));
        vsync <= ~(v_count >= (SCR_H + V_FRONT) && v_count < (SCR_H + V_FRONT + V_PULSE));
    end

    // 화면 표시 영역
    assign video_on = (h_count < SCR_W) && (v_count < SCR_H);

    // 현재 픽셀 좌표
    always @(posedge clk) begin
        pixel_x <= (h_count < SCR_W) ? h_count : 0;
        pixel_y <= (v_count < SCR_H) ? v_count : 0;
    end

endmodule
