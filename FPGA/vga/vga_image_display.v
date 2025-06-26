module vga_image_display_cycle (
    input  wire        sys_clk,    // 100 MHz
    input  wire        rstn,       // active-low reset
    input  wire        sw,         // 외부 스위치
    output wire        hsync,
    output wire        vsync,
    output reg   [3:0] vga_r,      // 4-bit VGA
    output reg   [3:0] vga_g,
    output reg   [3:0] vga_b
);
    // ----------------------------------------------------------------
    // 1) 해상도, 이미지, 확대 정보
    // ----------------------------------------------------------------
    localparam SCR_W   = 1920, SCR_H = 1080;
    localparam IMG_W   = 100,  IMG_H = 100;
    localparam SCALE   = 4;
    localparam DISP_W  = IMG_W * SCALE;  // 400
    localparam DISP_H  = IMG_H * SCALE;  // 400

    // A/B 공통 위치 (가로 200px 여백, 세로 중앙 정렬)
    localparam A_X = 200, 
               A_Y = (SCR_H - DISP_H) / 2;   // 340
    localparam B_X = SCR_W - 200 - DISP_W,  // 1920-200-400=1320
               B_Y = A_Y;

    // ----------------------------------------------------------------
    // 2) 클록 생성 (148.5 MHz) & 동기화
    // ----------------------------------------------------------------
    wire        clk_pixel, pixel_clk_locked;
    clk_wiz_0 u_clk_wiz (
        .clk_in1   (sys_clk),
        .reset    (~rstn),
        .clk_out1  (clk_pixel),
        .locked    (pixel_clk_locked)
    );
    wire active = rstn & pixel_clk_locked;

    wire        video_on;
    wire [11:0] px;
    wire [10:0] py;
    vga_sync u_sync (
        .clk       (clk_pixel),
        .resetn    (active),
        .hsync     (hsync),
        .vsync     (vsync),
        .video_on  (video_on),
        .pixel_x   (px),
        .pixel_y   (py)
    );

    // ----------------------------------------------------------------
    // 3) 스위치 엣지 디텍션 & 인덱스 순환 (A1→A2→A3→A1→…)
    // ----------------------------------------------------------------
    reg prev_sw;
    reg [1:0] idx;  // 0→1→2 (A1/A2/A3)
    always @(posedge clk_pixel or negedge rstn) begin
        if (!rstn) begin
            prev_sw <= 1'b0;
            idx     <= 2'd0;
        end else begin
            // sw가 1→0으로 떨어질 때마다 idx 증가
            if (prev_sw && !sw)
                idx <= (idx == 2) ? 2'd0 : idx + 1;
            prev_sw <= sw;
        end
    end

    // ----------------------------------------------------------------
    // 4) 확대를 위한 원본 좌표 계산
    // ----------------------------------------------------------------
    // A 영역, B 영역 판정
    wire within_A = video_on
                  && px >= A_X && px < A_X + DISP_W
                  && py >= A_Y && py < A_Y + DISP_H;
    wire within_B = video_on
                  && px >= B_X && px < B_X + DISP_W
                  && py >= B_Y && py < B_Y + DISP_H;

    // 화면→원본 맵 (÷SCALE)
    wire [6:0] src_x_A = (px - A_X) / SCALE;  // 0..99
    wire [6:0] src_y_A = (py - A_Y) / SCALE;
    wire [13:0] addr_A = src_y_A * IMG_W + src_x_A;

    wire [6:0] src_x_B = (px - B_X) / SCALE;
    wire [6:0] src_y_B = (py - B_Y) / SCALE;
    wire [13:0] addr_B = src_y_B * IMG_W + src_x_B;

    // ----------------------------------------------------------------
    // 5) ROM 인스턴스 (각각 12-bit COE 로 초기화된 블록 메모리)
    //    - a1, b1, a2, b2, a3, b3 총 6개
    // ----------------------------------------------------------------
    wire [11:0] imgA1, imgB1, imgA2, imgB2, imgA3, imgB3;

    blk_mem_gen_0 u_a1 (
        .clka  (clk_pixel), .ena(1'b1), .addra(addr_A), .douta(imgA1)
    );
    blk_mem_gen_1 u_b1 (
        .clka  (clk_pixel), .ena(1'b1), .addra(addr_B), .douta(imgB1)
    );
    blk_mem_gen_2 u_a2 (
        .clka  (clk_pixel), .ena(1'b1), .addra(addr_A), .douta(imgA2)
    );
    blk_mem_gen_3 u_b2 (
        .clka  (clk_pixel), .ena(1'b1), .addra(addr_B), .douta(imgB2)
    );
    blk_mem_gen_4 u_a3 (
        .clka  (clk_pixel), .ena(1'b1), .addra(addr_A), .douta(imgA3)
    );
    blk_mem_gen_5 u_b3 (
        .clka  (clk_pixel), .ena(1'b1), .addra(addr_B), .douta(imgB3)
    );

    // ----------------------------------------------------------------
    // 6) 현재 idx에 맞게 A/B 데이터 선택
    // ----------------------------------------------------------------
    wire [11:0] selA = (idx == 2'd0) ? imgA1
                       : (idx == 2'd1) ? imgA2
                                       : imgA3;
    wire [11:0] selB = (idx == 2'd0) ? imgB1
                       : (idx == 2'd1) ? imgB2
                                       : imgB3;

    // ----------------------------------------------------------------
    // 7) 출력 로직
    // ----------------------------------------------------------------
    always @(posedge clk_pixel) begin
        if (within_A) begin
            // A는 항상 표시
            vga_r <= selA[11:8];
            vga_g <= selA[ 7:4];
            vga_b <= selA[ 3:0];
        end
        else if (sw && within_B) begin
            // sw=1일 때만 B도 표시
            vga_r <= selB[11:8];
            vga_g <= selB[ 7:4];
            vga_b <= selB[ 3:0];
        end
        else begin
            // 나머지 배경은 검정
            vga_r <= 4'd0;
            vga_g <= 4'd0;
            vga_b <= 4'd0;
        end
    end

endmodule
