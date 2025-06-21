// File: top.v
module display_top (
    input  wire        clk,   // 100 MHz 입력 클럭
    input  wire        rst_n,  // Active-Low 리셋
    input  wire        switch_i,          // 스위치 (SW[0])
    output reg         LED,          // LED (LED[0])
     // VGA 출력
    output wire [3:0]  VGA_R,
    output wire [3:0]  VGA_G,
    output wire [3:0]  VGA_B,
    output wire        VGA_HS,
    output wire        VGA_VS
);
    localparam S_IDLE    = 4'b0001;
    localparam S_NN      = 4'b0010;
    localparam S_DISPLAY = 4'b0100;
    localparam S_DONE    = 4'b1000;

    reg [3:0] state, next_state;

    //------------------------------------------------
    // 상태 레지스터
    //------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;          // 리셋 시 IDLE
        else
            state <= next_state;
    end

    //------------------------------------------------
    // Next-state 로직
    //------------------------------------------------
    always @* begin
        // 기본값: 현재 상태 유지
        next_state = state;

        case (state)
            //------------------------------------------------
            S_IDLE: begin
                if (switch_i)             // if switch
                    next_state = S_NN;
            end
            //------------------------------------------------
            S_NN: begin
                if (total_done_i)         // if total_done
                    next_state = S_DISPLAY;
            end
            //------------------------------------------------
            S_DISPLAY: begin
                if (display_done_i)       // if display_done
                    next_state = S_DONE;
            end
            //------------------------------------------------
            S_DONE: begin
                if (switch_i)             // if switch
                    next_state = S_NN;    // 다시 NN 사이클
            end
        endcase
    end

 
    SPRNN_Top u_sprnn (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .memA_addr_i(memA_addr_i),
    .memB_addr_i(memB_addr_i),
    .wmem_addr_i(wmem_addr_i),
    .memA_d_i(memA_d_i),
    .memB_d_i(memB_d_i),
    .wmem_d_i(wmem_d_i),
    .wren_bias_i(wren_bias_i),
    .write_addr_bias_i(write_addr_bias_i),
    .write_data_bias_i(write_data_bias_i),
    .initial_SRAMw_done(initial_SRAMw_done),
    .initial_weight_done(initial_weight_done),
    .layer_num_o(layer_num_o),
    .layer_done_o(layer_done_o),
    .total_done_o(total_done_i)

);

  //==================================================
  // 1) Clock & VGA 타이밍
  //==================================================
  wire clk_pix, clk_locked;
  clk_wiz_0 u_clk (
    .clk_in1  (clk),
    .reset   (~rstn),
    .clk_pix  (clk_pix),
    .locked   (clk_locked)
  );
  wire active = rstn & clk_locked;

  wire        video_on;
  wire [11:0] pixel_x, pixel_y;
  vga_controller_1080p u_vga (
    .clk_pix   (clk_pix),
    .rstn      (active),
    .hsync     (vga_hs),
    .vsync     (vga_vs),
    .video_on  (video_on),
    .pixel_x   (pixel_x),
    .pixel_y   (pixel_y)
  );
  
  
  assign VGA_R = pixel_color[11:8];
  assign VGA_G = pixel_color[ 7:4];
  assign VGA_B = pixel_color[ 3:0];


endmodule

