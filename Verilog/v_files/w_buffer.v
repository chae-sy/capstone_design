`timescale 1ns / 1ps

module w_buffer #(
    parameter WIDTH_FSRAM_WL  = 128,   // SRAM?—?„œ ?•œ ë²ˆì— ?½?–´?˜¤?Š” ë¹„íŠ¸ ?­ (?˜ˆ: 8bitÃ—16ì±„ë„=128)
    parameter DATA_WIDTH      = 8,     // ?•œ ì±„ë„?‹¹ ?°?´?„° ?­
    parameter NUM_CHNL        = 16,    // ì±„ë„ ?ˆ˜
    parameter SIZE_BUFFER_H   = 3,     // ë²„í¼ ?„¸ë¡? ?¬ê¸? (?–‰ ê°œìˆ˜)
    parameter SIZE_BUFFER_W   = 3,     // ë²„í¼ ê°?ë¡? ?¬ê¸? (?—´ ê°œìˆ˜)
    parameter SIZE_KERNEL_H   = 3,     // ì»¤ë„ ?„¸ë¡? ?¬ê¸? (?˜ˆ: 3)
    parameter SIZE_KERNEL_W   = 3      // ì»¤ë„ ê°?ë¡? ?¬ê¸? (?˜ˆ: 3)
)(
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire                           wren,           // ?™¸ë¶??—?„œ feature ë¡œë“œ ?—ˆ?š©
    input  wire                           rden,           // ?°?´?„° ì¶œë ¥ ?—ˆ?š©
    input  wire [WIDTH_FSRAM_WL-1:0]      data_in,        // SRAM ?—?„œ ?½?–´?˜¨ 128bit
    input  wire                           layer_start,
    output reg [DATA_WIDTH*NUM_CHNL-1:0]  data_out,       // (ì¶œë ¥) ê°? ì±„ë„ë³„ë¡œ 8bit?”© ë¬¶ìŒ
    output reg                            w_buffer_done   // ë¦¬í„´: ì´ˆê¸° ?˜?Š” ?›„?† ë¡œë“œ/ì¶œë ¥?´ ??‚¬?Œ?„ ?•Œë¦?
);

    //================================================================
    // 1) ?‚´ë¶? ë²„í¼: 3D ë°°ì—´ ?„ ?–¸ (SIZE_BUFFER_H Ã— SIZE_BUFFER_W Ã— NUM_CHNL)
    //    ?†’ buffer_data[row][col][channel]
    //================================================================
    reg [NUM_CHNL*DATA_WIDTH-1:0] buffer_data [0:SIZE_BUFFER_H-1][0:SIZE_BUFFER_W-1];
    reg [WIDTH_FSRAM_WL-1:0] data_in_reg;

    //================================================================
    // 2) ?½ê¸?/?“°ê¸? ì¹´ìš´?„°: ì´ˆê¸° ë¡œë”©, ?›„?† ë¡œë”©, ì¶œë ¥ ?‹œ ê°ê° ?”°ë¡? ê´?ë¦?
    //    load_cnt : initial=1?´ë©? 0~8 (3Ã—3), is_initial=0 ?´ë©? 0~(SIZE_BUFFER_H-1)
    //    out_cnt  : 0~(SIZE_KERNEL_H*SIZE_KERNEL_W-1) ?™?•ˆ tap ì¶œë ¥
    //================================================================
    reg [5:0] load_cnt;  // ìµœë? 9 ?˜?Š” 3 ê¹Œì? ì¹´ìš´?Œ…(6ë¹„íŠ¸ë©? ì¶©ë¶„)
    reg [5:0] out_cnt, out_cnt_n;   // ìµœë? 9ê¹Œì? ì¹´ìš´?Œ…
    reg wren_d;
    // ? •?ˆ˜ ë°˜ë³µë¬¸ìš© ë³??ˆ˜
    integer r, c;

    //================================================================
    // 4) ë©”ì¸ always ë¸”ë¡: ë¦¬ì…‹, ?“°ê¸?(wren), ?½ê¸?(rden) ?ˆœ?œ¼ë¡? ë¶„ê¸°
    //================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            for (r = 0; r < SIZE_BUFFER_H; r = r + 1)
                for (c = 0; c < SIZE_BUFFER_W; c = c + 1)
                    buffer_data[r][c] <= {NUM_CHNL*DATA_WIDTH{1'b0}};
            load_cnt      <= 0;
            out_cnt       <= 0;
            w_buffer_done  <= 1'b0;
        end else begin
            w_buffer_done  <= 1'b0;
            out_cnt <= out_cnt_n;
            wren_d <= wren;
            if (wren) begin
                data_in_reg <= data_in;
            end
            // ?“°ê¸? ë¡œì§
            if (wren_d) begin
                if (load_cnt < SIZE_KERNEL_H * SIZE_KERNEL_W) begin
                    buffer_data[ load_cnt / SIZE_KERNEL_W ][ load_cnt % SIZE_KERNEL_W ] <= data_in_reg;
                    load_cnt <= load_cnt + 1;
                end
                if (load_cnt == (SIZE_KERNEL_H * SIZE_KERNEL_W - 1)) begin
                    w_buffer_done  <= 1'b1;
                    load_cnt <= 0;
                end
            end else begin
                load_cnt <= 0;
            end
        end
    end
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n | layer_start) begin
            data_out      = {NUM_CHNL*DATA_WIDTH{1'b0}};
        end
    end
    
    always @(*) begin
        out_cnt_n = out_cnt;
        if (rden) begin
            if (out_cnt < SIZE_KERNEL_H * SIZE_KERNEL_W) begin
                data_out = buffer_data[ out_cnt / SIZE_KERNEL_W ][ out_cnt % SIZE_KERNEL_W ];
                out_cnt_n = out_cnt + 1;
            end
            if (out_cnt == (SIZE_KERNEL_H * SIZE_KERNEL_W - 1)) begin
                out_cnt_n = 0;
            end
        end else begin
            out_cnt_n  = 0;
            data_out = {NUM_CHNL*DATA_WIDTH{1'b0}};
        end
    end

endmodule