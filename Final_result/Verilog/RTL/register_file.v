`timescale 1ns / 1ps

module regfile_sync #(
    parameter BITWIDTH            = 32,                        // ê°? ?›Œ?“œ?˜ ë¹„íŠ¸ ?­
    parameter NUM_WORD            = 16,                        // ?›Œ?“œ ê°œìˆ˜
    parameter LAYER_6_NUM_WORD    = 2,                         // layer_num == 1?¼ ?•Œë§? ?‚¬?š©?•  ?›Œ?“œ ê°œìˆ˜
    parameter DATA_WIDTH          = NUM_WORD * BITWIDTH,       // ? „ì²? ë©”ëª¨ë¦? ?­
    parameter ADDR_WIDTH          = 3                          // 
)(
    input  wire                     clk,        // ?´ë¡?
    input  wire                     rst_n,      // ë¹„ë™ê¸? ë¦¬ì…‹ (low active)
    // --- Write Port ---
    input  wire                     we,         // write enable
    input  wire [ADDR_WIDTH-1:0]    waddr,      // write address
    input  wire [DATA_WIDTH-1:0]    wdata,      // write data (?•œ ë²ˆì— NUM_WORD * BITWIDTHë¥? ?¨?„£?Œ)
    // --- Read Port ---
    input  wire [ADDR_WIDTH-1:0]    raddr,      // read address (?™ê¸°ì‹?œ¼ë¡? ìº¡ì²˜)
    input  wire                     rden,       // read enable: ?´ ?‹ ?˜¸ê°? 1ë¡? ? „?´(0?†’1)?  ?•Œë§ˆë‹¤ ?•œ ?›Œ?“œë¥? ?½?Œ
    output wire [BITWIDTH-1:0]      rdata,      // ìµœì¢… ?ï¿½ï¿½?ï¿½ï¿½?ï¿½ï¿½?ï¿½ï¿½?ï¿½ï¿½ ?ï¿½ï¿½ ?ï¿½ï¿½?ï¿½ï¿½ (BITWIDTH ?ï¿½ï¿½)
    // --- layer num ---
    input  wire [2:0]               layer_num
);

    //======================================================================
    // 1) ?‚´ë¶? ë©”ëª¨ë¦? ?„ ?–¸
    //======================================================================
    localparam DEPTH = (1 << ADDR_WIDTH);
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    integer i;

    //----------------------------------------------------------------------
    // 2) ?™ê¸°ì‹ ?“°ê¸? (Write)
    //----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1) begin
                mem[i] <= {DATA_WIDTH{1'b0}};
            end
        end else if (we) begin
            mem[waddr] <= wdata;
        end
    end

    //----------------------------------------------------------------------
    // 3) ?½ê¸? ì£¼ì†Œ ? ˆì§??Š¤?„° ìº¡ì²˜ (Read Address Register)
    //----------------------------------------------------------------------
    reg [ADDR_WIDTH-1:0] raddr_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            raddr_reg <= {ADDR_WIDTH{1'b0}};
        end else begin
            raddr_reg <= raddr;
        end
    end

    //----------------------------------------------------------------------
    // 4) rden ?—£ì§? ê°ì??š©: ?´? „ ?´ë¡ì˜ rden ê°’ì„ ???¥
    //----------------------------------------------------------------------
    reg prev_rden;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prev_rden <= 1'b0;
        end else begin
            prev_rden <= rden;
        end
    end

    //----------------------------------------------------------------------
    // 5) "rden?´ 0?†’1?œ¼ë¡? ? „?´?œ ?ˆœê°?"?—ë§? read_buf?— ë©”ëª¨ë¦? ë¡œë“œ
    //----------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] read_buf;
    always @(*) begin
        if (!rst_n) begin
            read_buf = {DATA_WIDTH{1'b0}};
        end 
        // rden?´ 0?†’1ë¡? ë°”ë?ŒëŠ” ?ˆœê°?: (rden=1 & prev_rden=0)
        else if (rden) begin
            read_buf = mem[raddr_reg];
        end
        // ê·? ?™¸(= rden ê³„ì† 1?´ê±°ë‚˜, ?˜¹?? rden=0?¸ ?ƒ?ƒœ)?—?Š” ë²„í¼ ?œ ì§?ë¥? ?•¨
    end

    //----------------------------------------------------------------------
    // 6) "rden?´ 0?†’1 ? „?´?œ ?ˆœê°?"ë§ˆë‹¤ cntë¥? 0?†’1?†’?¦â†’max-1?†’0 ?ˆœ?œ¼ë¡? ì¦ê??‹œ?‚¤ê¸?
    //----------------------------------------------------------------------
    reg [3:0] cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= 0;
        end 
        // rden?´ 0?†’1ë¡? ë°”ë?? ?•Œë§ˆë‹¤ cntë¥? +1 ?˜¹?? wrap-around
        else if (rden) begin
                // layer_num==1?¼ ?•Œ?Š” LAYER_1_NUM_WORD ê°œìˆ˜ë§Œí¼ë§? ?ˆœ?™˜
            if (layer_num == 6) begin
                    cnt <= 4'd0;
            end else begin
                // layer_num!=1?¼ ?•Œ?Š” NUM_WORD ê°œìˆ˜ë§Œí¼ë§? ?ˆœ?™˜
                if (cnt == (NUM_WORD - 1))
                    cnt <= 4'd0;
                else
                    cnt <= cnt + 4'd1;
            end
        end
        // rden?´ 1?´ ?•„?‹ˆê±°ë‚˜, rden?´ 1?¸?° ë°”ë¡œ ? „ ?´ë¡ì—” prev_rden?„ 1?¸ ê²½ìš°(cnt ì¦ë¶„ ì¡°ê±´ ?•„?‹˜) ?†’ cnt ?œ ì§?
    end

    //----------------------------------------------------------------------
    // 7) ?Š¬?¼?´?‹±?œ rdata ì¶œë ¥
    //    read_buf?— ???¥?œ DATA_WIDTH ?­ ? „ì²´ì—?„œ
    //    cnt*BITWIDTH ?œ„ì¹˜ë??„° BITWIDTH ?­ë§Œí¼ ?˜?¼?„œ ?‚´ë³´ëƒ„
    //----------------------------------------------------------------------
    assign rdata = read_buf[511-cnt * BITWIDTH -: BITWIDTH];

endmodule