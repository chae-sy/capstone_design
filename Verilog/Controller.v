`timescale 1ns / 1ps
//Controller
module Controller#(
    parameter           MEM_BIT_LEN = $clog2(100),
    //parameter MA_BIT_LEN = $clog2(16),
    //parameter MB_BIT_LEN = $clog2(20),
    parameter           WMEM_BIT_LEN = $clog2(144),
    parameter           NUM_CHANNEL = 32
)(
    input               rst_n,
    input               clk,

    input               weight_done,
    input               FE_done,.

    //Weight Memory
    output  reg         [WMEM_BIT_LEN-1:0] wmem_addr_o;
    output  reg         wmem_wenb_o,
    output  reg         wmem_enb_o,
    //Data Memory
    output  reg         [IM_BIT_LEN-1:0] mem_addr_o,
    output  reg         mem_wenb_o,
    output  reg         mem_enb_o,
    /*//Memory A 
    output reg [MA_BIT_LEN-1:0] ma_addr_o,
    output reg ma_web_o,
    output reg ma_ceb_o,
    //Memory B
    output reg [MB_BIT_LEN-1:0] mb_addr_o,
    output reg mb_web_o,
    output reg mb_ceb_o,*/ 

    //Weight Buffer
    output  reg         wei_buff_en_o,
    
    //PE array
    output  reg         pe_en_o,
    output  reg         pe_rst_o,
    //Output Buffer
    output  reg [31:0]  out_buf_en_o,
    output  reg         out_buf_sel_o,
    output  reg         out_buf_rst_o,
    //maxpool
    output  reg         pool_sel_o, //1 at layer 3

    //output  reg     comp_start_o,
    //output reg rf_sel_o,
    output reg [2:0]    layer,
    output done_o
);

    //FSM state
    localparam      S_IDLE      = 3'd0,
                    S_Layer1    = 3'd1,
                    S_Layer2    = 3'd2,
                    S_Layer3    = 3'd3,
                    S_Layer4    = 3'd4,  
                    S_Layer5    = 3'd5;

    reg     [2:0]           state,      state_n;
    reg     [2:0]           layer_num,  layer_num_n;

    // Weight layer parameter
    localparam              WEI_BIT      = 8; 
    localparam              WMEM_LEN     = 144; 

    // Layer Parameter
    localparam L1_INPUT_HORIZ = 10;
    localparam L1_INPUT_VERT = 29;
    localparam L1_WEIGHT_HORIZ = 4;
    localparam L1_WEIGHT_VERT = 10;
    localparam L1_STRIDE_HORIZ = 2;
    localparam L1_STRIDE_VERT = 1;
    localparam L1_INPUT_BIT_LEN = 8;
    localparam L1_WEIGHT_BIT_LEN = 32;
    localparam L1_NEXT_WEIGHT_VERT = 3;
    localparam L1_NEXET_INPUT_HORIZ = (L1_INPUT_HORIZ - L1_WEIGHT_HORIZ)/L1_STRIDE_HORIZ + 1;
    localparam L1_WEIGHT_VERT_BIT_LEN = $clog2(L1_WEIGHT_VERT);
    localparam L1_INPUT_VERT_BIT_LEN = $clog2(L1_INPUT_VERT);

    // Layer Wire
    wire    [MEM_BIT_LEN-1:0]   dmem_addr;
    //wire    [MA_BIT_LEN-1:0]ma_addr_l1;
    wire    [WMEM_BIT_LEN-1:0]  wmem_addr;
    wire                        dmem_enb;
    wire                        dmem_wenb;
    //wire ma_ceb_l1;
    //wire ma_web_l1;
    wire                        wmem_enb;
    wire                        wmem_wenb;
    wire                        pe_rst;
    wire                        pe_en;
    wire                        out_buf_rst;
    wire    [31:0]              out_buf_en;
    wire                        layer_done_o;
    
    wire    [31:0]              line_cnt;
    wire    [2:0]               layer_num;



    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;   
            layer_num       <= 3'b0;
        end
        else begin
            state           <= state_n;
            layer_num       <= layer_num_n;
        end
    end

    always @(*) begin
        state_n             = state;
        layer_num_n         = layer_num;

        case(state)
            S_IDLE: begin
                state_n             = S_Layer1;
                layer_num_n         = 3'd1;
            end
            S_Layer1: begin
                if(layer_done_o) begin
                    state_n             = S_Layer2;
                    layer_num_n         = 3'd2;
                end
            end
            S_Layer2: begin
                if(layer_done_o) begin
                    state_n             = S_Layer3;
                    layer_num_n         = 3'd3;
                end
            end
            S_Layer3: begin
                if(layer_done_o) begin
                    state_n             = S_Layer4;
                    layer_num_n         = 3'd4;
                end
            end
            S_Layer4: begin
                if(layer_done_o) begin
                    state_n             = S_Layer5;
                    layer_num_n         = 3'd5;
                end
            end
            S_Layer5: begin
                if(layer_done_o) begin
                    state_n             = S_IDLE;
                    layer_num_n         = 3'd0;
                end
            end

        endcase

    end

endmodule