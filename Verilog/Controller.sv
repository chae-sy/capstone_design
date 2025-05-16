`timescale 1ns / 1ps
//Controller
module Controller#(
    //parameter           MEM_BIT_LEN = $clog2(100),
    parameter           MA_BIT_LEN = $clog2(480000),
    parameter           MB_BIT_LEN = $clog2(480000),
    parameter           WMEM_BIT_LEN = $clog2(2304),
    parameter           NUM_CHANNEL = 16
)(
    input               rst_n,
    input               clk,
    input               initial_SRAMw_done,
    input               initial_weight_done,
    input               data_map_done,
    input               layer_done,
    //input               FE_done,
    //input               ReLU_done,
    //input               maxpool_done,

    //Weight Memory
    output  wire        [WMEM_BIT_LEN-1:0] wmem_addr_o,
    output  wire        wmem_wenb_o,
    output  wire        wmem_enb_o,
    /*Data Memory
    output  reg         [IM_BIT_LEN-1:0] mem_addr_o,
    output  reg         mem_wenb_o,
    output  reg         mem_enb_o,*/
    //Memory A 
    output  wire [MA_BIT_LEN-1:0] memA_addr_o,
    output  wire        memA_wenb_o,
    output  wire        memA_cenb_o,
    //Memory B
    output  wire [MB_BIT_LEN-1:0] memB_addr_o,
    output  wire        memB_wenb_o,
    output  wire        memB_cenb_o,

    //Input Buffer
    output  wire [31:0] in_buf_en_o,
    output  wire        in_buf_sel_o,
    output  wire        in_buf_rst_o,

    //Weight Buffer
    output  wire        wei_buff_en_o,
    
    //PE array
    output  wire        pe_en_o,
    output  wire        pe_rst_o,

    // ReLU
    output  wire        relu_en_o,

    //Output Buffer
    output  wire [31:0] out_buf_en_o,
    output  wire        out_buf_sel_o,
    output  wire        out_buf_rst_o,
    //maxpool
    output  wire        pool_sel_o, //1 at layer 3

    //output  reg     comp_start_o,
    //output reg rf_sel_o,
    output  wire [2:0]  layer_state,
    output  wire        done_o,
    output  wire        layer_done_o,
    output  reg         data_map_enb
);

    //FSM state
    localparam      S_IDLE          = 3'd0,
                    S_SRAM_W        = 3'd1,
                    S_Layer1        = 3'd2,
                    S_Layer2        = 3'd3,
                    S_Layer3        = 3'd4,
                    S_Layer4        = 3'd5,  
                    S_Layer5        = 3'd6,
                    S_data_mapping  = 3'd7;

    reg     [2:0]           state,      state_n;
    reg     [2:0]           layer_num,  layer_num_n;

    // Weight layer parameter
    localparam              WEI_BIT      = 8; 
    localparam              WMEM_LEN     = 144; 

    // Layer Parameter
    localparam      INPUT_HORIZ         = 15;
    localparam      INPUT_VERT          = 16;
    localparam      WEIGHT_HORIZ        = 9;
    localparam      WEIGHT_VERT         = 16;
    localparam      STRIDE_HORIZ        = 3;
    localparam      STRIDE_VERT         = 3;
    localparam      INPUT_BIT_LEN       = 8;
    localparam      WEIGHT_BIT_LEN      = 8;
    localparam      NEXT_WEIGHT_VERT    = 16;
    localparam      NEXET_INPUT_HORIZ   = (INPUT_HORIZ - WEIGHT_HORIZ)/STRIDE_HORIZ + 1;
    localparam      WEIGHT_VERT_BIT_LEN = $clog2(WEIGHT_VERT);
    localparam      INPUT_VERT_BIT_LEN  = $clog2(INPUT_VERT);

    // Layer Wire
    //wire    [MEM_BIT_LEN-1:0]   dmem_addr;
    reg    [MA_BIT_LEN-1:0]    memA_addr;
    reg    [MB_BIT_LEN-1:0]    memB_addr;
    reg    [WMEM_BIT_LEN-1:0]  wmem_addr;
    /*wire                        dmem_enb;
    wire                        dmem_wenb;*/
    reg                        memA_cenb;
    reg                        memA_wenb;
    reg                        memB_cenb;
    reg                        memB_wenb;
    reg                        wmem_enb;
    reg                        wmem_wenb;
    reg                        pe_rst;
    reg                        pe_en;
    reg                        wei_buff_en;
    reg                        in_buf_en;
    reg                        in_buf_rst;
    reg                        in_buf_sel;
    reg                        out_buf_rst;
    reg    [31:0]              out_buf_en;
    reg                        out_buf_sel;
    reg                        relu_en;
    reg                        pool_sel;
    
    reg                        done;
    //reg                        layer_done;
    
    reg    [31:0]              line_cnt;




    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;   
            layer_num       <= 3'b0;
        end
        else begin
            state           <= state_n;
            layer_num       <= layer_num_n;
        end
    end

    always_comb @(*) begin
        state_n                         = state;
        layer_num_n                     = layer_num;
        data_map_enb                    = 1;
        case(state)
            S_IDLE: begin
                state_n                 = S_SRAM_W;
                layer_num_n             = 3'd0;
            end
            S_SRAM_W: begin
                if(initial_SRAMw_done & initial_weight_done) begin
                    data_num_n          = 'd10404; // cov1 102*102
                    channel             = 'd2;
                    weight_num_n        = 'd16;
                    state_n             = S_Layer1;
                    layer_num_n         = 3'd1;
                end
            end
            S_Layer1: begin
                if(layer_done) begin
                    state_n             = S_Layer2;
                    layer_num_n         = 3'd2;
                end
            end
            S_Layer2: begin
                if(layer_done) begin
                    state_n             = S_Layer3;
                    layer_num_n         = 3'd3;
                end
            end
            S_Layer3: begin
                if(layer_done) begin
                    state_n             = S_Layer4;
                    layer_num_n         = 3'd4;
                end
            end
            S_Layer4: begin
                if(layer_done) begin
                    state_n             = S_Layer5;
                    layer_num_n         = 3'd5;
                end
            end
            S_Layer5: begin
                if(layer_done) begin
                    state_n             = S_data_mapping;
                    layer_num_n         = 3'd0;
                end
            end
            S_data_mapping: begin
                data_map_enb            = 0;
                if(data_map_done) begin
                    state_n             = S_IDLE;
                end

            end

        endcase

    end

    always_comb @(*) begin
    // Default value
    wmem_addr           = 0;
    memA_addr           = 0;
    memB_addr           = 0;
    
    wmem_wenb           = 1;
    wmem_enb            = 1;
    
    memA_wenb           = 1;
    memA_cenb           = 1;
    memB_wenb           = 1;
    memB_cenb           = 1;

    wei_buff_en         = 0;
    in_buf_en           = 0;
    in_buf_rst          = 0;
    in_buf_sel          = 0;
    pe_en               = 0;
    pe_rst              = 0;
    relu_en             = 0;
    out_buf_en          = 0;
    out_buf_sel         = 0;
    out_buf_rst         = 0;
    pool_sel            = 0;
    done                = 0;
    
    case (state)
        S_IDLE: begin

        end
        S_SRAM_W: begin
            wmem_enb            = 0;
            wmem_wenb           = 0;
            memA_cenb           = 0;
            memA_wenb           = 0;
        end
        S_Layer1, S_Layer2, S_Layer3, S_Layer4, S_Layer5: begin
            memA_cenb           = 0;
            memB_cenb           = 0;
            pe_en               = 1;
            wei_buff_en         = 1;
            in_buf_en           = 1;
            out_buf_en          = 1;
            if ((state == S_Layer1) | (state == S_Layer3) | (state == S_Layer5) ) begin
                memB_wenb       = 0;
                if (state == S_Layer3) begin
                    pool_sel    = 1;
                end
                else begin
                    pool_sel    = 0;
                end
            end
            else begin
                memA_wenb       = 0;
            end
            if (state != S_Layer5) begin
                relu_en         = 1;
            end
            else begin
                relu_en         = 0;
            end
        end
        S_data_mapping: begin
            memB_cenb           = 0;
            data_map_enb        = 0;
        end
    endcase
end

    //state output
    assign    wmem_addr_o     = wmem_addr;
    assign    wmem_wenb_o     = wmem_wenb;
    assign    wmem_enb_o      = wmem_enb;
    
    assign    memA_addr_o     = memA_addr;
    assign    memA_wenb_o     = memA_wenb;
    assign    memA_cenb_o     = memA_cenb;
    assign    memB_addr_o     = memB_addr;
    assign    memB_wenb_o     = memB_wenb;
    assign    memB_cenb_o     = memB_cenb;
        
    assign    in_buf_en_o     = in_buf_en;
    assign    in_buf_sel_o    = in_buf_sel;
    assign    in_buf_rst_o    = in_buf_rst;
    assign    wei_buff_en_o   = wei_buff_en;
    assign    pe_en_o         = pe_en;
    assign    pe_rst_o        = pe_rst;
    assign    relu_en_o       = relu_en;
    assign    out_buf_en_o    = out_buf_en;
    assign    out_buf_sel_o   = out_buf_sel;
    assign    out_buf_rst_o   = out_buf_rst;
    assign    layer_done_o    = layer_done;
    assign    pool_sel_o      = pool_sel;

    assign    layer_state     = state;
    assign    done_o          = (state == S_Layer5) && (layer_done_o);
    /*always @ (*) begin
        wmem_addr_o     = 0;
        wmem_wenb_o     = 1;
        wmem_enb_o      = 1;
    
        memA_addr_o     = 0;
        memA_wenb_o     = 1;
        memA_cenb_o     = 1;
        memB_addr_o     = 0;
        memB_wenb_o     = 1;
        memB_cenb_o     = 1;
        
        wei_buff_en_o   = 0;
        pe_en_o         = 0;
        pe_rst_o        = 0;
        relu_en_o       = 0;
        out_buf_en_o    = 0;
        out_buf_sel_o   = 0;
        out_buf_rst_o   = 0;
        pool_sel_o      = 0;

        layer           = 0;
        done_o          = 0;
        case(state)
            S_IDLE: begin
                state_n             = S_Layer1;
                layer_num_n         = 3'd1;
            end
            S_SRAM_W: begin
                wmem_addr_o     = wmem_addr;
                wmem_wenb_o     = wmem_wenb;
                wmem_enb_o      = wmem_enb;
            
                memA_addr_o     = memA_addr;
                memA_wenb_o     = memA_wenb;
                memA_cenb_o     = memA_cenb;
                memB_addr_o     = 0;
                memB_wenb_o     = 1;
                memB_cenb_o     = 1;

                wei_buff_en_o   = 0;
                pe_en_o         = 0;
                pe_rst_o        = 0;
                relu_en_o       = 0;
                out_buf_en_o    = 0;
                out_buf_sel_o   = 0;
                out_buf_rst_o   = 0;
                pool_sel_o      = 0;

                layer           = 0;
                done_o          = 0;
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
    end*/
    

endmodule
