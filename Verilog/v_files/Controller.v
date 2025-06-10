`timescale 1ns / 1ps
//Controller
module Controller#(
    parameter           NUM_CHANNEL = 16,
    parameter           NUM_COLOR = 3
)(
    input   wire        clk,
    input   wire        rst_n,
    
    input   wire        initial_SRAMw_done,
    input   wire        initial_weight_done,

    //Weight Memory
    output  wire [9:0]  wmem_addr_o,
    output  wire        wmem_wenb_o,
    output  wire        wmem_cenb_o,

    //Memory A 
    output  wire [15:0] memA_addr_o,
    output  wire        memA_wenb_o,
    output  wire        memA_cenb_o,
    //Memory B
    output  wire [15:0] memB_addr_o,
    output  wire        memB_wenb_o,
    output  wire        memB_cenb_o,

    //Weight Buffer
    output  wire        wei_buff_wren_o,
    output  wire        wei_buff_rden_o,
    
    // input buffer
    output  wire        in_buf_wren_r,
    output  wire        in_buf_wren_g,
    output  wire        in_buf_wren_b,
    output  wire        in_buf_rden_r,
    output  wire        in_buf_rden_g,
    output  wire        in_buf_rden_b,
    output  wire        is_initial_o,
    
    // output buffer
    output  wire        out_buf_wren_r,
    output  wire        out_buf_wren_g,
    output  wire        out_buf_wren_b,
    output  wire        out_buf_rden_r,
    output  wire        out_buf_rden_g,
    output  wire        out_buf_rden_b,
    input   wire        out_buf_done_i,
    
    //PE array
    output  wire        pe_en_o,
    input   wire        pe_done_i,

    //add tree
    output  wire        addtree_en_o,
    input   wire        addtree_done_i,

    // ReLU
    output  wire        relu_en_o,
    input   wire        relu_done_i,

    //maxpool
    output  wire        maxpool_en_o,
    input   wire        maxpool_done_i,
    output  wire [1:0]  color_o,
    
    output  wire        layer_done_o,
    output  wire        total_done_o, 
    output  wire [2:0]  layer_num_o,
    output  wire        layer_start_o
    
);

    //FSM state
    localparam      S_IDLE          = 4'd0,
                    S_SRAM_W        = 4'd1,
                    S_Layer1        = 4'd2,
                    S_Layer2        = 4'd3,
                    S_Layer3        = 4'd4,
                    S_Layer4        = 4'd5,  
                    S_Layer5        = 4'd6,
                    S_Layer6        = 4'd7;

    reg     [3:0]           state,      state_n;
    reg     [2:0]           layer_num,  layer_num_n;
    reg                     layer_start,  layer_start_n;
    reg     [15:0]          data_num,  data_num_n;
    reg     [8:0]           weight_num,  weight_num_n;
    reg     [4:0]           channel;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;   
            layer_num       <= 3'b0;
            weight_num      <= 'd16;
            layer_start     <= 1'b0;
        end
        else begin
            state           <= state_n;
            layer_num       <= layer_num_n;
            weight_num      <= weight_num_n;
            layer_start     <= layer_start_n;
        end
    end

    always @(*) begin
        state_n                         = state;
        layer_num_n                     = layer_num;
        channel                         = 'd16; 
        weight_num_n                    = weight_num;
        layer_start_n                   = layer_start;
        case(state)
            S_IDLE: begin
                state_n                 = S_SRAM_W;
                layer_num_n             = 3'd0;
            end
            S_SRAM_W: begin
                layer_start_n           = 1'b0;
                if(initial_SRAMw_done & initial_weight_done) begin
                    channel             = 'd2;
                    weight_num_n        = 'd16;
                    layer_start_n       = 1'b1;
                    state_n             = S_Layer1;
                    layer_num_n         = 3'd1;
                end
            end
            S_Layer1: begin
                layer_start_n           = 1'b0;
                if(layer_done) begin
                    channel             = 'd16;
                    weight_num_n        = 'd16;
                    layer_start_n       = 1'b1;
                    state_n             = S_Layer2;
                    layer_num_n         = 3'd2;
                end
            end
            S_Layer2: begin
                layer_start_n           = 1'b0;
                if(layer_done) begin
                    channel             = 'd16;
                    weight_num_n        = 'd16;
                    layer_start_n       = 1'b1;
                    state_n             = S_Layer3;
                    layer_num_n         = 3'd3;
                end
            end
            S_Layer3: begin
                layer_start_n           = 1'b0;
                if(layer_done) begin
                    channel             = 'd16;
                    weight_num_n        = 'd16;
                    layer_start_n       = 1'b1;
                    state_n             = S_Layer4;
                    layer_num_n         = 3'd4;
                end
            end
            S_Layer4: begin
                layer_start_n           = 1'b0;
                if(layer_done) begin
                    channel             = 'd16;
                    weight_num_n        = 'd16;
                    layer_start_n       = 1'b1;
                    state_n             = S_Layer5;
                    layer_num_n         = 3'd5;
                end
            end
            S_Layer5: begin
                layer_start_n           = 1'b0;
                if(layer_done) begin
                    channel             = 'd16;
                    weight_num_n        = 'd1;
                    layer_start_n       = 1'b1;
                    state_n             = S_Layer6;
                    layer_num_n         = 3'd6;
                end
            end
            S_Layer6: begin
                layer_start_n           = 1'b0;
                if(layer_done) begin
                    channel             = 'd1;
                    weight_num_n        = 'd16;
                    layer_start_n       = 1'b1;
                    state_n             = S_IDLE;
                    layer_num_n         = 3'd0;
                end
            end
        endcase
    end

    layer_pipeline  u_layer_pipeline(
        .clk                (clk),
        .rst_n              (rst_n),

        .wmem_addr_o        (wmem_addr_o),
        .wmem_wenb_o        (wmem_wenb_o),
        .wmem_cenb_o        (wmem_cenb_o),

        .memA_addr_o        (memA_addr_o),
        .memA_wenb_o        (memA_wenb_o),
        .memA_cenb_o        (memA_cenb_o),

        .memB_addr_o        (memB_addr_o),
        .memB_wenb_o        (memB_wenb_o),
        .memB_cenb_o        (memB_cenb_o),

        .wei_buff_wren_o    (wei_buff_wren_o),
        .wei_buff_rden_o    (wei_buff_rden_o),
        
        .in_buf_wren_r      (in_buf_wren_r),
        .in_buf_wren_g      (in_buf_wren_g),
        .in_buf_wren_b      (in_buf_wren_b),
        .in_buf_rden_r      (in_buf_rden_r),
        .in_buf_rden_g      (in_buf_rden_g),
        .in_buf_rden_b      (in_buf_rden_b),        
        .is_initial_o       (is_initial),
        
        .out_buf_wren_r     (out_buf_wren_r),
        .out_buf_wren_g     (out_buf_wren_g),
        .out_buf_wren_b     (out_buf_wren_b),
        .out_buf_rden_r     (out_buf_rden_r),
        .out_buf_rden_g     (out_buf_rden_g),
        .out_buf_rden_b     (out_buf_rden_b),
        .out_buf_done_i     (out_buf_done_i),
        
        .pe_en_o            (pe_en_o),
        .pe_done_i          (pe_done_i),

        .addtree_en_o       (addtree_en_o),
        .addtree_done_i     (addtree_done_i),

        .relu_en_o          (relu_en_o),
        .relu_done_i        (relu_done_i),

        .maxpool_en_o       (maxpool_en_o),
        .maxpool_done_i     (maxpool_done_i),
        .color_o            (color_o),
        
        .layer_num          (layer_num),
        .weight_num         (weight_num),
        .channel            (channel),
        .layer_start        (layer_start),
        .layer_done_o       (layer_done)
    );
    assign layer_num_o = layer_num;
    assign layer_done_o = layer_done;
    assign total_done_o = (state == S_Layer6) & layer_done;
    assign layer_start_o = layer_start;

endmodule