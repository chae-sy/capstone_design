`timescale 1ns / 1ps

module layer_mux #(parameter WORD_LENGTH = 128) (
    input [2:0] layer,
    input [WORD_LENGTH-1:0] im_q,
    input [WORD_LENGTH-1:0] ma_q,
    input [WORD_LENGTH-1:0] mb_q,
    output reg [WORD_LENGTH-1:0] data_o
    );

    always @ (*) begin
        case(layer)
            3'd1: data_o = im_q;
            3'd2: data_o = ma_q;
            3'd3: data_o = mb_q;
            3'd4: data_o = ma_q;
            3'd5: data_o = mb_q;
            default: data_o = 0;
        endcase
    end

endmodule