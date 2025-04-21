`timescale 1ns/1ps


module ib_ctrl (
    // Global input
    input rstb,
    input clk, 
    
    input wr_input_on,

    output reg [1:0] state,
    output reg [3:0] ib_addr,
    output reg ib_ceb,
    output reg ib_web,
    output reg calc_en
);
    localparam IDLE = 2'b00;
    localparam WRITE = 2'b10;
    localparam READ = 2'b11;

    always @ (posedge clk or negedge rstb) begin
        if (!rstb) begin
            state <= IDLE;
            ib_addr <= 0;
            ib_ceb <= 1;
            ib_web <= 1;
            calc_en <= 0;
        end
        else begin
            case(state) //synopsys full_case
                IDLE: begin
                    state <= (wr_input_on) ? WRITE : IDLE;
                    ib_addr <= 0;
                    ib_ceb <= 1;
                    ib_web <= 1;
                    calc_en <= 0;
                end
                WRITE: begin
                    state <= (!wr_input_on) ? READ : WRITE;
                    ib_addr <= 0;
                    ib_ceb <= (!wr_input_on) ? 0 : 1;
                    ib_web <= 1;
                    calc_en <= (!wr_input_on) ? 1 : 0;
                end
                READ: begin
                    if (ib_addr == 9) begin
                        ib_addr <= 0;
                        state <= IDLE;
                        ib_ceb <= 1;
                        ib_web <= 1;
                        calc_en <= 0;
                    end
                    else begin
                        ib_addr <= ib_addr + 1;
                        state <= READ;
                        ib_ceb <= 0;
                        ib_web <= 1;
                        calc_en <= 0;
                    end
                end
            endcase
        end
    end

endmodule