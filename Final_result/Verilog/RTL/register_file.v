`timescale 1ns / 1ps

module regfile_sync #(
    parameter BITWIDTH            = 32,                       
    parameter NUM_WORD            = 16,                     
    parameter LAYER_6_NUM_WORD    = 2,                      
    parameter DATA_WIDTH          = NUM_WORD * BITWIDTH,       
    parameter ADDR_WIDTH          = 3                          
)(
    input  wire                     clk,      
    input  wire                     rst_n,      
    // --- Write Port ---
    input  wire                     we,         // write enable
    input  wire [ADDR_WIDTH-1:0]    waddr,      // write address
    input  wire [DATA_WIDTH-1:0]    wdata,      // write data 
    // --- Read Port ---
    input  wire [ADDR_WIDTH-1:0]    raddr,      // read address 
    input  wire                     rden,       // read enable: 
    output wire [BITWIDTH-1:0]      rdata,      
    // --- layer num ---
    input  wire [2:0]               layer_num
);

    localparam DEPTH = (1 << ADDR_WIDTH);
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1) begin
                mem[i] <= {DATA_WIDTH{1'b0}};
            end
        end else if (we) begin
            mem[waddr] <= wdata;
        end
    end

    reg [ADDR_WIDTH-1:0] raddr_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            raddr_reg <= {ADDR_WIDTH{1'b0}};
        end else begin
            raddr_reg <= raddr;
        end
    end

    reg prev_rden;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prev_rden <= 1'b0;
        end else begin
            prev_rden <= rden;
        end
    end


    reg [DATA_WIDTH-1:0] read_buf;
    always @(*) begin
        if (!rst_n) begin
            read_buf = {DATA_WIDTH{1'b0}};
        end 
        else if (rden) begin
            read_buf = mem[raddr_reg];
        end
    end

    reg [3:0] cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= 0;
        end 
        else if (rden) begin
            if (layer_num == 6) begin
                    cnt <= 4'd0;
            end else begin
                if (cnt == (NUM_WORD - 1))
                    cnt <= 4'd0;
                else
                    cnt <= cnt + 4'd1;
            end
        end
    end


    assign rdata = read_buf[511-cnt * BITWIDTH -: BITWIDTH];

endmodule
