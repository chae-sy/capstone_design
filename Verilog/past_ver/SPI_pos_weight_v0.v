//---------------------------------------------
// Filename   : SPI.v 
// Author     : Taewook
// Created On : 2019-6-6
// Note       : spi_trig 0 -> 1, then spi starts
//---------------------------------------------
// Modified 2020-8-2
//  mosi triggers at posedge sclk

module SPI_pos_weight_v0
#(
    parameter LENGTH_ADDR = 9,
    parameter LENGTH_DATA = 24,
    parameter LENGTH_PCK = 34
)
(
    input   wire                    clk_fpga,
    input   wire                    reset_fpga,
    input   wire                    spi_trig,
    input   wire                    rw,
    input   wire [LENGTH_ADDR-1:0]  addr,
    input   wire [LENGTH_DATA-1:0]  data,
    input   wire                    spi_miso,
    
    output  reg                     spi_clk,
    output  reg                     spi_ss,
    output  reg                     spi_mosi,
    output  wire [2:0]              state_spi_obs,
    output  reg  [LENGTH_PCK-1:0]   spi_pck_read,
    output  reg                    wr_w_on
); 
//    localparam LENGTH_PCK    = 1+LENGTH_ADDR+LENGTH_DATA;
    localparam ST_SPI_IDLE   = 0;
    localparam ST_SPI_WAIT_0 = 1;
    localparam ST_SPI_WAIT_1 = 2;
    localparam ST_SPI_WAIT_2 = 3;
    localparam ST_SPI_DO     = 4;
    localparam ST_SPI_WAIT_3 = 5;
    localparam ST_SPI_WAIT_4 = 6;
    localparam ST_SPI_WAIT_5 = 7;


    
    reg  [LENGTH_PCK-1:0]           spi_pck_w;
    reg  [$clog2(2*LENGTH_PCK):0]   count_pck_w;
    reg  [$clog2(2*LENGTH_PCK):0]   count_bit_w;
    reg  [2:0]                      state_spi_w;
    assign  state_spi_obs_w = state_spi_w;

    always @(posedge clk_fpga or posedge reset_fpga) begin  
        if (reset_fpga) begin
            state_spi_w <= ST_SPI_IDLE;
            spi_clk <= 1;
            spi_ss <= 1;
            spi_mosi <= 0;
            spi_pck_w <= 0;
            count_pck_w <= 0;
            spi_pck_read <= 0;
            count_bit_w <= 0;
            wr_w_on <= 0;
        end
        else begin
            case (state_spi_w)
                ST_SPI_IDLE: begin //0
                    spi_clk <= 1;
                    spi_ss <= 1;
                    spi_mosi <= 0;
                    //spi_pck <= {rw, addr, data}; 
                    count_pck_w <= 0;
                    count_bit_w <= 0;
                    if (!spi_trig) begin
                        state_spi_w <= ST_SPI_WAIT_0;
                        wr_w_on <= 1;
                    end
                end
                ST_SPI_WAIT_0: begin //1
                    if (spi_trig) begin
                        spi_ss <= 0;
                        spi_pck_w <= {rw, addr, data}; 
                        state_spi_w <= ST_SPI_WAIT_1;
                    end
                end
                ST_SPI_WAIT_1: begin //2
                    count_pck_w <= count_pck_w + 1;
                    if ( count_pck_w == 3 ) begin
                        state_spi_w <= ST_SPI_WAIT_2;
                        spi_clk <= 0; // neg sclk / ST: idel -> cap
                        count_pck_w <= 0;
                    end
                end
                ST_SPI_WAIT_2: begin //3
                    state_spi_w <= ST_SPI_DO;
                    wr_w_on <= 0;
                    spi_clk <= 1; // for SPI__pos
                end
                ST_SPI_DO: begin //4
                    if (count_pck_w < 2*LENGTH_PCK+1) begin
                        count_pck_w <= count_pck_w + 1;
                        if (count_pck_w > 0) begin
                            spi_clk <= ~spi_clk; // 1. pisneg
                        end
                        if (!count_pck_w[0]) begin 
                            spi_mosi <= spi_pck_w[LENGTH_PCK-1];
                            spi_pck_w <= spi_pck_w << 1;
                        end
                        else begin
                            spi_pck_read <= {spi_pck_read[LENGTH_PCK-2:0], spi_miso}; 
                        end

                        if (count_pck_w[0]) begin  // counting bit num, for debugging
                            count_bit_w <= count_bit_w + 1; 
                        end
                    end
                         
                    else begin
                        spi_ss <= 1;
                        state_spi_w <= ST_SPI_WAIT_3;
                        spi_clk <= 1; //
                        count_pck_w <= 0; 
                    end
                end
                ST_SPI_WAIT_3 : begin //5
                    count_pck_w <= count_pck_w + 1;
                    if (count_pck_w < 1 + 2*5 + 2) begin // 1: cap -> store // 2*5 : store cnt==5 // 2 : store -> idle 
                        spi_clk <= ~spi_clk;
                    end
                    else begin
                        state_spi_w <= ST_SPI_IDLE;
                    end
               end
//                    spi_clk <= 0; // st:CAP -> store
//                    state_spi_w <= ST_SPI_WAIT_4;
//                end
//                ST_SPI_WAIT_4 : begin //6
//                    spi_clk <= 1; //
//                    state_spi_w <= ST_SPI_WAIT_5; 
//                end
//                ST_SPI_WAIT_5 : begin //7
//                    spi_clk <= 0; // st: store -> idle
//                    state_spi_w <= ST_SPI_IDLE;
//                end                
            endcase
        end
    end   
endmodule
