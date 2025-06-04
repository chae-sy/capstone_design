`timescale 1ns / 1ps
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
// + Progect : COMPASS
// + DATE    : 2024/7/11/Thu
// + IP      : BufferPump
//
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

module memory_w_v0   // Data Storage
#(
    parameter addr_width = 10, // 2^10 = 1024 addresses
    parameter data_width = 128, // 8bit * 16 words
    parameter wr_delay = 8
)
(	input		CLK,
	input		CEB,
	input		WEB,
	input	[addr_width-1:0]	A,
	input	[data_width-1:0]	D,
	output	[data_width-1:0]	Q
);
	localparam num_addr = 2**addr_width;
    reg	[data_width-1:0] mem_W [num_addr:0];
	reg	[data_width-1:0] mem_d;
	reg	[addr_width-1:0] temp_A;
	
	always @ (posedge CLK) begin
	   temp_A = A;
	   if( (!CEB)&(!WEB) ) begin
	       	mem_W[temp_A] = D;	//write
	   end
	   else if( (!CEB)&(WEB) ) begin
            mem_d <= mem_W[temp_A]; //read
	   end
	   else	begin
	       mem_d <= 'hx;	//read
	   end
	end	

	assign 	Q = mem_d;	

endmodule