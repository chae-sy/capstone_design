//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
// + Progect : COMPASS
// + DATE    : 2024/7/11/Thu
// + IP      : BufferPump
//
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

module input_buffer #(parameter ADDRESS_BIT = 4, WORD_LENGTH = 8)(  // Data Storage
	
	input		clk,
	input		ceb, //ren
	input		web, //wen
	input	[ADDRESS_BIT-1:0]	addr,
	input	[WORD_LENGTH-1:0]	d,
	output	[WORD_LENGTH-1:0]	q
);

	localparam ADDRESS = 2**ADDRESS_BIT;

	reg	[WORD_LENGTH-1:0] mem [ADDRESS-1:0];   
	reg	[WORD_LENGTH-1:0] mem_d;
	
	always @ (*) begin
		if((!ceb)&(!web))	mem[addr] = d;	//write
	end	

	always @ (posedge clk) begin
		if((!ceb)&(web))	mem_d <= mem[addr];//read
		else			mem_d <= mem_d;	//read
	end	

	assign 	q = mem_d;	

endmodule
