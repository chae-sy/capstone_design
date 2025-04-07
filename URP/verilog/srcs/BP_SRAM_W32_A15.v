//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
// + Progect : COMPASS
// + DATE    : 2024/7/11/Thu
// + IP      : BufferPump
//
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

module register_file #(parameter ADDRESS_BIT = 16, WORD_LENGTH = 128)(  // Data Storage
	
	input		CLK,
	input		CEB, //ren
	input		WEB, //wen
	input	[ADDRESS_BIT-1:0]	A,
	input	[WORD_LENGTH-1:0]	D,
	output	[WORD_LENGTH-1:0]	Q
);

	localparam ADDRESS = 2**ADDRESS_BIT;

	reg	[WORD_LENGTH-1:0] mem [ADDRESS-1:0];   // originally 32768 entries for 15bit address
	reg	[WORD_LENGTH-1:0] mem_d;
	
	always @ (*) begin
		if((!CEB)&(!WEB))	mem[A] = D;	//write
	end	

	always @ (posedge CLK) begin
		if((!CEB)&(WEB))	mem_d <= mem[A];//read
		else			mem_d <= mem_d;	//read
	end	

	assign 	Q = mem_d;	

endmodule
