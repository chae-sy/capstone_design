//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
// + Progect : COMPASS
// + DATE    : 2024/7/11/Thu
// + IP      : BufferPump
//
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

module F_SRAM_W32_A64 (  // Data Storage
	
	input		CLK,
	input		CEB,
	input		WEB,
	input	[9:0]	A,
	input	[127:0]	D,
	output	[127:0]	Q
);

	reg	[127:0] mem [1023:0];   // originally 32768 entries for 15bit address
	reg	[127:0] mem_d;
	
	always @ (*) begin
		if((!CEB)&(!WEB))	mem[A] = D;	//write
	end	

	always @ (posedge CLK) begin
	    // ? ??¡Æ ??? 

		if((!CEB)&(WEB))	mem_d <= mem[A];//read
//		else			mem_d <= 'hx;	//read
	end	

	assign 	Q = mem_d;	

endmodule