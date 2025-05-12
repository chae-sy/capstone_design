module SRAM_W32_A64 (  // Data Storage
	
	input		CLK,
	input		CEB,
	input		WEB,
	input	[5:0]	A,
	input	[31:0]	D,
	output	[31:0]	Q
);

	reg	[31:0] mem [1919:0];   // originally 32768 entries for 15bit address
	reg	[31:0] mem_d;
	
	always @ (*) begin
		if((!CEB)&(!WEB))	mem[A] = D;	//write
	end	

	always @ (posedge CLK) begin
		if((!CEB)&(WEB))	mem_d <= mem[A];//read
		else			mem_d <= 'hx;	//read
	end	

	assign 	Q = mem_d;	

endmodule
