`timescale 1ns/1ns

module comparator #(parameter INPUT_BITS = 8, NUM_CLASS = 7)
(
    clk, rstb,
    valid_in,

    //input_result
    in_data,

    //output_result
    decision,
	valid_out
);  

input clk, rstb, valid_in;
input signed [INPUT_BITS-1:0] in_data [0:NUM_CLASS-1];

output reg [3:0] decision;
output reg valid_out;

reg signed [INPUT_BITS-1:0] buffer [0:NUM_CLASS-1];
reg signed [INPUT_BITS-1:0] max;   
reg signed [INPUT_BITS-1:0] compare1_0, compare1_1, compare1_2, compare1_3,
							compare2_0, compare2_1;  

reg delay_count; 




always @ (*) begin
				// First compare
				if(buffer[0] >= buffer[1]) compare1_0 = buffer[0];
				else compare1_0 = buffer[1];
			
				if(buffer[2] >= buffer[3]) compare1_1 = buffer[2];
				else compare1_1 = buffer[3];

				if(buffer[4] >= buffer[5]) compare1_2 = buffer[4];
				else compare1_2 = buffer[5];

				compare1_3 = buffer[6];


				// Second compare
				if(compare1_0 >= compare1_1) compare2_0 = compare1_0;
				else compare2_0 = compare1_1;

				if(compare1_2 >= compare1_3) compare2_1 = compare1_2;
				else compare2_1 = compare1_3;



				// Third compare
				if(compare2_0 >= compare2_1) max = compare2_0;
				else max = compare2_1;



end


always @ (posedge clk or negedge rstb) begin
    if(!rstb) begin
        valid_out <= 0;
		// buf_count <= '0;
		delay_count <= 0;
		decision <= 4'b1111;
		// state <= 0;

    end
    else begin
		if(valid_in == 1) begin
			buffer <= in_data;
			valid_out <= 0;
			delay_count <= 1;
			
		end
		else begin
				if (delay_count) begin
					valid_out <= 1'b1;
					delay_count <= 1'b0;
				end
				else begin 
					valid_out <= 1'b0;
				end

				// Output Indexs
				if(max == buffer[0])
					decision <= 4'b0000;
				else if(max == buffer[1])
					decision <= 4'b0001;
				else if(max == buffer[2])
					decision <= 4'b0010;
				else if(max == buffer[3])
					decision <= 4'b0011;
				else if(max == buffer[4])
					decision <= 4'b0100;
				else if(max == buffer[5])
					decision <= 4'b0101;
				else if(max == buffer[6])
					decision <= 4'b0110;
				else 
					decision <= decision; // DO NOTHING on Useless Values
			end
		end
	end
	
endmodule