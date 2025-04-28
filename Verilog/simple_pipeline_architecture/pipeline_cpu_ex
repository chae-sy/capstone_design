`include "opcodes.v"

module CPU_W16_R4 #(
    parameter WORD_SIZE = 16,
    parameter REG_BITS  = 2
)
(
    // Global Signals
    input   wire    clk,      // global positive edge triggered clock
    input   wire    reset_n,  // global asynchronous negative triggered reset 

    // Instruction Memory IF (Read-only)
    input   wire    [WORD_SIZE-1:0]     imem_rdata_i,   // instruction read port
    output  wire    [WORD_SIZE-1:0]     imem_raddr_o,   // instruction read address

    // Data Memory IF (Read/Write)
    output  wire    [WORD_SIZE-1:0]     dmem_addr_o,    // data read address
    input   wire    [WORD_SIZE-1:0]     dmem_rdata_i,   // data read port
    output  wire    [WORD_SIZE-1:0]     dmem_wdata_o,   // data write port
    output  wire                        dmem_read_o,    // data read enable
    output  wire                        dmem_write_o,    // data write enable
    
    // Interrupt Handler IF (for debugging purpose)
    input   wire                        ir_ack_i,       // triggered when the given interrupt is properly handled
    output  wire                        ir_req_o,       // triggered when the internal interrupt occurs
    input   wire    [WORD_SIZE-1:0]     ir_msg_i,       // message transfer from interrupt handler (for RWD, not used)
    output  wire    [WORD_SIZE-1:0]     ir_msg_o       // message transfer to interrupt handler
);


    // Instruction Decoder IF
wire [3:0]           inst_opcode;
wire [1:0]           inst_rs;
wire [1:0]           inst_rt;
wire [1:0]           inst_rd;
wire [5:0]           inst_func;
wire [7:0]           inst_immed;
wire [11:0]          inst_taraddr;

    // Register File IF                   
wire                  rf_wr_enable;  // Register File: write enable
wire  [REG_BITS -1:0] rf_rd_reg1;    // Register File: read register 1
wire  [REG_BITS -1:0] rf_rd_reg2;    // Register File: read register 2
wire  [REG_BITS -1:0] rf_wr_reg;     // Register File: write register
wire  [WORD_SIZE-1:0] rf_rd_data1;   // Register File: read data 1
wire  [WORD_SIZE-1:0] rf_rd_data2;   // Register File: read data 2
wire  [WORD_SIZE-1:0] rf_wr_data;    // Register File: write wait

    // ALU IF
wire  [15:0] alu_operand1,  // ALU: first operand (or operand A) 
             alu_operand2;  // ALU: second operand (or operand B)
wire  [4 :0] alu_opcode;    // ALU: opcode (it may be different to the instruction's opcode)
wire [15:0] alu_result;    // ALU: output result
wire        alu_overflow;  // ALU: overflow or carry output
wire        bcond;  // ALU: Branch Condition Signal         
 
    // Control Unit IF 
wire [3:0]   contr_opcode;
wire [3:0]   IF_OP;
wire [5:0]   contr_func;
wire [4:0]   ALUOp;
wire        Branch,
            Jump,
            ALUSrc,
            RegDst,
            RegWrite,
            MemRead,
            MemtoReg,
            MemWrite,
            Jump_taraddr_select,
            Jump_reg;

    // Program Counter IF
wire         PCSrc,
             PCsource1,
             PCsource2,
             PCsource3,
             PCsource4,
             PCsource5;
reg [15:0]  inst_addr_nxt;
wire [15:0] target_addr;  
wire Hazard;

    // D- Flip Flop IF
    
reg [15:0] pc_f;
wire [15:0] pc_d;
wire [15:0] instruction_data_d;
wire  [WORD_SIZE-1:0] A_e,
                      B_e;
wire [15:0] Imm_e;
wire [1:0]           inst_rs_e;
wire [1:0]           inst_rt_e;
wire [1:0]           inst_rd_e;
wire [15:0] pc_e;
wire [1:0] dest;
wire RegDst_e,
     RegWrite_e,
     MemRead_e,
     MemWrite_e,
     MemtoReg_e,
     ALUSrc_e,
     Jump_reg_e;
wire [4:0] ALUOp_e;
wire [15:0] Aout_m;
wire [1:0] dest_m;
wire  [WORD_SIZE-1:0] B_m;
wire [15:0] pc_m;
wire RegWrite_m,
     MemRead_m,
     MemWrite_m,
     MemtoReg_m,
     Jump_reg_m;
wire [WORD_SIZE-1:0] MDR_w;
wire [15:0] Aout_w;
wire [1:0] dest_w;
wire [15:0] pc_w;
wire RegWrite_w,
     MemtoReg_w,
     Jump_reg_w;
     
reg [15:0] d_f1, q_f1;     
reg [15:0] d_f2, q_f2, q_inst;
reg [WORD_SIZE-1:0] d_f3, q_f3;
reg [WORD_SIZE-1:0] d_f4, q_f4;
reg [15:0] d_f5, q_f5;
reg [1:0] d_f6, q_f6;
reg [1:0] d_f7, q_f7;
reg [1:0] d_f8, q_f8;
reg [15:0] d_f31, q_f31;
reg d_f9, q_f9;
reg d_f10, q_f10;
reg d_f11, q_f11;
reg d_f12, q_f12;
reg d_f13, q_f13;
reg d_f14, q_f14;
reg d_f15, q_f15;
reg [4:0] d_f16, q_f16;
reg [15:0] d_f17, q_f17;
reg [1:0] d_f18, q_f18;
reg [WORD_SIZE-1:0] d_f19, q_f19;
reg [15:0] d_f32, q_f32;
reg d_f20, q_f20;
reg d_f21, q_f21;
reg d_f22, q_f22;
reg d_f23, q_f23;
reg d_f24, q_f24;
reg [WORD_SIZE-1:0] d_f25, q_f25;
reg [15:0] d_f26, q_f26;
reg [1:0] d_f27, q_f27;
reg [15:0] d_f33, q_f33;
reg d_f28, q_f28;
reg d_f29, q_f29;
reg d_f30, q_f30;

/********************************************************
    Reg Variable Declaration
********************************************************/

    // Program Counter, Interrupt Handle State
reg [1:0] state, state_nxt;
reg [15:0] inst_addr;
localparam IDLE = 0,
           ACTIVE = 1;
reg     datamem_disable;
reg     pc_disable;
reg [WORD_SIZE-1:0]    ir_msg;
reg [1:0] for_rt, for_rs, for_rs_branch;
reg ir_req;
reg hazard;
reg [3:0] num;
reg [1:0] flush, flush_nxt;

    // Part1: Instruction Memory Datapath
//////////////////////////////////////
///////////* IF stage *///////////////
//////////////////////////////////////
always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        inst_addr <= 'd0;
        hazard <= 'd0;
    end else begin
        inst_addr <= inst_addr_nxt;
    end
end
assign imem_raddr_o = inst_addr;

always @(*) begin
    inst_addr_nxt = inst_addr;
    pc_f = inst_addr+1;
    if (!pc_disable && !hazard) begin
        if (!PCSrc) inst_addr_nxt = target_addr;
        else begin
            if ((imem_rdata_i[15:12] == `OPCODE_JMP) || imem_rdata_i[15:12] == `OPCODE_JAL) begin
                inst_addr_nxt = pc_f[15:12] + imem_rdata_i[11:0];
            end
            else begin  
                inst_addr_nxt = pc_f;
            end
        end
    end
    else begin
        inst_addr_nxt = inst_addr;
    end
end    


// IF/ID (D-ff)
always @ (posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        q_f1 <= 0;
        q_f2 <= 16'bx;
    end 
    else begin 
        q_f1 <= d_f1;
        q_f2 <= d_f2;
    end
end
always @ (*) begin
     d_f1 = q_f1;
     d_f2 = q_f2;
    if (!hazard && !pc_disable) begin
        d_f1 = pc_f;
        d_f2 = imem_rdata_i;
    end
    else begin
        d_f1 = q_f1;
        d_f2 = q_f2;
    end
end

assign pc_d = q_f1;
assign instruction_data_d = q_f2;

//////////////////////////////////////
///////////* ID stage *///////////////
//////////////////////////////////////
assign inst_opcode = instruction_data_d[15:12];
assign inst_rs = instruction_data_d[11:10];
assign inst_rt = instruction_data_d[9:8];
assign inst_rd = instruction_data_d[7:6];
assign inst_func = instruction_data_d[5:0];
assign inst_immed = instruction_data_d[7:0];
assign inst_taraddr = instruction_data_d[11:0];

// hazard detection unit
always @(*) begin
    if ((inst_rs == dest) && (MemRead_e)) hazard = 1;
    else hazard = 0;
end

assign Hazard = hazard || flush;
//sign extend
reg [15:0] sign_extend;

always@(*) begin
    sign_extend <= {{8{instruction_data_d[7]}},inst_immed};
end

// branch target address
assign target_addr = (Jump_taraddr_select) ? ( (for_rs_branch == 1) ?  pc_e :(for_rs_branch == 2) ? pc_m : ((for_rs_branch == 3) ?  pc_w : rf_rd_data1 )) : (pc_d + sign_extend);

// branch prediction (pc + 1) test
always @ (posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        flush <= 0;
    end 
    else begin 
        flush <= flush_nxt;
    end
end

// flush signal
always@(*) begin
    flush_nxt = 0;
    if ( target_addr != pc_d && !PCSrc) flush_nxt = 1;
    else flush_nxt = 0;
end

// branch forwarding
always @ (*) begin
    for_rs_branch = 0;
    if ((inst_rs == inst_rd_e) && RegWrite_e) begin
        for_rs_branch = 1;
    end
    if ((inst_rs == dest_m) && RegWrite_m) begin
        for_rs_branch = 2;
    end
    else if ((inst_rs == dest_w) && RegWrite_w) begin
        for_rs_branch = 3;
    end
    else begin
    end
end
    // Part2: Register File Datapath
reg [1:0]           inst_rs_n;
reg [1:0]           inst_rt_n;

// read negative clock edge
always @( negedge clk) begin
    inst_rs_n = inst_rs;
    inst_rt_n = inst_rt;
end

assign rf_rd_reg1 = inst_rs_n;
assign rf_rd_reg2 = inst_rt_n;       

wire [15:0] rf_rd_data1_regfile, rf_rd_data2_regfile;
assign rf_rd_data1 = (rf_rd_reg1 == rf_wr_reg && rf_wr_enable) ? rf_wr_data : rf_rd_data1_regfile;
assign rf_rd_data2 = (rf_rd_reg2 == rf_wr_reg && rf_wr_enable) ? rf_wr_data : rf_rd_data2_regfile;

    // Register File 
    RegisterFile #(
        .WORD_SIZE  (WORD_SIZE), 
        .REG_BITS   (REG_BITS)
    ) rf_w16_r4 (
.clk(clk),.reset_n(reset_n),.wr_enable(rf_wr_enable),
.rd_reg1(rf_rd_reg1),.rd_reg2(rf_rd_reg2),.wr_reg(rf_wr_reg),
.rd_data1(rf_rd_data1_regfile),.rd_data2(rf_rd_data2_regfile),.wr_data(rf_wr_data)
);

// branch condition        
assign PCsource1 = (inst_opcode == 0) ? ((rf_rd_data1 != rf_rd_data2) ? 0 : 1) : 1;
assign PCsource2 = (inst_opcode == 1) ? ((rf_rd_data1 == rf_rd_data2) ? 0 : 1) : 1;
assign PCsource3 = (inst_opcode == 2) ? ((rf_rd_data1 > 0) ? 0 : 1) : 1;
assign PCsource4 = (inst_opcode == 3) ? ((rf_rd_data1 < 0) ? 0 : 1) : 1;
assign PCsource5 = !Jump_taraddr_select;
assign PCSrc = (PCsource1 && PCsource2 && PCsource3 && PCsource4 && PCsource5);

// ID/EX (D-ff)
always @ (posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        q_f3 <= 16'b0;
        q_f4 <= 16'b0;
        q_f5 <= 16'b0;
        q_f6 <= 2'bx;
        q_f7 <= 2'bx;
        q_f8 <= 2'bx;
        q_f31 <= 0;
    end 
    else begin 
        q_f3 <= d_f3;
        q_f4 <= d_f4;
        q_f5 <= d_f5;
        q_f6 <= d_f6;
        q_f7 <= d_f7;
        q_f8 <= d_f8;
        q_f31 <= d_f31;
    end
end
always @ (*) begin
     d_f3 = rf_rd_data1;
     d_f4 = rf_rd_data2;
     d_f5 = sign_extend;
     d_f6 = inst_rs;
     d_f7 = inst_rt;
     d_f8 = inst_rd;
     d_f31 = pc_d;
end

assign A_e = q_f3;
assign B_e = q_f4;
assign Imm_e = q_f5;
assign inst_rs_e = q_f6;
assign inst_rt_e = q_f7;
assign inst_rd_e = q_f8;
assign pc_e = q_f31;


// control unit D-ff
always @ (posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        q_f9 <= 0;
        q_f10 <= 0;
        q_f11 <= 0;
        q_f12 <= 0;
        q_f13 <= 0;
        q_f14 <= 0;
        q_f15 <= 0;
        q_f16 <= 5'bx;
    end 
    else begin 
        q_f9 <= d_f9;
        q_f10 <= d_f10;
        q_f11 <= d_f11;
        q_f12 <= d_f12;
        q_f13 <= d_f13;
        q_f14 <= d_f14;
        q_f15 <= d_f15;
        q_f16 <= d_f16;
    end
end
always @ (*) begin
     d_f9 = RegDst;
     d_f10 = RegWrite;
     d_f11 = MemRead;
     d_f12 = MemWrite;
     d_f13 = MemtoReg;
     d_f14 = ALUSrc;
     d_f15 = Jump_reg;
     d_f16 = ALUOp;
end

assign RegDst_e = q_f9;
assign RegWrite_e = q_f10;
assign MemRead_e = q_f11;
assign MemWrite_e = q_f12;
assign MemtoReg_e = q_f13;
assign ALUSrc_e = q_f14;
assign Jump_reg_e = q_f15;
assign ALUOp_e = q_f16;

//////////////////////////////////////
////////////* EX stage *//////////////
//////////////////////////////////////

// forwarding
always @ (*) begin
    for_rs = 0;
    if ((inst_rs_e == dest_m) && RegWrite_m) begin
        for_rs = 1;
    end
    else if ((inst_rs_e == dest_w) && RegWrite_w) begin
        for_rs = 2;
    end
    else begin
    end
end
always @ (*) begin
    for_rt = 0;
    if  ((inst_rt_e == dest_m) && RegWrite_m) begin
        for_rt = 1;
    end
    else if ((inst_rt_e == dest_w) && RegWrite_w) begin
        for_rt = 2;
    end
    else begin
    end
end

// Part3: ALU Datapath
assign alu_operand1 = (for_rs == 1) ? Aout_m : ((for_rs == 2) ?  rf_wr_data : A_e);
assign alu_operand2 = (ALUSrc_e) ? Imm_e : ((for_rt == 1) ? Aout_m : ((for_rt == 2) ? rf_wr_data : B_e));
assign alu_opcode = ALUOp_e;

// write reg mux
assign dest = (Jump_reg_e) ? 2'd2 : ((RegDst_e) ? inst_rd_e : inst_rt_e);

    // Arithmetic Logic Unit
    ALU alu_w16_m (
   .A_i(alu_operand1),.B_i(alu_operand2),
   .OP_i(alu_opcode),
   .C_i(1'b0),
   .F_o(alu_result),
   .C_o(alu_overflow), .bcond(bcond)
    );
    
// EX/MEM (D-ff)
always @ (posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        q_f17 <= 16'b0;
        q_f18 <= 2'bx;
        q_f19 <= 16'b0;
        q_f32 <= 0;
    end 
    else begin 
        q_f17 <= d_f17;
        q_f18 <= d_f18;
        q_f19 <= d_f19;
        q_f32 <= d_f32;
    end
end
always @ (*) begin
     d_f17 = alu_result;
     d_f18 = dest;
     d_f19 = B_e;
     d_f32 = pc_e;
end

assign Aout_m = q_f17;
assign dest_m = q_f18;
assign B_m = q_f19;
assign pc_m = q_f32;

// control unit D-ff
always @ (posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        q_f20 <= 0;
        q_f21 <= 0;
        q_f22 <= 0;
        q_f23 <= 0;
        q_f24 <= 0;
    end 
    else begin 
        q_f20 <= d_f20;
        q_f21 <= d_f21;
        q_f22 <= d_f22;
        q_f23 <= d_f23;
        q_f24 <= d_f24;
    end
end
always @ (*) begin
     d_f20 = RegWrite_e;
     d_f21 = MemRead_e;
     d_f22 = MemWrite_e;
     d_f23 = MemtoReg_e;
     d_f24 = Jump_reg_e;
end

assign RegWrite_m = q_f20;
assign MemRead_m = q_f21;
assign MemWrite_m = q_f22;
assign MemtoReg_m = q_f23;
assign Jump_reg_m = q_f24;

//////////////////////////////////////
////////////* MEM stage *//////////////
//////////////////////////////////////

    //Data Memory Unit
assign dmem_addr_o = (( MemRead_m || MemWrite_m ) ? Aout_m : 15'bx );
assign dmem_wdata_o = ( MemWrite_m) ? B_m : 0;
assign dmem_read_o = MemRead_m;
assign dmem_write_o = (datamem_disable) ? 15'bx : MemWrite_m;

// MEM/WB (D-ff)
always @ (posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        q_f25 <= 16'b0;
        q_f26 <= 16'b0;
        q_f27 <= 2'bx;
        q_f33 <= 0;
    end 
    else begin 
        q_f25 <= d_f25;
        q_f26 <= d_f26;
        q_f27 <= d_f27;
        q_f33 <= d_f33;
    end
end
always @ (*) begin
     d_f25 = dmem_rdata_i;
     d_f26 = (Jump_reg_w) ? (B_m) : Aout_m;
     d_f27 = dest_m;
     d_f33 = pc_m;
end

assign MDR_w = q_f25;
assign Aout_w = q_f26;
assign dest_w = q_f27;
assign pc_w = q_f33;

// control unit D-ff
always @ (posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        q_f28 <= 0;
        q_f29 <= 0;
        q_f30 <= 0;
    end 
    else begin 
        q_f28 <= d_f28;
        q_f29 <= d_f29;
        q_f30 <= d_f30;
    end
end
always @ (*) begin
     d_f28 = RegWrite_m;
     d_f29 = MemtoReg_m;
     d_f30 = Jump_reg_m;
end

assign RegWrite_w = q_f28;
assign MemtoReg_w = q_f29;
assign Jump_reg_w = q_f30;

//////////////////////////////////////
////////////* WB stage *//////////////
//////////////////////////////////////
assign rf_wr_reg = dest_w; 
assign rf_wr_data =  ((RegWrite_w) ? ((MemtoReg_w) ? MDR_w : Aout_w) : 15'bx);
assign rf_wr_enable = (RegWrite_w) ? 1'b1 : 1'b0;


    // Part4: Control Unit Datapath
assign contr_opcode = inst_opcode;
assign contr_func = inst_func;
assign ir_req_o = ir_req;

    // Control Unit 
    ControlUnit ctrl_unit (
    .con_in(contr_opcode),
    .funct(contr_func),
    .JUMP(Jump),
    .hazard(Hazard),
    .Branch(Branch),
    .MemRead(MemRead),
    .MemtoReg(MemtoReg),
    .ALUOp(ALUOp),
    .MemWrite(MemWrite),
    .ALUSrc(ALUSrc),
    .RegDst(RegDst),
    .RegWrite(RegWrite),
    .Jump_taraddr_select(Jump_taraddr_select),
    .Jump_reg(Jump_reg)
    );

    
    // Part6: Program Counter & Interrupt Handler with FSM
    
// WWD check FSM    
always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        state <= IDLE;
        num <= 0;
    end else begin
        state <= state_nxt;
    end
end

always @(*) begin
    state_nxt = state;
    case (state)
        IDLE: begin 
            if (instruction_data_d[5:0] == `FUNC_WWD) begin
                num = num + 1;
                if ((!ir_ack_i) && !((inst_rs == dest) && (MemRead_e))) begin
                    ir_req = 1;
                    state_nxt = ACTIVE;
                end
            end
            else begin  
                ir_req = 0;
                num = 0;
                state_nxt = IDLE;
            end
        end
        ACTIVE: begin
            if (ir_ack_i) begin 
                state_nxt = IDLE;
                ir_req = 0;
            end 
            else begin
            end
        end
   endcase
end     

always@(*) begin
    datamem_disable = 0;
    pc_disable = 0;
       case (state)
           IDLE: begin
                if(ir_req ) begin
                    // WWD forwarding
                    if (instruction_data_d[11:10] == dest && RegWrite_e) ir_msg = alu_result;
                    else if (instruction_data_d[11:10] == dest_m && RegWrite_m) ir_msg = MemRead_m ? dmem_rdata_i : Aout_m; 
                    else if (instruction_data_d[11:10]== dest_w && RegWrite_w) ir_msg = rf_wr_data;
                    else ir_msg = rf_rd_data1;
                    datamem_disable = 1;
                    pc_disable = 1;
                end
            end
           ACTIVE: begin
               if (ir_ack_i) begin 
                    datamem_disable = 0;
                    pc_disable = 0;
               end 
               else begin
                    datamem_disable = 1;
                    pc_disable = 1;
               end
           end
       endcase
end 
assign ir_msg_o = ir_msg;

endmodule
