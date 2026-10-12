from bitstring import BitArray

class GemminiISA:
        
    def __init__ (self, inst, rs1, rs2):
        
        self.decode_info = {}
        
        self.inst_r = inst
        self.rs1_r = rs1
        self.rs2_r = rs2
        
        self.inst = BitArray(uint=int(inst, 10), length=32)
        self.rs1  = BitArray(uint=int(rs1, 10), length=64)
        self.rs2  = BitArray(uint=int(rs2, 10), length=64)
        self.funct = self.inst[0:7].uint
    
    def get_type(self):
    
        if(self.funct==0): instruction = Config(self.inst_r, self.rs1_r, self.rs2_r)
        elif(self.funct==1 or self.funct==2 or self.funct==14): instruction = Mvin(self.inst_r, self.rs1_r, self.rs2_r)
        elif(self.funct==3): instruction = Mvout(self.inst_r, self.rs1_r, self.rs2_r)
        elif(self.funct==4 or self.funct==5): instruction = Compute(self.inst_r, self.rs1_r, self.rs2_r)
        elif(self.funct==6): instruction = Preload(self.inst_r, self.rs1_r, self.rs2_r)
        elif(self.funct==7): instruction = Flush(self.inst_r, self.rs1_r, self.rs2_r)
        
        return instruction
    
    def decode(self):
        
        instruction = self.get_type() # 어떤 명령어인지 확인하고
        self.decode_info = instruction.decode() # 그 명령어에 따라 decode 진행
        
        return self.decode_info
    
    def get_latency(self, stage_info):
        
        instruction = self.get_type()        
        latency_info = instruction.get_latency(stage_info)
        
        return latency_info
        
        
        
    
class Config(GemminiISA):
    
    def decode(self):
        config_op = self.rs1[-2:].uint
        if(config_op==0): return self.decode_config_ex()
        elif(config_op==1): return self.decode_config_mvin()
        elif(config_op==2): return self.decode_config_mvout()
        else: return self.decode_config_norm()
        
    def decode_config_ex(self):
        dataflow = self.rs1[-3:-2].uint
        act = self.rs1[-4:-3].uint
        A_T = self.rs1[-9:-8].uint
        B_T = self.rs1[-10:-9].uint
        spad_stride = self.rs1[-32:-16].uint
        
        self.decode_info['inst'] = "config_ex"
        self.decode_info['act'] = act
        self.decode_info['A_T'] = A_T
        self.decode_info['B_T'] = B_T
        self.decode_info['spad_stride'] = spad_stride
        
        return self.decode_info
        
    def decode_config_mvin(self):
        Type = self.rs1[-5:-3].uint
        spad_stride = self.rs1[-32: -16].uint
        dram_stride = self.rs2[-32:].uint
        
        self.decode_info['inst'] = "config_mvin{0}".format(Type)
        self.decode_info['spad_stride'] = spad_stride
        self.decode_info['dram_stride'] = dram_stride
        
        return self.decode_info
    
    def decode_config_mvout(self):
        dram_stride = self.rs2[-32:].uint
        self.decode_info['inst'] = "config_mvout"
        self.decode_info['dram_stride'] = dram_stride
        
        return self.decode_info
        
    def config_norm(self):
        self.decode_info['inst'] = "config_norm"
        
        return self.decode_info
    
    def get_latency(self, stage_info):
        
        return None
            
            
        
class Mvin(GemminiISA):
    
    def decode(self):
        
        dram_addr = hex(self.rs1.uint)
        spad_addr = hex(self.rs2[-32:].uint)
        num_col = self.rs2[-48:-32].uint
        num_row = self.rs2[-64:-48].uint
        
        if(self.funct==1):
            self.decode_info['inst'] = 'mvin2'
        elif(self.funct==2):
            self.decode_info['inst'] = 'mvin1'
        else:
            self.decode_info['inst'] = 'mvin3'
        self.decode_info['dram_addr'] = dram_addr
        self.decode_info['spad_addr'] = spad_addr
        self.decode_info['num_col'] = num_col
        self.decode_info['num_row'] = num_row
        
        return self.decode_info
    
    def get_latency(self, stage_info):
#여기 수정함
        return {'Type': 'mvin', 'latency' : (stage_info['LD_CTRL_EXECUTE'], stage_info['LEAVE_LD_CTRL']-stage_info['LD_CTRL_EXECUTE'])}
        
class Mvout(GemminiISA):
    
    def decode(self):
        
        dram_addr = hex(self.rs1.uint)
        spad_addr = hex(self.rs2[-32:].uint)
        num_col = self.rs2[-48:-32].uint
        num_row = self.rs2[-64:-48].uint
        
        self.decode_info['inst'] = 'mvout'
        self.decode_info['dram_addr'] = dram_addr
        self.decode_info['spad_addr'] = spad_addr
        self.decode_info['num_col'] = num_col
        self.decode_info['num_row'] = num_row    

        return self.decode_info
    
    def get_latency(self, stage_info):
#여기 수정함
        return {'Type': 'mvout', 'latency' : (stage_info['ST_CTRL_EXECUTE'], stage_info['ROB_COMPLETE']-stage_info['ST_CTRL_EXECUTE'])}

class Flush(GemminiISA):
    
    def decode(self):
        
        self.decode_info['inst'] = 'flush'
        
        return self.decode_info
    
    def get_latency(self, stage_info):
        
        return {'Type': False, 'latency': (stage_info['ROB_ALLOC'], stage_info['ROB_COMPLETE'])}

class Compute(GemminiISA):
    
    def decode(self):
        
        spad_addr_A = hex(self.rs1[-32:].uint)
        num_col_A = self.rs1[-48:-32].uint
        num_row_A = self.rs1[-64:-48].uint
                
        spad_addr_BD = hex(self.rs2[-32:].uint)
        num_col_BD = self.rs2[-48:-32].uint
        num_row_BD = self.rs2[-64:-48].uint
        
        if(self.funct == 4):
            self.decode_info['inst'] = 'compute_and_flip'
        else:
            self.decode_info['inst'] = 'compute_and_stay'
        self.decode_info['spad_addr_A'] = spad_addr_A
        self.decode_info['num_col_A'] = num_col_A
        self.decode_info['num_row_A'] = num_row_A
        self.decode_info['spad_addr_BD'] = spad_addr_BD
        self.decode_info['num_col_BD'] = num_col_BD
        self.decode_info['num_row_BD'] = num_row_BD
        
        return self.decode_info
    
    def get_latency(self, stage_info):
        
        return {'Type': 'compute', 'latency' : (stage_info['ENTER_EX_CTRL'], stage_info['LEAVE_EX_CTRL']-stage_info['ENTER_EX_CTRL'])}
        
class Preload(GemminiISA):
    
    def decode(self):
        
        spad_addr_D = hex(self.rs1[-32:].uint)
        num_col_D = self.rs1[-48:-32].uint
        num_row_D = self.rs1[-64:-48].uint
        
        spad_addr_C = hex(self.rs2[-32:].uint)
        num_col_C = self.rs2[-48:-32].uint
        num_row_C = self.rs2[-64:-48].uint
        
        self.decode_info['inst'] = 'preload'
        
        self.decode_info['spad_addr_D'] = spad_addr_D
        self.decode_info['num_col_D'] = num_col_D
        self.decode_info['num_row_D'] = num_row_D
        self.decode_info['spad_addr_C'] = spad_addr_C
        self.decode_info['num_col_C'] = num_col_C
        self.decode_info['num_row_C'] = num_row_C
        
        return self.decode_info
    
    def get_latency(self, stage_info):
        
        return None
        


