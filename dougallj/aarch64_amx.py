# IDA (disassembler) and Hex-Rays (decompiler) plugin for Apple AMX
#
# WIP research. (This was edited to add more info after someone posted it to
# Hacker News. Click "Revisions" to see full changes.)
#
# Copyright (c) 2020 dougallj


# Based on Python port of VMX intrinsics plugin:
# Copyright (c) 2019 w4kfu - Synacktiv

# Based on AArch64 8.3-A Pointer Authentication plugin:
# Copyright (c) 2018 Eloi Benoist-Vanderbeken - Synacktiv
# Copyright (c) 2018 xerub

# TODO: XZR can be an operand, but I don't handle that correctly in
# the decompuler yet.


# AMX: Apple Matrix coprocessor
#
# This is an undocumented arm64 ISA extension present on the Apple M1. These
# instructions have been reversed from Accelerate (vImage, libBLAS, libBNNS,
# libvDSP and libLAPACK all use them), and by experimenting with their
# behaviour on the M1. Apple has not published a compiler, assembler, or
# disassembler, but by callling into the public Accelerate framework
# APIs you can get the performance benefits (fast multiplication of big
# matrices). This is separate from the Apple Neural Engine.
#
# Warning: This is a work in progress, some of this is going to be incorrect.
#
# This may actually be very similar to Intel Advanced Matrix Extension (AMX),
# making the name collision even more confusing, but it's not a bad place to
# look for some idea of what's probably going on.
#
#
# WIP simulator/hardware tests are at:
# https://gist.github.com/dougallj/7cba721da1a94da725ee37c1e9cd1f21
#
#
# The coprocessor state consists of two 0x200 byte "registers", amx0 ("x")
# and amx1 ("y"), and one 0x1000 byte register amx2 ("z"). (Apple headers
# describe x, y, and z as register groups, where each row of 64-bytes is a
# "register", and describe only "z" as being "64 registers in an M-by-N
# matrix". They also describe a 64-bit AMX_STATE_T_EL1 register, which
# presumably records if AMX is enabled or not, but possibly other state
# too.)
#
# Each is typically loaded/stored from memory in rows of 0x40 bytes,
# although in some operations the registers can be indexed by byte offsets.
#
#
# AMX instructions are of the form:
#
#   0x00201000 | ((op & 0x1F) << 5) | (operand & 0x1F)
#
# AMX must be explicitly enabled using op=17, operand=0 and disabled using
# op=17, operand=1. In Accelerate, these instructions are always prefixed
# by three nops. What could go wrong?
#
# If instructions other than "enable" are executed when AMX is not enabled,
# they are treated as illegal instructions.
#
#
# All other operations (op=0-16 and op=18-22) seem to take a 64-bit register
# number (X0-X30 or 31=XZR) as the operand.
#
# This register is typically a bitfield containing further parameters to the
# operation. For example, loads and stores have a 56-bit address in bits 0
# through 55, a 5-bit register offset (in units of 0x40) in bits 56
# through 61, and a 1-bit flag in bit 62 (acting as an 0x40 byte load/store
# when zero, or an 0x80 byte (but aligned) load/store when one).
#
# My best guess at the names is based on:
# https://www.realworldtech.com/forum/?threadid=187087&curpostid=187120
#
# ops 0 through 7 are loads/stores:
#
#   0 is load  amx0 (amxldx)
#   1 is load  amx1 (amxldy)
#   2 is store amx0 (amxstx)
#   3 is store amx1 (amxsty)
#   4 is load  amx2 (amxldz)
#   5 is store amx2 (amxstz)
#
# 6 and 7 load and store amx2, but in a different order, and
# always as 0x40 bytes (bit 62 is ignored)
#
#   6 also loads amx2  (amxldzi)
#   7 also stores amx2 (amxstzi)
#   but they use halves of two registers in amx2
#    row index 0 = amx2[0].low and amx2[1].low interleaved
#    row index 1 = amx2[0].high and amx2[1].high interleaved
#    row index 2 = amx2[2].low and amx2[3].low interleaved
#    row index 3 = amx2[2].high and amx2[3].high interleaved
#   etc.
#
# Other operations do not touch memory, and usually have their result in
# amx2 (z), but 8 and 9 have their result in amx0 and amx1 (x/y), and 22 seems
# to have its result in row 0 (bytes 0 through 0x3F) of amx0.
#
# op8: extract row or move to x, result in amx0 (amxextrx)
#
#    move a 64-byte row from z or y to x
#    operands:
#      x offset in bytes = (argument >> 10) & 0x1FF
#      z offset in rows = (argument >> 20) & 63
#      move from y = argument >> 27) & 1
#    if moving from y, the x offset is rounded down to 0x40 bytes (so it can only
#    store to a row, rather than an arbitrary byte offset in x)
#
#    TODO: other bits
#
# op9: extract column or move to y, result in amx1/amx0 (amxextry)
#
#   move a 64-byte column from z to x or y, or move a 64-byte row from x to y
#
#      y offset in bytes = argument & 0x1FF
#      z offset in columns = (argument >> 20) & 63
#      move from x = (argument >> 27) & 1
#
#   TODO: many other bits factor into how the layout and order of columns is
#   determined, and which register is the destination. i'd like to finish
#   reversing it before trying to specify it, but my current understanding
#   is recorded in amx_state_extry at:
#
#    https://gist.github.com/dougallj/7cba721da1a94da725ee37c1e9cd1f21
#
# op10: multiply and add 64-bit floats (amxfma64)
# 
#   similar to op14, but 8x8 double multiplies for 64 results, added
#   (in groups of 8) to every 8th row of register "z" (z0, z8, z16).
#   no "32-bit mode" flag (?)
# 
# op11: multiply and subtract 64-bit floats (amxfms64)
# 
#   same as op10, but subtracting
# 
# op12: multiply and add 32-bit floats (amxfma32)
# 
#   similar to op14, but 16x16 float multiplies for 256 results, added
#   (in groups of 16) to every 4th row of register "z" (z0, z4, z8).
#   no "32-bit mode" flag (?)
# 
# op13: multiply and subtract 32-bit floats (amxfms32)
# 
#   same as op12, but subtracting
# 
# op14: multiply and add 16-bit signed integers (amxmac16)
# 
#   input two vectors of 32 16-bit values, one from register "x" and the other
#   from register "y". register "z" is the output, but may also be considered an
#   input for "add" operations.
#  
#   each value in the first vector is multiplied with each value in the second
#   vector (giving 32 * 32 = 1024 results), and each result is added to the value
#   in register "z". (although a bit in the input register may be set to skip
#   the addition, and simply store the result, which is typically used on the
#   first iteration of an accumulating loop.)
# 
#   operands:
#     input offset in x (byte): (argument & 0x1FF)
#     input offset in y (byte): ((argument >> 10) & 0x1FF)
#     row offset in z: (argument >> 20) & 63
#     clear "z" flag (don't add): (argument >> 27) & 1
#     skip y input (don't mul): (argument >> 28) & 1
#     skip x input (don't mul): (argument >> 29) & 1
#     row disable: (argument >> 32) & 0x7F
#     col disable: (argument >> 41) & 0x7F
#     32-bit mode: (argument >> 62) & 1
#     vector (non-matrix) multiply add (16x16->16 in one row): (argument >> 63) & 1
#     TODO: there are operands in other bits that still need to be reversed
#  
#   bit 62 makes output 32-bit ints, rather than 16-bit ints
#  
#   if bit 62 is zero, the output is in every second row, and if bit 27 is also
#   set, only every second row gets zeroed (old values remain in the other rows)
#
#   row/column disable skips the operation for certain entries in the row/column:
#     if disable is 0: process all entries
#     if disable is 1: process only every second entry (starting from the index 1)
#     if disable is 2: process only every second entry (starting from the index 0)
#     if (disable & 0x60) is 0x20: process only the entry at index "ignore & 0x1F"
#     if (disable & 0x60) is 0x40: process only the first "ignore & 0x1F" entries
#     if (disable & 0x60) is 0x60: process only the last "ignore & 0x1F" entries
#
#   for 32-bit output (sign extend all inputs to 32-bit):
#   z += [
#     [x0, x2, x4, x6, x8, x10, x12, x14, x16, x18, x20, x22, x24, x26, x28, x30] * y0,
#     [x1, x3, x5, x7, x9, x11, x13, x15, x17, x19, x21, x23, x25, x27, x29, x31] * y0,
#     [x0, x2, x4, x6, x8, x10, x12, x14, x16, x18, x20, x22, x24, x26, x28, x30] * y1,
#     [x1, x3, x5, x7, x9, x11, x13, x15, x17, x19, x21, x23, x25, x27, x29, x31] * y1,
#     [x0, x2, x4, x6, x8, x10, x12, x14, x16, x18, x20, x22, x24, x26, x28, x30] * y2,
#     [x1, x3, x5, x7, x9, x11, x13, x15, x17, x19, x21, x23, x25, x27, x29, x31] * y2,
#     ...
#   ]
# 
#   note that this works well with the "store z interleaved operation" to get the values out
#   in order.
# 
#   for 16-bit output (although the zeroes aren't really "added" just skipped):
#   z += [
#     [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31] * y0,
#     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31] * y1,
#     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31] * y2,
#     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     ...
#   ]
#
#
# op15: multiply and add 16-bit floats (amxfma16)
#   (same as op14, but for 16-bit floats)
#   bit 62 makes output 32-bit floats, rather than 16-bit floats
#  
# op16: multiply and subtract 16-bit floats (amxfms16)
#   (same as op15, but subtracting from register "z" instead of adding)
#
#  17 is enable/disable
#  18 does an operation, result in amx2  (vecint)
#   vector multiply 16-bit integers? (doesn't mac16 have a flag for this?)
#   z0[i] += x0[i] + y0[i]
#
#  19 does an operation, result in amx2  (vecfp)
#   vector multiply 16-bit floats? (doesn't mac16 have a flag for this?)
#   z0[i] += x0[i] + y0[i]
#
#  20 does an operation, result in amx2  (matint)
#   16-bit integer matrix multiply? (doesn't fma16 do this?)
#
#  21 does an operation, result in amx2  (matfp)
#   16-bit float matrix multiply? (doesn't fma16 do this?)
#
#  22 does an operation, result in amx0[0]  (genlut)
#
#   with xzr as input it takes 16 signed 32-bit integers from amx0[0] as input,
#   generates a 64-bit output:
#    [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -> 0xffffffffffffffff
#    [0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1] -> 0xf0
#    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] -> 0xfedcba9876543210
#
#
#
# Performance characteristics:
#
# This is trickier and still needs some work, but it appears to function as a
# co-processor, with operations being posted to it from the main processor.
# Because it doesn't go through the main processors out-of-order execution, it
# can be beneficial to add prefetch instructions (which do) immediately before
# AMX stores (and loads).
#
# All AMX instructions are sent via the store ports. Non-load/store AMX
# instructions can be fused with other non-load/store AMX instructions,
# to only use one store port.
#
# There is probably only one AMX coprocessor on the M1, as multiple threads
# trying to use AMX at the same time cause a slow-down. (It presumably must
# store the registers for each core?)
#
# There is some out-of-order execution capability on the co-processor (currently
# estimated at a 28 to 32 operation buffer, with very low confidence).
#
# An FMA typically has a 4-cycle latency, but a throughput of 1 per cycle
# (e.g. 16*16 = 256 32-bit float fused multiply-adds per cycle). This throughput
# is only possible if the destinations can be independent (using a z offset), so
# operations which use too much of the Z register will have lower throughput, and
# operations which use less may have higher throughput (TODO: test).
#
# Loads and stores seem to be the bottleneck. They can generate faults on bad
# addresses/alignments as expected, but appear to go via L2, with slight penalties
# for data in L1.
#
# Mixing loads and stores on the main processor with co-processor loads and stores
# causes big slowdowns, presumably as some kinds of barriers are needed to ensure a
# consistent view of memory.
#
# At best I've seen loads of 0x80 bytes to x and 0x80 bytes to y in 9 cycles
# (running in a loop).
#
# Because of the out-of-order capabilities, performing four fmas fits within
# this 9-cycle window at essentially no extra cost, so the following can run in
# a loop at 9 cycles per iteration:
#
#	AMX_LDX(load_addr | 0x4000000000000000);
#	AMX_LDY(load_addr | 0x4000000000000080);
#	AMX_FMA32(0x000000);
#	AMX_FMA32(0x110000);
#	AMX_FMA32(0x200040);
#	AMX_FMA32(0x310040);
#
# (this is accumulating a 32x32 tile within a larger matrix multiply)
#
#
#
# Slowdowns from mixing loads/stores:
#
#   nop/add before AMXLDX: 9 cycles/iter
#   str before AMXLDX (no-aliasing): 47 cycles/iter
#   str before AMXLDX (aliasing): 93 or 103 cycles/iter
#   ldr before AMXLDX (no-"aliasing"): 11 cycles/iter
#   ldr before AMXLDX ("aliasing"): 66 cycles/iter
#
#   nop/add before AMXSTX: 28 cycles/iter
#   str before AMXSTX (no-aliasing): 48 cycles/iter
#   str before AMXSTX (aliasing): 115 cycles/iter
#   ldr before AMXSTX (no-aliasing): 31 cycles/iter
#   ldr before AMXSTX (aliasing): 112 cycles/iter

import idaapi
import ida_hexrays


AMX_NONE = 0
AMX_OP0 = 1
AMX_OP1 = 2
AMX_OP2 = 3
AMX_OP3 = 4
AMX_OP4 = 5
AMX_OP5 = 6
AMX_OP6 = 7
AMX_OP7 = 8
AMX_OP8 = 9
AMX_OP9 = 10
AMX_OP10 = 11
AMX_OP11 = 12
AMX_OP12 = 13
AMX_OP13 = 14
AMX_OP14 = 15
AMX_OP15 = 16
AMX_OP16 = 17
AMX_OP17 = 18
AMX_OP18 = 19
AMX_OP19 = 20
AMX_OP20 = 21
AMX_OP21 = 22
AMX_OP22 = 23

OP_NAMES = {
	AMX_OP0: "AMXLDX",
	AMX_OP1: "AMXLDY",
	AMX_OP2: "AMXSTX",
	AMX_OP3: "AMXSTY",
	AMX_OP4: "AMXLDZ",
	AMX_OP5: "AMXSTZ",
	AMX_OP6: "AMXLDZI",
	AMX_OP7: "AMXSTZI",
	AMX_OP8: "AMXEXTRX", # amxextrx?
	AMX_OP9: "AMXEXTRY", # amxextry?
	AMX_OP10: "AMXFMA64",
	AMX_OP11: "AMXFMS64",
	AMX_OP12: "AMXFMA32",
	AMX_OP13: "AMXFMS32",
	AMX_OP14: "AMXMAC16",
	AMX_OP15: "AMXFMA16",
	AMX_OP16: "AMXFMS16",
	AMX_OP17: "AMX17", # amxset / amxclr
	AMX_OP18: "AMXVECINT",
	AMX_OP19: "AMXVECFP",
	AMX_OP20: "AMXMATINT",
	AMX_OP21: "AMXMATFP",
	AMX_OP22: "AMXGENLUT",
}

OP_INTRINSIC_NAMES = {
	AMX_OP0:  "__amx_ldx",
	AMX_OP1:  "__amx_ldy",
	AMX_OP2:  "__amx_stx",
	AMX_OP3:  "__amx_sty",
	AMX_OP4:  "__amx_ldz",
	AMX_OP5:  "__amx_stz",
	AMX_OP6:  "__amx_ldzi",
	AMX_OP7:  "__amx_stzi",
	AMX_OP8:  "__amx_extrx",
	AMX_OP9:  "__amx_extry",
	AMX_OP10: "__amx_fma64",
	AMX_OP11: "__amx_fms64",
	AMX_OP12: "__amx_fma32",
	AMX_OP13: "__amx_fms32",
	AMX_OP14: "__amx_mac16",
	AMX_OP15: "__amx_fma16",
	AMX_OP16: "__amx_fms16",
	AMX_OP17: "__amx_op17", # amxset / amxclr
	AMX_OP18: "__amx_vecint",
	AMX_OP19: "__amx_vecfp",
	AMX_OP20: "__amx_matint",
	AMX_OP21: "__amx_matfp",
	AMX_OP22: "__amx_genlut",
}

def decode_AMX(d, insn):
	if (d & 0xfffffC00) == 0x00201000:
		Xr = d & 31
		m = (d >> 5) & 31
		if m <= AMX_OP22 - AMX_OP0:
			#insn.itype = idaapi.ARM_nop
			insn.itype = idaapi.ARM_hlt
			insn.segpref = 14
			if m == 17:
				insn.Op1.type = idaapi.o_imm
				insn.Op1.value = Xr
				insn.Op1.dtype = idaapi.dt_byte
			else:
				insn.Op1.type = idaapi.o_reg
				insn.Op1.reg = Xr + 129
				insn.Op1.dtype = idaapi.dt_qword
			insn.insnpref = AMX_OP0 + m
			insn.size = 4
		return True
	return False

class Aarch64AMXHook(idaapi.IDP_Hooks):
	CUSTOM_INSTRUCTIONS = {idaapi.ARM_hlt}
	INDENT = 16
	def ev_ana_insn(self, outctx):
		return outctx.size if decode_AMX(idaapi.get_dword(outctx.ea), outctx) else 0

	def ev_emu_insn(self, insn):
		if insn.itype != idaapi.ARM_brk:
			return False
		return True

	def ev_out_mnem(self, outctx):
		if outctx.insn.itype in self.CUSTOM_INSTRUCTIONS:
			mnem = OP_NAMES.get(ord(outctx.insn.insnpref), None)
			if mnem is not None:
				outctx.out_custom_mnem(mnem, self.INDENT)
				return 1
		return 0

class MicroInstruction(ida_hexrays.minsn_t):

	def __init__(self, opcode, ea):
		ida_hexrays.minsn_t.__init__(self, ea)
		self.opcode = opcode
		self.l.zero()
		self.r.zero()
		self.d.zero()

class CallBuilder():

	def __init__(self, cdg, name, return_type=idaapi.tinfo_t(idaapi.BT_VOID)):
		self.emitted = False
		self.cdg = cdg
		self.callinfo = ida_hexrays.mcallinfo_t()
		self.callinfo.callee = idaapi.BADADDR
		self.callinfo.solid_args = 0
		self.callinfo.call_spd = 0
		self.callinfo.stkargs_top = 0
		self.callinfo.cc = idaapi.CM_CC_FASTCALL
		self.callinfo.return_type = return_type
		self.callinfo.flags = idaapi.FCI_SPLOK | idaapi.FCI_FINAL | idaapi.FCI_PROP
		self.callinfo.role = idaapi.ROLE_UNK

		glbhigh_off = cdg.mba.get_stack_region().off + cdg.mba.get_stack_region().size
		# what memory is visible to the call : GLBLOW - GLBHIGH
		self.callinfo.visible_memory.add(ida_hexrays.ivl_t(0x00, 0x100000))
		self.callinfo.visible_memory.add(ida_hexrays.ivl_t(glbhigh_off, 0xFFFFFFFFFFFFFFFF - glbhigh_off))
		# spoiled locations : GLBLOW - GLBHIGH
		self.callinfo.spoiled.mem.add(ida_hexrays.ivl_t(0x00, 0x100000))
		self.callinfo.spoiled.mem.add(ida_hexrays.ivl_t(glbhigh_off, 0xFFFFFFFFFFFFFFFF - glbhigh_off))

		self.callins = MicroInstruction(ida_hexrays.m_call, self.cdg.insn.ea)
		self.callins.l.make_helper(name)
		self.callins.d.t = ida_hexrays.mop_f
		self.callins.d.size = 0
		self.callins.d.f = self.callinfo

		if (return_type.is_void()):
			self.ins = self.callins
		else:
			self.callins.d.size = return_type.get_size()
			self.ins = MicroInstruction(ida_hexrays.m_mov, self.cdg.insn.ea)
			self.ins.l.t = ida_hexrays.mop_d
			self.ins.l.d = self.callins
			self.ins.l.size = self.callins.d.size
			self.ins.d.t = ida_hexrays.mop_r
			self.ins.d.r = 0x00
			self.ins.d.size = self.callins.d.size

	def add_register_argument(self, t, operand):
		ca = ida_hexrays.mcallarg_t()
		ca.t = idaapi.mop_r
		ca.r = operand
		ca.type = t
		ca.size = t.get_size()
		self.callinfo.args.push_back(ca)
		self.callinfo.solid_args += 1

	def set_return_register(self, reg):
		self.ins.d.r = reg

	def emit(self):
		if self.emitted == False:
			self.cdg.mb.insert_into_block(self.ins, self.cdg.mb.tail)
			self.emitted = True

class AMXFilter(ida_hexrays.microcode_filter_t):
	def __init__(self):
		ida_hexrays.microcode_filter_t.__init__(self)
		ida_hexrays.install_microcode_filter(self, True)

	def match(self, cdg):
		return cdg.insn.itype == idaapi.ARM_hlt and cdg.insn.insnpref != AMX_NONE

	def apply(self, cdg):
		op = ord(cdg.insn.insnpref)
		intrinsic_name = OP_INTRINSIC_NAMES.get(op, '__amx%d' % op)
		if cdg.insn.Op1.type == idaapi.o_reg:
			builder = CallBuilder(cdg, intrinsic_name)
			builder.add_register_argument(idaapi.tinfo_t(idaapi.BT_INT64 | idaapi.BTMT_UNSIGNED), cdg.load_operand(0))
			builder.emit()
		elif cdg.insn.Op1.type == idaapi.o_imm:
			if op == AMX_OP17 and cdg.insn.Op1.value == 0:
				builder = CallBuilder(cdg, '__amx_begin')
				builder.emit()
			elif op == AMX_OP17 and cdg.insn.Op1.value == 1:
				builder = CallBuilder(cdg, '__amx_end')
				builder.emit()
			else:
				builder = CallBuilder(cdg, '%s_%d' % (intrinsic_name, cdg.insn.Op1.value))
				builder.emit()

		return idaapi.MERR_OK


class Aarch64AMXPlugin(idaapi.plugin_t):
	flags = idaapi.PLUGIN_PROC | idaapi.PLUGIN_HIDE
	comment = "Aarch64 Apple AMX extension"
	wanted_hotkey = ""
	help = "Runs transparently"
	wanted_name = "Aarch64 AMX"
	hook = None
	enabled = 1

	def init(self):
		if idaapi.ph_get_id() != idaapi.PLFM_ARM or idaapi.BADADDR <= 0xFFFFFFFF:
			return idaapi.PLUGIN_SKIP
		if not ida_hexrays.init_hexrays_plugin():
			print("[-] {0} : no decompiler available, skipping".format(self.wanted_name))
			return idaapi.PLUGIN_SKIP
		print "%s init"%self.comment
		self.hook = Aarch64AMXHook()
		self.hook.hook()
		self.filter = AMXFilter()
		return idaapi.PLUGIN_KEEP

	def run():
		pass

	def term(self):
		if self.hook is not None:
			self.hook.unhook()
		print "%s unloaded"%self.comment

def PLUGIN_ENTRY():
	return Aarch64AMXPlugin()