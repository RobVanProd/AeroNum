	.text
	.file	"aero_compiler"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$280, %rsp                      # imm = 0x118
	.cfi_def_cfa_offset 288
	movabsq	$4607182418800017408, %r10      # imm = 0x3FF0000000000000
	movq	%r10, 272(%rsp)
	movabsq	$4611686018427387904, %rax      # imm = 0x4000000000000000
	movq	%rax, 264(%rsp)
	movabsq	$4613937818241073152, %rcx      # imm = 0x4008000000000000
	movq	%rcx, 256(%rsp)
	movabsq	$4616189618054758400, %r8       # imm = 0x4010000000000000
	movq	%r8, 248(%rsp)
	movabsq	$4617315517961601024, %r9       # imm = 0x4014000000000000
	movq	%r9, 240(%rsp)
	movabsq	$4618441417868443648, %r11      # imm = 0x4018000000000000
	movq	%r11, 232(%rsp)
	movabsq	$4619567317775286272, %rsi      # imm = 0x401C000000000000
	movq	%rsi, 224(%rsp)
	movabsq	$4620693217682128896, %rdi      # imm = 0x4020000000000000
	movq	%rdi, 216(%rsp)
	movabsq	$4621256167635550208, %rdx      # imm = 0x4022000000000000
	movq	%rdx, 208(%rsp)
	movq	%rdx, 200(%rsp)
	movq	%rdi, 192(%rsp)
	movq	%rsi, 184(%rsp)
	movq	%r11, 176(%rsp)
	movq	%r9, 168(%rsp)
	movq	%r8, 160(%rsp)
	movq	%rcx, 152(%rsp)
	movq	%rax, 144(%rsp)
	movq	%r10, 136(%rsp)
	movq	%r10, 128(%rsp)
	movq	%rax, 120(%rsp)
	movq	%rcx, 112(%rsp)
	movq	%r8, 104(%rsp)
	movq	%r9, 96(%rsp)
	movq	%r11, 88(%rsp)
	movq	%rax, 80(%rsp)
	movq	%rcx, 72(%rsp)
	movabsq	$4621819117588971520, %rdx      # imm = 0x4024000000000000
	movq	%rdx, 64(%rsp)
	movabsq	$4626322717216342016, %rdx      # imm = 0x4034000000000000
	movq	%rdx, 56(%rsp)
	movabsq	$4629137466983448576, %rdi      # imm = 0x403E000000000000
	movq	%rdi, 48(%rsp)
	movabsq	$4630826316843712512, %rdx      # imm = 0x4044000000000000
	movq	%rdx, 40(%rsp)
	movabsq	$4632233691727265792, %rdx      # imm = 0x4049000000000000
	movq	%rdx, 32(%rsp)
	movq	%rax, 24(%rsp)
	movq	%rcx, 16(%rsp)
	movq	%r8, 8(%rsp)
	movq	%r9, (%rsp)
	movq	%rax, -8(%rsp)
	movabsq	$4624633867356078080, %rax      # imm = 0x402E000000000000
	movq	%rax, -16(%rsp)
	movabsq	$4627730092099895296, %rax      # imm = 0x4039000000000000
	movq	%rax, -24(%rsp)
	movabsq	$4630122629401935872, %rax      # imm = 0x4041800000000000
	movq	%rax, -32(%rsp)
	movabsq	$4631530004285489152, %rcx      # imm = 0x4046800000000000
	movq	%rcx, -40(%rsp)
	movabsq	$4632937379169042432, %rcx      # imm = 0x404B800000000000
	movq	%rcx, -48(%rsp)
	movabsq	$4647134273306886144, %rcx      # imm = 0x407DF00000000000
	movq	%rcx, -56(%rsp)
	movabsq	$4650934185492480000, %rcx      # imm = 0x408B700000000000
	movq	%rcx, -64(%rsp)
	movabsq	$4648013882609106944, %rcx      # imm = 0x4081100000000000
	movq	%rcx, -72(%rsp)
	movq	%rdi, -80(%rsp)
	movabsq	$4629700416936869888, %rcx      # imm = 0x4040000000000000
	movq	%rcx, -88(%rsp)
	movabsq	$4624070917402656768, %rcx      # imm = 0x402C000000000000
	movq	%rcx, -96(%rsp)
	movabsq	$4630967054332067840, %rcx      # imm = 0x4044800000000000
	movq	%rcx, -104(%rsp)
	movq	%rax, -112(%rsp)
	movabsq	$4644565814144401408, %rax      # imm = 0x4074D00000000000
	movq	%rax, -120(%rsp)
	movq	%rdi, -128(%rsp)
	movl	$30, %eax
	addq	$280, %rsp                      # imm = 0x118
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
