	.text
	.file	"aero_compiler"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$392, %rsp                      # imm = 0x188
	.cfi_def_cfa_offset 400
	movabsq	$4621819117588971520, %r9       # imm = 0x4024000000000000
	movq	%r9, 384(%rsp)
	movabsq	$4624633867356078080, %rsi      # imm = 0x402E000000000000
	movq	%rsi, 376(%rsp)
	movabsq	$4622945017495814144, %r8       # imm = 0x4028000000000000
	movq	%r8, 368(%rsp)
	movabsq	$4625759767262920704, %rdi      # imm = 0x4032000000000000
	movq	%rdi, 360(%rsp)
	movq	%rsi, 352(%rsp)
	movabsq	$4626885667169763328, %rax      # imm = 0x4036000000000000
	movq	%rax, 344(%rsp)
	movq	%rdi, 336(%rsp)
	movabsq	$4628011567076605952, %rcx      # imm = 0x403A000000000000
	movq	%rcx, 328(%rsp)
	movabsq	$4626322717216342016, %rcx      # imm = 0x4034000000000000
	movq	%rcx, 320(%rsp)
	movabsq	$4629137466983448576, %rdx      # imm = 0x403E000000000000
	movq	%rdx, 312(%rsp)
	movq	%rax, 304(%rsp)
	movabsq	$4629841154425225216, %rax      # imm = 0x4040800000000000
	movq	%rax, 296(%rsp)
	movabsq	$4627730092099895296, %rax      # imm = 0x4039000000000000
	movq	%rax, 288(%rsp)
	movabsq	$4630544841867001856, %rax      # imm = 0x4043000000000000
	movq	%rax, 280(%rsp)
	movabsq	$4628574517030027264, %rax      # imm = 0x403C000000000000
	movq	%rax, 272(%rsp)
	movabsq	$4631107791820423168, %rax      # imm = 0x4045000000000000
	movq	%rax, 264(%rsp)
	movq	%rdx, 256(%rsp)
	movabsq	$4631530004285489152, %rax      # imm = 0x4046800000000000
	movq	%rax, 248(%rsp)
	movabsq	$4629700416936869888, %rax      # imm = 0x4040000000000000
	movq	%rax, 240(%rsp)
	movabsq	$4631952216750555136, %rax      # imm = 0x4048000000000000
	movq	%rax, 232(%rsp)
	movabsq	$4607182418800017408, %rax      # imm = 0x3FF0000000000000
	movq	%rax, 224(%rsp)
	movq	$0, 216(%rsp)
	movq	%rax, 208(%rsp)
	movq	%r9, 200(%rsp)
	movq	%r8, 192(%rsp)
	movq	%rsi, 184(%rsp)
	movq	%rdi, 176(%rsp)
	movq	%rcx, 168(%rsp)
	movabsq	$4617315517961601024, %rcx      # imm = 0x4014000000000000
	movq	%rcx, 160(%rsp)
	movabsq	$4618441417868443648, %rcx      # imm = 0x4018000000000000
	movq	%rcx, 152(%rsp)
	movabsq	$4619567317775286272, %rcx      # imm = 0x401C000000000000
	movq	%rcx, 144(%rsp)
	movabsq	$4620693217682128896, %rcx      # imm = 0x4020000000000000
	movq	%rcx, 136(%rsp)
	movq	%r9, 128(%rsp)
	movabsq	$4645040803167600640, %rcx      # imm = 0x4076800000000000
	movq	%rcx, 120(%rsp)
	movabsq	$4630263366890291200, %rcx      # imm = 0x4042000000000000
	movq	%rcx, 112(%rsp)
	movabsq	$4645058395353645056, %rdx      # imm = 0x4076900000000000
	movq	%rdx, 104(%rsp)
	movq	%rcx, 96(%rsp)
	movabsq	$4660236053863464960, %rdx      # imm = 0x40AC7C0000000000
	movq	%rdx, 88(%rsp)
	movabsq	$4661524681491218432, %rdx      # imm = 0x40B1100000000000
	movq	%rdx, 80(%rsp)
	movabsq	$4662715452584099840, %rdx      # imm = 0x40B54B0000000000
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movabsq	$4613937818241073152, %rcx      # imm = 0x4008000000000000
	movq	%rcx, 48(%rsp)
	movabsq	$4627448617123184640, %rcx      # imm = 0x4038000000000000
	movq	%rcx, 40(%rsp)
	movabsq	$4666008489909288960, %rcx      # imm = 0x40C0FE0000000000
	movq	%rcx, -128(%rsp)
	movabsq	$4635681760191971328, %rcx      # imm = 0x4055400000000000
	movq	%rcx, 32(%rsp)
	movabsq	$4636174341401214976, %rcx      # imm = 0x4057000000000000
	movq	%rcx, 24(%rsp)
	movabsq	$4639270566145032192, %rdx      # imm = 0x4062000000000000
	movq	%rdx, 16(%rsp)
	movabsq	$4625196817309499392, %rdx      # imm = 0x4030000000000000
	movq	%rdx, 8(%rsp)
	movabsq	$4621256167635550208, %rdx      # imm = 0x4022000000000000
	movq	%rdx, (%rsp)
	movabsq	$4636315078889570304, %rdx      # imm = 0x4057800000000000
	movq	%rdx, -8(%rsp)
	movabsq	$4636737291354636288, %rdx      # imm = 0x4059000000000000
	movq	%rdx, -16(%rsp)
	movq	%rax, -24(%rsp)
	movabsq	$4636596553866280960, %rdx      # imm = 0x4058800000000000
	movq	%rdx, -32(%rsp)
	movabsq	$4654311885213007872, %rdx      # imm = 0x4097700000000000
	movq	%rdx, -40(%rsp)
	movabsq	$4632233691727265792, %rdx      # imm = 0x4049000000000000
	movq	%rdx, -48(%rsp)
	movq	%rax, -56(%rsp)
	movq	%rcx, -64(%rsp)
	movabsq	$4635963235168681984, %rcx      # imm = 0x4056400000000000
	movq	%rcx, -72(%rsp)
	movabsq	$4635822497680326656, %rcx      # imm = 0x4055C00000000000
	movq	%rcx, -80(%rsp)
	movq	%rax, -88(%rsp)
	movq	%rax, -96(%rsp)
	movq	%rax, -104(%rsp)
	movq	%rax, -112(%rsp)
	movq	%rax, -120(%rsp)
	cvttsd2si	-128(%rsp), %eax
	addq	$392, %rsp                      # imm = 0x188
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
