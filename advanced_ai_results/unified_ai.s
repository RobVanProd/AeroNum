	.text
	.file	"aero_compiler"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$1344, %rsp                     # imm = 0x540
	.cfi_def_cfa_offset 1360
	.cfi_offset %rbx, -16
	movabsq	$4607182418800017408, %rax      # imm = 0x3FF0000000000000
	movq	%rax, 1336(%rsp)
	movq	%rax, 1328(%rsp)
	movq	$0, 1320(%rsp)
	movabsq	$4656616461584826368, %rcx      # imm = 0x409FA00000000000
	movq	%rcx, 1312(%rsp)
	movq	%rax, 1304(%rsp)
	movq	%rax, 1296(%rsp)
	movq	%rax, 1288(%rsp)
	movq	%rax, 1280(%rsp)
	movq	%rax, 1272(%rsp)
	movq	%rax, 1264(%rsp)
	movq	%rax, 1256(%rsp)
	movq	%rax, 1248(%rsp)
	movq	%rax, 1240(%rsp)
	movq	%rax, 1232(%rsp)
	movq	%rax, 1224(%rsp)
	movq	%rax, 1216(%rsp)
	movabsq	$4616189618054758400, %r9       # imm = 0x4010000000000000
	movq	%r9, 1208(%rsp)
	movabsq	$4652895714236432384, %rcx      # imm = 0x4092680000000000
	movq	%rcx, 1200(%rsp)
	movq	%rax, 1192(%rsp)
	movq	%rax, 1184(%rsp)
	movabsq	$4638707616191610880, %rcx      # imm = 0x4060000000000000
	movq	%rcx, 1176(%rsp)
	movabsq	$4634204016564240384, %rcx      # imm = 0x4050000000000000
	movq	%rcx, 1168(%rsp)
	movabsq	$4640959416005296128, %rcx      # imm = 0x4068000000000000
	movq	%rcx, 1160(%rsp)
	movabsq	$4629700416936869888, %rcx      # imm = 0x4040000000000000
	movq	%rcx, 1152(%rsp)
	movabsq	$4658479034282278912, %rcx      # imm = 0x40A63E0000000000
	movq	%rcx, 1144(%rsp)
	movabsq	$4653142004841054208, %rcx      # imm = 0x4093480000000000
	movq	%rcx, 1136(%rsp)
	movabsq	$4648198600562573312, %rcx      # imm = 0x4081B80000000000
	movq	%rcx, 1128(%rsp)
	movabsq	$4619567317775286272, %r8       # imm = 0x401C000000000000
	movq	%r8, 1120(%rsp)
	movabsq	$4635822497680326656, %rcx      # imm = 0x4055C00000000000
	movq	%rcx, 1112(%rsp)
	movq	%rax, 1104(%rsp)
	movabsq	$4627730092099895296, %rdx      # imm = 0x4039000000000000
	movq	%rdx, 1096(%rsp)
	movabsq	$4634063279075885056, %rdx      # imm = 0x404F800000000000
	movq	%rdx, 1088(%rsp)
	movq	%rax, 1080(%rsp)
	movq	%rax, 1072(%rsp)
	movq	%rax, 1064(%rsp)
	movq	%rax, 1056(%rsp)
	movq	%rax, 1048(%rsp)
	movq	%rax, 1040(%rsp)
	movabsq	$4632515166703976448, %r10      # imm = 0x404A000000000000
	movq	%r10, 1032(%rsp)
	movabsq	$4635189178982727680, %rsi      # imm = 0x4053800000000000
	movq	%rsi, 1024(%rsp)
	movq	%rax, 1016(%rsp)
	movq	%rax, 1008(%rsp)
	movq	%rax, 1000(%rsp)
	movq	%rax, 992(%rsp)
	movq	%rax, 984(%rsp)
	movq	%rax, 976(%rsp)
	movq	%rax, 968(%rsp)
	movabsq	$4613937818241073152, %rdx      # imm = 0x4008000000000000
	movq	%rdx, 960(%rsp)
	movq	%rax, 952(%rsp)
	movq	%rax, 944(%rsp)
	movabsq	$4636174341401214976, %rdi      # imm = 0x4057000000000000
	movq	%rdi, 936(%rsp)
	movq	%rax, 928(%rsp)
	movq	%rax, 920(%rsp)
	movabsq	$4643721389214269440, %rdx      # imm = 0x4071D00000000000
	movq	%rdx, 912(%rsp)
	movq	%rcx, 904(%rsp)
	movq	%rax, 896(%rsp)
	movabsq	$4635963235168681984, %r11      # imm = 0x4056400000000000
	movq	%r11, 888(%rsp)
	movq	%rax, 880(%rsp)
	movq	%rax, 872(%rsp)
	movq	%rax, 864(%rsp)
	movq	%rax, 856(%rsp)
	movq	%rax, 848(%rsp)
	movq	%rax, 840(%rsp)
	movq	%rax, 832(%rsp)
	movq	%rax, 824(%rsp)
	movq	%rax, 816(%rsp)
	movq	%rax, 808(%rsp)
	movabsq	$4611686018427387904, %rdx      # imm = 0x4000000000000000
	movq	%rdx, 800(%rsp)
	movq	%rax, 792(%rsp)
	movq	%rax, 784(%rsp)
	movq	%rax, 776(%rsp)
	movq	%rcx, 768(%rsp)
	movq	%rax, 760(%rsp)
	movq	%rax, 752(%rsp)
	movq	%rax, 744(%rsp)
	movabsq	$4635681760191971328, %rdx      # imm = 0x4055400000000000
	movq	%rdx, 736(%rsp)
	movabsq	$4634415122796773376, %rbx      # imm = 0x4050C00000000000
	movq	%rbx, 728(%rsp)
	movq	%rsi, 720(%rsp)
	movq	%rdi, 712(%rsp)
	movabsq	$4642437159633027072, %rbx      # imm = 0x406D400000000000
	movq	%rbx, 704(%rsp)
	movabsq	$4646729653027864576, %rbx      # imm = 0x407C800000000000
	movq	%rbx, 696(%rsp)
	movabsq	$4650151333213503488, %rbx      # imm = 0x4088A80000000000
	movq	%rbx, 688(%rsp)
	movabsq	$4638355772470722560, %rbx      # imm = 0x405EC00000000000
	movq	%rbx, 680(%rsp)
	movq	%rax, 672(%rsp)
	movq	%rax, 664(%rsp)
	movq	%rax, 656(%rsp)
	movq	%rax, 648(%rsp)
	movq	%rax, 640(%rsp)
	movq	%rax, 632(%rsp)
	movq	%rax, 624(%rsp)
	movq	%rax, 616(%rsp)
	movq	%rax, 608(%rsp)
	movq	%rax, 600(%rsp)
	movq	%rax, 592(%rsp)
	movq	%rax, 584(%rsp)
	movq	%rcx, 576(%rsp)
	movabsq	$4635470653959438336, %rbx      # imm = 0x4054800000000000
	movq	%rbx, 568(%rsp)
	movq	%rsi, 560(%rsp)
	movq	%rdx, 552(%rsp)
	movq	%rax, 544(%rsp)
	movq	%rax, 536(%rsp)
	movq	%rax, 528(%rsp)
	movq	%rax, 520(%rsp)
	movq	%rax, 512(%rsp)
	movq	%rax, 504(%rsp)
	movq	%rax, 496(%rsp)
	movq	%rax, 488(%rsp)
	movq	%rax, 480(%rsp)
	movq	%rax, 472(%rsp)
	movq	%rax, 464(%rsp)
	movq	%rax, 456(%rsp)
	movq	%rax, 448(%rsp)
	movq	%rax, 440(%rsp)
	movq	%rax, 432(%rsp)
	movq	%rax, 424(%rsp)
	movq	%rax, 416(%rsp)
	movq	%rax, 408(%rsp)
	movq	%rax, 400(%rsp)
	movq	%rax, 392(%rsp)
	movq	%r9, 384(%rsp)
	movq	%rax, 376(%rsp)
	movq	%rax, 368(%rsp)
	movq	%rax, 360(%rsp)
	movabsq	$4638496509959077888, %rbx      # imm = 0x405F400000000000
	movq	%rbx, 352(%rsp)
	movq	%r11, 344(%rsp)
	movq	%rsi, 336(%rsp)
	movq	%rdx, 328(%rsp)
	movq	%rcx, 320(%rsp)
	movq	%rcx, 312(%rsp)
	movq	%r10, 304(%rsp)
	movq	%rdi, 296(%rsp)
	movq	%rcx, 288(%rsp)
	movq	%r11, 280(%rsp)
	movq	%rdx, 272(%rsp)
	movq	%rdi, 264(%rsp)
	movabsq	$4636315078889570304, %rsi      # imm = 0x4057800000000000
	movq	%rsi, 256(%rsp)
	movq	%rax, 248(%rsp)
	movq	%rax, 240(%rsp)
	movq	%rax, 232(%rsp)
	movabsq	$4635892866424504320, %rsi      # imm = 0x4056000000000000
	movq	%rsi, 224(%rsp)
	movq	%rax, 216(%rsp)
	movq	%r11, 208(%rsp)
	movq	%rcx, 200(%rsp)
	movq	%rdx, 192(%rsp)
	movq	%r8, 184(%rsp)
	movq	%rax, 176(%rsp)
	movq	%rax, 168(%rsp)
	movq	%rax, 160(%rsp)
	movq	%rax, 152(%rsp)
	movq	%rax, 144(%rsp)
	movq	%rax, 136(%rsp)
	movq	%rax, 128(%rsp)
	movq	%rax, 120(%rsp)
	movq	%rax, 112(%rsp)
	movq	%rax, 104(%rsp)
	movq	%rax, 96(%rsp)
	movq	%rax, 88(%rsp)
	movq	%rax, 80(%rsp)
	movq	%rax, 72(%rsp)
	movq	%rax, 64(%rsp)
	movq	%rax, 56(%rsp)
	movq	%rax, 48(%rsp)
	movq	%rax, 40(%rsp)
	movq	%rax, 32(%rsp)
	movq	%rax, 24(%rsp)
	movq	%rax, 16(%rsp)
	movq	%rax, 8(%rsp)
	movq	%rax, (%rsp)
	movq	%rax, -8(%rsp)
	movq	%rax, -16(%rsp)
	movq	%rax, -24(%rsp)
	movq	%rax, -32(%rsp)
	movq	%rax, -40(%rsp)
	movq	%r11, -48(%rsp)
	movq	%rcx, -56(%rsp)
	movq	%rdx, -64(%rsp)
	movq	%rax, -72(%rsp)
	movq	%rax, -80(%rsp)
	movq	%rax, -88(%rsp)
	movq	%rax, -96(%rsp)
	movq	%rax, -104(%rsp)
	movq	%rax, -112(%rsp)
	movq	%rax, -120(%rsp)
	movq	%rax, -128(%rsp)
	movl	$1, %eax
	addq	$1344, %rsp                     # imm = 0x540
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
