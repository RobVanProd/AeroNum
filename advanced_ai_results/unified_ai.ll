; ModuleID = "aero_compiler"
source_filename = "aero_compiler"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare i32 @printf(i8*, ...)

define i32 @main() {
entry:
  %ptr0 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr0, align 8
  %ptr1 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr1, align 8
  %ptr2 = alloca double, align 8
  store double 0x0000000000000000, double* %ptr2, align 8
  %ptr3 = alloca double, align 8
  store double 0x409FA00000000000, double* %ptr3, align 8
  %ptr4 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr4, align 8
  %ptr5 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr5, align 8
  %ptr6 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr6, align 8
  %ptr7 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr7, align 8
  %ptr8 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr8, align 8
  %ptr9 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr9, align 8
  %ptr10 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr10, align 8
  %ptr11 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr11, align 8
  %ptr12 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr12, align 8
  %ptr13 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr13, align 8
  %ptr14 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr14, align 8
  %ptr15 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr15, align 8
  %ptr16 = alloca double, align 8
  store double 0x4010000000000000, double* %ptr16, align 8
  %ptr17 = alloca double, align 8
  store double 0x4092680000000000, double* %ptr17, align 8
  %ptr18 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr18, align 8
  %ptr19 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr19, align 8
  %ptr20 = alloca double, align 8
  store double 0x4060000000000000, double* %ptr20, align 8
  %ptr21 = alloca double, align 8
  store double 0x4050000000000000, double* %ptr21, align 8
  %ptr22 = alloca double, align 8
  store double 0x4068000000000000, double* %ptr22, align 8
  %ptr23 = alloca double, align 8
  store double 0x4040000000000000, double* %ptr23, align 8
  %ptr24 = alloca double, align 8
  store double 0x40A63E0000000000, double* %ptr24, align 8
  %ptr25 = alloca double, align 8
  store double 0x4093480000000000, double* %ptr25, align 8
  %ptr26 = alloca double, align 8
  store double 0x4081B80000000000, double* %ptr26, align 8
  %ptr27 = alloca double, align 8
  store double 0x401C000000000000, double* %ptr27, align 8
  %ptr28 = alloca double, align 8
  store double 0x4055C00000000000, double* %ptr28, align 8
  %ptr29 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr29, align 8
  %ptr30 = alloca double, align 8
  store double 0x4039000000000000, double* %ptr30, align 8
  %ptr31 = alloca double, align 8
  store double 0x404F800000000000, double* %ptr31, align 8
  %ptr32 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr32, align 8
  %ptr33 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr33, align 8
  %ptr34 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr34, align 8
  %ptr35 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr35, align 8
  %ptr36 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr36, align 8
  %ptr37 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr37, align 8
  %ptr38 = alloca double, align 8
  store double 0x404A000000000000, double* %ptr38, align 8
  %ptr39 = alloca double, align 8
  store double 0x4053800000000000, double* %ptr39, align 8
  %ptr40 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr40, align 8
  %ptr41 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr41, align 8
  %ptr42 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr42, align 8
  %ptr43 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr43, align 8
  %ptr44 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr44, align 8
  %ptr45 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr45, align 8
  %ptr46 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr46, align 8
  %ptr47 = alloca double, align 8
  store double 0x4008000000000000, double* %ptr47, align 8
  %ptr48 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr48, align 8
  %ptr49 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr49, align 8
  %ptr50 = alloca double, align 8
  store double 0x4057000000000000, double* %ptr50, align 8
  %ptr51 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr51, align 8
  %ptr52 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr52, align 8
  %ptr53 = alloca double, align 8
  store double 0x4071D00000000000, double* %ptr53, align 8
  %ptr54 = alloca double, align 8
  store double 0x4055C00000000000, double* %ptr54, align 8
  %ptr55 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr55, align 8
  %ptr56 = alloca double, align 8
  store double 0x4056400000000000, double* %ptr56, align 8
  %ptr57 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr57, align 8
  %ptr58 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr58, align 8
  %ptr59 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr59, align 8
  %ptr60 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr60, align 8
  %ptr61 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr61, align 8
  %ptr62 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr62, align 8
  %ptr63 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr63, align 8
  %ptr64 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr64, align 8
  %ptr65 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr65, align 8
  %ptr66 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr66, align 8
  %ptr67 = alloca double, align 8
  store double 0x4000000000000000, double* %ptr67, align 8
  %ptr68 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr68, align 8
  %ptr69 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr69, align 8
  %ptr70 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr70, align 8
  %ptr71 = alloca double, align 8
  store double 0x4055C00000000000, double* %ptr71, align 8
  %ptr72 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr72, align 8
  %ptr73 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr73, align 8
  %ptr74 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr74, align 8
  %ptr75 = alloca double, align 8
  store double 0x4055400000000000, double* %ptr75, align 8
  %ptr76 = alloca double, align 8
  store double 0x4050C00000000000, double* %ptr76, align 8
  %ptr77 = alloca double, align 8
  store double 0x4053800000000000, double* %ptr77, align 8
  %ptr78 = alloca double, align 8
  store double 0x4057000000000000, double* %ptr78, align 8
  %ptr79 = alloca double, align 8
  store double 0x406D400000000000, double* %ptr79, align 8
  %ptr80 = alloca double, align 8
  store double 0x407C800000000000, double* %ptr80, align 8
  %ptr81 = alloca double, align 8
  store double 0x4088A80000000000, double* %ptr81, align 8
  %ptr82 = alloca double, align 8
  store double 0x405EC00000000000, double* %ptr82, align 8
  %ptr83 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr83, align 8
  %ptr84 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr84, align 8
  %ptr85 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr85, align 8
  %ptr86 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr86, align 8
  %ptr87 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr87, align 8
  %ptr88 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr88, align 8
  %ptr89 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr89, align 8
  %ptr90 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr90, align 8
  %ptr91 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr91, align 8
  %ptr92 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr92, align 8
  %ptr93 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr93, align 8
  %ptr94 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr94, align 8
  %ptr95 = alloca double, align 8
  store double 0x4055C00000000000, double* %ptr95, align 8
  %ptr96 = alloca double, align 8
  store double 0x4054800000000000, double* %ptr96, align 8
  %ptr97 = alloca double, align 8
  store double 0x4053800000000000, double* %ptr97, align 8
  %ptr98 = alloca double, align 8
  store double 0x4055400000000000, double* %ptr98, align 8
  %ptr99 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr99, align 8
  %ptr100 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr100, align 8
  %ptr101 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr101, align 8
  %ptr102 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr102, align 8
  %ptr103 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr103, align 8
  %ptr104 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr104, align 8
  %ptr105 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr105, align 8
  %ptr106 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr106, align 8
  %ptr107 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr107, align 8
  %ptr108 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr108, align 8
  %ptr109 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr109, align 8
  %ptr110 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr110, align 8
  %ptr111 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr111, align 8
  %ptr112 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr112, align 8
  %ptr113 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr113, align 8
  %ptr114 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr114, align 8
  %ptr115 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr115, align 8
  %ptr116 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr116, align 8
  %ptr117 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr117, align 8
  %ptr118 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr118, align 8
  %ptr119 = alloca double, align 8
  store double 0x4010000000000000, double* %ptr119, align 8
  %ptr120 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr120, align 8
  %ptr121 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr121, align 8
  %ptr122 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr122, align 8
  %ptr123 = alloca double, align 8
  store double 0x405F400000000000, double* %ptr123, align 8
  %ptr124 = alloca double, align 8
  store double 0x4056400000000000, double* %ptr124, align 8
  %ptr125 = alloca double, align 8
  store double 0x4053800000000000, double* %ptr125, align 8
  %ptr126 = alloca double, align 8
  store double 0x4055400000000000, double* %ptr126, align 8
  %ptr127 = alloca double, align 8
  store double 0x4055C00000000000, double* %ptr127, align 8
  %ptr128 = alloca double, align 8
  store double 0x4055C00000000000, double* %ptr128, align 8
  %ptr129 = alloca double, align 8
  store double 0x404A000000000000, double* %ptr129, align 8
  %ptr130 = alloca double, align 8
  store double 0x4057000000000000, double* %ptr130, align 8
  %ptr131 = alloca double, align 8
  store double 0x4055C00000000000, double* %ptr131, align 8
  %ptr132 = alloca double, align 8
  store double 0x4056400000000000, double* %ptr132, align 8
  %ptr133 = alloca double, align 8
  store double 0x4055400000000000, double* %ptr133, align 8
  %ptr134 = alloca double, align 8
  store double 0x4057000000000000, double* %ptr134, align 8
  %ptr135 = alloca double, align 8
  store double 0x4057800000000000, double* %ptr135, align 8
  %ptr136 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr136, align 8
  %ptr137 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr137, align 8
  %ptr138 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr138, align 8
  %ptr139 = alloca double, align 8
  store double 0x4056000000000000, double* %ptr139, align 8
  %ptr140 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr140, align 8
  %ptr141 = alloca double, align 8
  store double 0x4056400000000000, double* %ptr141, align 8
  %ptr142 = alloca double, align 8
  store double 0x4055C00000000000, double* %ptr142, align 8
  %ptr143 = alloca double, align 8
  store double 0x4055400000000000, double* %ptr143, align 8
  %ptr144 = alloca double, align 8
  store double 0x401C000000000000, double* %ptr144, align 8
  %ptr145 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr145, align 8
  %ptr146 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr146, align 8
  %ptr147 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr147, align 8
  %ptr148 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr148, align 8
  %ptr149 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr149, align 8
  %ptr150 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr150, align 8
  %ptr151 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr151, align 8
  %ptr152 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr152, align 8
  %ptr153 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr153, align 8
  %ptr154 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr154, align 8
  %ptr155 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr155, align 8
  %ptr156 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr156, align 8
  %ptr157 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr157, align 8
  %ptr158 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr158, align 8
  %ptr159 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr159, align 8
  %ptr160 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr160, align 8
  %ptr161 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr161, align 8
  %ptr162 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr162, align 8
  %ptr163 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr163, align 8
  %ptr164 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr164, align 8
  %ptr165 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr165, align 8
  %ptr166 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr166, align 8
  %ptr167 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr167, align 8
  %ptr168 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr168, align 8
  %ptr169 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr169, align 8
  %ptr170 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr170, align 8
  %ptr171 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr171, align 8
  %ptr172 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr172, align 8
  %ptr173 = alloca double, align 8
  store double 0x4056400000000000, double* %ptr173, align 8
  %ptr174 = alloca double, align 8
  store double 0x4055C00000000000, double* %ptr174, align 8
  %ptr175 = alloca double, align 8
  store double 0x4055400000000000, double* %ptr175, align 8
  %ptr176 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr176, align 8
  %ptr177 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr177, align 8
  %ptr178 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr178, align 8
  %ptr179 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr179, align 8
  %ptr180 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr180, align 8
  %ptr181 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr181, align 8
  %ptr182 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr182, align 8
  %ptr183 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr183, align 8
  %reg0 = load double, double* %ptr172, align 8
  %reg1 = fptosi double %reg0 to i32
  ret i32 %reg1
}

