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
  store double 0x4000000000000000, double* %ptr1, align 8
  %ptr2 = alloca double, align 8
  store double 0x4008000000000000, double* %ptr2, align 8
  %ptr3 = alloca double, align 8
  store double 0x4010000000000000, double* %ptr3, align 8
  %ptr4 = alloca double, align 8
  store double 0x4014000000000000, double* %ptr4, align 8
  %ptr5 = alloca double, align 8
  store double 0x4018000000000000, double* %ptr5, align 8
  %ptr6 = alloca double, align 8
  store double 0x401C000000000000, double* %ptr6, align 8
  %ptr7 = alloca double, align 8
  store double 0x4020000000000000, double* %ptr7, align 8
  %ptr8 = alloca double, align 8
  store double 0x4022000000000000, double* %ptr8, align 8
  %ptr9 = alloca double, align 8
  store double 0x4022000000000000, double* %ptr9, align 8
  %ptr10 = alloca double, align 8
  store double 0x4020000000000000, double* %ptr10, align 8
  %ptr11 = alloca double, align 8
  store double 0x401C000000000000, double* %ptr11, align 8
  %ptr12 = alloca double, align 8
  store double 0x4018000000000000, double* %ptr12, align 8
  %ptr13 = alloca double, align 8
  store double 0x4014000000000000, double* %ptr13, align 8
  %ptr14 = alloca double, align 8
  store double 0x4010000000000000, double* %ptr14, align 8
  %ptr15 = alloca double, align 8
  store double 0x4008000000000000, double* %ptr15, align 8
  %ptr16 = alloca double, align 8
  store double 0x4000000000000000, double* %ptr16, align 8
  %ptr17 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr17, align 8
  %ptr18 = alloca double, align 8
  store double 0x3FF0000000000000, double* %ptr18, align 8
  %ptr19 = alloca double, align 8
  store double 0x4000000000000000, double* %ptr19, align 8
  %ptr20 = alloca double, align 8
  store double 0x4008000000000000, double* %ptr20, align 8
  %ptr21 = alloca double, align 8
  store double 0x4010000000000000, double* %ptr21, align 8
  %ptr22 = alloca double, align 8
  store double 0x4014000000000000, double* %ptr22, align 8
  %ptr23 = alloca double, align 8
  store double 0x4018000000000000, double* %ptr23, align 8
  %ptr24 = alloca double, align 8
  store double 0x4000000000000000, double* %ptr24, align 8
  %ptr25 = alloca double, align 8
  store double 0x4008000000000000, double* %ptr25, align 8
  %ptr26 = alloca double, align 8
  store double 0x4024000000000000, double* %ptr26, align 8
  %ptr27 = alloca double, align 8
  store double 0x4034000000000000, double* %ptr27, align 8
  %ptr28 = alloca double, align 8
  store double 0x403E000000000000, double* %ptr28, align 8
  %ptr29 = alloca double, align 8
  store double 0x4044000000000000, double* %ptr29, align 8
  %ptr30 = alloca double, align 8
  store double 0x4049000000000000, double* %ptr30, align 8
  %ptr31 = alloca double, align 8
  store double 0x4000000000000000, double* %ptr31, align 8
  %ptr32 = alloca double, align 8
  store double 0x4008000000000000, double* %ptr32, align 8
  %ptr33 = alloca double, align 8
  store double 0x4010000000000000, double* %ptr33, align 8
  %ptr34 = alloca double, align 8
  store double 0x4014000000000000, double* %ptr34, align 8
  %ptr35 = alloca double, align 8
  store double 0x4000000000000000, double* %ptr35, align 8
  %ptr36 = alloca double, align 8
  store double 0x402E000000000000, double* %ptr36, align 8
  %ptr37 = alloca double, align 8
  store double 0x4039000000000000, double* %ptr37, align 8
  %ptr38 = alloca double, align 8
  store double 0x4041800000000000, double* %ptr38, align 8
  %ptr39 = alloca double, align 8
  store double 0x4046800000000000, double* %ptr39, align 8
  %ptr40 = alloca double, align 8
  store double 0x404B800000000000, double* %ptr40, align 8
  %ptr41 = alloca double, align 8
  store double 0x407DF00000000000, double* %ptr41, align 8
  %ptr42 = alloca double, align 8
  store double 0x408B700000000000, double* %ptr42, align 8
  %ptr43 = alloca double, align 8
  store double 0x4081100000000000, double* %ptr43, align 8
  %ptr44 = alloca double, align 8
  store double 0x403E000000000000, double* %ptr44, align 8
  %ptr45 = alloca double, align 8
  store double 0x4040000000000000, double* %ptr45, align 8
  %ptr46 = alloca double, align 8
  store double 0x402C000000000000, double* %ptr46, align 8
  %ptr47 = alloca double, align 8
  store double 0x4044800000000000, double* %ptr47, align 8
  %ptr48 = alloca double, align 8
  store double 0x4041800000000000, double* %ptr48, align 8
  %ptr49 = alloca double, align 8
  store double 0x4074D00000000000, double* %ptr49, align 8
  %reg0 = load double, double* %ptr44, align 8
  %ptr50 = alloca double, align 8
  store double %reg0, double* %ptr50, align 8
  %reg1 = load double, double* %ptr50, align 8
  %reg2 = fptosi double %reg1 to i32
  ret i32 %reg2
}

