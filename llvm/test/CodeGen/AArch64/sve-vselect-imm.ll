; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve < %s | FileCheck %s

define <vscale x 16 x i8> @sel_8_positive(<vscale x 16 x i1> %p) {
; CHECK-LABEL: sel_8_positive:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.b, p0/z, #3 // =0x3
; CHECK-NEXT:    ret
  %sel = select <vscale x 16 x i1> %p, <vscale x 16 x i8> splat (i8 3), <vscale x 16 x i8> zeroinitializer
  ret <vscale x 16 x i8> %sel
}

define <vscale x 8 x i16> @sel_16_positive(<vscale x 8 x i1> %p) {
; CHECK-LABEL: sel_16_positive:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.h, p0/z, #3 // =0x3
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %p, <vscale x 8 x i16> splat (i16 3), <vscale x 8 x i16> zeroinitializer
  ret <vscale x 8 x i16> %sel
}

define <vscale x 4 x i32> @sel_32_positive(<vscale x 4 x i1> %p) {
; CHECK-LABEL: sel_32_positive:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.s, p0/z, #3 // =0x3
; CHECK-NEXT:    ret
  %sel = select <vscale x 4 x i1> %p, <vscale x 4 x i32> splat (i32 3), <vscale x 4 x i32> zeroinitializer
  ret <vscale x 4 x i32> %sel
}

define <vscale x 2 x i64> @sel_64_positive(<vscale x 2 x i1> %p) {
; CHECK-LABEL: sel_64_positive:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.d, p0/z, #3 // =0x3
; CHECK-NEXT:    ret
  %sel = select <vscale x 2 x i1> %p, <vscale x 2 x i64> splat (i64 3), <vscale x 2 x i64> zeroinitializer
  ret <vscale x 2 x i64> %sel
}

define <vscale x 16 x i8> @sel_8_negative(<vscale x 16 x i1> %p) {
; CHECK-LABEL: sel_8_negative:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.b, p0/z, #-128 // =0xffffffffffffff80
; CHECK-NEXT:    ret
  %sel = select <vscale x 16 x i1> %p, <vscale x 16 x i8> splat (i8 -128), <vscale x 16 x i8> zeroinitializer
  ret <vscale x 16 x i8> %sel
}

define <vscale x 8 x i16> @sel_16_negative(<vscale x 8 x i1> %p) {
; CHECK-LABEL: sel_16_negative:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.h, p0/z, #-128 // =0xffffffffffffff80
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %p, <vscale x 8 x i16> splat (i16 -128), <vscale x 8 x i16> zeroinitializer
  ret <vscale x 8 x i16> %sel
}

define <vscale x 4 x i32> @sel_32_negative(<vscale x 4 x i1> %p) {
; CHECK-LABEL: sel_32_negative:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.s, p0/z, #-128 // =0xffffffffffffff80
; CHECK-NEXT:    ret
  %sel = select <vscale x 4 x i1> %p, <vscale x 4 x i32> splat (i32 -128), <vscale x 4 x i32> zeroinitializer
  ret <vscale x 4 x i32> %sel
}

define <vscale x 2 x i64> @sel_64_negative(<vscale x 2 x i1> %p) {
; CHECK-LABEL: sel_64_negative:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.d, p0/z, #-128 // =0xffffffffffffff80
; CHECK-NEXT:    ret
  %sel = select <vscale x 2 x i1> %p, <vscale x 2 x i64> splat (i64 -128), <vscale x 2 x i64> zeroinitializer
  ret <vscale x 2 x i64> %sel
}

define <vscale x 8 x i16> @sel_16_shifted(<vscale x 8 x i1> %p) {
; CHECK-LABEL: sel_16_shifted:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.h, p0/z, #512 // =0x200
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %p, <vscale x 8 x i16> splat (i16 512), <vscale x 8 x i16> zeroinitializer
  ret <vscale x 8 x i16> %sel
}

define <vscale x 4 x i32> @sel_32_shifted(<vscale x 4 x i1> %p) {
; CHECK-LABEL: sel_32_shifted:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.s, p0/z, #512 // =0x200
; CHECK-NEXT:    ret
  %sel = select <vscale x 4 x i1> %p, <vscale x 4 x i32> splat (i32 512), <vscale x 4 x i32> zeroinitializer
  ret <vscale x 4 x i32> %sel
}

define <vscale x 2 x i64> @sel_64_shifted(<vscale x 2 x i1> %p) {
; CHECK-LABEL: sel_64_shifted:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.d, p0/z, #512 // =0x200
; CHECK-NEXT:    ret
  %sel = select <vscale x 2 x i1> %p, <vscale x 2 x i64> splat (i64 512), <vscale x 2 x i64> zeroinitializer
  ret <vscale x 2 x i64> %sel
}

; TODO: We could actually use something like "cpy z0.b, p0/z, #-128". But it's
; a little tricky to prove correctness: we're using the predicate with the
; wrong width, so we'd have to prove the bits which would normally be unused
; are actually zero.
define <vscale x 8 x i16> @sel_16_illegal_wrong_extension(<vscale x 8 x i1> %p) {
; CHECK-LABEL: sel_16_illegal_wrong_extension:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movi v0.2d, #0000000000000000
; CHECK-NEXT:    mov z1.h, #128 // =0x80
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %p, <vscale x 8 x i16> splat (i16 128), <vscale x 8 x i16> zeroinitializer
  ret <vscale x 8 x i16> %sel
}

define <vscale x 4 x i32> @sel_32_illegal_wrong_extension(<vscale x 4 x i1> %p) {
; CHECK-LABEL: sel_32_illegal_wrong_extension:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movi v0.2d, #0000000000000000
; CHECK-NEXT:    mov z1.s, #128 // =0x80
; CHECK-NEXT:    mov z0.s, p0/m, z1.s
; CHECK-NEXT:    ret
  %sel = select <vscale x 4 x i1> %p, <vscale x 4 x i32> splat (i32 128), <vscale x 4 x i32> zeroinitializer
  ret <vscale x 4 x i32> %sel
}

define <vscale x 2 x i64> @sel_64_illegal_wrong_extension(<vscale x 2 x i1> %p) {
; CHECK-LABEL: sel_64_illegal_wrong_extension:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movi v0.2d, #0000000000000000
; CHECK-NEXT:    mov z1.d, #128 // =0x80
; CHECK-NEXT:    mov z0.d, p0/m, z1.d
; CHECK-NEXT:    ret
  %sel = select <vscale x 2 x i1> %p, <vscale x 2 x i64> splat (i64 128), <vscale x 2 x i64> zeroinitializer
  ret <vscale x 2 x i64> %sel
}

define <vscale x 8 x i16> @sel_16_illegal_shifted(<vscale x 8 x i1> %p) {
; CHECK-LABEL: sel_16_illegal_shifted:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movi v0.2d, #0000000000000000
; CHECK-NEXT:    mov w8, #513 // =0x201
; CHECK-NEXT:    mov z1.h, w8
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %p, <vscale x 8 x i16> splat (i16 513), <vscale x 8 x i16> zeroinitializer
  ret <vscale x 8 x i16> %sel
}

define <vscale x 4 x i32> @sel_32_illegal_shifted(<vscale x 4 x i1> %p) {
; CHECK-LABEL: sel_32_illegal_shifted:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movi v0.2d, #0000000000000000
; CHECK-NEXT:    mov w8, #513 // =0x201
; CHECK-NEXT:    mov z1.s, w8
; CHECK-NEXT:    mov z0.s, p0/m, z1.s
; CHECK-NEXT:    ret
  %sel = select <vscale x 4 x i1> %p, <vscale x 4 x i32> splat (i32 513), <vscale x 4 x i32> zeroinitializer
  ret <vscale x 4 x i32> %sel
}

define <vscale x 2 x i64> @sel_64_illegal_shifted(<vscale x 2 x i1> %p) {
; CHECK-LABEL: sel_64_illegal_shifted:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movi v0.2d, #0000000000000000
; CHECK-NEXT:    mov w8, #513 // =0x201
; CHECK-NEXT:    mov z1.d, x8
; CHECK-NEXT:    mov z0.d, p0/m, z1.d
; CHECK-NEXT:    ret
  %sel = select <vscale x 2 x i1> %p, <vscale x 2 x i64> splat (i64 513), <vscale x 2 x i64> zeroinitializer
  ret <vscale x 2 x i64> %sel
}

define <vscale x 16 x i8> @sel_merge_8_positive(<vscale x 16 x i1> %p, <vscale x 16 x i8> %in) {
; CHECK-LABEL: sel_merge_8_positive:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.b, p0/m, #3 // =0x3
; CHECK-NEXT:    ret
  %sel = select <vscale x 16 x i1> %p, <vscale x 16 x i8> splat (i8 3), <vscale x 16 x i8> %in
  ret <vscale x 16 x i8> %sel
}

define <vscale x 8 x i16> @sel_merge_16_positive(<vscale x 8 x i1> %p, <vscale x 8 x i16> %in) {
; CHECK-LABEL: sel_merge_16_positive:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.h, p0/m, #3 // =0x3
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %p, <vscale x 8 x i16> splat (i16 3), <vscale x 8 x i16> %in
  ret <vscale x 8 x i16> %sel
}

define <vscale x 4 x i32> @sel_merge_32_positive(<vscale x 4 x i1> %p, <vscale x 4 x i32> %in) {
; CHECK-LABEL: sel_merge_32_positive:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.s, p0/m, #3 // =0x3
; CHECK-NEXT:    ret
  %sel = select <vscale x 4 x i1> %p, <vscale x 4 x i32> splat (i32 3), <vscale x 4 x i32> %in
  ret <vscale x 4 x i32> %sel
}

define <vscale x 2 x i64> @sel_merge_64_positive(<vscale x 2 x i1> %p, <vscale x 2 x i64> %in) {
; CHECK-LABEL: sel_merge_64_positive:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.d, p0/m, #3 // =0x3
; CHECK-NEXT:    ret
  %sel = select <vscale x 2 x i1> %p, <vscale x 2 x i64> splat (i64 3), <vscale x 2 x i64> %in
  ret <vscale x 2 x i64> %sel
}

define <vscale x 16 x i8> @sel_merge_8_negative(<vscale x 16 x i1> %p, <vscale x 16 x i8> %in) {
; CHECK-LABEL: sel_merge_8_negative:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.b, p0/m, #-128 // =0xffffffffffffff80
; CHECK-NEXT:    ret
  %sel = select <vscale x 16 x i1> %p, <vscale x 16 x i8> splat (i8 -128), <vscale x 16 x i8> %in
  ret <vscale x 16 x i8> %sel
}

define <vscale x 8 x i16> @sel_merge_16_negative(<vscale x 8 x i1> %p, <vscale x 8 x i16> %in) {
; CHECK-LABEL: sel_merge_16_negative:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.h, p0/m, #-128 // =0xffffffffffffff80
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %p, <vscale x 8 x i16> splat (i16 -128), <vscale x 8 x i16> %in
  ret <vscale x 8 x i16> %sel
}

define <vscale x 4 x i32> @sel_merge_32_negative(<vscale x 4 x i1> %p, <vscale x 4 x i32> %in) {
; CHECK-LABEL: sel_merge_32_negative:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.s, p0/m, #-128 // =0xffffffffffffff80
; CHECK-NEXT:    ret
  %sel = select <vscale x 4 x i1> %p, <vscale x 4 x i32> splat (i32 -128), <vscale x 4 x i32> %in
  ret <vscale x 4 x i32> %sel
}

define <vscale x 2 x i64> @sel_merge_64_negative(<vscale x 2 x i1> %p, <vscale x 2 x i64> %in) {
; CHECK-LABEL: sel_merge_64_negative:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.d, p0/m, #-128 // =0xffffffffffffff80
; CHECK-NEXT:    ret
  %sel = select <vscale x 2 x i1> %p, <vscale x 2 x i64> splat (i64 -128), <vscale x 2 x i64> %in
  ret <vscale x 2 x i64> %sel
}

define <vscale x 16 x i8> @sel_merge_8_zero(<vscale x 16 x i1> %p, <vscale x 16 x i8> %in) {
; CHECK-LABEL: sel_merge_8_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.b, p0/m, #0 // =0x0
; CHECK-NEXT:    ret
  %sel = select <vscale x 16 x i1> %p, <vscale x 16 x i8> zeroinitializer, <vscale x 16 x i8> %in
  ret <vscale x 16 x i8> %sel
}

define <vscale x 8 x i16> @sel_merge_16_zero(<vscale x 8 x i1> %p, <vscale x 8 x i16> %in) {
; CHECK-LABEL: sel_merge_16_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.h, p0/m, #0 // =0x0
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %p, <vscale x 8 x i16> zeroinitializer, <vscale x 8 x i16> %in
  ret <vscale x 8 x i16> %sel
}

define <vscale x 4 x i32> @sel_merge_32_zero(<vscale x 4 x i1> %p, <vscale x 4 x i32> %in) {
; CHECK-LABEL: sel_merge_32_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.s, p0/m, #0 // =0x0
; CHECK-NEXT:    ret
  %sel = select <vscale x 4 x i1> %p, <vscale x 4 x i32> zeroinitializer, <vscale x 4 x i32> %in
  ret <vscale x 4 x i32> %sel
}

define <vscale x 2 x i64> @sel_merge_64_zero(<vscale x 2 x i1> %p, <vscale x 2 x i64> %in) {
; CHECK-LABEL: sel_merge_64_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.d, p0/m, #0 // =0x0
; CHECK-NEXT:    ret
  %sel = select <vscale x 2 x i1> %p, <vscale x 2 x i64> zeroinitializer, <vscale x 2 x i64> %in
  ret <vscale x 2 x i64> %sel
}

define <vscale x 8 x half> @sel_merge_nxv8f16_zero(<vscale x 8 x i1> %p, <vscale x 8 x half> %in) {
; CHECK-LABEL: sel_merge_nxv8f16_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.h, p0/m, #0 // =0x0
; CHECK-NEXT:    ret
%sel = select <vscale x 8 x i1> %p, <vscale x 8 x half> zeroinitializer, <vscale x 8 x half> %in
ret <vscale x 8 x half> %sel
}

define <vscale x 4 x half> @sel_merge_nx4f16_zero(<vscale x 4 x i1> %p, <vscale x 4 x half> %in) {
; CHECK-LABEL: sel_merge_nx4f16_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.s, p0/m, #0 // =0x0
; CHECK-NEXT:    ret
%sel = select <vscale x 4 x i1> %p, <vscale x 4 x half> zeroinitializer, <vscale x 4 x half> %in
ret <vscale x 4 x half> %sel
}

define <vscale x 2 x half> @sel_merge_nx2f16_zero(<vscale x 2 x i1> %p, <vscale x 2 x half> %in) {
; CHECK-LABEL: sel_merge_nx2f16_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.d, p0/m, #0 // =0x0
; CHECK-NEXT:    ret
%sel = select <vscale x 2 x i1> %p, <vscale x 2 x half> zeroinitializer, <vscale x 2 x half> %in
ret <vscale x 2 x half> %sel
}

define <vscale x 4 x float> @sel_merge_nx4f32_zero(<vscale x 4 x i1> %p, <vscale x 4 x float> %in) {
; CHECK-LABEL: sel_merge_nx4f32_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.s, p0/m, #0 // =0x0
; CHECK-NEXT:    ret
%sel = select <vscale x 4 x i1> %p, <vscale x 4 x float> zeroinitializer, <vscale x 4 x float> %in
ret <vscale x 4 x float> %sel
}

define <vscale x 2 x float> @sel_merge_nx2f32_zero(<vscale x 2 x i1> %p, <vscale x 2 x float> %in) {
; CHECK-LABEL: sel_merge_nx2f32_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.d, p0/m, #0 // =0x0
; CHECK-NEXT:    ret
%sel = select <vscale x 2 x i1> %p, <vscale x 2 x float> zeroinitializer, <vscale x 2 x float> %in
ret <vscale x 2 x float> %sel
}

define <vscale x 2 x double> @sel_merge_nx2f64_zero(<vscale x 2 x i1> %p, <vscale x 2 x double> %in) {
; CHECK-LABEL: sel_merge_nx2f64_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.d, p0/m, #0 // =0x0
; CHECK-NEXT:    ret
%sel = select <vscale x 2 x i1> %p, <vscale x 2 x double> zeroinitializer, <vscale x 2 x double> %in
ret <vscale x 2 x double> %sel
}

define <vscale x 8 x half> @sel_merge_nxv8f16_negative_zero(<vscale x 8 x i1> %p, <vscale x 8 x half> %in) {
; CHECK-LABEL: sel_merge_nxv8f16_negative_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #32768 // =0x8000
; CHECK-NEXT:    mov z1.h, w8
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
%sel = select <vscale x 8 x i1> %p, <vscale x 8 x half> splat (half -0.0), <vscale x 8 x half> %in
ret <vscale x 8 x half> %sel
}

define <vscale x 4 x half> @sel_merge_nx4f16_negative_zero(<vscale x 4 x i1> %p, <vscale x 4 x half> %in) {
; CHECK-LABEL: sel_merge_nx4f16_negative_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #32768 // =0x8000
; CHECK-NEXT:    mov z1.h, w8
; CHECK-NEXT:    mov z0.s, p0/m, z1.s
; CHECK-NEXT:    ret
%sel = select <vscale x 4 x i1> %p, <vscale x 4 x half> splat (half -0.0), <vscale x 4 x half> %in
ret <vscale x 4 x half> %sel
}

define <vscale x 2 x half> @sel_merge_nx2f16_negative_zero(<vscale x 2 x i1> %p, <vscale x 2 x half> %in) {
; CHECK-LABEL: sel_merge_nx2f16_negative_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #32768 // =0x8000
; CHECK-NEXT:    mov z1.h, w8
; CHECK-NEXT:    mov z0.d, p0/m, z1.d
; CHECK-NEXT:    ret
%sel = select <vscale x 2 x i1> %p, <vscale x 2 x half> splat (half -0.0), <vscale x 2 x half> %in
ret <vscale x 2 x half> %sel
}

define <vscale x 4 x float> @sel_merge_nx4f32_negative_zero(<vscale x 4 x i1> %p, <vscale x 4 x float> %in) {
; CHECK-LABEL: sel_merge_nx4f32_negative_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #-2147483648 // =0x80000000
; CHECK-NEXT:    mov z1.s, w8
; CHECK-NEXT:    mov z0.s, p0/m, z1.s
; CHECK-NEXT:    ret
%sel = select <vscale x 4 x i1> %p, <vscale x 4 x float> splat (float -0.0), <vscale x 4 x float> %in
ret <vscale x 4 x float> %sel
}

define <vscale x 2 x float> @sel_merge_nx2f32_negative_zero(<vscale x 2 x i1> %p, <vscale x 2 x float> %in) {
; CHECK-LABEL: sel_merge_nx2f32_negative_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #-2147483648 // =0x80000000
; CHECK-NEXT:    mov z1.s, w8
; CHECK-NEXT:    mov z0.d, p0/m, z1.d
; CHECK-NEXT:    ret
%sel = select <vscale x 2 x i1> %p, <vscale x 2 x float> splat (float -0.0), <vscale x 2 x float> %in
ret <vscale x 2 x float> %sel
}

define <vscale x 2 x double> @sel_merge_nx2f64_negative_zero(<vscale x 2 x i1> %p, <vscale x 2 x double> %in) {
; CHECK-LABEL: sel_merge_nx2f64_negative_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov x8, #-9223372036854775808 // =0x8000000000000000
; CHECK-NEXT:    mov z1.d, x8
; CHECK-NEXT:    mov z0.d, p0/m, z1.d
; CHECK-NEXT:    ret
%sel = select <vscale x 2 x i1> %p, <vscale x 2 x double> splat (double -0.0), <vscale x 2 x double> %in
ret <vscale x 2 x double> %sel
}

define <vscale x 8 x i16> @sel_merge_16_shifted(<vscale x 8 x i1> %p, <vscale x 8 x i16> %in) {
; CHECK-LABEL: sel_merge_16_shifted:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.h, p0/m, #512 // =0x200
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %p, <vscale x 8 x i16> splat (i16 512), <vscale x 8 x i16> %in
  ret <vscale x 8 x i16> %sel
}

define <vscale x 4 x i32> @sel_merge_32_shifted(<vscale x 4 x i1> %p, <vscale x 4 x i32> %in) {
; CHECK-LABEL: sel_merge_32_shifted:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.s, p0/m, #512 // =0x200
; CHECK-NEXT:    ret
  %sel = select <vscale x 4 x i1> %p, <vscale x 4 x i32> splat (i32 512), <vscale x 4 x i32> %in
  ret <vscale x 4 x i32> %sel
}

define <vscale x 2 x i64> @sel_merge_64_shifted(<vscale x 2 x i1> %p, <vscale x 2 x i64> %in) {
; CHECK-LABEL: sel_merge_64_shifted:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z0.d, p0/m, #512 // =0x200
; CHECK-NEXT:    ret
  %sel = select <vscale x 2 x i1> %p, <vscale x 2 x i64> splat (i64 512), <vscale x 2 x i64> %in
  ret <vscale x 2 x i64> %sel
}

; TODO: We could actually use something like "cpy z0.b, p0/m, #-128". But it's
; a little tricky to prove correctness: we're using the predicate with the
; wrong width, so we'd have to prove the bits which would normally be unused
; are actually zero.
define <vscale x 8 x i16> @sel_merge_16_illegal_wrong_extension(<vscale x 8 x i1> %p, <vscale x 8 x i16> %in) {
; CHECK-LABEL: sel_merge_16_illegal_wrong_extension:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z1.h, #128 // =0x80
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %p, <vscale x 8 x i16> splat (i16 128), <vscale x 8 x i16> %in
  ret <vscale x 8 x i16> %sel
}

define <vscale x 4 x i32> @sel_merge_32_illegal_wrong_extension(<vscale x 4 x i1> %p, <vscale x 4 x i32> %in) {
; CHECK-LABEL: sel_merge_32_illegal_wrong_extension:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z1.s, #128 // =0x80
; CHECK-NEXT:    mov z0.s, p0/m, z1.s
; CHECK-NEXT:    ret
  %sel = select <vscale x 4 x i1> %p, <vscale x 4 x i32> splat (i32 128), <vscale x 4 x i32> %in
  ret <vscale x 4 x i32> %sel
}

define <vscale x 2 x i64> @sel_merge_64_illegal_wrong_extension(<vscale x 2 x i1> %p, <vscale x 2 x i64> %in) {
; CHECK-LABEL: sel_merge_64_illegal_wrong_extension:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov z1.d, #128 // =0x80
; CHECK-NEXT:    mov z0.d, p0/m, z1.d
; CHECK-NEXT:    ret
  %sel = select <vscale x 2 x i1> %p, <vscale x 2 x i64> splat (i64 128), <vscale x 2 x i64> %in
  ret <vscale x 2 x i64> %sel
}

define <vscale x 8 x i16> @sel_merge_16_illegal_shifted(<vscale x 8 x i1> %p, <vscale x 8 x i16> %in) {
; CHECK-LABEL: sel_merge_16_illegal_shifted:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #513 // =0x201
; CHECK-NEXT:    mov z1.h, w8
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %p, <vscale x 8 x i16> splat (i16 513), <vscale x 8 x i16> %in
  ret <vscale x 8 x i16> %sel
}

define <vscale x 4 x i32> @sel_merge_32_illegal_shifted(<vscale x 4 x i1> %p, <vscale x 4 x i32> %in) {
; CHECK-LABEL: sel_merge_32_illegal_shifted:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #513 // =0x201
; CHECK-NEXT:    mov z1.s, w8
; CHECK-NEXT:    mov z0.s, p0/m, z1.s
; CHECK-NEXT:    ret
  %sel = select <vscale x 4 x i1> %p, <vscale x 4 x i32> splat (i32 513), <vscale x 4 x i32> %in
  ret <vscale x 4 x i32> %sel
}

define <vscale x 2 x i64> @sel_merge_64_illegal_shifted(<vscale x 2 x i1> %p, <vscale x 2 x i64> %in) {
; CHECK-LABEL: sel_merge_64_illegal_shifted:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #513 // =0x201
; CHECK-NEXT:    mov z1.d, x8
; CHECK-NEXT:    mov z0.d, p0/m, z1.d
; CHECK-NEXT:    ret
  %sel = select <vscale x 2 x i1> %p, <vscale x 2 x i64> splat (i64 513), <vscale x 2 x i64> %in
  ret <vscale x 2 x i64> %sel
}
