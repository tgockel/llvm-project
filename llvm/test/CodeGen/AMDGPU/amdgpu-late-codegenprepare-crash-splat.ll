; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 5
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-late-codegenprepare %s  | FileCheck %s

; This crashed because the PHI with a splat was rejected, but then we marked the PHI
; as visited and tried to convert one of its user afterwards.

define amdgpu_kernel void @widget(ptr %arg, ptr %arg1, ptr %arg2) {
; CHECK-LABEL: define amdgpu_kernel void @widget(
; CHECK-SAME: ptr [[ARG:%.*]], ptr [[ARG1:%.*]], ptr [[ARG2:%.*]]) {
; CHECK-NEXT:  [[BB:.*]]:
; CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[ARG]], align 4
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 [[TMP0]] to i1
; CHECK-NEXT:    [[ARG1_LOAD:%.*]] = load <4 x i8>, ptr [[ARG1]], align 4
; CHECK-NEXT:    [[ARG2_LOAD:%.*]] = load i64, ptr [[ARG2]], align 4
; CHECK-NEXT:    br label %[[BB_1:.*]]
; CHECK:       [[BB_1]]:
; CHECK-NEXT:    [[PHI:%.*]] = phi ptr [ null, %[[BB]] ], [ [[ARG1]], %[[BB_6:.*]] ]
; CHECK-NEXT:    [[PHI4:%.*]] = phi <4 x i8> [ splat (i8 1), %[[BB]] ], [ [[PHI15:%.*]], %[[BB_6]] ]
; CHECK-NEXT:    br i1 [[TMP1]], label %[[BB_2:.*]], label %[[BB_6]]
; CHECK:       [[BB_2]]:
; CHECK-NEXT:    [[PHI7:%.*]] = phi <4 x i8> [ [[PHI13:%.*]], %[[BB_5:.*]] ], [ [[PHI4]], %[[BB_1]] ]
; CHECK-NEXT:    br i1 [[TMP1]], label %[[BB_4:.*]], label %[[BB_5]]
; CHECK:       [[BB_3:.*]]:
; CHECK-NEXT:    br i1 [[TMP1]], label %[[BB_4]], label %[[BB_EXIT:.*]]
; CHECK:       [[BB_4]]:
; CHECK-NEXT:    [[PHI11:%.*]] = phi <4 x i8> [ [[PHI7]], %[[BB_3]] ], [ zeroinitializer, %[[BB_2]] ]
; CHECK-NEXT:    store <4 x i8> [[PHI11]], ptr [[PHI]], align 1
; CHECK-NEXT:    br label %[[BB_5]]
; CHECK:       [[BB_5]]:
; CHECK-NEXT:    [[PHI13]] = phi <4 x i8> [ zeroinitializer, %[[BB_4]] ], [ [[PHI7]], %[[BB_2]] ]
; CHECK-NEXT:    br i1 [[TMP1]], label %[[BB_2]], label %[[BB_6]]
; CHECK:       [[BB_6]]:
; CHECK-NEXT:    [[PHI15]] = phi <4 x i8> [ [[ARG1_LOAD]], %[[BB_1]] ], [ zeroinitializer, %[[BB_5]] ]
; CHECK-NEXT:    br label %[[BB_1]]
; CHECK:       [[BB_EXIT]]:
; CHECK-NEXT:    ret void
;
bb:
  %ld = load i32, ptr %arg, align 4
  %ld.trunc = trunc i32 %ld to i1
  %arg1.load = load <4 x i8>, ptr %arg1, align 4
  %arg2.load = load i64, ptr %arg2, align 4
  br label %bb.1

bb.1:
  %phi = phi ptr [ null, %bb ], [ %arg1, %bb.6 ]
  %phi4 = phi <4 x i8> [ splat (i8 1), %bb ], [ %phi15, %bb.6 ]
  br i1 %ld.trunc, label %bb.2, label %bb.6

bb.2:
  %phi7 = phi <4 x i8> [ %phi13, %bb.5 ], [ %phi4, %bb.1 ]
  br i1 %ld.trunc, label %bb.4, label %bb.5

bb.3:
  br i1 %ld.trunc, label %bb.4, label %bb.exit

bb.4:
  %phi11 = phi <4 x i8> [ %phi7, %bb.3 ], [ zeroinitializer, %bb.2 ]
  store <4 x i8> %phi11, ptr %phi, align 1
  br label %bb.5

bb.5:
  %phi13 = phi <4 x i8> [ zeroinitializer, %bb.4 ], [ %phi7, %bb.2 ]
  br i1 %ld.trunc, label %bb.2, label %bb.6

bb.6:
  %phi15 = phi <4 x i8> [ %arg1.load, %bb.1 ], [ zeroinitializer, %bb.5 ]
  br label %bb.1

bb.exit:
  ret void
}
