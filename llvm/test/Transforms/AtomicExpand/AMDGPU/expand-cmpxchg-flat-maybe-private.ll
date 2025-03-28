; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 5
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=atomic-expand %s | FileCheck %s

define { i16, i1 } @cmpxchg_flat_agent_i16(ptr %ptr, i16 %val, i16 %swap) {
; CHECK-LABEL: define { i16, i1 } @cmpxchg_flat_agent_i16(
; CHECK-SAME: ptr [[PTR:%.*]], i16 [[VAL:%.*]], i16 [[SWAP:%.*]]) {
; CHECK-NEXT:    [[ALIGNEDADDR:%.*]] = call ptr @llvm.ptrmask.p0.i64(ptr [[PTR]], i64 -4)
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[PTR]] to i64
; CHECK-NEXT:    [[PTRLSB:%.*]] = and i64 [[TMP1]], 3
; CHECK-NEXT:    [[TMP2:%.*]] = shl i64 [[PTRLSB]], 3
; CHECK-NEXT:    [[SHIFTAMT:%.*]] = trunc i64 [[TMP2]] to i32
; CHECK-NEXT:    [[MASK:%.*]] = shl i32 65535, [[SHIFTAMT]]
; CHECK-NEXT:    [[INV_MASK:%.*]] = xor i32 [[MASK]], -1
; CHECK-NEXT:    [[TMP3:%.*]] = zext i16 [[SWAP]] to i32
; CHECK-NEXT:    [[TMP4:%.*]] = shl i32 [[TMP3]], [[SHIFTAMT]]
; CHECK-NEXT:    [[TMP5:%.*]] = zext i16 [[VAL]] to i32
; CHECK-NEXT:    [[TMP6:%.*]] = shl i32 [[TMP5]], [[SHIFTAMT]]
; CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr [[ALIGNEDADDR]], align 4
; CHECK-NEXT:    [[TMP8:%.*]] = and i32 [[TMP7]], [[INV_MASK]]
; CHECK-NEXT:    br label %[[PARTWORD_CMPXCHG_LOOP:.*]]
; CHECK:       [[PARTWORD_CMPXCHG_LOOP]]:
; CHECK-NEXT:    [[TMP9:%.*]] = phi i32 [ [[TMP8]], [[TMP0:%.*]] ], [ [[TMP15:%.*]], %[[PARTWORD_CMPXCHG_FAILURE:.*]] ]
; CHECK-NEXT:    [[TMP10:%.*]] = or i32 [[TMP9]], [[TMP4]]
; CHECK-NEXT:    [[TMP11:%.*]] = or i32 [[TMP9]], [[TMP6]]
; CHECK-NEXT:    [[TMP12:%.*]] = cmpxchg ptr [[ALIGNEDADDR]], i32 [[TMP11]], i32 [[TMP10]] syncscope("agent") monotonic seq_cst, align 4
; CHECK-NEXT:    [[TMP13:%.*]] = extractvalue { i32, i1 } [[TMP12]], 0
; CHECK-NEXT:    [[TMP14:%.*]] = extractvalue { i32, i1 } [[TMP12]], 1
; CHECK-NEXT:    br i1 [[TMP14]], label %[[PARTWORD_CMPXCHG_END:.*]], label %[[PARTWORD_CMPXCHG_FAILURE]]
; CHECK:       [[PARTWORD_CMPXCHG_FAILURE]]:
; CHECK-NEXT:    [[TMP15]] = and i32 [[TMP13]], [[INV_MASK]]
; CHECK-NEXT:    [[TMP16:%.*]] = icmp ne i32 [[TMP9]], [[TMP15]]
; CHECK-NEXT:    br i1 [[TMP16]], label %[[PARTWORD_CMPXCHG_LOOP]], label %[[PARTWORD_CMPXCHG_END]]
; CHECK:       [[PARTWORD_CMPXCHG_END]]:
; CHECK-NEXT:    [[SHIFTED:%.*]] = lshr i32 [[TMP13]], [[SHIFTAMT]]
; CHECK-NEXT:    [[EXTRACTED:%.*]] = trunc i32 [[SHIFTED]] to i16
; CHECK-NEXT:    [[TMP17:%.*]] = insertvalue { i16, i1 } poison, i16 [[EXTRACTED]], 0
; CHECK-NEXT:    [[TMP18:%.*]] = insertvalue { i16, i1 } [[TMP17]], i1 [[TMP14]], 1
; CHECK-NEXT:    ret { i16, i1 } [[TMP18]]
;
  %result = cmpxchg ptr %ptr, i16 %val, i16 %swap syncscope("agent") monotonic seq_cst
  ret { i16, i1 } %result
}

define { i16, i1 } @cmpxchg_flat_agent_i16_align4(ptr %ptr, i16 %val, i16 %swap) {
; CHECK-LABEL: define { i16, i1 } @cmpxchg_flat_agent_i16_align4(
; CHECK-SAME: ptr [[PTR:%.*]], i16 [[VAL:%.*]], i16 [[SWAP:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = zext i16 [[SWAP]] to i32
; CHECK-NEXT:    [[TMP2:%.*]] = zext i16 [[VAL]] to i32
; CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr [[PTR]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = and i32 [[TMP3]], -65536
; CHECK-NEXT:    br label %[[PARTWORD_CMPXCHG_LOOP:.*]]
; CHECK:       [[PARTWORD_CMPXCHG_LOOP]]:
; CHECK-NEXT:    [[TMP5:%.*]] = phi i32 [ [[TMP4]], [[TMP0:%.*]] ], [ [[TMP11:%.*]], %[[PARTWORD_CMPXCHG_FAILURE:.*]] ]
; CHECK-NEXT:    [[TMP6:%.*]] = or i32 [[TMP5]], [[TMP1]]
; CHECK-NEXT:    [[TMP7:%.*]] = or i32 [[TMP5]], [[TMP2]]
; CHECK-NEXT:    [[TMP8:%.*]] = cmpxchg ptr [[PTR]], i32 [[TMP7]], i32 [[TMP6]] syncscope("agent") monotonic seq_cst, align 4
; CHECK-NEXT:    [[TMP9:%.*]] = extractvalue { i32, i1 } [[TMP8]], 0
; CHECK-NEXT:    [[TMP10:%.*]] = extractvalue { i32, i1 } [[TMP8]], 1
; CHECK-NEXT:    br i1 [[TMP10]], label %[[PARTWORD_CMPXCHG_END:.*]], label %[[PARTWORD_CMPXCHG_FAILURE]]
; CHECK:       [[PARTWORD_CMPXCHG_FAILURE]]:
; CHECK-NEXT:    [[TMP11]] = and i32 [[TMP9]], -65536
; CHECK-NEXT:    [[TMP12:%.*]] = icmp ne i32 [[TMP5]], [[TMP11]]
; CHECK-NEXT:    br i1 [[TMP12]], label %[[PARTWORD_CMPXCHG_LOOP]], label %[[PARTWORD_CMPXCHG_END]]
; CHECK:       [[PARTWORD_CMPXCHG_END]]:
; CHECK-NEXT:    [[EXTRACTED:%.*]] = trunc i32 [[TMP9]] to i16
; CHECK-NEXT:    [[TMP13:%.*]] = insertvalue { i16, i1 } poison, i16 [[EXTRACTED]], 0
; CHECK-NEXT:    [[TMP14:%.*]] = insertvalue { i16, i1 } [[TMP13]], i1 [[TMP10]], 1
; CHECK-NEXT:    ret { i16, i1 } [[TMP14]]
;
  %result = cmpxchg ptr %ptr, i16 %val, i16 %swap syncscope("agent") monotonic seq_cst, align 4
  ret { i16, i1 } %result
}

define { i32, i1 } @cmpxchg_flat_agent_i32(ptr %ptr, i32 %val, i32 %swap) {
; CHECK-LABEL: define { i32, i1 } @cmpxchg_flat_agent_i32(
; CHECK-SAME: ptr [[PTR:%.*]], i32 [[VAL:%.*]], i32 [[SWAP:%.*]]) {
; CHECK-NEXT:    [[RESULT:%.*]] = cmpxchg ptr [[PTR]], i32 [[VAL]], i32 [[SWAP]] syncscope("agent") monotonic seq_cst, align 4
; CHECK-NEXT:    ret { i32, i1 } [[RESULT]]
;
  %result = cmpxchg ptr %ptr, i32 %val, i32 %swap syncscope("agent") monotonic seq_cst
  ret { i32, i1 } %result
}

define { i64, i1 } @cmpxchg_flat_agent_i64(ptr %ptr, i64 %val, i64 %swap) {
; CHECK-LABEL: define { i64, i1 } @cmpxchg_flat_agent_i64(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL:%.*]], i64 [[SWAP:%.*]]) {
; CHECK-NEXT:    [[IS_PRIVATE:%.*]] = call i1 @llvm.amdgcn.is.private(ptr [[PTR]])
; CHECK-NEXT:    br i1 [[IS_PRIVATE]], label %[[ATOMICRMW_PRIVATE:.*]], label %[[ATOMICRMW_GLOBAL:.*]]
; CHECK:       [[ATOMICRMW_PRIVATE]]:
; CHECK-NEXT:    [[TMP3:%.*]] = addrspacecast ptr [[PTR]] to ptr addrspace(5)
; CHECK-NEXT:    [[TMP4:%.*]] = load i64, ptr addrspace(5) [[TMP3]], align 8
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i64 [[TMP4]], [[VAL]]
; CHECK-NEXT:    [[TMP6:%.*]] = select i1 [[TMP5]], i64 [[SWAP]], i64 [[TMP4]]
; CHECK-NEXT:    store i64 [[TMP6]], ptr addrspace(5) [[TMP3]], align 8
; CHECK-NEXT:    [[TMP7:%.*]] = insertvalue { i64, i1 } poison, i64 [[TMP4]], 0
; CHECK-NEXT:    [[TMP8:%.*]] = insertvalue { i64, i1 } [[TMP7]], i1 [[TMP5]], 1
; CHECK-NEXT:    br label %[[ATOMICRMW_PHI:.*]]
; CHECK:       [[ATOMICRMW_GLOBAL]]:
; CHECK-NEXT:    [[TMP9:%.*]] = cmpxchg ptr [[PTR]], i64 [[VAL]], i64 [[SWAP]] syncscope("agent") monotonic seq_cst, align 8, !noalias.addrspace [[META0:![0-9]+]]
; CHECK-NEXT:    br label %[[ATOMICRMW_PHI]]
; CHECK:       [[ATOMICRMW_PHI]]:
; CHECK-NEXT:    [[RESULT:%.*]] = phi { i64, i1 } [ [[TMP8]], %[[ATOMICRMW_PRIVATE]] ], [ [[TMP9]], %[[ATOMICRMW_GLOBAL]] ]
; CHECK-NEXT:    br label %[[ATOMICRMW_END:.*]]
; CHECK:       [[ATOMICRMW_END]]:
; CHECK-NEXT:    ret { i64, i1 } [[RESULT]]
;
  %result = cmpxchg ptr %ptr, i64 %val, i64 %swap syncscope("agent") monotonic seq_cst
  ret { i64, i1 } %result
}

define { i64, i1 } @cmpxchg_flat_agent_i64_volatile(ptr %ptr, i64 %val, i64 %swap) {
; CHECK-LABEL: define { i64, i1 } @cmpxchg_flat_agent_i64_volatile(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL:%.*]], i64 [[SWAP:%.*]]) {
; CHECK-NEXT:    [[IS_PRIVATE:%.*]] = call i1 @llvm.amdgcn.is.private(ptr [[PTR]])
; CHECK-NEXT:    br i1 [[IS_PRIVATE]], label %[[ATOMICRMW_PRIVATE:.*]], label %[[ATOMICRMW_GLOBAL:.*]]
; CHECK:       [[ATOMICRMW_PRIVATE]]:
; CHECK-NEXT:    [[TMP3:%.*]] = addrspacecast ptr [[PTR]] to ptr addrspace(5)
; CHECK-NEXT:    [[TMP4:%.*]] = load i64, ptr addrspace(5) [[TMP3]], align 8
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i64 [[TMP4]], [[VAL]]
; CHECK-NEXT:    [[TMP6:%.*]] = select i1 [[TMP5]], i64 [[SWAP]], i64 [[TMP4]]
; CHECK-NEXT:    store i64 [[TMP6]], ptr addrspace(5) [[TMP3]], align 8
; CHECK-NEXT:    [[TMP7:%.*]] = insertvalue { i64, i1 } poison, i64 [[TMP4]], 0
; CHECK-NEXT:    [[TMP8:%.*]] = insertvalue { i64, i1 } [[TMP7]], i1 [[TMP5]], 1
; CHECK-NEXT:    br label %[[ATOMICRMW_PHI:.*]]
; CHECK:       [[ATOMICRMW_GLOBAL]]:
; CHECK-NEXT:    [[TMP9:%.*]] = cmpxchg volatile ptr [[PTR]], i64 [[VAL]], i64 [[SWAP]] syncscope("agent") monotonic seq_cst, align 8, !noalias.addrspace [[META0]]
; CHECK-NEXT:    br label %[[ATOMICRMW_PHI]]
; CHECK:       [[ATOMICRMW_PHI]]:
; CHECK-NEXT:    [[RESULT:%.*]] = phi { i64, i1 } [ [[TMP8]], %[[ATOMICRMW_PRIVATE]] ], [ [[TMP9]], %[[ATOMICRMW_GLOBAL]] ]
; CHECK-NEXT:    br label %[[ATOMICRMW_END:.*]]
; CHECK:       [[ATOMICRMW_END]]:
; CHECK-NEXT:    ret { i64, i1 } [[RESULT]]
;
  %result = cmpxchg volatile ptr %ptr, i64 %val, i64 %swap syncscope("agent") monotonic seq_cst
  ret { i64, i1 } %result
}

define { i16, i1 } @cmpxchg_flat_agent_i16__noprivate(ptr %ptr, i16 %val, i16 %swap) {
; CHECK-LABEL: define { i16, i1 } @cmpxchg_flat_agent_i16__noprivate(
; CHECK-SAME: ptr [[PTR:%.*]], i16 [[VAL:%.*]], i16 [[SWAP:%.*]]) {
; CHECK-NEXT:    [[ALIGNEDADDR:%.*]] = call ptr @llvm.ptrmask.p0.i64(ptr [[PTR]], i64 -4)
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[PTR]] to i64
; CHECK-NEXT:    [[PTRLSB:%.*]] = and i64 [[TMP1]], 3
; CHECK-NEXT:    [[TMP2:%.*]] = shl i64 [[PTRLSB]], 3
; CHECK-NEXT:    [[SHIFTAMT:%.*]] = trunc i64 [[TMP2]] to i32
; CHECK-NEXT:    [[MASK:%.*]] = shl i32 65535, [[SHIFTAMT]]
; CHECK-NEXT:    [[INV_MASK:%.*]] = xor i32 [[MASK]], -1
; CHECK-NEXT:    [[TMP3:%.*]] = zext i16 [[SWAP]] to i32
; CHECK-NEXT:    [[TMP4:%.*]] = shl i32 [[TMP3]], [[SHIFTAMT]]
; CHECK-NEXT:    [[TMP5:%.*]] = zext i16 [[VAL]] to i32
; CHECK-NEXT:    [[TMP6:%.*]] = shl i32 [[TMP5]], [[SHIFTAMT]]
; CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr [[ALIGNEDADDR]], align 4
; CHECK-NEXT:    [[TMP8:%.*]] = and i32 [[TMP7]], [[INV_MASK]]
; CHECK-NEXT:    br label %[[PARTWORD_CMPXCHG_LOOP:.*]]
; CHECK:       [[PARTWORD_CMPXCHG_LOOP]]:
; CHECK-NEXT:    [[TMP9:%.*]] = phi i32 [ [[TMP8]], [[TMP0:%.*]] ], [ [[TMP15:%.*]], %[[PARTWORD_CMPXCHG_FAILURE:.*]] ]
; CHECK-NEXT:    [[TMP10:%.*]] = or i32 [[TMP9]], [[TMP4]]
; CHECK-NEXT:    [[TMP11:%.*]] = or i32 [[TMP9]], [[TMP6]]
; CHECK-NEXT:    [[TMP12:%.*]] = cmpxchg ptr [[ALIGNEDADDR]], i32 [[TMP11]], i32 [[TMP10]] syncscope("agent") monotonic seq_cst, align 4
; CHECK-NEXT:    [[TMP13:%.*]] = extractvalue { i32, i1 } [[TMP12]], 0
; CHECK-NEXT:    [[TMP14:%.*]] = extractvalue { i32, i1 } [[TMP12]], 1
; CHECK-NEXT:    br i1 [[TMP14]], label %[[PARTWORD_CMPXCHG_END:.*]], label %[[PARTWORD_CMPXCHG_FAILURE]]
; CHECK:       [[PARTWORD_CMPXCHG_FAILURE]]:
; CHECK-NEXT:    [[TMP15]] = and i32 [[TMP13]], [[INV_MASK]]
; CHECK-NEXT:    [[TMP16:%.*]] = icmp ne i32 [[TMP9]], [[TMP15]]
; CHECK-NEXT:    br i1 [[TMP16]], label %[[PARTWORD_CMPXCHG_LOOP]], label %[[PARTWORD_CMPXCHG_END]]
; CHECK:       [[PARTWORD_CMPXCHG_END]]:
; CHECK-NEXT:    [[SHIFTED:%.*]] = lshr i32 [[TMP13]], [[SHIFTAMT]]
; CHECK-NEXT:    [[EXTRACTED:%.*]] = trunc i32 [[SHIFTED]] to i16
; CHECK-NEXT:    [[TMP17:%.*]] = insertvalue { i16, i1 } poison, i16 [[EXTRACTED]], 0
; CHECK-NEXT:    [[TMP18:%.*]] = insertvalue { i16, i1 } [[TMP17]], i1 [[TMP14]], 1
; CHECK-NEXT:    ret { i16, i1 } [[TMP18]]
;
  %result = cmpxchg ptr %ptr, i16 %val, i16 %swap syncscope("agent") monotonic seq_cst, !noalias.addrspace !0
  ret { i16, i1 } %result
}

define { i32, i1 } @cmpxchg_flat_agent_i32__noprivate(ptr %ptr, i32 %val, i32 %swap) {
; CHECK-LABEL: define { i32, i1 } @cmpxchg_flat_agent_i32__noprivate(
; CHECK-SAME: ptr [[PTR:%.*]], i32 [[VAL:%.*]], i32 [[SWAP:%.*]]) {
; CHECK-NEXT:    [[RESULT:%.*]] = cmpxchg ptr [[PTR]], i32 [[VAL]], i32 [[SWAP]] syncscope("agent") monotonic seq_cst, align 4, !noalias.addrspace [[META0]]
; CHECK-NEXT:    ret { i32, i1 } [[RESULT]]
;
  %result = cmpxchg ptr %ptr, i32 %val, i32 %swap syncscope("agent") monotonic seq_cst, !noalias.addrspace !0
  ret { i32, i1 } %result
}

define { i64, i1 } @cmpxchg_flat_agent_i64__noprivate(ptr %ptr, i64 %val, i64 %swap) {
; CHECK-LABEL: define { i64, i1 } @cmpxchg_flat_agent_i64__noprivate(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL:%.*]], i64 [[SWAP:%.*]]) {
; CHECK-NEXT:    [[RESULT:%.*]] = cmpxchg ptr [[PTR]], i64 [[VAL]], i64 [[SWAP]] syncscope("agent") monotonic seq_cst, align 8, !noalias.addrspace [[META0]]
; CHECK-NEXT:    ret { i64, i1 } [[RESULT]]
;
  %result = cmpxchg ptr %ptr, i64 %val, i64 %swap syncscope("agent") monotonic seq_cst, !noalias.addrspace !0
  ret { i64, i1 } %result
}

define { i64, i1 } @cmpxchg_flat_agent_i64__nolocal(ptr %ptr, i64 %val, i64 %swap) {
; CHECK-LABEL: define { i64, i1 } @cmpxchg_flat_agent_i64__nolocal(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL:%.*]], i64 [[SWAP:%.*]]) {
; CHECK-NEXT:    [[IS_PRIVATE:%.*]] = call i1 @llvm.amdgcn.is.private(ptr [[PTR]])
; CHECK-NEXT:    br i1 [[IS_PRIVATE]], label %[[ATOMICRMW_PRIVATE:.*]], label %[[ATOMICRMW_GLOBAL:.*]]
; CHECK:       [[ATOMICRMW_PRIVATE]]:
; CHECK-NEXT:    [[TMP1:%.*]] = addrspacecast ptr [[PTR]] to ptr addrspace(5)
; CHECK-NEXT:    [[TMP2:%.*]] = load i64, ptr addrspace(5) [[TMP1]], align 8
; CHECK-NEXT:    [[TMP3:%.*]] = icmp eq i64 [[TMP2]], [[VAL]]
; CHECK-NEXT:    [[TMP4:%.*]] = select i1 [[TMP3]], i64 [[SWAP]], i64 [[TMP2]]
; CHECK-NEXT:    store i64 [[TMP4]], ptr addrspace(5) [[TMP1]], align 8
; CHECK-NEXT:    [[TMP5:%.*]] = insertvalue { i64, i1 } poison, i64 [[TMP2]], 0
; CHECK-NEXT:    [[TMP6:%.*]] = insertvalue { i64, i1 } [[TMP5]], i1 [[TMP3]], 1
; CHECK-NEXT:    br label %[[ATOMICRMW_PHI:.*]]
; CHECK:       [[ATOMICRMW_GLOBAL]]:
; CHECK-NEXT:    [[TMP7:%.*]] = cmpxchg ptr [[PTR]], i64 [[VAL]], i64 [[SWAP]] syncscope("agent") monotonic seq_cst, align 8, !noalias.addrspace [[META0]]
; CHECK-NEXT:    br label %[[ATOMICRMW_PHI]]
; CHECK:       [[ATOMICRMW_PHI]]:
; CHECK-NEXT:    [[RESULT:%.*]] = phi { i64, i1 } [ [[TMP6]], %[[ATOMICRMW_PRIVATE]] ], [ [[TMP7]], %[[ATOMICRMW_GLOBAL]] ]
; CHECK-NEXT:    br label %[[ATOMICRMW_END:.*]]
; CHECK:       [[ATOMICRMW_END]]:
; CHECK-NEXT:    ret { i64, i1 } [[RESULT]]
;
  %result = cmpxchg ptr %ptr, i64 %val, i64 %swap syncscope("agent") monotonic seq_cst, !noalias.addrspace !1
  ret { i64, i1 } %result
}

define { i64, i1 } @cmpxchg_flat_agent_i64_mmra(ptr %ptr, i64 %val, i64 %swap) {
; CHECK-LABEL: define { i64, i1 } @cmpxchg_flat_agent_i64_mmra(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL:%.*]], i64 [[SWAP:%.*]]) {
; CHECK-NEXT:    [[IS_PRIVATE:%.*]] = call i1 @llvm.amdgcn.is.private(ptr [[PTR]])
; CHECK-NEXT:    br i1 [[IS_PRIVATE]], label %[[ATOMICRMW_PRIVATE:.*]], label %[[ATOMICRMW_GLOBAL:.*]]
; CHECK:       [[ATOMICRMW_PRIVATE]]:
; CHECK-NEXT:    [[TMP1:%.*]] = addrspacecast ptr [[PTR]] to ptr addrspace(5)
; CHECK-NEXT:    [[TMP2:%.*]] = load i64, ptr addrspace(5) [[TMP1]], align 8
; CHECK-NEXT:    [[TMP3:%.*]] = icmp eq i64 [[TMP2]], [[VAL]]
; CHECK-NEXT:    [[TMP4:%.*]] = select i1 [[TMP3]], i64 [[SWAP]], i64 [[TMP2]]
; CHECK-NEXT:    store i64 [[TMP4]], ptr addrspace(5) [[TMP1]], align 8
; CHECK-NEXT:    [[TMP5:%.*]] = insertvalue { i64, i1 } poison, i64 [[TMP2]], 0
; CHECK-NEXT:    [[TMP6:%.*]] = insertvalue { i64, i1 } [[TMP5]], i1 [[TMP3]], 1
; CHECK-NEXT:    br label %[[ATOMICRMW_PHI:.*]]
; CHECK:       [[ATOMICRMW_GLOBAL]]:
; CHECK-NEXT:    [[TMP7:%.*]] = cmpxchg ptr [[PTR]], i64 [[VAL]], i64 [[SWAP]] syncscope("agent") monotonic seq_cst, align 8, !mmra [[META1:![0-9]+]], !noalias.addrspace [[META0]]
; CHECK-NEXT:    br label %[[ATOMICRMW_PHI]]
; CHECK:       [[ATOMICRMW_PHI]]:
; CHECK-NEXT:    [[RESULT:%.*]] = phi { i64, i1 } [ [[TMP6]], %[[ATOMICRMW_PRIVATE]] ], [ [[TMP7]], %[[ATOMICRMW_GLOBAL]] ]
; CHECK-NEXT:    br label %[[ATOMICRMW_END:.*]]
; CHECK:       [[ATOMICRMW_END]]:
; CHECK-NEXT:    ret { i64, i1 } [[RESULT]]
;
  %result = cmpxchg ptr %ptr, i64 %val, i64 %swap syncscope("agent") monotonic seq_cst, !mmra !4
  ret { i64, i1 } %result
}

define { i64, i1 } @cmpxchg_flat_agent_i64_mmra_noprivate(ptr %ptr, i64 %val, i64 %swap) {
; CHECK-LABEL: define { i64, i1 } @cmpxchg_flat_agent_i64_mmra_noprivate(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL:%.*]], i64 [[SWAP:%.*]]) {
; CHECK-NEXT:    [[IS_PRIVATE:%.*]] = call i1 @llvm.amdgcn.is.private(ptr [[PTR]])
; CHECK-NEXT:    br i1 [[IS_PRIVATE]], label %[[ATOMICRMW_PRIVATE:.*]], label %[[ATOMICRMW_GLOBAL:.*]]
; CHECK:       [[ATOMICRMW_PRIVATE]]:
; CHECK-NEXT:    [[TMP1:%.*]] = addrspacecast ptr [[PTR]] to ptr addrspace(5)
; CHECK-NEXT:    [[TMP2:%.*]] = load i64, ptr addrspace(5) [[TMP1]], align 8
; CHECK-NEXT:    [[TMP3:%.*]] = icmp eq i64 [[TMP2]], [[VAL]]
; CHECK-NEXT:    [[TMP4:%.*]] = select i1 [[TMP3]], i64 [[SWAP]], i64 [[TMP2]]
; CHECK-NEXT:    store i64 [[TMP4]], ptr addrspace(5) [[TMP1]], align 8
; CHECK-NEXT:    [[TMP5:%.*]] = insertvalue { i64, i1 } poison, i64 [[TMP2]], 0
; CHECK-NEXT:    [[TMP6:%.*]] = insertvalue { i64, i1 } [[TMP5]], i1 [[TMP3]], 1
; CHECK-NEXT:    br label %[[ATOMICRMW_PHI:.*]]
; CHECK:       [[ATOMICRMW_GLOBAL]]:
; CHECK-NEXT:    [[TMP7:%.*]] = cmpxchg ptr [[PTR]], i64 [[VAL]], i64 [[SWAP]] syncscope("agent") monotonic seq_cst, align 8, !mmra [[META1]], !noalias.addrspace [[META0]]
; CHECK-NEXT:    br label %[[ATOMICRMW_PHI]]
; CHECK:       [[ATOMICRMW_PHI]]:
; CHECK-NEXT:    [[RESULT:%.*]] = phi { i64, i1 } [ [[TMP6]], %[[ATOMICRMW_PRIVATE]] ], [ [[TMP7]], %[[ATOMICRMW_GLOBAL]] ]
; CHECK-NEXT:    br label %[[ATOMICRMW_END:.*]]
; CHECK:       [[ATOMICRMW_END]]:
; CHECK-NEXT:    ret { i64, i1 } [[RESULT]]
;
  %result = cmpxchg ptr %ptr, i64 %val, i64 %swap syncscope("agent") monotonic seq_cst, !noalias.addrspace !1, !mmra !4
  ret { i64, i1 } %result
}

; may alias private, wrapped range
define { i64, i1 } @cmpxchg_flat_agent_i64__noalias_addrspace_edge_case0(ptr %ptr, i64 %val, i64 %swap) {
; CHECK-LABEL: define { i64, i1 } @cmpxchg_flat_agent_i64__noalias_addrspace_edge_case0(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL:%.*]], i64 [[SWAP:%.*]]) {
; CHECK-NEXT:    [[IS_PRIVATE:%.*]] = call i1 @llvm.amdgcn.is.private(ptr [[PTR]])
; CHECK-NEXT:    br i1 [[IS_PRIVATE]], label %[[ATOMICRMW_PRIVATE:.*]], label %[[ATOMICRMW_GLOBAL:.*]]
; CHECK:       [[ATOMICRMW_PRIVATE]]:
; CHECK-NEXT:    [[TMP1:%.*]] = addrspacecast ptr [[PTR]] to ptr addrspace(5)
; CHECK-NEXT:    [[TMP2:%.*]] = load i64, ptr addrspace(5) [[TMP1]], align 8
; CHECK-NEXT:    [[TMP3:%.*]] = icmp eq i64 [[TMP2]], [[VAL]]
; CHECK-NEXT:    [[TMP4:%.*]] = select i1 [[TMP3]], i64 [[SWAP]], i64 [[TMP2]]
; CHECK-NEXT:    store i64 [[TMP4]], ptr addrspace(5) [[TMP1]], align 8
; CHECK-NEXT:    [[TMP5:%.*]] = insertvalue { i64, i1 } poison, i64 [[TMP2]], 0
; CHECK-NEXT:    [[TMP6:%.*]] = insertvalue { i64, i1 } [[TMP5]], i1 [[TMP3]], 1
; CHECK-NEXT:    br label %[[ATOMICRMW_PHI:.*]]
; CHECK:       [[ATOMICRMW_GLOBAL]]:
; CHECK-NEXT:    [[TMP7:%.*]] = cmpxchg ptr [[PTR]], i64 [[VAL]], i64 [[SWAP]] syncscope("agent") monotonic seq_cst, align 8, !noalias.addrspace [[META0]]
; CHECK-NEXT:    br label %[[ATOMICRMW_PHI]]
; CHECK:       [[ATOMICRMW_PHI]]:
; CHECK-NEXT:    [[RESULT:%.*]] = phi { i64, i1 } [ [[TMP6]], %[[ATOMICRMW_PRIVATE]] ], [ [[TMP7]], %[[ATOMICRMW_GLOBAL]] ]
; CHECK-NEXT:    br label %[[ATOMICRMW_END:.*]]
; CHECK:       [[ATOMICRMW_END]]:
; CHECK-NEXT:    ret { i64, i1 } [[RESULT]]
;
  %result = cmpxchg ptr %ptr, i64 %val, i64 %swap syncscope("agent") monotonic seq_cst, !noalias.addrspace !6
  ret { i64, i1 } %result
}

; covers private case, but private isn't the low value.
define { i64, i1 } @cmpxchg_flat_agent_i64__no_2_6(ptr %ptr, i64 %val, i64 %swap) {
; CHECK-LABEL: define { i64, i1 } @cmpxchg_flat_agent_i64__no_2_6(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL:%.*]], i64 [[SWAP:%.*]]) {
; CHECK-NEXT:    [[IS_PRIVATE:%.*]] = call i1 @llvm.amdgcn.is.private(ptr [[PTR]])
; CHECK-NEXT:    br i1 [[IS_PRIVATE]], label %[[ATOMICRMW_PRIVATE:.*]], label %[[ATOMICRMW_GLOBAL:.*]]
; CHECK:       [[ATOMICRMW_PRIVATE]]:
; CHECK-NEXT:    [[TMP1:%.*]] = addrspacecast ptr [[PTR]] to ptr addrspace(5)
; CHECK-NEXT:    [[TMP2:%.*]] = load i64, ptr addrspace(5) [[TMP1]], align 8
; CHECK-NEXT:    [[TMP3:%.*]] = icmp eq i64 [[TMP2]], [[VAL]]
; CHECK-NEXT:    [[TMP4:%.*]] = select i1 [[TMP3]], i64 [[SWAP]], i64 [[TMP2]]
; CHECK-NEXT:    store i64 [[TMP4]], ptr addrspace(5) [[TMP1]], align 8
; CHECK-NEXT:    [[TMP5:%.*]] = insertvalue { i64, i1 } poison, i64 [[TMP2]], 0
; CHECK-NEXT:    [[TMP6:%.*]] = insertvalue { i64, i1 } [[TMP5]], i1 [[TMP3]], 1
; CHECK-NEXT:    br label %[[ATOMICRMW_PHI:.*]]
; CHECK:       [[ATOMICRMW_GLOBAL]]:
; CHECK-NEXT:    [[TMP7:%.*]] = cmpxchg ptr [[PTR]], i64 [[VAL]], i64 [[SWAP]] syncscope("agent") monotonic seq_cst, align 8, !noalias.addrspace [[META0]]
; CHECK-NEXT:    br label %[[ATOMICRMW_PHI]]
; CHECK:       [[ATOMICRMW_PHI]]:
; CHECK-NEXT:    [[RESULT:%.*]] = phi { i64, i1 } [ [[TMP6]], %[[ATOMICRMW_PRIVATE]] ], [ [[TMP7]], %[[ATOMICRMW_GLOBAL]] ]
; CHECK-NEXT:    br label %[[ATOMICRMW_END:.*]]
; CHECK:       [[ATOMICRMW_END]]:
; CHECK-NEXT:    ret { i64, i1 } [[RESULT]]
;
  %result = cmpxchg ptr %ptr, i64 %val, i64 %swap syncscope("agent") monotonic seq_cst, !noalias.addrspace !7
  ret { i64, i1 } %result
}

define { i64, i1 } @cmpxchg_flat_agent_i64__no_2_3_5(ptr %ptr, i64 %val, i64 %swap) {
; CHECK-LABEL: define { i64, i1 } @cmpxchg_flat_agent_i64__no_2_3_5(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL:%.*]], i64 [[SWAP:%.*]]) {
; CHECK-NEXT:    [[RESULT:%.*]] = cmpxchg ptr [[PTR]], i64 [[VAL]], i64 [[SWAP]] syncscope("agent") monotonic seq_cst, align 8, !noalias.addrspace [[META4:![0-9]+]]
; CHECK-NEXT:    ret { i64, i1 } [[RESULT]]
;
  %result = cmpxchg ptr %ptr, i64 %val, i64 %swap syncscope("agent") monotonic seq_cst, !noalias.addrspace !8
  ret { i64, i1 } %result
}

!0 = !{i32 5, i32 6}
!1 = !{i32 3, i32 4}
!2 = !{!"foo", !"bar"}
!3 = !{!"bux", !"baz"}
!4 = !{!2, !3}
!5 = !{}
!6 = !{i32 6, i32 5}
!7 = !{i32 2, i32 6}
!8 = !{i32 2, i32 4, i32 5, i32 6}

;.
; CHECK: [[META0]] = !{i32 5, i32 6}
; CHECK: [[META1]] = !{[[META2:![0-9]+]], [[META3:![0-9]+]]}
; CHECK: [[META2]] = !{!"foo", !"bar"}
; CHECK: [[META3]] = !{!"bux", !"baz"}
; CHECK: [[META4]] = !{i32 2, i32 4, i32 5, i32 6}
;.
