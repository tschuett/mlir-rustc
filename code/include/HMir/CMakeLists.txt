set(LLVM_TARGET_DEFINITIONS HMirAttr.td)
mlir_tablegen(HMirEnumAttr.h.inc -gen-enum-decls)
mlir_tablegen(HMirEnumAttr.cpp.inc -gen-enum-defs)
mlir_tablegen(HMirAttr.h.inc --gen-attrdef-decls)
mlir_tablegen(HMirAttr.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS HMirOps.td)
mlir_tablegen(HMirOps.h.inc -gen-op-decls)
mlir_tablegen(HMirOps.cpp.inc -gen-op-defs)
mlir_tablegen(HMirTypes.h.inc --gen-typedef-decls)
mlir_tablegen(HMirTypes.cpp.inc --gen-typedef-defs)
add_public_tablegen_target(HMirOpsIncGen)


set(LLVM_TARGET_DEFINITIONS HMirOpsInterfaces.td)
mlir_tablegen(HMirOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(HMirOpsInterfaces.cpp.inc -gen-op-interface-defs)
