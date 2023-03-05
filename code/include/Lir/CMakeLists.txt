set(LLVM_TARGET_DEFINITIONS LirAttr.td)
mlir_tablegen(LirEnumAttr.h.inc -gen-enum-decls)
mlir_tablegen(LirEnumAttr.cpp.inc -gen-enum-defs)
mlir_tablegen(LirAttr.h.inc --gen-attrdef-decls)
mlir_tablegen(LirAttr.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS LirOps.td)
mlir_tablegen(LirOps.h.inc -gen-op-decls)
mlir_tablegen(LirOps.cpp.inc -gen-op-defs)
mlir_tablegen(LirTypes.h.inc --gen-typedef-decls)
mlir_tablegen(LirTypes.cpp.inc --gen-typedef-defs)
mlir_tablegen(LirDialect.h.inc -gen-dialect-decls --dialect=lir)
mlir_tablegen(LirDialect.cpp.inc -gen-dialect-defs --dialect=lir)
add_public_tablegen_target(LirOpsIncGen)


set(LLVM_TARGET_DEFINITIONS LirOpsInterfaces.td)
mlir_tablegen(LirOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(LirOpsInterfaces.cpp.inc -gen-op-interface-defs)
