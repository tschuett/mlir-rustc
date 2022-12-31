#add_mlir_dialect(MirOps mir)
#add_mlir_doc(Mir Mir Dialects/ -gen-dialect-doc)


set(LLVM_TARGET_DEFINITIONS Mir.td)

mlir_tablegen(MirOps.h.inc -gen-op-decls)
mlir_tablegen(MirOps.cpp.inc -gen-op-defs)
mlir_tablegen(MirDialect.h.inc -gen-dialect-decls)
mlir_tablegen(MirDialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(MirOpsIncGen)

#mlir_tablegen(MirOps.h.inc -gen-op-decls)
#mlir_tablegen(MirOps.cpp.inc -gen-op-defs)


set(LLVM_TARGET_DEFINITIONS MirOpsInterfaces.td)
mlir_tablegen(MirOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(MirOpsInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MirOpInterfacesIncGen)
