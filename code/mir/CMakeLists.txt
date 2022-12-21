add_mlir_dialect(MirOps mir)
add_mlir_doc(MirOps MirDialect Dialects/ -gen-dialect-doc)


add_mlir_dialect_library(Mir
  DEPENDS
  MirOpsIncGen
  MirTransformsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRBar
  <some-other-library>
  )
