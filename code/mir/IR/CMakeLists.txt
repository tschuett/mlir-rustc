add_mlir_dialect_library(Mir
  MirDialect.cpp
  DEPENDS
#  MirOpsIncGen
#  MirTransformsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
#  MLIRBar
#  <some-other-library>
  )

target_include_directories(Mir PRIVATE ../../include)
