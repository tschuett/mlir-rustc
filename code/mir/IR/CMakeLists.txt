#add_mlir_dialect_library(Mir
#  MirDialect.cpp
#  DEPENDS
##  MirOpsIncGen
##  MirTransformsIncGen
#
#  LINK_COMPONENTS
#  Core
#
#  LINK_LIBS PUBLIC
##  MLIRBar
##  <some-other-library>
#  )


add_library(MirDialect
  MirDialect.cpp
  )

target_include_directories(MirDialect PRIVATE ../../include)
