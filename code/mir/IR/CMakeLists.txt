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
  MirOps.cpp
  )

target_include_directories(MirDialect PRIVATE ../../include)

add_dependencies(MirDialect MirOpsIncGen ) #MirOpInterfacesIncGen

target_link_libraries(MirDialect MLIRLLVMDialect)
