add_library(analysis
           MemorySSA.cpp
           AliasAnalysis.cpp
           Attributer.cpp
           DependencyGraph.cpp
           DependencyGraphNode.cpp
           AANoop.cpp
           MemorySSANodes.cpp
           Abstractelement.cpp
           Loops.cpp
           PointsTo.cpp
           Cycles.cpp
           )

target_include_directories(analysis PRIVATE  ../include)


target_link_libraries(analysis )
