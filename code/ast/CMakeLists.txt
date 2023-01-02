add_library(ast
           OuterAttribute.cpp
           Module.cpp
           InnerAttribute.cpp
           UseDeclaration.cpp
           ClippyAttribute.cpp
           FunctionSignature.cpp
           Function.cpp
           BlockExpression.cpp
           PrimitiveTypes.cpp
           )


target_include_directories(ast PRIVATE  ../include)


target_link_libraries(ast )
