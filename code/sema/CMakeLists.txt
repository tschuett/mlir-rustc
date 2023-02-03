add_library(sema
           Sema.cpp
           Module.cpp
           Item.cpp
           VisItem.cpp
           OuterAttributes.cpp
           Function.cpp
           TypeChecking.cpp
           Mappings.cpp
           BlockExpression.cpp
           ExhaustivenessCheck.cpp
           AttributeAnalyzer.cpp
           LetStatement.cpp
           ExpressionStatement.cpp
           MacroInvocation.cpp
           Reachability.cpp
           )

target_include_directories(sema PRIVATE  ../include)


target_link_libraries(sema )
