add_library(sema
           Sema.cpp
           Module.cpp
           Item.cpp
           VisItem.cpp
           OuterAttributes.cpp
           Function.cpp
           BlockExpression.cpp
           ExhaustivenessCheck.cpp
           AttributeAnalyzer.cpp
           LetStatement.cpp
           ExpressionStatement.cpp
           MacroInvocation.cpp
           Reachability.cpp
           CallExpression.cpp
           MethodCallExpression.cpp
           InfiniteLoopExpression.cpp
           OperatorExpression.cpp
           BlockExpressionVisitor.cpp
           LiteralExpression.cpp
           Expression.cpp
           LoopExpression.cpp
           ItemVisitor.cpp
           ModuleTree.cpp
           NormalizeUseTree.cpp
           )

target_include_directories(sema PRIVATE  ../include)

add_subdirectory(Resolver)
add_subdirectory(TypeChecking)

target_link_libraries(sema Resolver TypeChecking)

