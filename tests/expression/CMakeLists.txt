include(GoogleTest)

add_executable(ExpressionTests
        ExpressionTests.cpp
        Expression.cpp
        ReturnExpressions.cpp
        PathExpression.cpp
        FunctionParameter.cpp
        OperatorExpression.cpp
        ArithmeticOrLogicalExpression.cpp
)

llvm_map_components_to_libnames(llvm_libs Support)

target_link_libraries(ExpressionTests lexer parser ${llvm_libs} ${GTest_LDFLAGS})

target_include_directories(ExpressionTests PUBLIC ../../code/parser ../../code/include ${GTest_INCLUDE_DIRS})

gtest_discover_tests(ExpressionTests)
