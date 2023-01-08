include(GoogleTest)

add_executable(LexerTests
        LexerTests.cpp
        Lexer.cpp
        ArithmeticOrLogical.cpp
)

message(STATUS "Using GTest in: ${GTest_INCLUDE_DIRS}")


target_link_libraries(LexerTests lexer ${GTest_LDFLAGS})
target_include_directories(LexerTests PUBLIC ../../code/include ${GTest_INCLUDE_DIRS})

gtest_discover_tests(LexerTests)

