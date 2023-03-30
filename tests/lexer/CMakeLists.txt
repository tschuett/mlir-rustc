include(GoogleTest)

add_executable(LexerTests
        LexerTests.cpp
        Lexer.cpp
        ArithmeticOrLogical.cpp
)

target_link_libraries(LexerTests lexer GTest::gtest GTest::gtest_main)
target_include_directories(LexerTests PUBLIC ../../code/include ${GTEST_INCLUDE_DIRS})

gtest_discover_tests(LexerTests)

#/usr/local/Cellar/googletest/1.12.1_1/include
