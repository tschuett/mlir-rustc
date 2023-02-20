add_library(parser
           Parser.cpp
           Util.cpp
           Function.cpp
           Attributes.cpp
           PatternNoTopAlt.cpp
           Expression.cpp
           Enumeration.cpp
           Closure.cpp
           Statement.cpp
           Struct.cpp
           Pattern.cpp
           BareFunctionType.cpp
           Path.cpp
           Pointer.cpp
           QualifiedPathType.cpp
           RawPointerType.cpp
           Tuple.cpp
           TypeNoBounds.cpp
           Statements.cpp
           Loops.cpp
           Types.cpp
           Item.cpp
           MatchExpression.cpp
           UseDeclaration.cpp
           Trait.cpp
           )


target_include_directories(parser PRIVATE  ../include)


target_link_libraries(parser ast lexer)

