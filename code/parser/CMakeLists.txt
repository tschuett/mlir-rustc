add_library(parser
           Parser.cpp
           UseTree.cpp
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
           Item.cpp
           UseDeclaration.cpp
           Trait.cpp
           MatchExpression.cpp
           Types.cpp
           ArithmeticOrLogicalExpression.cpp
           MacroRules.cpp
           DelimTokenTree.cpp
           ParseItem.cpp
           VisItem.cpp
           )


target_include_directories(parser PRIVATE  ../include)


target_link_libraries(parser ast lexer)

