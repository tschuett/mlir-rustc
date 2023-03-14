add_library(parser
           Parser.cpp
           UseTree.cpp
           Util.cpp
           Function.cpp
           Attributes.cpp
           PatternNoTopAlt.cpp
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
           VisItem.cpp
           CheckExpressionWithoutBlock.cpp
           ComplexType.cpp
#           ExpressionExceptStruct.cpp
           ExpressionUnaryOrBinary.cpp
           Precedence.cpp
           ParseExpression.cpp
           ErrorStack.cpp
           Expression.cpp
           ParseItem.cpp
           FunctionStack.cpp
           ExpressionWithoutBlock.cpp
           StructPattern.cpp
           Range.cpp
           )


target_include_directories(parser PRIVATE  ../include)


target_link_libraries(parser ast lexer)


target_precompile_headers(parser PRIVATE
                                 ../include/Parser/Parser.h
                                 ../include/Parser/ErrorStack.h
                                 ../include/Parser/Precedence.h
                                 ../include/Parser/Restrictions.h
                                 ../include/Lexer/Token.h
                                 ../include/Lexer/KeyWords.h
                                 ../include/AST/AST.h
                                 ../include/AST/ASTVisitor.h
                                 ../include/AST/Expression.h
)
