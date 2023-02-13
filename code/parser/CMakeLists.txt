add_library(parser
           Parser.cpp
           Util.cpp
           )


target_include_directories(parser PRIVATE  ../include)


target_link_libraries(parser ast lexer)

#target_precompile_headers(parser PRIVATE Modules.h Visibility.h Attributes.h Item.h Function.h UseDeclaration.h SimplePath.h Util.h Type.h FunctionParameters.h WhereClause.h Generics.h Patterns.h BlockExpression.h WhereClause.h IdentifierPattern.h PrimitiveType.h Statement.h ExpressionWithoutBlock.h ReturnExpression.h ExpressionStatement.h ExpressionWithBlock.h Expression.h OperatorExpression.h ArithmeticOrLogicalExpression.h LiteralExpression.h PathExpression.h PathInExpression.h QualifiedPathInExpression.h FunctionParam.h PathExprSegment.h GenericArgs.h PathIdentSegment.h Statements.h OperatorFeeding.h PatternNoTopAlt.h LiteralPattern.h SelfParam.h ShorthandSelf.h TypedSelf.h)
