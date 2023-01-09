add_library(parser
           Parser.cpp
           Attributes.cpp
           UseDeclaration.cpp
           Visibility.cpp
           Modules.cpp
           Item.cpp
           Function.cpp
           SimplePath.cpp
           Util.cpp
           Type.cpp
           Patterns.cpp
           Generics.cpp
           WhereClause.cpp
           FunctionParameters.cpp
           BlockExpression.cpp
           IdentifierPattern.cpp
           PrimitiveType.cpp
           Statement.cpp
           ExpressionWithoutBlock.cpp
           ExpressionWithBlock.cpp
           ReturnExpression.cpp
           ExpressionStatement.cpp
           Expression.cpp
           OperatorExpression.cpp
           ArithmeticOrLogicalExpression.cpp
           LiteralExpression.cpp
           PathExpression.cpp
           PathInExpression.cpp
           QualifiedPathInExpression.cpp
           FunctionParam.cpp
           PathExprSegment.cpp
           GenericArgs.cpp
           PathIdentSegment.cpp
           Statements.cpp
           OperatorFeeding.cpp
           PatternNoTopAlt.cpp
           LiteralPattern.cpp
           SelfParam.cpp
           ShorthandSelf.cpp
           TypedSelf.cpp
           )


target_include_directories(parser PRIVATE  ../include)


target_link_libraries(parser ast)

target_precompile_headers(parser PRIVATE Modules.h Visibility.h Attributes.h Item.h Function.h UseDeclaration.h SimplePath.h Util.h Type.h FunctionParameters.h WhereClause.h Generics.h Patterns.h BlockExpression.h WhereClause.h IdentifierPattern.h PrimitiveType.h Statement.h ExpressionWithoutBlock.h ReturnExpression.h ExpressionStatement.h ExpressionWithBlock.h Expression.h OperatorExpression.h ArithmeticOrLogicalExpression.h LiteralExpression.h PathExpression.h PathInExpression.h QualifiedPathInExpression.h FunctionParam.h PathExprSegment.h GenericArgs.h PathIdentSegment.h Statements.h OperatorFeeding.h PatternNoTopAlt.h LiteralPattern.h SelfParam.h ShorthandSelf.h TypedSelf.h)
