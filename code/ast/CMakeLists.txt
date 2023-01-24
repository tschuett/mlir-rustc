add_library(ast
           OuterAttribute.cpp
           Module.cpp
           InnerAttribute.cpp
           UseDeclaration.cpp
           ClippyAttribute.cpp
           FunctionSignature.cpp
           Function.cpp
           BlockExpression.cpp
           PrimitiveTypes.cpp
           Visibility.cpp
           SimplePath.cpp
           SimplePathSegment.cpp
           UseTree.cpp
           WhereClause.cpp
           FunctionQualifiers.cpp
           FunctionParameters.cpp
           ReturnExpression.cpp
           ExpressionStatement.cpp
           LiteralExpression.cpp
           GenericArgs.cpp
           PathInExpression.cpp
           PathExprSegment.cpp
           PathExpression.cpp
           Statements.cpp
           FunctionParam.cpp
           IdentifierPattern.cpp
           ShorthandSelf.cpp
           SelfParam.cpp
           TypedSelf.cpp
           NegationExpression.cpp
           VariableDeclaration.cpp
           BorrowExpression.cpp
           ArithmeticOrLogicalExpression.cpp
           Item.cpp
           OuterAttributes.cpp
           IfExpression.cpp
           IfLetExpression.cpp
           Pattern.cpp
           RestPattern.cpp
           TuplePattern.cpp
           IfExpression.cpp
           LetStatement.cpp
           LetStatementParam.cpp
           PredicateLoopExpression.cpp
           AssignmentExpression.cpp
           ComparisonExpression.cpp
           )


target_include_directories(ast PRIVATE  ../include)


target_link_libraries(ast )
