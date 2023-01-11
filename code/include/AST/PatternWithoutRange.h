#pragma once

//#include "AST/AST.h"
//#include "AST/PatternNoTopAlt.h"
//#include "Location.h"
//
//namespace rust_compiler::ast {
//
//enum class PatternWithoutRangeKind {
//  LiteralPattern,
//  IdentifierPattern,
//  WildcardPattern,
//  RestPattern,
//  ReferencePattern,
//  StructPattern,
//  TupleStructPatern,
//  TuplePattern,
//  GroupedPattern,
//  SlicePattern,
//  PathPattern,
//  MacroInvocation
//};
//
//class PatternWithoutRange : public PatternNoTopAlt {
//
//public:
//  PatternWithoutRange(Location loc, PatternWithoutRangeKind kind)
//      : PatternNoTopAlt(loc, PatternNoTopAltKind::PatternWithoutRange),
//        kind(kind){};
//
//protected:
//  PatternWithoutRangeKind kind;
//};
//
//} // namespace rust_compiler::ast
