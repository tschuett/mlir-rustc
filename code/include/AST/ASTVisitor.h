#pragma once

#include "AST/Abi.h"
#include "AST/AsClause.h"
#include "AST/CrateRef.h"
#include "AST/Decls.h"
#include "AST/Enumeration.h"
#include "AST/Function.h"
#include "AST/FunctionParam.h"
#include "AST/FunctionParamPattern.h"
#include "AST/FunctionParameters.h"
#include "AST/FunctionQualifiers.h"
#include "AST/FunctionReturnType.h"
#include "AST/Implementation.h"
#include "AST/MacroItem.h"
#include "AST/Module.h"
#include "AST/OuterAttribute.h"
#include "AST/SelfParam.h"
#include "AST/ShorthandSelf.h"
#include "AST/Struct.h"
#include "AST/TypedSelf.h"
#include "AST/UseDeclaration.h"
#include "AST/VisItem.h"

namespace rust_compiler::ast {

class ASTVisitor {
public:
  virtual ~ASTVisitor() = default;

  virtual void visit(OuterAttribute &outer) = 0;
  virtual void visit(VisItem &visItem) = 0;
  virtual void visit(MacroItem &macroItem) = 0;

  // VisItem
  virtual void visit(Module &module) = 0;
  virtual void visit(ExternCrate &externCrate) = 0;
  virtual void visit(UseDeclaration &useDeclaration) = 0;
  virtual void visit(Function &function) = 0;
  virtual void visit(TypeAlias &typeAlias) = 0;
  virtual void visit(Struct &structItem) = 0;
  virtual void visit(Enumeration &enumItem) = 0;
  virtual void visit(Union &unionItem) = 0;
  virtual void visit(ConstantItem &constantItem) = 0;
  virtual void visit(StaticItem &staticItem) = 0;
  virtual void visit(Trait &trait) = 0;
  virtual void visit(Implementation &implementation) = 0;
  virtual void visit(ExternBlock &externBlock) = 0;

  // Extern Crate
  virtual void visit(CrateRef &crateRef) = 0;
  virtual void visit(AsClause &asClause) = 0;

  // UseDeclaration
  virtual void visit(use_tree::UseTree &useTree) = 0;

  // Function
  virtual void visit(FunctionQualifiers &qualifiers) = 0;
  virtual void visit(Abi &abi) = 0;
  virtual void visit(FunctionParameters &parameters) = 0;
  virtual void visit(SelfParam &selfParam) = 0;
  virtual void visit(ShorthandSelf &shortHandSelf) = 0;
  virtual void visit(TypedSelf &typedSelf) = 0;
  virtual void visit(FunctionParam &functionParam) = 0;
  virtual void visit(FunctionParamPattern &functionParamPattern) = 0;
  virtual void visit(FunctionReturnType &returnType) = 0;

  // Struct
  virtual void visit(class StructStruct &structStruct) = 0;
  virtual void visit(class TupleStruct &tupleStruct) = 0;
  virtual void visit(StructFields &structFields) = 0;
  virtual void visit(StructField &structField) = 0;
  virtual void visit(TupleFields &tupleFields) = 0;
  virtual void visit(TupleField &tupleField) = 0;

  // Enumeration
  virtual void visit(EnumItems &enumItems) = 0;
  virtual void visit(EnumItem &enumItem) = 0;
  virtual void visit(EnumItemTuple &enumItemTuple) = 0;
  virtual void visit(EnumItemStruct &enumItemStruct) = 0;
  virtual void visit(EnumItemDiscriminant &enumItemDiscriminant) = 0;
};

} // namespace rust_compiler::ast
