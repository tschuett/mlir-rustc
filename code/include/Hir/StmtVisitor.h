#pragma once

namespace rust_compiler::hir {

class StmtVisitor {
public:
  virtual ~StmtVisitor() = default;

  virtual void visit(EnumItemTuple &) = 0;
  virtual void visit(EnumItemStruct &) = 0;
  virtual void visit(EnumItem &item) = 0;
  virtual void visit(TupleStruct &tuple_struct) = 0;
  virtual void visit(EnumItemDiscriminant &) = 0;
  virtual void visit(TypePathSegmentFunction &segment) = 0;
  virtual void visit(TypePath &path) = 0;
  virtual void visit(QualifiedPathInType &path) = 0;
  virtual void visit(Module &module) = 0;
  virtual void visit(ExternCrate &crate) = 0;
  virtual void visit(UseDeclaration &use_decl) = 0;
  virtual void visit(Function &function) = 0;
  virtual void visit(TypeAlias &type_alias) = 0;
  virtual void visit(StructStruct &struct_item) = 0;
  virtual void visit(Enum &enum_item) = 0;
  virtual void visit(Union &union_item) = 0;
  virtual void visit(ConstantItem &const_item) = 0;
  virtual void visit(StaticItem &static_item) = 0;
  virtual void visit(Trait &trait) = 0;
  virtual void visit(ImplBlock &impl) = 0;
  virtual void visit(ExternBlock &block) = 0;
  virtual void visit(EmptyStmt &stmt) = 0;
  virtual void visit(LetStmt &stmt) = 0;
  virtual void visit(ExprStmtWithoutBlock &stmt) = 0;
  virtual void visit(ExprStmtWithBlock &stmt) = 0;
};

} // namespace rust_compiler::hir
