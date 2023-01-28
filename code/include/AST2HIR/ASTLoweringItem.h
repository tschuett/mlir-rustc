#pragma once

#include "AST/Module.h"

namespace rust_compiler::ast2hir {

class ASTLoweringItem : public ASTLoweringBase {
  using Rust::HIR::ASTLoweringBase::visit;

public:
  static HIR::Item *translate(AST::Item *item);

  void visit(AST::Module &module) override;
  void visit(AST::TypeAlias &alias) override;
  void visit(AST::TupleStruct &struct_decl) override;
  void visit(AST::StructStruct &struct_decl) override;
  void visit(AST::Enum &enum_decl) override;
  void visit(AST::Union &union_decl) override;
  void visit(AST::StaticItem &var) override;
  void visit(AST::ConstantItem &constant) override;
  void visit(AST::Function &function) override;
  void visit(AST::InherentImpl &impl_block) override;
  void visit(AST::Trait &trait) override;
  void visit(AST::TraitImpl &impl_block) override;
  void visit(AST::ExternBlock &extern_block) override;

private:
  ASTLoweringItem() : translated(nullptr) {}

  HIR::Item *translated;
};


} // namespace rust_compiler::ast2hir
