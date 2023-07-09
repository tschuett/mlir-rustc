#include "Mangler/Mangler.h"

#include "AST/Crate.h"
#include "AST/Function.h"
#include "AST/Implementation.h"
#include "AST/InherentImpl.h"
#include "AST/Module.h"
#include "AST/Struct.h"
#include "AST/StructStruct.h"
#include "AST/TraitImpl.h"
#include "AST/Types/ArrayType.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypePath.h"
#include "AST/VisItem.h"
#include "Lexer/Identifier.h"
#include "llvm/Support/ErrorHandling.h"

#include <llvm/Support/raw_ostream.h>
#include <sstream>
#include <string>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;

namespace rust_compiler::mangler {

// std::string
// Mangler::mangleFreestandingFunction(std::span<const ast::VisItem *> path,
//                                     ast::Crate *crate) {}
//
// std::string Mangler::mangleMethod(std::span<const ast::VisItem *> path,
//                                   ast::Crate *crate) {}
//
// std::string Mangler::mangleStatic(std::span<const ast::VisItem *> path,
//                                   ast::Crate *crate) {}
//
// std::string Mangler::mangleClosure(std::span<const ast::VisItem *> path,
//                                    ast::Crate *crate,
//                                    ast::ClosureExpression *closure) {}

std::string Mangler::mangle(std::span<const ast::VisItem *> path,
                            ast::Crate *crate) {
  std::vector<std::string> tags;
  std::string symbolPath;
  llvm::raw_string_ostream mangledName(symbolPath);

  // _R0 FIXME

  // tags

  // instantiating-crate
  mangledName << "C" << std::to_string(crate->getCrateName().size())
              << crate->getCrateName();

  for (const ast::VisItem *it : path) {
    switch (it->getKind()) {
    case VisItemKind::Module: {
      const ast::Module *module = static_cast<const Module *>(it);
      Identifier name = module->getModuleName();
      if (name.isASCII()) {
        std::string ascii = name.toString();
        mangledName << std::to_string(ascii.size()) << ascii;
      } else {
        // FIXME
      }
      tags.push_back("Nt");
      break;
    }
    case VisItemKind::ExternCrate: {
      break;
    }
    case VisItemKind::UseDeclaration: {
      llvm_unreachable("we don't mangle use declarations");
      break;
    }
    case VisItemKind::Function: {
      const Function *fun = static_cast<const Function *>(it);
      Identifier name = fun->getName();
      if (name.isASCII()) {
        std::string ascii = name.toString();
        mangledName << std::to_string(ascii.size()) << ascii;
      } else {
        // FIXME
      }
      tags.push_back("Nv");
      break;
    }
    case VisItemKind::TypeAlias: {
      break;
    }
    case VisItemKind::Struct: {
      const Struct *stru = static_cast<const Struct *>(it);
      switch (stru->getKind()) {
      case StructKind::StructStruct2: {
        const StructStruct *struc2 = static_cast<const StructStruct *>(stru);
        Identifier name = struc2->getIdentifier();
        if (name.isASCII()) {
          std::string ascii = name.toString();
          mangledName << std::to_string(ascii.size()) << ascii;
        } else {
        }
        tags.push_back("Nu");
        break;
      }
      case StructKind::TupleStruct2: {
        break;
      }
      }
      break;
    }
    case VisItemKind::Enumeration: {
      break;
    }
    case VisItemKind::Union: {
      break;
    }
    case VisItemKind::ConstantItem: {
      break;
    }
    case VisItemKind::StaticItem: {
      break;
    }
    case VisItemKind::Trait: {
      break;
    }
    case VisItemKind::Implementation: {
      const ast::Implementation *implementation =
          static_cast<const ast::Implementation *>(it);
      switch (implementation->getKind()) {
      case ImplementationKind::InherentImpl: {
        const ast::InherentImpl *impl =
            static_cast<const ast::InherentImpl *>(it);
        mangledName << mangleType(impl->getType().get());
        tags.push_back("M");
        break;
      }
      case ImplementationKind::TraitImpl: {
        const ast::TraitImpl *impl = static_cast<const ast::TraitImpl *>(it);
        mangledName << mangleType(impl->getType().get());
        mangledName << mangleType(impl->getTypePath().get());
        tags.push_back("X");
        break;
      }
      }
      break;
    }
    case VisItemKind::ExternBlock: {
      break;
    }
    }
  }

  std::reverse(tags.begin(), tags.end());

  // _R0 FIXME

  // tags

  std::string result;

  llvm::raw_string_ostream mangled(result);

  mangled << "_R0";
  for (const std::string& tag: tags)
    mangled << tag;

  mangled << mangledName.str();

  // vendor-specific-suffix
  mangled << "rust_compiler";

  return mangled.str();
}

std::string Mangler::mangleType(ast::types::TypeExpression *type) {
  switch (type->getKind()) {
  case TypeExpressionKind::TypeNoBounds: {
    TypeNoBounds *noBounds = static_cast<TypeNoBounds *>(type);
    switch (noBounds->getKind()) {
    case TypeNoBoundsKind::ParenthesizedType:
      break;
    case TypeNoBoundsKind::ImplTraitType:
      break;
    case TypeNoBoundsKind::ImplTraitTypeOneBound:
      break;
    case TypeNoBoundsKind::TraitObjectTypeOneBound:
      break;
    case TypeNoBoundsKind::TypePath: {
      TypePath *path = static_cast<TypePath *>(type);
      return mangleTypePath(path);
      break;
    }
    case TypeNoBoundsKind::TupleType: {
      break;
    }
    case TypeNoBoundsKind::NeverType: {
      break;
    }
    case TypeNoBoundsKind::RawPointerType: {
      break;
    }
    case TypeNoBoundsKind::ReferenceType: {
      break;
    }
    case TypeNoBoundsKind::ArrayType: {
      ArrayType *array = static_cast<ArrayType *>(type);
      std::string result;
      llvm::raw_string_ostream mangled(result);
      //mangled << "A" << mangleType(array->getType().get()) << 
      return mangled.str();
      break;
    }
    case TypeNoBoundsKind::SliceType: {
      llvm_unreachable("no slice types");
      break;
    }
    case TypeNoBoundsKind::InferredType:
      llvm_unreachable("no infered types");
      break;
    case TypeNoBoundsKind::QualifiedPathInType: {
      break;
    }
    case TypeNoBoundsKind::BareFunctionType: {
      break;
    }
    case TypeNoBoundsKind::MacroInvocation:
      break;
    }
    break;
  }
  case TypeExpressionKind::ImplTraitType: {
    break;
  }
  case TypeExpressionKind::TraitObjectType: {
    break;
  }
  }
}

std::string Mangler::mangleTypePath(ast::types::TypePath *) {}

} // namespace rust_compiler::mangler
