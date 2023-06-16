#include "AST/AttributeParser.h"
#include "AST/StructField.h"
#include "AST/StructFields.h"
#include "AST/StructStruct.h"
#include "AST/TupleStruct.h"
#include "Sema/Sema.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

// FIXME repr, align and packed modifiers
/// [repr(packed(x))] #[repr(align(x))] #[repr(packed)]

void Sema::analyzeStructStruct(ast::StructStruct *str) {
  llvm::errs() << "found outer attributes on a struct : "
               << str->getOuterAttributes().size() << "\n";
  for (OuterAttribute &outer : str->getOuterAttributes()) {
    SimplePath path = outer.getAttr().getPath();

    outer.getAttr().parseMetaItem();
    if (path.getNrOfSegments() == 1)
      llvm::errs() << "x" << path.getSegment(0).asString() << "x"
                   << "\n";

    const std::vector<std::shared_ptr<MetaItemInner>> &inner =
        outer.getAttr().getInput().getMetaItems();
    llvm::errs() << "found meta item inners: " << inner.size() << "\n";
    for (const auto &item : inner) {
      if (isReprAttribute(path)) {
        if (item->isMetaItem()) {
          MetaItem *meta = static_cast<MetaItem *>(item.get());
          switch (meta->getKind()) {
          case MetaItemKind::MetaNameValueString: {
            assert(false);
          }
          case MetaItemKind::MetaListPaths: {
            assert(false);
          }
          case MetaItemKind::MetaListNameValueString: {
            assert(false);
          }
          case MetaItemKind::MetaWord: {
            MetaWord *word = static_cast<MetaWord *>(meta);
            llvm::errs() << "meta word: " << word->getIdentifier().toString()
                         << "\n";
            break;
          }
          case MetaItemKind::MetaItemSequence: {
            assert(false);
          }
          case MetaItemKind::MetaLiteralExpression: {
            assert(false);
          }
          case MetaItemKind::MetaItemPathLit: {
            assert(false);
          }
          case MetaItemKind::MetaItemPath: {
            assert(false);
          }
          }
        }
      }
    }
  }
  //  StructFields fields = str->getFields();
  //
  //  size_t size = 0;
  //  for (const StructField& field: fields.getFields()) {
  //    auto[alignment, size] =
  //    getAlignmentAndSizeOfType(field.getType().get());
  //  }
}

void Sema::analyzeTupleStruct(ast::TupleStruct *tuple) {}

} // namespace rust_compiler::sema
