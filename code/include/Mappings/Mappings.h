#pragma once

#include "Basic/Ids.h"
#include "AST/Crate.h"

#include <map>
#include <string>
#include <string_view>

namespace rust_compiler::mappings {

class Mappings {
public:
  static Mappings *get();
  ~Mappings();

  basic::CrateNum getCrateNum(std::string_view name);
  basic::CrateNum getCurrentCrate();

  basic::NodeId getNextNodeId();

  ast::Crate &insertAstCrate(std::shared_ptr<ast::Crate> crate,
                             basic::CrateNum crateNum);

private:
  Mappings();

  basic::CrateNum currentCrateNum;

  std::map<basic::CrateNum, std::string> crateNames;
};

} // namespace rust_compiler::mappings
