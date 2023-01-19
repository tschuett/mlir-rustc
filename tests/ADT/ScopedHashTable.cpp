#include "ADT/ScopedHashTable.h"

#include "Util.h"
#include "gtest/gtest.h"

using namespace rust_compiler::adt;

TEST(ScopedHashTableTest, CheckTrivial) {

  [[maybe_unused]] ScopedHashTable<std::string, std::string> table;
};

TEST(ScopedHashTableTest, CheckTrivialScope) {

  ScopedHashTable<std::string, std::string> table;

  { ScopedHashTableScope<std::string, std::string> scope{table}; }
};

TEST(ScopedHashTableTest, CheckInsert) {

  ScopedHashTable<std::string, std::string> table;

  {
    ScopedHashTableScope<std::string, std::string> scope{table};
    table.insert("foo", "bar");
  }
};

TEST(ScopedHashTableTest, CheckSimple) {

  ScopedHashTable<std::string, std::string> table;

  {
    ScopedHashTableScope<std::string, std::string> scope{table};
    table.insert("foo", "bar");
  }
};

TEST(ScopedHashTableTest, CheckContains) {

  ScopedHashTable<std::string, std::string> table;

  {
    ScopedHashTableScope<std::string, std::string> scope{table};
    EXPECT_FALSE(table.contains("foo"));
  }
};

TEST(ScopedHashTableTest, CheckInsertContains) {

  ScopedHashTable<std::string, std::string> table;

  {
    ScopedHashTableScope<std::string, std::string> scope{table};
    table.insert("foo", "bar");
    EXPECT_TRUE(table.contains("foo"));
  }

  // EXPECT_FALSE(table.contains("foo"));
};

TEST(ScopedHashTableTest, CheckSimpleWithScope) {

  ScopedHashTable<std::string, std::string> table;

  {
    ScopedHashTableScope<std::string, std::string> scope{table};
    table.insert("foo", "bar");

    EXPECT_TRUE(table.contains("foo"));
    {
      ScopedHashTableScope<std::string, std::string> scope2{table};
      table.insert("foo2", "bar");

      EXPECT_TRUE(table.contains("foo2"));
      EXPECT_TRUE(table.contains("foo"));
    }
    EXPECT_FALSE(table.contains("foo2"));
  }
};
