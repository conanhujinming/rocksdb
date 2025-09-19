// Copyright (c) 2024, Your Name. All rights reserved.
//
// This source code is licensed under both the GPLv2 (found in the
// COPYING file in the root directory) and Apache 2.0 License
// (found in the LICENSE.Apache file in the root directory).

#pragma once

#include "rocksdb/memtablerep.h"
#include "columnar_memtable.h"
#include "memory/arena.h"

namespace ROCKSDB_NAMESPACE {

class ColumnarRepFactory : public MemTableRepFactory {
 public:
  explicit ColumnarRepFactory(size_t active_block_size_bytes = 16 * 1024 * 48,
                              size_t num_shards = 16,
                              bool enable_compaction = false);

  ~ColumnarRepFactory() override {}

  using MemTableRepFactory::CreateMemTableRep;

  MemTableRep* CreateMemTableRep(const MemTableRep::KeyComparator&, Allocator*,
                                 const SliceTransform*, Logger* logger) override;

  const char* Name() const override { return "ColumnarRepFactory"; }

  bool IsInsertConcurrentlySupported() const override { return true; }

 private:
  size_t active_block_size_bytes_;
  size_t num_shards_;
  bool enable_compaction_;
};

class ColumnarRep : public MemTableRep {
 public:
  ColumnarRep(std::shared_ptr<ColumnarMemTable> table, Allocator* allocator);

  ~ColumnarRep() override;

  KeyHandle Allocate(const size_t len, char** buf) override;

  void Insert(KeyHandle handle) override;

  void InsertConcurrently(KeyHandle handle) override;

  bool Contains(const char* key) const override;

  void Get(const LookupKey& k, void* callback_args,
           bool (*callback_func)(void* arg, const char* entry)) override;

  size_t ApproximateMemoryUsage() override;

  Iterator* GetIterator(Arena* arena = nullptr) override;
  
  bool SupportsBatchInsert() const override { return true; }
  void InsertBatch(const std::vector<std::pair<Slice, Slice>>& batch) override;

 private:
  // The actual memtable implementation
  std::shared_ptr<ColumnarMemTable> table_;

  // A simple arena for Get() calls to reconstruct entries
  mutable Arena get_arena_;
  mutable std::mutex get_arena_mutex_;

  void DecodeAndInsert(KeyHandle handle);
};

}  // namespace ROCKSDB_NAMESPACE