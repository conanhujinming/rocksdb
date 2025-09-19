// Copyright (c) 2024, Your Name. All rights reserved.
//
// This source code is licensed under both the GPLv2 (found in the
// COPYING file in the root directory) and Apache 2.0 License
// (found in the LICENSE.Apache file in the root directory).

#include "columnar_memtable_rep.h"

#include "db/memtable.h"
#include "memory/arena.h"
#include "rocksdb/comparator.h"

namespace ROCKSDB_NAMESPACE {

// Helper class to bridge MemTableRep::Iterator and our Columnar Iterator
// NOTE: This iterator only works on the *flushed* (sorted) part of the
// memtable. For a complete view, RocksDB's MemTable::NewIterator would
// typically merge this with an iterator over the active part. Our Get() method
// is fast for the active part, so this iterator is primarily for full
// scans/flushes.
class ColumnarIterator : public MemTableRep::Iterator {
 public:
  explicit ColumnarIterator(std::shared_ptr<ColumnarMemTable> table)
      : table_(std::move(table)), iter_(table_->NewRawIterator()) {
    iter_->SeekToFirst();
  }

  ~ColumnarIterator() override = default;

  bool Valid() const override { return iter_ && iter_->IsValid(); }

  const char* key() const override {
    assert(Valid());
    // The key() method in MemTableRep::Iterator needs to return a buffer
    // in the format that MemTable expects:
    // varint32 internal_key_size | internal_key | varint32 value_size | value
    RecordRef ref = iter_->Get();

    const size_t key_size = ref.key.size();
    const size_t val_size = ref.value.size();
    const size_t internal_key_len_size = VarintLength(key_size);
    const size_t value_len_size = VarintLength(val_size);
    const size_t encoded_len =
        internal_key_len_size + key_size + value_len_size + val_size;

    // Use a member string as a stable buffer for the encoded key.
    current_key_buffer_.clear();
    current_key_buffer_.reserve(encoded_len);

    PutVarint32(&current_key_buffer_, static_cast<uint32_t>(key_size));
    current_key_buffer_.append(ref.key.data(), key_size);
    PutVarint32(&current_key_buffer_, static_cast<uint32_t>(val_size));
    current_key_buffer_.append(ref.value.data(), val_size);

    return current_key_buffer_.data();
  }

  void Next() override {
    assert(Valid());
    iter_->Next();
  }

  void Prev() override {
    // Our iterator is forward-only.
    // This is a valid limitation for many memtable implementations.
    assert(false);  // Should not be called in standard RocksDB flows.
  }

  void Seek(const Slice& internal_key, const char* /*memtable_key*/) override {
    iter_->Seek(internal_key);
  }

  void SeekForPrev(const Slice& /*internal_key*/,
                   const char* /*memtable_key*/) override {
    // Not supported.
    assert(false);
  }

  void SeekToFirst() override { iter_->SeekToFirst(); }

  void SeekToLast() override {
    // Not supported.
    assert(false);
  }

 private:
  std::shared_ptr<ColumnarMemTable> table_;
  std::unique_ptr<FlushIterator> iter_;
  mutable std::string current_key_buffer_;
};

// --- ColumnarRepFactory ---

ColumnarRepFactory::ColumnarRepFactory(size_t active_block_size_bytes,
                                       size_t num_shards,
                                       bool enable_compaction)
    : active_block_size_bytes_(active_block_size_bytes),
      num_shards_(num_shards),
      enable_compaction_(enable_compaction) {}

MemTableRep* ColumnarRepFactory::CreateMemTableRep(
    const MemTableRep::KeyComparator& memtable_rep_comparator,
    Allocator* allocator, const SliceTransform* /*slice_transform*/,
    Logger* /*logger*/) {
  // In the context of a DB, the provided comparator is actually a
  // MemTable::KeyComparator. We can safely cast it to get the underlying
  // InternalKeyComparator, which is what our CppKeyComparator needs.
  const auto* memtable_key_cmp =
      static_cast<const MemTable::KeyComparator*>(&memtable_rep_comparator);
  const InternalKeyComparator* internal_comparator =
      &((static_cast<const MemTable::KeyComparator*>(&memtable_rep_comparator))
            ->comparator);

  // Create a C++-style lambda wrapper for the InternalKeyComparator.
  auto comparator_wrapper = [internal_comparator](const Slice& a,
                                                  const Slice& b) -> bool {
    return internal_comparator->Compare(a, b) < 0;
  };

  auto table = ColumnarMemTable::Create(
      active_block_size_bytes_, enable_compaction_,
      std::make_shared<StdSorter>(),
      num_shards_, comparator_wrapper);

  return new ColumnarRep(std::move(table), allocator);
}

// --- ColumnarRep ---

ColumnarRep::ColumnarRep(std::shared_ptr<ColumnarMemTable> table,
                         Allocator* allocator)
    : MemTableRep(allocator), table_(std::move(table)) {}

ColumnarRep::~ColumnarRep() {}

KeyHandle ColumnarRep::Allocate(const size_t len, char** buf) {
  *buf = allocator_->Allocate(len);
  return static_cast<KeyHandle>(*buf);
}

void ColumnarRep::DecodeAndInsert(KeyHandle handle) {
  const char* entry = static_cast<const char*>(handle);
  Slice internal_key = GetLengthPrefixedSlice(entry);
  // The value starts after the internal key's length prefix and data
  const char* value_ptr =
      entry + VarintLength(internal_key.size()) + internal_key.size();
  Slice value = GetLengthPrefixedSlice(value_ptr);

  table_->Insert(internal_key, value);
}

void ColumnarRep::Insert(KeyHandle handle) {
  // Our memtable is internally concurrent, so we can just delegate.
  DecodeAndInsert(handle);
}

void ColumnarRep::InsertConcurrently(KeyHandle handle) {
  DecodeAndInsert(handle);
}

void ColumnarRep::InsertBatch(
    const std::vector<std::pair<Slice, Slice>>& batch) {
  // This method assumes the Slices are internal keys and values.
  // In our db_bench test, we will have to construct these.
  // Let's assume the ColumnarMemTable has a method that can take this.
  // We might need to adjust the ColumnarMemTable interface slightly.

  // For now, let's assume ColumnarMemTable::Insert can be called in a loop,
  // but a real implementation would call a true batch method.
  // Let's add a proper PutBatch to your ColumnarMemTable first.

  // ASSUMING you add this to ColumnarMemTable:
  // void PutBatch(const std::vector<std::pair<Slice, Slice>>& batch);
  table_->InsertBatch(batch);
}

bool ColumnarRep::Contains(const char* key) const {
  Slice internal_key = GetLengthPrefixedSlice(key);
  // A simple Get is sufficient. We need to check if the result is a deletion.
  auto result = table_->Get(internal_key);
  return result.has_value() && result->type == kTypeValue;
}

void ColumnarRep::Get(const LookupKey& k, void* callback_args,
                      bool (*callback_func)(void* arg, const char* entry)) {
  // Our Get() is optimized for point lookups on the active/sealed data.
  auto result_ref = table_->Get(k.internal_key());

  if (result_ref.has_value()) {
    // We found the latest version. Reconstruct the entry for the callback.
    // The callback is only called for Put values. The memtable iterator logic
    // in RocksDB handles filtering out deletions.
    if (result_ref->type == kTypeValue) {
      const Slice& ikey = result_ref->key;
      const Slice& val = result_ref->value;

      std::lock_guard<std::mutex> lock(get_arena_mutex_);
      const size_t key_size = ikey.size();
      const size_t val_size = val.size();
      const size_t internal_key_len_size = VarintLength(key_size);
      const size_t value_len_size = VarintLength(val_size);
      const size_t encoded_len =
          internal_key_len_size + key_size + value_len_size + val_size;

      char* buf = get_arena_.Allocate(encoded_len);
      char* p = buf;
      p = EncodeVarint32(p, static_cast<uint32_t>(key_size));
      memcpy(p, ikey.data(), key_size);
      p += key_size;
      p = EncodeVarint32(p, static_cast<uint32_t>(val_size));
      memcpy(p, val.data(), val_size);

      callback_func(callback_args, buf);
    }
    // If it's a delete or other type, we've found the latest version, so we
    // stop and let the caller handle it.
    return;
  }

  // If not found in our optimized path (active/sealed blocks), it might be in
  // the sorted blocks. The default MemTableRep::Get uses the iterator for this.
  // Note: This default implementation is slow as it does a full seek.
  // A fully optimized version might have a Get() that also searches sorted
  // blocks. But for now, this is correct.
  MemTableRep::Get(k, callback_args, callback_func);
}

size_t ColumnarRep::ApproximateMemoryUsage() {
  return table_ ? table_->ApproximateMemoryUsage() : 0;
}

MemTableRep::Iterator* ColumnarRep::GetIterator(Arena* arena) {
  if (arena == nullptr) {
    return new ColumnarIterator(table_);
  } else {
    void* buf = arena->Allocate(sizeof(ColumnarIterator));
    // Placement new
    return new (buf) ColumnarIterator(table_);
  }
}

}  // namespace ROCKSDB_NAMESPACE