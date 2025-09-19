// Copyright (c) 2024, Your Name. All rights reserved.
//
// A high-performance, sharded, header-only columnar memtable designed for
// integration with RocksDB. It features thread-local allocation, concurrent
// hash-based indexing for active data, and a background thread for sorting and
// compacting sealed data blocks.
//
// --- ROCKSDB-FRIENDLY REVISION 3 (PERFORMANCE OPTIMIZATION) ---
// Changes from previous revision:
// 1. MAJOR ITERATOR REFACTOR: Replaced the inefficient UnsortedBlockIterator.
//    The new strategy collects all records from unsorted blocks (active/sealed)
//    into a single vector, sorts it ONCE, and uses a new efficient
//    VectorRecordIterator. This drastically reduces iterator creation overhead.
// 2. CACHE-FRIENDLY ARENA: Reworked ColumnarRecordArena's internal chunk
//    management from a vector-of-pointers to an intrusive linked list. This
//    improves data locality and speeds up the CollectRecordsFromArena process
//    for the background thread.

#ifndef COLUMNAR_MEMTABLE_H
#define COLUMNAR_MEMTABLE_H

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#define XXH_INLINE_ALL
#include "db/dbformat.h"
#include "rocksdb/comparator.h"
#include "rocksdb/slice.h"
#include "util/xxhash.h"

namespace ROCKSDB_NAMESPACE {

// --- Comparator Type Definition ---
using CppKeyComparator = std::function<bool(const Slice& a, const Slice& b)>;

// --- Forward Declarations ---
struct RecordRef;
struct StoredRecord;
class Sorter;
class SortedColumnarBlock;
class FlashActiveBlock;
class BloomFilter;
class ColumnarRecordArena;
class ColumnarMemTable;
class FlushIterator;

// --- Core Utility Structures (No changes) ---

struct XXHasher {
  std::size_t operator()(const Slice& key) const noexcept {
    return XXH3_64bits(key.data(), key.size());
  }
};

class SpinLock {
 public:
  void lock() noexcept {
    for (;;) {
      if (!lock_.exchange(true, std::memory_order_acquire)) {
        return;
      }
      while (lock_.load(std::memory_order_relaxed)) {
        __builtin_ia32_pause();
      }
    }
  }
  void unlock() noexcept { lock_.store(false, std::memory_order_release); }

 private:
  std::atomic<bool> lock_ = {false};
};

struct RecordRef {
  Slice key;
  Slice value;
  ValueType type;
};

// --- Bloom Filter (No changes) ---
class BloomFilter {
 public:
  explicit BloomFilter(size_t num_entries, double false_positive_rate = 0.01);
  void Add(const Slice& key);
  bool MayContain(const Slice& key) const;
  size_t ApproximateMemoryUsage() const { return bits_.capacity() / 8; }

 private:
  static std::array<uint64_t, 2> Hash(const Slice& key);
  std::vector<bool> bits_;
  int num_hashes_;
};
inline BloomFilter::BloomFilter(size_t n, double p) {
  if (n == 0) n = 1;
  size_t m = static_cast<size_t>(-1.44 * n * std::log(p));
  bits_ = std::vector<bool>((m + 7) & ~7, false);
  num_hashes_ = static_cast<int>(0.7 * (double(bits_.size()) / n));
  if (num_hashes_ < 1) num_hashes_ = 1;
  if (num_hashes_ > 8) num_hashes_ = 8;
}
inline void BloomFilter::Add(const Slice& key) {
  auto h = Hash(key);
  for (int i = 0; i < num_hashes_; ++i) {
    uint64_t hash = h[0] + i * h[1];
    if (!bits_.empty()) bits_[hash % bits_.size()] = true;
  }
}
inline bool BloomFilter::MayContain(const Slice& key) const {
  if (bits_.empty()) return true;
  auto h = Hash(key);
  for (int i = 0; i < num_hashes_; ++i) {
    uint64_t hash = h[0] + i * h[1];
    if (!bits_[hash % bits_.size()]) return false;
  }
  return true;
}
inline std::array<uint64_t, 2> BloomFilter::Hash(const Slice& key) {
  XXH128_hash_t const hash_val = XXH3_128bits(key.data(), key.size());
  return {hash_val.low64, hash_val.high64};
}

// --- Columnar MemTable Components ---

struct StoredRecord {
  RecordRef record;
  std::atomic<bool> ready{false};
};

// --- ThreadIdManager (No changes) ---
class ThreadIdManager {
 public:
  static constexpr size_t kMaxThreads = 256;
  static uint32_t GetId() {
    thread_local static std::size_t h = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return static_cast<uint32_t>(h % kMaxThreads);
  }

 private:
  struct ThreadIdRecycler {
    uint32_t id;
    ThreadIdRecycler() {
      std::lock_guard<SpinLock> lock(pool_lock_);
      if (!recycled_ids_.empty()) {
        id = recycled_ids_.front();
        recycled_ids_.pop_front();
      } else {
        id = next_id_.fetch_add(1, std::memory_order_relaxed);
        if (id >= kMaxThreads) {
          next_id_.fetch_sub(1, std::memory_order_relaxed);
          throw std::runtime_error("Exceeded kMaxThreads. Increase constant.");
        }
      }
    }
    ~ThreadIdRecycler() {
      std::lock_guard<SpinLock> lock(pool_lock_);
      recycled_ids_.push_back(id);
    }
  };
  static inline std::atomic<uint32_t> next_id_{0};
  static inline std::deque<uint32_t> recycled_ids_;
  static inline SpinLock pool_lock_;
};

// --- ColumnarRecordArena (OPTIMIZED for CACHE LOCALITY) ---
class ColumnarRecordArena {
 private:
  friend class ColumnarMemTable;

 public:
  ColumnarRecordArena();
  ~ColumnarRecordArena();
  struct DataChunk {
    static constexpr size_t kRecordCapacity = 256;
    static constexpr size_t kBufferCapacity = 32 * 1024;
    std::atomic<uint64_t> positions_{0};
    std::array<StoredRecord, kRecordCapacity> records;
    alignas(64) char buffer[kBufferCapacity];
    // --- OPTIMIZATION: Intrusive linked list for cache-friendly traversal ---
    DataChunk* next_chunk_in_block{nullptr};
  };

  struct alignas(64) ThreadLocalData {
    // --- OPTIMIZATION: Changed from vector of unique_ptr to raw pointers for a
    // linked list ---
    DataChunk* head_chunk = nullptr;
    DataChunk* current_chunk = nullptr;
    SpinLock chunk_switch_lock;
    ThreadLocalData() { AddNewChunk(); }
    void AddNewChunk() {
      auto* new_chunk = new DataChunk();
      if (current_chunk) {
        current_chunk->next_chunk_in_block = new_chunk;
      } else {
        head_chunk = new_chunk;
      }
      current_chunk = new_chunk;
    }
  };
  const StoredRecord* AllocateAndAppend(const Slice& internal_key,
                                        const Slice& value);
  std::vector<const StoredRecord*> AllocateAndAppendBatch(
      const std::vector<std::pair<Slice, Slice>>& batch);
  size_t size() const { return size_.load(std::memory_order_acquire); }
  size_t ApproximateMemoryUsage() const {
    return memory_usage_.load(std::memory_order_relaxed);
  }
  uint32_t GetMaxThreadIdSeen() const {
    return max_tid_seen_.load(std::memory_order_acquire);
  }
  const std::array<std::atomic<ThreadLocalData*>, ThreadIdManager::kMaxThreads>&
  GetAllTlsData() const {
    return all_tls_data_;
  }

 private:
  ThreadLocalData* GetTlsData();
  std::array<std::atomic<ThreadLocalData*>, ThreadIdManager::kMaxThreads>
      all_tls_data_{};
  std::vector<ThreadLocalData*> owned_tls_data_;
  SpinLock owner_lock_;
  std::atomic<size_t> size_;
  std::atomic<size_t> memory_usage_{0};
  std::atomic<uint32_t> max_tid_seen_{0};
};

// Definitions for ColumnarRecordArena methods (updated for cache optimization)
inline ColumnarRecordArena::ColumnarRecordArena() : size_(0) {}
inline ColumnarRecordArena::~ColumnarRecordArena() {
  std::lock_guard<SpinLock> lock(owner_lock_);
  for (auto* tls_data : owned_tls_data_) {
    // --- OPTIMIZATION: Manually delete the linked list of chunks ---
    DataChunk* current = tls_data->head_chunk;
    while (current) {
      DataChunk* next = current->next_chunk_in_block;
      delete current;
      current = next;
    }
    delete tls_data;
  }
}
inline ColumnarRecordArena::ThreadLocalData* ColumnarRecordArena::GetTlsData() {
  uint32_t tid = ThreadIdManager::GetId();
  uint32_t current_max = max_tid_seen_.load(std::memory_order_relaxed);
  while (tid > current_max) {
    if (max_tid_seen_.compare_exchange_weak(current_max, tid,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {
      break;
    }
  }
  ThreadLocalData* my_data = all_tls_data_[tid].load(std::memory_order_acquire);
  if (my_data == nullptr) {
    auto* new_data = new ThreadLocalData();
    ThreadLocalData* expected_null = nullptr;
    if (all_tls_data_[tid].compare_exchange_strong(expected_null, new_data,
                                                   std::memory_order_release,
                                                   std::memory_order_acquire)) {
      std::lock_guard<SpinLock> lock(owner_lock_);
      owned_tls_data_.push_back(new_data);
      my_data = new_data;
      memory_usage_.fetch_add(sizeof(ThreadLocalData) + sizeof(DataChunk),
                              std::memory_order_relaxed);
    } else {
      delete new_data
          ->current_chunk;  // The new chunk was not linked, delete it.
      delete new_data;
      my_data = expected_null;
    }
  }
  return my_data;
}
inline const StoredRecord* ColumnarRecordArena::AllocateAndAppend(
    const Slice& internal_key, const Slice& value) {
  ThreadLocalData* tls_data = GetTlsData();
  size_t required_size = internal_key.size() + value.size();
  if (required_size > DataChunk::kBufferCapacity) return nullptr;
  uint32_t record_idx;
  uint32_t buffer_offset;
  DataChunk* chunk;
  while (true) {
    chunk = tls_data->current_chunk;
    uint64_t old_pos = chunk->positions_.load(std::memory_order_relaxed);
    while (true) {
      uint32_t old_ridx = static_cast<uint32_t>(old_pos >> 32);
      uint32_t old_bpos = static_cast<uint32_t>(old_pos);
      if (old_ridx >= DataChunk::kRecordCapacity ||
          old_bpos + required_size > DataChunk::kBufferCapacity)
        break;
      uint64_t new_pos = (static_cast<uint64_t>(old_ridx + 1) << 32) |
                         (old_bpos + required_size);
      if (chunk->positions_.compare_exchange_weak(old_pos, new_pos,
                                                  std::memory_order_acq_rel)) {
        record_idx = old_ridx;
        buffer_offset = old_bpos;
        goto allocation_success;
      }
    }
    std::lock_guard<SpinLock> lock(tls_data->chunk_switch_lock);
    if (chunk == tls_data->current_chunk) {
      tls_data->AddNewChunk();
      memory_usage_.fetch_add(sizeof(DataChunk), std::memory_order_relaxed);
    }
  }
allocation_success:
  char* key_mem = chunk->buffer + buffer_offset;
  memcpy(key_mem, internal_key.data(), internal_key.size());
  char* val_mem = key_mem + internal_key.size();
  memcpy(val_mem, value.data(), value.size());
  StoredRecord& record_slot = chunk->records[record_idx];
  record_slot.record.key = Slice(key_mem, internal_key.size());
  record_slot.record.value = Slice(val_mem, value.size());
  record_slot.record.type = ExtractValueType(internal_key);
  record_slot.ready.store(true, std::memory_order_release);
  size_.fetch_add(1, std::memory_order_release);
  memory_usage_.fetch_add(required_size + sizeof(StoredRecord),
                          std::memory_order_relaxed);
  return &record_slot;
}
inline std::vector<const StoredRecord*>
ColumnarRecordArena::AllocateAndAppendBatch(
    const std::vector<std::pair<Slice, Slice>>& batch) {
  std::vector<const StoredRecord*> results;
  if (batch.empty()) return results;
  results.reserve(batch.size());
  ThreadLocalData* tls_data = GetTlsData();
  size_t batch_offset = 0;
  while (batch_offset < batch.size()) {
    DataChunk* chunk = tls_data->current_chunk;
    uint64_t old_pos = chunk->positions_.load(std::memory_order_relaxed);
    uint32_t allocated_record_idx, allocated_buffer_pos;
    uint32_t records_to_alloc = 0;
    size_t buffer_needed = 0;
    while (true) {
      uint32_t old_ridx = static_cast<uint32_t>(old_pos >> 32);
      uint32_t old_bpos = static_cast<uint32_t>(old_pos);
      records_to_alloc = 0;
      buffer_needed = 0;
      for (size_t i = batch_offset; i < batch.size(); ++i) {
        const auto& [key, value] = batch[i];
        size_t item_size = key.size() + value.size();
        if (old_ridx + records_to_alloc < DataChunk::kRecordCapacity &&
            old_bpos + buffer_needed + item_size <=
                DataChunk::kBufferCapacity) {
          records_to_alloc++;
          buffer_needed += item_size;
        } else {
          break;
        }
      }
      if (records_to_alloc == 0) break;
      uint64_t new_pos =
          (static_cast<uint64_t>(old_ridx + records_to_alloc) << 32) |
          (old_bpos + buffer_needed);
      if (chunk->positions_.compare_exchange_weak(old_pos, new_pos,
                                                  std::memory_order_acq_rel)) {
        allocated_record_idx = old_ridx;
        allocated_buffer_pos = old_bpos;
        goto batch_allocation_success;
      }
    }
    {
      std::lock_guard<SpinLock> lock(tls_data->chunk_switch_lock);
      if (chunk == tls_data->current_chunk) {
        const auto& [key, value] = batch[batch_offset];
        if (key.size() + value.size() > DataChunk::kBufferCapacity) {
          batch_offset++;
          continue;
        }
        tls_data->AddNewChunk();
        memory_usage_.fetch_add(sizeof(DataChunk), std::memory_order_relaxed);
      }
      continue;
    }
  batch_allocation_success:
    size_t current_buffer_offset_in_batch = 0;
    for (uint32_t i = 0; i < records_to_alloc; ++i) {
      const auto& [key, value] = batch[batch_offset + i];
      char* key_mem =
          chunk->buffer + allocated_buffer_pos + current_buffer_offset_in_batch;
      memcpy(key_mem, key.data(), key.size());
      char* val_mem = key_mem + key.size();
      memcpy(val_mem, value.data(), value.size());
      StoredRecord& record_slot = chunk->records[allocated_record_idx + i];
      record_slot.record = {{key_mem, key.size()}, {val_mem, value.size()}};
      record_slot.record.type = ExtractValueType(key);
      record_slot.ready.store(true, std::memory_order_release);
      results.push_back(&record_slot);
      current_buffer_offset_in_batch += key.size() + value.size();
    }
    size_.fetch_add(records_to_alloc, std::memory_order_release);
    memory_usage_.fetch_add(
        buffer_needed + records_to_alloc * sizeof(StoredRecord),
        std::memory_order_relaxed);
    batch_offset += records_to_alloc;
  }
  return results;
}

// Helper function updated for cache-friendly arena
inline void CollectRecordsFromArena(const ColumnarRecordArena& arena,
                                    std::vector<const StoredRecord*>& records) {
  uint32_t active_threads = arena.GetMaxThreadIdSeen() + 1;
  if (active_threads > ThreadIdManager::kMaxThreads)
    active_threads = ThreadIdManager::kMaxThreads;

  for (uint32_t thread_idx = 0; thread_idx < active_threads; ++thread_idx) {
    const auto* tls_data =
        arena.GetAllTlsData()[thread_idx].load(std::memory_order_acquire);
    if (!tls_data) continue;

    // --- OPTIMIZATION: Traverse the linked list of chunks for better cache
    // locality ---
    for (const auto* chunk = tls_data->head_chunk; chunk != nullptr;
         chunk = chunk->next_chunk_in_block) {
      uint64_t final_pos = chunk->positions_.load(std::memory_order_relaxed);
      uint32_t max_idx = static_cast<uint32_t>(final_pos >> 32);
      if (max_idx > ColumnarRecordArena::DataChunk::kRecordCapacity)
        max_idx = ColumnarRecordArena::DataChunk::kRecordCapacity;

      for (uint32_t i = 0; i < max_idx; ++i) {
        const auto& record_slot = chunk->records[i];
        while (!record_slot.ready.load(std::memory_order_acquire)) {
          __builtin_ia32_pause();
        }
        records.push_back(&record_slot);
      }
    }
  }
}

// ConcurrentStringHashMap (No changes) ---
class ConcurrentStringHashMap {
 public:
  ConcurrentStringHashMap(const ConcurrentStringHashMap&) = delete;
  ConcurrentStringHashMap& operator=(const ConcurrentStringHashMap&) = delete;
  explicit ConcurrentStringHashMap(size_t build_size);
  void Insert(const Slice& key, const StoredRecord* new_record);
  const StoredRecord* Find(const Slice& key) const;

 private:
  static constexpr uint8_t EMPTY_TAG = 0xFF, LOCKED_TAG = 0xFE;
  struct alignas(64) Slot {
    std::atomic<uint8_t> tag;
    uint64_t full_hash;
    Slice key;
    std::atomic<const StoredRecord*> record;
  };
  static size_t calculate_power_of_2(size_t n) {
    if (n == 0) return 1;
    return 1UL << (64 - __builtin_clzll(n - 1));
  }
  std::unique_ptr<Slot[]> slots_;
  size_t capacity_, capacity_mask_;
  XXHasher hasher_;
};
inline ConcurrentStringHashMap::ConcurrentStringHashMap(size_t build_size) {
  size_t capacity = calculate_power_of_2(build_size * 1.5 + 64);
  capacity_ = capacity;
  capacity_mask_ = capacity - 1;
  slots_ = std::make_unique<Slot[]>(capacity_);
  for (size_t i = 0; i < capacity_; ++i) {
    slots_[i].tag.store(EMPTY_TAG, std::memory_order_relaxed);
    slots_[i].record.store(nullptr, std::memory_order_relaxed);
  }
}
inline void ConcurrentStringHashMap::Insert(const Slice& key,
                                            const StoredRecord* new_record) {
  uint64_t hash = hasher_(key);
  uint8_t tag = static_cast<uint8_t>(hash >> 56);
  if (tag >= LOCKED_TAG) tag = 0;
  size_t pos = hash & capacity_mask_;
  const size_t initial_pos = pos;
  while (true) {
    uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);
    if (current_tag == tag && slots_[pos].full_hash == hash &&
        slots_[pos].key == key) {
      slots_[pos].record.store(new_record, std::memory_order_release);
      return;
    }
    if (current_tag == EMPTY_TAG) {
      uint8_t expected_empty = EMPTY_TAG;
      if (slots_[pos].tag.compare_exchange_strong(expected_empty, LOCKED_TAG,
                                                  std::memory_order_acq_rel)) {
        slots_[pos].key = key;
        slots_[pos].full_hash = hash;
        slots_[pos].record.store(new_record, std::memory_order_relaxed);
        slots_[pos].tag.store(tag, std::memory_order_release);
        return;
      }
      continue;
    }
    pos = (pos + 1) & capacity_mask_;
    if (pos == initial_pos) {
      throw std::runtime_error("ConcurrentStringHashMap is full.");
    }
  }
}
inline const StoredRecord* ConcurrentStringHashMap::Find(
    const Slice& key) const {
  uint64_t hash = hasher_(key);
  uint8_t tag = static_cast<uint8_t>(hash >> 56);
  if (tag >= LOCKED_TAG) tag = 0;
  size_t pos = hash & capacity_mask_;
  const size_t initial_pos = pos;
  do {
    uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);
    if (current_tag == EMPTY_TAG) return nullptr;
    if (current_tag == tag && slots_[pos].full_hash == hash &&
        slots_[pos].key == key) {
      const StoredRecord* rec =
          slots_[pos].record.load(std::memory_order_acquire);
      if (rec && rec->ready.load(std::memory_order_acquire)) {
        return rec;
      }
      return nullptr;
    }
    pos = (pos + 1) & capacity_mask_;
  } while (pos != initial_pos);
  return nullptr;
}

// FlashActiveBlock (added CollectRecords) ---
class FlashActiveBlock {
  friend class ColumnarMemTable;
  friend class SortedColumnarBlock;

 public:
  explicit FlashActiveBlock(size_t cap) : index_(cap) {}
  ~FlashActiveBlock() = default;
  bool TryAdd(const Slice& internal_key, const Slice& value) {
    if (is_sealed()) return false;
    const StoredRecord* record_ptr =
        data_log_.AllocateAndAppend(internal_key, value);
    if (record_ptr) {
      index_.Insert(record_ptr->record.key, record_ptr);
      return true;
    }
    return false;
  }
  bool TryAddBatch(const std::vector<std::pair<Slice, Slice>>& batch) {
    if (is_sealed()) return false;
    auto record_ptrs = data_log_.AllocateAndAppendBatch(batch);
    if (is_sealed()) return false;
    for (const auto* record_ptr : record_ptrs) {
      if (record_ptr) {
        index_.Insert(record_ptr->record.key, record_ptr);
      }
    }
    return true;
  }
  std::optional<RecordRef> Get(const Slice& internal_key) const {
    const StoredRecord* record_ptr = index_.Find(internal_key);
    return record_ptr ? std::optional<RecordRef>(record_ptr->record)
                      : std::nullopt;
  }
  void CollectRecords(std::vector<const StoredRecord*>& records) const {
    CollectRecordsFromArena(data_log_, records);
  }
  size_t size() const { return data_log_.size(); }
  size_t ApproximateMemoryUsage() const {
    return data_log_.ApproximateMemoryUsage() + sizeof(index_);
  }
  void Seal() { sealed_.store(true, std::memory_order_release); }
  bool is_sealed() const { return sealed_.load(std::memory_order_acquire); }

 private:
  ColumnarRecordArena data_log_;
  ConcurrentStringHashMap index_;
  std::atomic<bool> sealed_{false};
};

// Sorter and implementations (No changes) ---
class Sorter {
 public:
  virtual ~Sorter() = default;
  virtual void Sort(std::vector<const StoredRecord*>& records,
                    const CppKeyComparator& comp) const = 0;
};
class StdSorter : public Sorter {
 public:
  void Sort(std::vector<const StoredRecord*>& records,
            const CppKeyComparator& comp) const override {
    if (records.empty()) return;
    std::stable_sort(records.begin(), records.end(),
                     [&](const StoredRecord* a, const StoredRecord* b) {
                       return comp(a->record.key, b->record.key);
                     });
  }
};
class ParallelRadixSorter : public Sorter {
 public:
  void Sort(std::vector<const StoredRecord*>& records,
            const CppKeyComparator& comp) const override {
    if (records.empty()) return;
    if (!comp) {
      StdSorter fallback;
      fallback.Sort(records, comp);
      return;
    }
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;
    radix_sort_msd_parallel(records.begin(), records.end(), 0, num_threads);
  }

 private:
  static constexpr size_t kSequentialSortThreshold = 2048;
  static constexpr size_t kRadixAlphabetSize = 256;
  using Iterator = std::vector<const StoredRecord*>::iterator;
  static inline int get_char_at(const Slice& s, size_t depth) {
    return depth < s.size() ? static_cast<unsigned char>(s[depth]) : -1;
  }
  void radix_sort_msd_sequential(Iterator begin, Iterator end,
                                 size_t depth) const {
    if (std::distance(begin, end) <= 1) return;
    if (static_cast<size_t>(std::distance(begin, end)) <=
        kSequentialSortThreshold) {
      std::stable_sort(
          begin, end, [&](const StoredRecord* a, const StoredRecord* b) {
            const Slice& key_a = a->record.key;
            const Slice& key_b = b->record.key;
            size_t offset_a = std::min(depth, key_a.size());
            size_t offset_b = std::min(depth, key_b.size());
            Slice sub_a(key_a.data() + offset_a, key_a.size() - offset_a);
            Slice sub_b(key_b.data() + offset_b, key_b.size() - offset_b);
            return sub_a.compare(sub_b) < 0;
          });
      return;
    }
    std::vector<const StoredRecord*> buckets[kRadixAlphabetSize];
    std::vector<const StoredRecord*> finished_strings;
    for (auto it = begin; it != end; ++it) {
      int char_code = get_char_at((*it)->record.key, depth);
      if (char_code == -1)
        finished_strings.push_back(*it);
      else
        buckets[char_code].push_back(*it);
    }
    auto current = begin;
    std::copy(finished_strings.begin(), finished_strings.end(), current);
    current += finished_strings.size();
    for (size_t i = 0; i < kRadixAlphabetSize; ++i) {
      if (!buckets[i].empty()) {
        auto bucket_begin = current;
        std::copy(buckets[i].begin(), buckets[i].end(), bucket_begin);
        current += buckets[i].size();
        radix_sort_msd_sequential(bucket_begin, current, depth + 1);
      }
    }
  }
  void radix_sort_msd_parallel(Iterator begin, Iterator end, size_t depth,
                               unsigned int num_threads) const {
    const size_t size = std::distance(begin, end);
    if (size <= kSequentialSortThreshold || num_threads <= 1) {
      radix_sort_msd_sequential(begin, end, depth);
      return;
    }
    std::vector<size_t> bucket_counts(kRadixAlphabetSize + 1, 0);
    for (auto it = begin; it != end; ++it)
      bucket_counts[get_char_at((*it)->record.key, depth) + 1]++;
    std::vector<size_t> bucket_offsets(kRadixAlphabetSize + 2, 0);
    for (size_t i = 0; i < kRadixAlphabetSize + 1; ++i)
      bucket_offsets[i + 1] = bucket_offsets[i] + bucket_counts[i];
    std::vector<const StoredRecord*> sorted_output(size);
    std::vector<size_t> current_offsets = bucket_offsets;
    for (auto it = begin; it != end; ++it) {
      const StoredRecord* val = *it;
      int char_code = get_char_at(val->record.key, depth);
      sorted_output[current_offsets[char_code + 1]++] = val;
    }
    std::copy(sorted_output.begin(), sorted_output.end(), begin);
    std::vector<std::future<void>> futures;
    for (size_t i = 1; i < kRadixAlphabetSize + 1; ++i) {
      size_t bucket_size = bucket_counts[i];
      if (bucket_size == 0) continue;
      Iterator bucket_begin = begin + bucket_offsets[i];
      Iterator bucket_end = begin + bucket_offsets[i + 1];
      if (futures.size() < num_threads - 1 &&
          bucket_size > kSequentialSortThreshold) {
        futures.push_back(std::async(
            std::launch::async,
            [this, bucket_begin, bucket_end, depth, num_threads, &futures] {
              unsigned int threads_for_child = std::max(
                  1u, num_threads / (unsigned int)(futures.size() + 1));
              radix_sort_msd_parallel(bucket_begin, bucket_end, depth + 1,
                                      threads_for_child);
            }));
      } else {
        radix_sort_msd_sequential(bucket_begin, bucket_end, depth + 1);
      }
    }
    for (auto& f : futures) f.get();
  }
};

// SortedColumnarBlock (No changes) ---
class SortedColumnarBlock {
 public:
  class Iterator;
  static constexpr size_t kSparseIndexSampleRate = 16;
  explicit SortedColumnarBlock(std::shared_ptr<FlashActiveBlock> block,
                               const Sorter& sorter,
                               const CppKeyComparator& comp,
                               bool build_bloom_filter = true)
      : source_block_(std::move(block)), comparator_(comp) {
    source_block_->CollectRecords(sorted_records_);
    sorter.Sort(sorted_records_, comparator_);
    if (sorted_records_.empty()) {
      min_key_ = {};
      max_key_ = {};
      return;
    }
    min_key_ = sorted_records_.front()->record.key;
    max_key_ = sorted_records_.back()->record.key;
    if (build_bloom_filter && !sorted_records_.empty()) {
      bloom_filter_ = std::make_unique<BloomFilter>(sorted_records_.size());
      for (const auto* rec : sorted_records_) {
        bloom_filter_->Add(ExtractUserKey(rec->record.key));
      }
    }
    sparse_index_.reserve(sorted_records_.size() / kSparseIndexSampleRate + 1);
    for (size_t i = 0; i < sorted_records_.size(); i += kSparseIndexSampleRate)
      sparse_index_.emplace_back(sorted_records_[i]->record.key, i);
  }
  bool MayContain(const Slice& key) const {
    if (empty() || comparator_(key, min_key_) || comparator_(max_key_, key))
      return false;
    if (!bloom_filter_) return true;
    return bloom_filter_->MayContain(ExtractUserKey(key));
  }
  std::optional<RecordRef> Get(const Slice& key) const {
    if (!MayContain(key)) return std::nullopt;
    size_t pos = FindFirstGreaterOrEqual(key);
    if (pos >= sorted_records_.size()) return std::nullopt;
    const StoredRecord* record_ptr = sorted_records_[pos];
    if (ExtractUserKey(record_ptr->record.key) != ExtractUserKey(key))
      return std::nullopt;
    return record_ptr->record;
  }
  Iterator NewIterator() const;
  bool empty() const { return sorted_records_.empty(); }
  size_t size() const { return sorted_records_.size(); }
  size_t ApproximateMemoryUsage() const {
    size_t usage = sizeof(*this);
    if (source_block_) usage += source_block_->ApproximateMemoryUsage();
    usage += sorted_records_.capacity() * sizeof(StoredRecord*);
    if (bloom_filter_) usage += bloom_filter_->ApproximateMemoryUsage();
    usage += sparse_index_.capacity() * sizeof(std::pair<Slice, size_t>);
    return usage;
  }

 private:
  friend class Iterator;
  size_t FindFirstGreaterOrEqual(const Slice& key) const {
    auto sparse_it = std::lower_bound(
        sparse_index_.begin(), sparse_index_.end(), key,
        [&](const auto& a, auto b) { return comparator_(a.first, b); });
    size_t start_pos =
        (sparse_it == sparse_index_.begin()) ? 0 : (sparse_it - 1)->second;
    auto it = std::lower_bound(sorted_records_.begin() + start_pos,
                               sorted_records_.end(), key,
                               [&](const StoredRecord* rec, const Slice& k) {
                                 return comparator_(rec->record.key, k);
                               });
    return std::distance(sorted_records_.begin(), it);
  }
  std::shared_ptr<FlashActiveBlock> source_block_;
  std::vector<const StoredRecord*> sorted_records_;
  Slice min_key_, max_key_;
  std::unique_ptr<BloomFilter> bloom_filter_;
  std::vector<std::pair<Slice, size_t>> sparse_index_;
  CppKeyComparator comparator_;
};

// SortedColumnarBlock::Iterator (No changes) ---
class SortedColumnarBlock::Iterator {
 public:
  explicit Iterator(const SortedColumnarBlock* b) : block_(b), pos_(0) {}
  RecordRef operator*() const { return block_->sorted_records_[pos_]->record; }
  void Next() { ++pos_; }
  bool IsValid() const {
    return block_ && pos_ < block_->sorted_records_.size();
  }
  void Seek(const Slice& key) { pos_ = block_->FindFirstGreaterOrEqual(key); }
  void SeekToFirst() { pos_ = 0; }

 private:
  const SortedColumnarBlock* block_;
  size_t pos_;
};
inline SortedColumnarBlock::Iterator SortedColumnarBlock::NewIterator() const {
  return Iterator(this);
}

// --- OPTIMIZATION: REFACTORED ITERATOR SYSTEM ---
class MemTableIterator {
 public:
  virtual ~MemTableIterator() = default;
  virtual bool IsValid() const = 0;
  virtual void Next() = 0;
  virtual void Seek(const Slice& key) = 0;
  virtual void SeekToFirst() = 0;
  virtual RecordRef Get() const = 0;
};

class SortedBlockIterator : public MemTableIterator {
 public:
  explicit SortedBlockIterator(const SortedColumnarBlock* block)
      : iter_(block->NewIterator()) {}
  bool IsValid() const override { return iter_.IsValid(); }
  void Next() override { iter_.Next(); }
  void Seek(const Slice& key) override { iter_.Seek(key); }
  void SeekToFirst() override { iter_.SeekToFirst(); }
  RecordRef Get() const override { return *iter_; }

 private:
  SortedColumnarBlock::Iterator iter_;
};

// --- OPTIMIZATION: New iterator for a pre-sorted vector of records. Replaces
// UnsortedBlockIterator. ---
class VectorRecordIterator : public MemTableIterator {
 public:
  explicit VectorRecordIterator(std::vector<const StoredRecord*> records,
                                CppKeyComparator comp)
      : sorted_records_(std::move(records)),
        comparator_(std::move(comp)),
        pos_(0) {}

  bool IsValid() const override { return pos_ < sorted_records_.size(); }
  void Next() override { ++pos_; }
  RecordRef Get() const override { return sorted_records_[pos_]->record; }
  void SeekToFirst() override { pos_ = 0; }
  void Seek(const Slice& key) override {
    auto it =
        std::lower_bound(sorted_records_.begin(), sorted_records_.end(), key,
                         [&](const StoredRecord* rec, const Slice& k) {
                           return comparator_(rec->record.key, k);
                         });
    pos_ = std::distance(sorted_records_.begin(), it);
  }

 private:
  std::vector<const StoredRecord*> sorted_records_;
  CppKeyComparator comparator_;
  size_t pos_;
};

// --- FlushIterator (No changes, but now more efficient due to its sources) ---
class FlushIterator {
 public:
  explicit FlushIterator(std::vector<std::unique_ptr<MemTableIterator>> sources,
                         const CppKeyComparator& comp)
      : comparator_(comp),
        sources_(std::move(sources)),
        min_heap_(HeapNodeComparator(comp)) {
    SeekToFirst();
  }
  bool IsValid() const { return !min_heap_.empty(); }
  RecordRef Get() const { return min_heap_.top().record; }
  void Next() {
    if (!IsValid()) return;
    HeapNode n = min_heap_.top();
    min_heap_.pop();
    sources_[n.source_index]->Next();
    if (sources_[n.source_index]->IsValid()) {
      min_heap_.push({sources_[n.source_index]->Get(), n.source_index});
    }
  }
  void Seek(const Slice& key) {
    min_heap_ = std::priority_queue<HeapNode, std::vector<HeapNode>,
                                    HeapNodeComparator>(
        HeapNodeComparator(comparator_));
    for (size_t i = 0; i < sources_.size(); ++i) {
      sources_[i]->Seek(key);
      if (sources_[i]->IsValid()) min_heap_.push({sources_[i]->Get(), i});
    }
  }
  void SeekToFirst() {
    min_heap_ = std::priority_queue<HeapNode, std::vector<HeapNode>,
                                    HeapNodeComparator>(
        HeapNodeComparator(comparator_));
    for (size_t i = 0; i < sources_.size(); ++i) {
      sources_[i]->SeekToFirst();
      if (sources_[i]->IsValid()) min_heap_.push({sources_[i]->Get(), i});
    }
  }

 private:
  struct HeapNode {
    RecordRef record;
    size_t source_index;
  };
  struct HeapNodeComparator {
    CppKeyComparator comp_;
    explicit HeapNodeComparator(CppKeyComparator comp)
        : comp_(std::move(comp)) {}
    bool operator()(const HeapNode& a, const HeapNode& b) const {
      return comp_(b.record.key, a.record.key);
    }
  };
  CppKeyComparator comparator_;
  std::vector<std::unique_ptr<MemTableIterator>> sources_;
  std::priority_queue<HeapNode, std::vector<HeapNode>, HeapNodeComparator>
      min_heap_;
};

// CompactingIterator (No changes) ---
class CompactingIterator {
 public:
  explicit CompactingIterator(std::unique_ptr<FlushIterator> s,
                              const CppKeyComparator& comp)
      : source_(std::move(s)), comp_(comp) {
    FindNext();
  }
  bool IsValid() const { return is_valid_; }
  RecordRef Get() const { return current_record_; }
  void Next() { FindNext(); }
  void Seek(const Slice& key) {
    source_->Seek(key);
    FindNext();
  }
  void SeekToFirst() {
    source_->SeekToFirst();
    FindNext();
  }

 private:
  void FindNext() {
    while (source_->IsValid()) {
      RecordRef latest_record = source_->Get();
      Slice user_key = ExtractUserKey(latest_record.key);
      source_->Next();
      while (source_->IsValid() &&
             ExtractUserKey(source_->Get().key) == user_key)
        source_->Next();
      if (latest_record.type == kTypeValue) {
        current_record_ = latest_record;
        is_valid_ = true;
        return;
      }
    }
    is_valid_ = false;
  }
  std::unique_ptr<FlushIterator> source_;
  RecordRef current_record_;
  bool is_valid_ = false;
  CppKeyComparator comp_;
};

// --- ColumnarMemTable (core logic with new iterator strategy) ---
class ColumnarMemTable : public std::enable_shared_from_this<ColumnarMemTable> {
 public:
  ~ColumnarMemTable() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      stop_background_thread_ = true;
    }
    queue_cond_.notify_one();
    if (background_thread_.joinable()) background_thread_.join();
  }
  ColumnarMemTable(const ColumnarMemTable&) = delete;
  ColumnarMemTable& operator=(const ColumnarMemTable&) = delete;

  void Insert(const Slice& internal_key, const Slice& value) {
    const size_t shard_idx = GetShardIdx(internal_key);
    auto current_block = GetActiveBlockForThread(shard_idx);
    while (!current_block->TryAdd(internal_key, value))
      current_block = GetActiveBlockForThread(shard_idx, true);
    if (current_block->size() >= active_block_threshold_)
      SealActiveBlockIfNeeded(shard_idx);
  }
  void InsertBatch(const std::vector<std::pair<Slice, Slice>>& batch) {
    if (batch.empty()) return;
    std::vector<std::vector<std::pair<Slice, Slice>>> sharded_batches(
        num_shards_);
    for (const auto& [key, value] : batch)
      sharded_batches[GetShardIdx(key)].emplace_back(key, value);
    std::vector<std::future<void>> futures;
    futures.reserve(num_shards_);
    for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
      if (sharded_batches[shard_idx].empty()) continue;
      futures.emplace_back(std::async(
          std::launch::async,
          [this, shard_idx, &sub_batch = sharded_batches[shard_idx]] {
            auto current_block = GetActiveBlockForThread(shard_idx);
            if (!current_block->TryAddBatch(sub_batch)) {
              current_block = GetActiveBlockForThread(shard_idx, true);
              for (const auto& [key, value] : sub_batch) {
                while (!current_block->TryAdd(key, value))
                  current_block = GetActiveBlockForThread(shard_idx, true);
              }
            }
            if (current_block->size() >= active_block_threshold_)
              SealActiveBlockIfNeeded(shard_idx);
          }));
    }
    for (auto& f : futures) f.get();
  }
  std::optional<RecordRef> Get(const Slice& internal_key) const {
    const size_t shard_idx = GetShardIdx(internal_key);
    auto active_block = GetActiveBlockForThread(shard_idx);
    if (auto r = active_block->Get(internal_key)) return r;
    auto s = GetImmutableStateForThread(shard_idx);
    if (s->sealed_blocks) {
      for (auto it = s->sealed_blocks->rbegin(); it != s->sealed_blocks->rend();
           ++it) {
        if (auto r = (*it)->Get(internal_key)) return r;
      }
    }
    if (s->blocks) {
      for (auto it = s->blocks->rbegin(); it != s->blocks->rend(); ++it) {
        if (auto r = (*it)->Get(internal_key)) return r;
      }
    }
    return std::nullopt;
  }

  // --- OPTIMIZATION: Complete rewrite of iterator creation logic ---
  std::unique_ptr<FlushIterator> NewRawIterator() {
    std::vector<std::unique_ptr<MemTableIterator>> all_iterators;
    std::vector<const StoredRecord*> unsorted_records;

    for (const auto& shard_ptr : shards_) {
      auto s = std::atomic_load(&shard_ptr->immutable_state_);
      auto ab = std::atomic_load(&shard_ptr->active_block_);

      // 1. Add iterators for already sorted blocks (cheap)
      if (s->blocks) {
        for (const auto& block : *s->blocks) {
          if (!block->empty())
            all_iterators.push_back(
                std::make_unique<SortedBlockIterator>(block.get()));
        }
      }

      // 2. Collect pointers from all unsorted blocks (active and sealed)
      if (s->sealed_blocks) {
        for (const auto& block : *s->sealed_blocks) {
          if (block->size() > 0) block->CollectRecords(unsorted_records);
        }
      }
      if (ab->size() > 0) ab->CollectRecords(unsorted_records);
    }

    // 3. If there's any unsorted data, sort it ONCE and create a single
    // iterator for it.
    if (!unsorted_records.empty()) {
      std::stable_sort(unsorted_records.begin(), unsorted_records.end(),
                       [comp = this->comparator_](const StoredRecord* a,
                                                  const StoredRecord* b) {
                         return comp(a->record.key, b->record.key);
                       });
      all_iterators.push_back(std::make_unique<VectorRecordIterator>(
          std::move(unsorted_records), comparator_));
    }

    // 4. Create the final merging iterator
    return std::make_unique<FlushIterator>(std::move(all_iterators),
                                           comparator_);
  }

  std::unique_ptr<CompactingIterator> NewCompactingIterator() {
    auto raw_iterator = NewRawIterator();
    return std::make_unique<CompactingIterator>(std::move(raw_iterator),
                                                comparator_);
  }
  size_t ApproximateMemoryUsage() const {
    size_t total = 0;
    for (const auto& shard : shards_) {
      auto active = std::atomic_load(&shard->active_block_);
      total += active->ApproximateMemoryUsage();
      auto immutable = std::atomic_load(&shard->immutable_state_);
      if (immutable->sealed_blocks) {
        for (const auto& block : *immutable->sealed_blocks)
          total += block->ApproximateMemoryUsage();
      }
      if (immutable->blocks) {
        for (const auto& block : *immutable->blocks)
          total += block->ApproximateMemoryUsage();
      }
    }
    return total;
  }
  void PrepareForFlush() {
    {
      std::lock_guard<std::mutex> queue_lock(queue_mutex_);
      for (size_t i = 0; i < num_shards_; ++i) SealActiveBlockIfNeeded(i, true);
    }
    queue_cond_.notify_one();
  }
  const CppKeyComparator& GetComparator() const { return comparator_; }
  static std::shared_ptr<ColumnarMemTable> Create(
      size_t active_block_size_bytes = 4 * 1024 * 1024,
      bool enable_compaction = false,
      std::shared_ptr<Sorter> sorter = std::make_shared<ParallelRadixSorter>(),
      size_t num_shards = 16, CppKeyComparator comparator = nullptr) {
    struct MakeSharedEnabler : public ColumnarMemTable {
      MakeSharedEnabler(size_t ab, bool ec, std::shared_ptr<Sorter> s,
                        size_t ns, CppKeyComparator c)
          : ColumnarMemTable(ab, ec, std::move(s), ns, std::move(c)) {}
    };
    auto table = std::make_shared<MakeSharedEnabler>(
        active_block_size_bytes, enable_compaction, std::move(sorter),
        num_shards, std::move(comparator));
    table->StartBackgroundThread();
    return table;
  }
  void StartBackgroundThread() {
    background_thread_ =
        std::thread([this]() { this->BackgroundWorkerLoop(); });
  }

 private:
  explicit ColumnarMemTable(size_t active_block_size_bytes,
                            bool enable_compaction,
                            std::shared_ptr<Sorter> sorter, size_t num_shards,
                            CppKeyComparator comparator)
      : active_block_threshold_(
            std::max((size_t)1,
                     active_block_size_bytes / 128)),  // Adjusted average size
        enable_compaction_(enable_compaction),
        sorter_(std::move(sorter)),
        comparator_(std::move(comparator)),
        num_shards_(num_shards > 0 ? 1UL << (63 - __builtin_clzll(num_shards))
                                   : 1),
        shard_mask_(num_shards_ - 1) {
    for (size_t i = 0; i < num_shards_; ++i)
      shards_.push_back(std::make_unique<Shard>(active_block_threshold_));
  }
  struct ImmutableState {
    using SortedBlockList =
        std::vector<std::shared_ptr<const SortedColumnarBlock>>;
    using SealedBlockList = std::vector<std::shared_ptr<FlashActiveBlock>>;
    std::shared_ptr<const SealedBlockList> sealed_blocks;
    std::shared_ptr<const SortedBlockList> blocks;
    ImmutableState()
        : sealed_blocks(std::make_shared<const SealedBlockList>()),
          blocks(std::make_shared<const SortedBlockList>()) {}
  };
  struct alignas(64) Shard {
    std::shared_ptr<FlashActiveBlock> active_block_;
    std::shared_ptr<const ImmutableState> immutable_state_;
    std::atomic<uint64_t> version_{0};
    SpinLock seal_mutex_;
    Shard(size_t active_block_threshold) {
      active_block_ =
          std::make_shared<FlashActiveBlock>(active_block_threshold);
      immutable_state_ = std::make_shared<const ImmutableState>();
    }
  };
  struct BackgroundWorkItem {
    std::shared_ptr<FlashActiveBlock> block;
    size_t shard_idx;
  };
  void SealActiveBlockIfNeeded(size_t shard_idx, bool force = false) {
    auto& shard = *shards_[shard_idx];
    auto current_b_sp = std::atomic_load(&shard.active_block_);
    bool should_seal =
        force ? (current_b_sp->size() > 0 && !current_b_sp->is_sealed())
              : (current_b_sp->size() >= active_block_threshold_ &&
                 !current_b_sp->is_sealed());
    if (!should_seal) return;
    std::shared_ptr<FlashActiveBlock> sealed_block;
    {
      std::lock_guard<SpinLock> lock(shard.seal_mutex_);
      current_b_sp = std::atomic_load(&shard.active_block_);
      should_seal =
          force ? (current_b_sp->size() > 0 && !current_b_sp->is_sealed())
                : (current_b_sp->size() >= active_block_threshold_ &&
                   !current_b_sp->is_sealed());
      if (!should_seal) return;
      current_b_sp->Seal();
      sealed_block = current_b_sp;
      auto new_active_block =
          std::make_shared<FlashActiveBlock>(active_block_threshold_);
      auto old_s = std::atomic_load(&shard.immutable_state_);
      auto new_s = std::make_shared<ImmutableState>();
      new_s->blocks = old_s->blocks;
      auto new_sealed_list = std::make_shared<ImmutableState::SealedBlockList>(
          *old_s->sealed_blocks);
      new_sealed_list->push_back(sealed_block);
      new_s->sealed_blocks = new_sealed_list;
      std::atomic_exchange(&shard.active_block_, new_active_block);
      std::atomic_store(&shard.immutable_state_,
                        std::shared_ptr<const ImmutableState>(new_s));
      shard.version_.fetch_add(1, std::memory_order_release);
    }
    {
      std::lock_guard<std::mutex> ql(queue_mutex_);
      sealed_blocks_queue_.push_back({std::move(sealed_block), shard_idx});
    }
    if (!force) queue_cond_.notify_one();
  }
  void BackgroundWorkerLoop() {
    while (true) {
      std::vector<BackgroundWorkItem> work_items;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cond_.wait(lock, [this] {
          return !sealed_blocks_queue_.empty() || stop_background_thread_;
        });
        if (stop_background_thread_ && sealed_blocks_queue_.empty()) return;
        work_items.swap(sealed_blocks_queue_);
      }
      std::map<size_t, std::vector<std::shared_ptr<FlashActiveBlock>>>
          work_by_shard;
      for (auto& item : work_items) {
        if (item.block)
          work_by_shard[item.shard_idx].push_back(std::move(item.block));
      }
      for (auto const& [shard_idx, blocks] : work_by_shard) {
        try {
          ProcessBlocksForShard(shard_idx, blocks);
        } catch (const std::exception& e) {
          fprintf(stderr, "Exception in background thread for shard %zu: %s\n",
                  shard_idx, e.what());
        }
      }
    }
  }
  void ProcessBlocksForShard(
      size_t shard_idx,
      const std::vector<std::shared_ptr<FlashActiveBlock>>& sealed_blocks) {
    if (sealed_blocks.empty()) return;
    std::vector<std::shared_ptr<const SortedColumnarBlock>> new_sorted_blocks;
    new_sorted_blocks.reserve(sealed_blocks.size());
    for (const auto& sealed_b : sealed_blocks) {
      if (sealed_b->size() == 0) continue;
      new_sorted_blocks.push_back(std::make_shared<const SortedColumnarBlock>(
          sealed_b, *sorter_, comparator_, !enable_compaction_));
    }
    auto& shard = *shards_[shard_idx];
    std::lock_guard<SpinLock> lock(shard.seal_mutex_);
    auto old_s = std::atomic_load(&shard.immutable_state_);
    auto new_s = std::make_shared<ImmutableState>();
    auto new_sorted_list = std::make_shared<ImmutableState::SortedBlockList>();
    if (old_s->blocks) *new_sorted_list = *old_s->blocks;
    new_sorted_list->insert(new_sorted_list->end(), new_sorted_blocks.begin(),
                            new_sorted_blocks.end());
    new_s->blocks = std::move(new_sorted_list);
    auto new_sealed_list = std::make_shared<ImmutableState::SealedBlockList>();
    if (old_s->sealed_blocks) {
      for (const auto& b : *old_s->sealed_blocks) {
        bool was_processed = false;
        for (const auto& pb : sealed_blocks)
          if (b == pb) {
            was_processed = true;
            break;
          }
        if (!was_processed) new_sealed_list->push_back(b);
      }
    }
    new_s->sealed_blocks = std::move(new_sealed_list);
    std::atomic_store(&shard.immutable_state_,
                      std::shared_ptr<const ImmutableState>(new_s));
  }
  size_t GetShardIdx(const Slice& key) const {
    return hasher_(ExtractUserKey(key)) & shard_mask_;
  }
  std::shared_ptr<FlashActiveBlock> GetActiveBlockForThread(
      size_t shard_idx, bool force_refresh = false) const {
    thread_local std::vector<std::shared_ptr<FlashActiveBlock>>
        active_block_cache;
    thread_local std::vector<uint64_t> last_seen_version;
    if (active_block_cache.size() != num_shards_) {
      active_block_cache.resize(num_shards_);
      last_seen_version.resize(num_shards_, (uint64_t)-1);
    }
    const auto& shard = *shards_[shard_idx];
    uint64_t current_version = shard.version_.load(std::memory_order_acquire);
    if (force_refresh || active_block_cache[shard_idx] == nullptr ||
        last_seen_version[shard_idx] != current_version) {
      active_block_cache[shard_idx] = std::atomic_load(&shard.active_block_);
      last_seen_version[shard_idx] = current_version;
    }
    return active_block_cache[shard_idx];
  }
  std::shared_ptr<const ImmutableState> GetImmutableStateForThread(
      size_t shard_idx, bool force_refresh = false) const {
    thread_local std::vector<std::shared_ptr<const ImmutableState>>
        immutable_state_cache;
    thread_local std::vector<uint64_t> last_seen_version;
    if (immutable_state_cache.size() != num_shards_) {
      immutable_state_cache.resize(num_shards_);
      last_seen_version.resize(num_shards_, (uint64_t)-1);
    }
    const auto& shard = *shards_[shard_idx];
    uint64_t current_version = shard.version_.load(std::memory_order_acquire);
    if (force_refresh || immutable_state_cache[shard_idx] == nullptr ||
        last_seen_version[shard_idx] != current_version) {
      immutable_state_cache[shard_idx] =
          std::atomic_load(&shard.immutable_state_);
      last_seen_version[shard_idx] = current_version;
    }
    return immutable_state_cache[shard_idx];
  }
  const size_t active_block_threshold_;
  const bool enable_compaction_;
  std::shared_ptr<Sorter> sorter_;
  CppKeyComparator comparator_;
  const size_t num_shards_;
  const size_t shard_mask_;
  std::vector<std::unique_ptr<Shard>> shards_;
  XXHasher hasher_;
  std::vector<BackgroundWorkItem> sealed_blocks_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cond_;
  std::thread background_thread_;
  std::atomic<bool> stop_background_thread_{false};
};

}  // namespace ROCKSDB_NAMESPACE
#endif  // COLUMNAR_MEMTABLE_H