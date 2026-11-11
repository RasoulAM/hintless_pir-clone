// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hintless_simplepir/server.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "hintless_simplepir/database_hwy.h"
#include "hintless_simplepir/parameters.h"
#include "hintless_simplepir/serialization.pb.h"
#include "hintless_simplepir/utils.h"
#include "lwe/lwe_symmetric_encryption.h"
#include "lwe/types.h"
#include "shell_encryption/prng/single_thread_chacha_prng.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/status_macros.h"

namespace hintless_pir {
namespace hintless_simplepir {

namespace {

// Returns an error if `params` uses an invalid PRNG type.
inline absl::Status CheckForValidPrngType(const Parameters& params) {
  if (!(params.prng_type == rlwe::PRNG_TYPE_HKDF ||
        params.prng_type == rlwe::PRNG_TYPE_CHACHA)) {
    return absl::InvalidArgumentError("Invalid PRNG type in `params`.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<Server>> Server::Create(
    const Parameters& params) {
  RLWE_RETURN_IF_ERROR(CheckForValidPrngType(params));

  // Create RLWE contexts, one per plaintext modulus in `ts`.
  auto const& rlwe_params = params.linpir_params;
  int num_linpir_instances = rlwe_params.ts.size();
  std::vector<std::unique_ptr<const RlweRnsContext>> rlwe_contexts;
  rlwe_contexts.reserve(num_linpir_instances);
  for (int i = 0; i < num_linpir_instances; ++i) {
    RLWE_ASSIGN_OR_RETURN(
        auto rlwe_context,
        RlweRnsContext::CreateForBfvFiniteFieldEncoding(
            rlwe_params.log_n, rlwe_params.qs, /*ps=*/{}, rlwe_params.ts[i]));
    rlwe_contexts.push_back(
        std::make_unique<const RlweRnsContext>(std::move(rlwe_context)));
  }

  // Create a Database object holding the database and hint matrices.
  RLWE_ASSIGN_OR_RETURN(auto database, Database::Create(params));

  return absl::WrapUnique(
      new Server(params, std::move(database), std::move(rlwe_contexts)));
}

absl::StatusOr<std::unique_ptr<Server>> Server::CreateWithRandomDatabaseRecords(
    const Parameters& params) {
  RLWE_RETURN_IF_ERROR(CheckForValidPrngType(params));

  // Create RLWE contexts, one per plaintext modulus in `ts`.
  auto const& rlwe_params = params.linpir_params;
  int num_linpir_instances = rlwe_params.ts.size();
  std::vector<std::unique_ptr<const RlweRnsContext>> rlwe_contexts;
  rlwe_contexts.reserve(num_linpir_instances);
  for (int i = 0; i < num_linpir_instances; ++i) {
    RLWE_ASSIGN_OR_RETURN(
        auto rlwe_context,
        RlweRnsContext::CreateForBfvFiniteFieldEncoding(
            rlwe_params.log_n, rlwe_params.qs, /*ps=*/{}, rlwe_params.ts[i]));
    rlwe_contexts.push_back(
        std::make_unique<const RlweRnsContext>(std::move(rlwe_context)));
  }

  // Create a Databas holding random records.
  RLWE_ASSIGN_OR_RETURN(auto database, Database::CreateRandom(params));

  return absl::WrapUnique(
      new Server(params, std::move(database), std::move(rlwe_contexts)));
}

absl::Status Server::GeneratePublicParams() {
  int num_linpir_instances = params_.linpir_params.ts.size();
  if (params_.prng_type == rlwe::PRNG_TYPE_HKDF) {
    // Sample PRNG seeds for LWE "A" matrix and LinPIR.
    RLWE_ASSIGN_OR_RETURN(prng_seed_lwe_query_pad_,
                          rlwe::SingleThreadHkdfPrng::GenerateSeed());
    prng_seed_linpir_ct_pads_.clear();
    prng_seed_linpir_ct_pads_.resize(num_linpir_instances);
    for (int i = 0; i < num_linpir_instances; ++i) {
      RLWE_ASSIGN_OR_RETURN(prng_seed_linpir_ct_pads_[i],
                            rlwe::SingleThreadHkdfPrng::GenerateSeed());
    }
    RLWE_ASSIGN_OR_RETURN(prng_seed_linpir_gk_pad_,
                          rlwe::SingleThreadHkdfPrng::GenerateSeed());
    // Generate the LWE "A" matrix.
    RLWE_ASSIGN_OR_RETURN(auto prng, rlwe::SingleThreadHkdfPrng::Create(
                                         prng_seed_lwe_query_pad_));
    RLWE_ASSIGN_OR_RETURN(
        auto pad,
        lwe::ExpandPad(params_.db_cols, params_.lwe_secret_dim, prng.get()));
    lwe_query_pad_ = std::make_unique<const lwe::Matrix>(std::move(pad));
  } else {
    RLWE_ASSIGN_OR_RETURN(prng_seed_lwe_query_pad_,
                          rlwe::SingleThreadChaChaPrng::GenerateSeed());
    prng_seed_linpir_ct_pads_.clear();
    prng_seed_linpir_ct_pads_.resize(num_linpir_instances);
    for (int i = 0; i < num_linpir_instances; ++i) {
      RLWE_ASSIGN_OR_RETURN(prng_seed_linpir_ct_pads_[i],
                            rlwe::SingleThreadChaChaPrng::GenerateSeed());
    }
    RLWE_ASSIGN_OR_RETURN(prng_seed_linpir_gk_pad_,
                          rlwe::SingleThreadChaChaPrng::GenerateSeed());

    RLWE_ASSIGN_OR_RETURN(auto prng, rlwe::SingleThreadChaChaPrng::Create(
                                         prng_seed_lwe_query_pad_));
    RLWE_ASSIGN_OR_RETURN(
        auto pad,
        lwe::ExpandPad(params_.db_cols, params_.lwe_secret_dim, prng.get()));
    lwe_query_pad_ = std::make_unique<const lwe::Matrix>(std::move(pad));
  }
  return absl::OkStatus();
}

namespace {

// Given `matrix` with mod-q entries, returns `matrix` mod p, where modular
// numbers are in balanced representation.
template <typename Integer>
std::vector<std::vector<Integer>> EncodeLweMatrix(
    const Database::LweMatrix& matrix, Integer q, Integer p) {
  Integer q_half = q >> 1;
  int num_rows = matrix.size();
  int num_cols = matrix[0].size();
  std::vector<std::vector<Integer>> matrix_mod_p(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    matrix_mod_p[i].reserve(num_cols);
    for (int j = 0; j < num_cols; ++j) {
      Integer x = static_cast<Integer>(matrix[i][j]);
      matrix_mod_p[i].push_back(ConvertModulus(x, q, p, q_half));
    }
  }
  return matrix_mod_p;
}

}  // namespace


namespace {

constexpr char kStateFile[] = "/tmp/hintless_db.bin";

int64_t MicrosSince(std::chrono::high_resolution_clock::time_point t) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::high_resolution_clock::now() - t)
      .count();
}

// Writes raw bytes of trivially copyable values / arrays.
class StateWriter {
 public:
  explicit StateWriter(std::ofstream* out) : out_(out) {}
  void Write(const void* src, size_t n) {
    out_->write(reinterpret_cast<const char*>(src), n);
  }
  template <typename T>
  void Put(const T& v) { Write(&v, sizeof(T)); }

 private:
  std::ofstream* out_;
};

// Reads the state file either through a staging buffer holding the whole
// file (memcpy into destinations), or directly into destination buffers
// with read(2) (option 3).
class StateReader {
 public:
  StateReader(int fd, size_t size, bool direct) : fd_(fd), direct_(direct) {
    if (!direct_) {
      buffer_.reset(new char[size]);
      char* p = buffer_.get();
      size_t left = size;
      while (left > 0) {
        ssize_t r = read(fd_, p, left);
        if (r <= 0) break;
        p += r;
        left -= r;
      }
      ptr_ = buffer_.get();
    }
  }
  void Read(void* dst, size_t n) {
    if (direct_) {
      char* p = reinterpret_cast<char*>(dst);
      while (n > 0) {
        ssize_t r = read(fd_, p, n);
        if (r <= 0) break;
        p += r;
        n -= r;
      }
    } else {
      std::memcpy(dst, ptr_, n);
      ptr_ += n;
    }
  }
  template <typename T>
  T Get() {
    T v;
    Read(&v, sizeof(T));
    return v;
  }
  void ReleaseBuffer() { buffer_.reset(); }

 private:
  int fd_;
  bool direct_;
  std::unique_ptr<char[]> buffer_;
  const char* ptr_ = nullptr;
};

template <typename ModularInt>
void WritePoly(StateWriter& w, const rlwe::RnsPolynomial<ModularInt>& poly) {
  auto coeffs = poly.Coeffs();
  w.Put<size_t>(coeffs.size());
  for (const auto& vec : coeffs) {
    w.Put<size_t>(vec.size());
    w.Write(vec.data(), vec.size() * sizeof(ModularInt));
  }
}

template <typename ModularInt, typename Context>
rlwe::RnsPolynomial<ModularInt> ReadPoly(StateReader& r, const Context* ctx) {
  size_t num_vecs = r.Get<size_t>();
  std::vector<std::vector<ModularInt>> coeffs(num_vecs);
  for (size_t v = 0; v < num_vecs; ++v) {
    size_t n = r.Get<size_t>();
    auto zero =
        ModularInt::ImportZero(ctx->MainPrimeModuli()[v]->ModParams());
    coeffs[v].resize(n, zero);
    r.Read(coeffs[v].data(), n * sizeof(ModularInt));
  }
  return rlwe::RnsPolynomial<ModularInt>::Create(std::move(coeffs), true)
      .value();
}

// Option 2b: writes an NTT-form RNS polynomial whose coefficients are small
// integers (|v| < 2^31, identical across RNS limbs) as int32 coefficients.
template <typename ModularInt>
void WriteSmallPoly(StateWriter& w, const rlwe::RnsPolynomial<ModularInt>& poly,
                    absl::Span<const rlwe::PrimeModulus<ModularInt>* const> moduli) {
  rlwe::RnsPolynomial<ModularInt> c = poly;
  if (c.IsNttForm()) c.ConvertToCoeffForm(moduli).IgnoreError();
  auto coeffs = c.Coeffs();
  const auto* p0 = moduli[0]->ModParams();
  const auto q0 = p0->modulus;
  std::vector<int32_t> vals(coeffs[0].size());
  for (size_t j = 0; j < vals.size(); ++j) {
    auto x = coeffs[0][j].ExportInt(p0);
    int64_t v = x > q0 / 2 ? static_cast<int64_t>(x) - static_cast<int64_t>(q0)
                           : static_cast<int64_t>(x);
    if (v >= (int64_t{1} << 31) || v < -(int64_t{1} << 31)) {
      std::cerr << "WriteSmallPoly: coefficient too large" << std::endl;
      std::abort();
    }
    // Sanity check: the same small integer in every RNS limb.
    for (size_t i = 1; i < coeffs.size(); ++i) {
      const auto* pi = moduli[i]->ModParams();
      int64_t qi = static_cast<int64_t>(pi->modulus);
      auto expect = static_cast<uint64_t>(v < 0 ? qi + v : v);
      if (coeffs[i][j].ExportInt(pi) != expect) {
        std::cerr << "WriteSmallPoly: limbs disagree" << std::endl;
        std::abort();
      }
    }
    vals[j] = static_cast<int32_t>(v);
  }
  w.Put<size_t>(vals.size());
  w.Write(vals.data(), vals.size() * sizeof(int32_t));
}

template <typename ModularInt>
rlwe::RnsPolynomial<ModularInt> ReadSmallPoly(
    StateReader& r,
    absl::Span<const rlwe::PrimeModulus<ModularInt>* const> moduli) {
  size_t n = r.Get<size_t>();
  std::vector<int32_t> vals(n);
  r.Read(vals.data(), n * sizeof(int32_t));
  std::vector<std::vector<ModularInt>> coeffs(moduli.size());
  for (size_t i = 0; i < moduli.size(); ++i) {
    const auto* pi = moduli[i]->ModParams();
    int64_t qi = static_cast<int64_t>(pi->modulus);
    coeffs[i].reserve(n);
    for (size_t j = 0; j < n; ++j) {
      int64_t v = vals[j];
      coeffs[i].push_back(
          ModularInt::ImportInt(static_cast<uint64_t>(v < 0 ? qi + v : v), pi)
              .value());
    }
  }
  auto poly =
      rlwe::RnsPolynomial<ModularInt>::Create(std::move(coeffs), false).value();
  poly.ConvertToNttForm(moduli).IgnoreError();
  return poly;
}

template <typename T>
void WriteMatrix(StateWriter& w, const std::vector<std::vector<T>>& m) {
  size_t rows = m.size();
  size_t cols = m.empty() ? 0 : m[0].size();
  w.Put(rows);
  w.Put(cols);
  for (const auto& row : m) w.Write(row.data(), cols * sizeof(T));
}

template <typename T>
std::vector<std::vector<T>> ReadMatrix(StateReader& r) {
  size_t rows = r.Get<size_t>();
  size_t cols = r.Get<size_t>();
  std::vector<std::vector<T>> m(rows);
  for (auto& row : m) {
    row.resize(cols);
    r.Read(row.data(), cols * sizeof(T));
  }
  return m;
}

}  // namespace

// File layout (sections present depend on the SSD options):
//   data matrices (always)
//   [LWE query pad]            unless ssd_seed_state_in_memory
//   [hint matrices]            unless (ssd_seed_state_in_memory && !compact)
//   per LinPIR context / shard: [BFV diagonals] unless compact, pad products
void Server::DumpStateToDisk() {
  std::ofstream out(kStateFile, std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "Failed to open " << kStateFile << " for writing!" << std::endl;
    return;
  }
  StateWriter w(&out);
  const bool opt1 = params_.ssd_seed_state_in_memory;
  const bool opt2 = params_.ssd_compact_linpir;

  w.Put<size_t>(database_->Data().size());
  for (const auto& matrix : database_->Data()) WriteMatrix(w, matrix);

  if (!opt1) {
    size_t rows = lwe_query_pad_->rows();
    size_t cols = lwe_query_pad_->cols();
    w.Put(rows);
    w.Put(cols);
    w.Write(lwe_query_pad_->data(), rows * cols * sizeof(lwe::Integer));
  }

  if (!opt1 || opt2) {
    w.Put<size_t>(database_->Hints().size());
    for (const auto& hint : database_->Hints()) WriteMatrix(w, hint);
  }

  w.Put<size_t>(linpir_databases_.size());
  for (size_t k = 0; k < linpir_databases_.size(); ++k) {
    w.Put<size_t>(linpir_databases_[k].size());
    for (const auto& db : linpir_databases_[k]) {
      if (!opt2) {
        const auto& diagonals = db->Diagonals();
        w.Put<size_t>(diagonals.size());
        auto moduli = rlwe_contexts_[k]->MainPrimeModuli();
        for (const auto& block : diagonals) {
          w.Put<size_t>(block.size());
          for (const auto& poly : block) {
            if (params_.ssd_compact_linpir_coeff) {
              WriteSmallPoly<RlweModularInt>(w, poly, moduli);
            } else {
              WritePoly(w, poly);
            }
          }
        }
      }
      const auto& pads = db->PadInnerProducts();
      w.Put<size_t>(pads.size());
      for (const auto& poly : pads) WritePoly(w, poly);
    }
  }

  size_t total_size = out.tellp();
  out.close();

  // Evict the file from the page cache so loading reads from the SSD.
  int fd = open(kStateFile, O_RDWR);
  if (fd != -1) {
    fdatasync(fd);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
  }
  std::cout << "Server Long term Storage: " << total_size << " bytes"
            << std::endl;
  std::cout << "SSD options: seed_state_in_memory=" << opt1
            << " compact_linpir=" << opt2
            << " compact_linpir_coeff=" << params_.ssd_compact_linpir_coeff
            << " direct_read=" << params_.ssd_direct_read << std::endl;
  if (opt1) {
    size_t seed_state_bytes = static_cast<size_t>(lwe_query_pad_->rows()) *
                              lwe_query_pad_->cols() * sizeof(lwe::Integer);
    for (const auto& s : linpir_servers_) {
      seed_state_bytes += s->PreprocessedStateBytes();
    }
    std::cout << "Server Seed-Derived State In Memory: " << seed_state_bytes
              << " bytes" << std::endl;
  }
}

void Server::LoadStateFromDisk() {
  const bool opt1 = params_.ssd_seed_state_in_memory;
  const bool opt2 = params_.ssd_compact_linpir;
  const bool opt3 = params_.ssd_direct_read;

  int fd = open(kStateFile, O_RDONLY);
  if (fd == -1) {
    std::cerr << "Failed to open " << kStateFile << " for reading!" << std::endl;
    return;
  }
  off_t size = lseek(fd, 0, SEEK_END);
  lseek(fd, 0, SEEK_SET);

  auto start = std::chrono::high_resolution_clock::now();
  StateReader r(fd, size, opt3);
  int64_t read_us = opt3 ? 0 : MicrosSince(start);
  auto start_parse = std::chrono::high_resolution_clock::now();

  // Data matrices.
  size_t num_data = r.Get<size_t>();
  std::vector<std::vector<std::vector<absl::uint128>>> data(num_data);
  for (auto& m : data) m = ReadMatrix<absl::uint128>(r);
  database_->SetData(std::move(data));

  // LWE query pad.
  if (!opt1) {
    size_t rows = r.Get<size_t>();
    size_t cols = r.Get<size_t>();
    lwe::Matrix pad(rows, cols);
    r.Read(pad.data(), rows * cols * sizeof(lwe::Integer));
    lwe_query_pad_ = std::make_unique<const lwe::Matrix>(std::move(pad));
  }

  // Hint matrices.
  if (!opt1 || opt2) {
    size_t num_hints = r.Get<size_t>();
    std::vector<Database::LweMatrix> hints(num_hints);
    for (auto& h : hints) h = ReadMatrix<lwe::Integer>(r);
    database_->SetHints(std::move(hints));
  }

  // LinPIR databases.
  int64_t derive_us = 0;
  size_t num_contexts = r.Get<size_t>();
  linpir_databases_.clear();
  linpir_databases_.resize(num_contexts);
  if (!opt1) {
    linpir_servers_.clear();
    linpir_servers_.resize(num_contexts);
  }
  RlweInteger lwe_modulus = RlweInteger{1} << params_.lwe_modulus_bit_size;
  for (size_t k = 0; k < num_contexts; ++k) {
    size_t num_shards = r.Get<size_t>();
    linpir_databases_[k].reserve(num_shards);
    for (size_t s = 0; s < num_shards; ++s) {
      std::unique_ptr<LinPirDatabase> db;
      if (opt2) {
        // Re-encode the BFV diagonals from the 32-bit hint.
        auto start_derive = std::chrono::high_resolution_clock::now();
        RlweInteger t = rlwe_contexts_[k]->PlaintextModulus();
        auto hint_mod_t =
            EncodeLweMatrix(database_->Hints()[s], lwe_modulus, t);
        db = LinPirDatabase::Create(params_.linpir_params,
                                    rlwe_contexts_[k].get(), hint_mod_t)
                 .value();
        derive_us += MicrosSince(start_derive);
      } else {
        db = LinPirDatabase::CreateEmpty(rlwe_contexts_[k].get()).value();
        size_t num_blocks = r.Get<size_t>();
        std::vector<std::vector<rlwe::RnsPolynomial<RlweModularInt>>> diags(
            num_blocks);
        auto moduli = rlwe_contexts_[k]->MainPrimeModuli();
        for (auto& block : diags) {
          size_t num_polys = r.Get<size_t>();
          block.reserve(num_polys);
          for (size_t p = 0; p < num_polys; ++p) {
            if (params_.ssd_compact_linpir_coeff) {
              auto start_derive = std::chrono::high_resolution_clock::now();
              block.push_back(ReadSmallPoly<RlweModularInt>(r, moduli));
              derive_us += MicrosSince(start_derive);
            } else {
              block.push_back(
                  ReadPoly<RlweModularInt>(r, rlwe_contexts_[k].get()));
            }
          }
        }
        db->SetDiagonals(std::move(diags));
      }
      size_t num_pads = r.Get<size_t>();
      std::vector<rlwe::RnsPolynomial<RlweModularInt>> pads;
      pads.reserve(num_pads);
      for (size_t p = 0; p < num_pads; ++p) {
        pads.push_back(ReadPoly<RlweModularInt>(r, rlwe_contexts_[k].get()));
      }
      db->SetPadInnerProducts(std::move(pads));
      linpir_databases_[k].push_back(std::move(db));
    }

    std::vector<LinPirDatabase*> ptrs;
    for (auto& p : linpir_databases_[k]) ptrs.push_back(p.get());
    if (opt1) {
      // The seed-derived LinPIR server state is resident; just attach.
      linpir_servers_[k]->SetDatabases(std::move(ptrs));
    } else {
      linpir_servers_[k] =
          LinPirServer::Create(params_.linpir_params, rlwe_contexts_[k].get(),
                               ptrs, prng_seed_linpir_ct_pads_[k],
                               prng_seed_linpir_gk_pad_)
              .value();
    }
  }
  close(fd);
  r.ReleaseBuffer();
  int64_t parse_us = MicrosSince(start_parse) - derive_us;

  // Without option 1 the LinPIR server's seed-derived pads were not stored
  // and must be recomputed before answering.
  int64_t rehydrate_us = 0;
  if (!opt1) {
    auto start_rehydrate = std::chrono::high_resolution_clock::now();
    for (size_t k = 0; k < num_contexts; ++k) {
      linpir_servers_[k]->Preprocess().IgnoreError();
    }
    rehydrate_us = MicrosSince(start_rehydrate);
  }
  int64_t total_us = MicrosSince(start);

  if (opt3) {
    std::cout << "Server Load Read+Parse Time: " << parse_us << " us"
              << std::endl;
  } else {
    std::cout << "Server Load Read Time: " << read_us << " us" << std::endl;
    std::cout << "Server Load Parse Time: " << parse_us << " us" << std::endl;
  }
  std::cout << "Server Load Derive Time: " << derive_us << " us" << std::endl;
  std::cout << "Server Load Rehydrate Time: " << rehydrate_us << " us"
            << std::endl;
  std::cout << "Server Total Load Time: " << total_us << " us" << std::endl;
}

absl::Status Server::Preprocess() {
  // Refresh the PRNG seeds.
  RLWE_RETURN_IF_ERROR(GeneratePublicParams());

  // Make sure the hint is up to date.
  RLWE_RETURN_IF_ERROR(database_->UpdateLweQueryPad(lwe_query_pad_.get()));
  RLWE_RETURN_IF_ERROR(database_->UpdateHints());

  RlweInteger lwe_modulus = RlweInteger{1} << params_.lwe_modulus_bit_size;
  size_t num_shards = database_->NumShards();

  // Create LinPir databases (holding the preprocessed hints) and servers.
  linpir_servers_.clear();
  linpir_servers_.resize(rlwe_contexts_.size());
  linpir_databases_.clear();
  linpir_databases_.resize(rlwe_contexts_.size());
  for (int k = 0; k < rlwe_contexts_.size(); ++k) {
    RlweInteger plaintext_modulus = rlwe_contexts_[k]->PlaintextModulus();

    // One LinPir database per shard, for the current plaintext modulus.
    std::vector<std::unique_ptr<LinPirDatabase>> linpir_databases_mod_tk;
    linpir_databases_mod_tk.reserve(num_shards);
    for (const Database::LweMatrix& hint : database_->Hints()) {
      std::vector<std::vector<RlweInteger>> hint_mod_tk =
          EncodeLweMatrix(hint, lwe_modulus, plaintext_modulus);
      RLWE_ASSIGN_OR_RETURN(
          auto linpir_database,
          LinPirDatabase::Create(params_.linpir_params, rlwe_contexts_[k].get(),
                                 hint_mod_tk));
      linpir_databases_mod_tk.push_back(std::move(linpir_database));
    }
    std::vector<LinPirDatabase*> linpir_databases_ptrs;
    std::transform(linpir_databases_mod_tk.begin(),
                   linpir_databases_mod_tk.end(),
                   std::back_inserter(linpir_databases_ptrs),
                   [](auto& ptr) { return ptr.get(); });
    RLWE_ASSIGN_OR_RETURN(
        auto linpir_server_mod_tk,
        LinPirServer::Create(params_.linpir_params, rlwe_contexts_[k].get(),
                             linpir_databases_ptrs,
                             prng_seed_linpir_ct_pads_[k],
                             prng_seed_linpir_gk_pad_));
    RLWE_RETURN_IF_ERROR(linpir_server_mod_tk->Preprocess());

    linpir_databases_[k] = std::move(linpir_databases_mod_tk);
    linpir_servers_[k] = std::move(linpir_server_mod_tk);
  }


  // Dump state to disk if requested
  if (params_.write_to_file) {
    DumpStateToDisk();
  
    // Clear memory to force loading from disk on HandleRequest. With option
    // 1, the seed-derived state (LWE query pad, LinPIR server pads) stays
    // resident; it does not depend on the database.
    linpir_databases_.clear();
    if (!params_.ssd_seed_state_in_memory) {
      linpir_servers_.clear();
      lwe_query_pad_.reset();
    }
    database_->SetHints({});
    database_->SetData({});
  }
  
  return absl::OkStatus();
}

absl::StatusOr<HintlessPirResponse> Server::HandleRequest(
    const HintlessPirRequest& request) {
  auto start = std::chrono::high_resolution_clock::now();
  if (params_.write_to_file && (linpir_databases_.empty() || linpir_databases_[0].empty())) {
      LoadStateFromDisk();
  }

  if (!IsPreprocessed()) {
    return absl::FailedPreconditionError("Server has not been preprocessed.");
  }
  HintlessPirResponse response;
  // Handle the LWE part of the request.
  Database::LweVector ct_query_vector =
      DeserializeLweCiphertext(request.ct_query_vector());
  RLWE_ASSIGN_OR_RETURN(std::vector<Database::LweVector> ct_records,
                        database_->InnerProductWith(ct_query_vector));
  for (auto& ct_record : ct_records) {
    *response.add_ct_records() = SerializeLweCiphertext(ct_record);
  }

  // Handle the LinPIR requests.
  int num_linpir_requests = request.linpir_ct_bs_size();
  if (num_linpir_requests != linpir_servers_.size()) {
    return absl::InvalidArgumentError(
        "`request` contains unexpected number of LinPir requests.");
  }
  for (int k = 0; k < num_linpir_requests; ++k) {
    RLWE_ASSIGN_OR_RETURN(LinPirResponse linpir_response,
                          linpir_servers_[k]->HandleRequest(
                              request.linpir_ct_bs(k), request.linpir_gk_bs()));
    *response.add_linpir_responses() = std::move(linpir_response);
  }


  auto latency_end = std::chrono::high_resolution_clock::now();
  auto latency_duration = std::chrono::duration_cast<std::chrono::microseconds>(latency_end - start).count();
  std::cout << "Server Response Latency: " << latency_duration << " us" << std::endl;
  return response;
}

HintlessPirServerPublicParams Server::GetPublicParams() const {
  HintlessPirServerPublicParams output;
  output.set_prng_seed_lwe_query_pad(prng_seed_lwe_query_pad_);
  for (auto const& prng_seed : prng_seed_linpir_ct_pads_) {
    *output.add_prng_seed_linpir_ct_pads() = prng_seed;
  }
  output.set_prng_seed_linpir_gk_pad(prng_seed_linpir_gk_pad_);
  return output;
}

}  // namespace hintless_simplepir
}  // namespace hintless_pir
