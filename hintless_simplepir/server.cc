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


void Server::DumpStateToDisk() {
    std::ofstream out("/tmp/hintless_db.bin", std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open /tmp/hintless_db.bin for writing!" << std::endl;
        return;
    }

    // 0. Write database_->Data()
    size_t num_shards = database_->Data().size();
    out.write(reinterpret_cast<const char*>(&num_shards), sizeof(num_shards));
    for (const auto& matrix : database_->Data()) {
        size_t rows = matrix.size();
        size_t cols = matrix.empty() ? 0 : matrix[0].size();
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        for (const auto& row : matrix) {
            out.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(absl::uint128));
        }
    }
    
    // 1. Write lwe_query_pad_
    size_t rows = lwe_query_pad_->rows();
    size_t cols = lwe_query_pad_->cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char*>(lwe_query_pad_->data()), rows * cols * sizeof(lwe::Integer));

    // 2. Write database_->hints_
    size_t num_hints = database_->Hints().size();
    out.write(reinterpret_cast<const char*>(&num_hints), sizeof(num_hints));
    for (const auto& hint : database_->Hints()) {
        rows = hint.size();
        cols = hint.empty() ? 0 : hint[0].size();
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        for (const auto& row : hint) {
            out.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(lwe::Integer));
        }
    }
    
    // 3. Write linpir_databases_
    size_t num_contexts = linpir_databases_.size();
    out.write(reinterpret_cast<const char*>(&num_contexts), sizeof(num_contexts));
    for (size_t k = 0; k < num_contexts; ++k) {
        size_t num_shards = linpir_databases_[k].size();
        out.write(reinterpret_cast<const char*>(&num_shards), sizeof(num_shards));
        for (size_t s = 0; s < num_shards; ++s) {
            auto& db = linpir_databases_[k][s];
            
            // Diagonals
            const auto& diagonals = db->Diagonals();
            size_t num_blocks = diagonals.size();
            out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
            for (const auto& block : diagonals) {
                size_t num_polys = block.size();
                out.write(reinterpret_cast<const char*>(&num_polys), sizeof(num_polys));
                for (const auto& poly : block) {
                    auto coeffs = poly.Coeffs();
                    size_t num_coeffs_vecs = coeffs.size();
                    out.write(reinterpret_cast<const char*>(&num_coeffs_vecs), sizeof(num_coeffs_vecs));
                    for (const auto& vec : coeffs) {
                        size_t vec_size = vec.size();
                        out.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
                        out.write(reinterpret_cast<const char*>(vec.data()), vec_size * sizeof(RlweModularInt));
                    }
                }
            }
            
            // Pad Inner Products
            const auto& pads = db->PadInnerProducts();
            size_t num_pads = pads.size();
            out.write(reinterpret_cast<const char*>(&num_pads), sizeof(num_pads));
            for (const auto& poly : pads) {
                auto coeffs = poly.Coeffs();
                size_t num_coeffs_vecs = coeffs.size();
                out.write(reinterpret_cast<const char*>(&num_coeffs_vecs), sizeof(num_coeffs_vecs));
                for (const auto& vec : coeffs) {
                    size_t vec_size = vec.size();
                    out.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
                    out.write(reinterpret_cast<const char*>(vec.data()), vec_size * sizeof(RlweModularInt));
                }
            }
        }
    }
    
    size_t total_size = out.tellp();
    out.close();

    // Cache eviction via POSIX APIs
    int fd = open("/tmp/hintless_db.bin", O_RDWR);
    if (fd != -1) {
        fdatasync(fd);
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
    }
    
    std::cout << "Server Long term Storage: " << total_size << " bytes" << std::endl;
}

void Server::LoadStateFromDisk() {
    int fd = open("/tmp/hintless_db.bin", O_RDONLY);
    if (fd == -1) {
        std::cerr << "Failed to open /tmp/hintless_db.bin for reading!" << std::endl;
        return;
    }
    off_t size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    
    auto start_load = std::chrono::high_resolution_clock::now();
    
    char* buffer = new char[size];
    char* write_ptr = buffer;
    size_t bytes_left = size;
    while (bytes_left > 0) {
        ssize_t bytes_read = read(fd, write_ptr, bytes_left);
        if (bytes_read <= 0) break;
        write_ptr += bytes_read;
        bytes_left -= bytes_read;
    }
    close(fd);
    
    auto end_load = std::chrono::high_resolution_clock::now();
    auto duration_load = std::chrono::duration_cast<std::chrono::microseconds>(end_load - start_load).count();
    std::cout << "Server Data Load Time: " << duration_load << " us" << std::endl;
    
    const char* ptr = buffer;
    
    // 0. Read database_->Data()
    size_t num_data_matrices;
    std::memcpy(&num_data_matrices, ptr, sizeof(num_data_matrices)); ptr += sizeof(num_data_matrices);
    std::vector<std::vector<std::vector<absl::uint128>>> data(num_data_matrices);
    for (size_t i = 0; i < num_data_matrices; ++i) {
        size_t rows, cols;
        std::memcpy(&rows, ptr, sizeof(rows)); ptr += sizeof(rows);
        std::memcpy(&cols, ptr, sizeof(cols)); ptr += sizeof(cols);
        data[i].resize(rows);
        for (size_t r = 0; r < rows; ++r) {
            data[i][r].resize(cols);
            std::memcpy(data[i][r].data(), ptr, cols * sizeof(absl::uint128)); ptr += cols * sizeof(absl::uint128);
        }
    }
    database_->SetData(std::move(data));
    
    // 1. Read lwe_query_pad_
    size_t rows, cols;
    std::memcpy(&rows, ptr, sizeof(rows)); ptr += sizeof(rows);
    std::memcpy(&cols, ptr, sizeof(cols)); ptr += sizeof(cols);
    lwe::Matrix pad(rows, cols);
    std::memcpy(pad.data(), ptr, rows * cols * sizeof(lwe::Integer)); ptr += rows * cols * sizeof(lwe::Integer);
    lwe_query_pad_ = std::make_unique<const lwe::Matrix>(std::move(pad));
    
    // 2. Read database_->hints_
    size_t num_hints;
    std::memcpy(&num_hints, ptr, sizeof(num_hints)); ptr += sizeof(num_hints);
    std::vector<Database::LweMatrix> hints(num_hints);
    for (size_t i = 0; i < num_hints; ++i) {
        std::memcpy(&rows, ptr, sizeof(rows)); ptr += sizeof(rows);
        std::memcpy(&cols, ptr, sizeof(cols)); ptr += sizeof(cols);
        hints[i].resize(rows);
        for (size_t r = 0; r < rows; ++r) {
            hints[i][r].resize(cols);
            std::memcpy(hints[i][r].data(), ptr, cols * sizeof(lwe::Integer)); ptr += cols * sizeof(lwe::Integer);
        }
    }
    database_->SetHints(std::move(hints));
    
    // 3. Read linpir_databases_
    size_t num_contexts;
    std::memcpy(&num_contexts, ptr, sizeof(num_contexts)); ptr += sizeof(num_contexts);
    linpir_databases_.clear();
    linpir_databases_.resize(num_contexts);
    linpir_servers_.clear();
    linpir_servers_.resize(num_contexts);
    for (size_t k = 0; k < num_contexts; ++k) {
        size_t num_shards;
        std::memcpy(&num_shards, ptr, sizeof(num_shards)); ptr += sizeof(num_shards);
        linpir_databases_[k].reserve(num_shards);
        for (size_t s = 0; s < num_shards; ++s) {
            auto db_empty = LinPirDatabase::CreateEmpty(rlwe_contexts_[k].get()).value();
            
            // Diagonals
            size_t num_blocks;
            std::memcpy(&num_blocks, ptr, sizeof(num_blocks)); ptr += sizeof(num_blocks);
            std::vector<std::vector<rlwe::RnsPolynomial<RlweModularInt>>> diagonals(num_blocks);
            for (size_t b = 0; b < num_blocks; ++b) {
                size_t num_polys;
                std::memcpy(&num_polys, ptr, sizeof(num_polys)); ptr += sizeof(num_polys);
                diagonals[b].reserve(num_polys);
                for (size_t p = 0; p < num_polys; ++p) {
                    size_t num_coeffs_vecs;
                    std::memcpy(&num_coeffs_vecs, ptr, sizeof(num_coeffs_vecs)); ptr += sizeof(num_coeffs_vecs);
                    std::vector<std::vector<RlweModularInt>> coeffs(num_coeffs_vecs);
                    for (size_t v = 0; v < num_coeffs_vecs; ++v) {
                        size_t vec_size;
                        std::memcpy(&vec_size, ptr, sizeof(vec_size)); ptr += sizeof(vec_size);
                        auto dummy = RlweModularInt::ImportZero(rlwe_contexts_[k]->MainPrimeModuli()[v]->ModParams()); coeffs[v].resize(vec_size, dummy);
                        std::memcpy(coeffs[v].data(), ptr, vec_size * sizeof(RlweModularInt)); ptr += vec_size * sizeof(RlweModularInt);
                    }
                    diagonals[b].push_back(rlwe::RnsPolynomial<RlweModularInt>::Create(std::move(coeffs), true).value());
                }
            }
            db_empty->SetDiagonals(std::move(diagonals));
            
            // Pad Inner Products
            size_t num_pads;
            std::memcpy(&num_pads, ptr, sizeof(num_pads)); ptr += sizeof(num_pads);
            std::vector<rlwe::RnsPolynomial<RlweModularInt>> pads;
            pads.reserve(num_pads);
            for (size_t p = 0; p < num_pads; ++p) {
                size_t num_coeffs_vecs;
                std::memcpy(&num_coeffs_vecs, ptr, sizeof(num_coeffs_vecs)); ptr += sizeof(num_coeffs_vecs);
                std::vector<std::vector<RlweModularInt>> coeffs(num_coeffs_vecs);
                for (size_t v = 0; v < num_coeffs_vecs; ++v) {
                    size_t vec_size;
                    std::memcpy(&vec_size, ptr, sizeof(vec_size)); ptr += sizeof(vec_size);
                    auto dummy = RlweModularInt::ImportZero(rlwe_contexts_[k]->MainPrimeModuli()[v]->ModParams()); coeffs[v].resize(vec_size, dummy);
                    std::memcpy(coeffs[v].data(), ptr, vec_size * sizeof(RlweModularInt)); ptr += vec_size * sizeof(RlweModularInt);
                }
                pads.push_back(rlwe::RnsPolynomial<RlweModularInt>::Create(std::move(coeffs), true).value());
            }
            db_empty->SetPadInnerProducts(std::move(pads));
            
            linpir_databases_[k].push_back(std::move(db_empty));
        }
        
        // Re-create server
        std::vector<LinPirDatabase*> linpir_databases_ptrs;
        std::transform(linpir_databases_[k].begin(),
                       linpir_databases_[k].end(),
                       std::back_inserter(linpir_databases_ptrs),
                       [](auto& p) { return p.get(); });
        linpir_servers_[k] = LinPirServer::Create(params_.linpir_params, rlwe_contexts_[k].get(),
                             linpir_databases_ptrs,
                             prng_seed_linpir_ct_pads_[k],
                             prng_seed_linpir_gk_pad_).value();
        // Set preprocessed flag on LinPirServer by calling Preprocess? 
        // Wait, LinPirServer::Preprocess() creates random polynomials.
        // We actually only need the linpir server to HandleRequest.
        // Let's call Preprocess, but we must override or just avoid it.
        // Actually, LinPirServer::Preprocess() does NOT modify Database. It just sets up server's own state.
        // Wait, we don't save LinPirServer's internal state!
    }
    
    delete[] buffer;

    // Time spent deserializing the buffer into in-memory structures (after the
    // raw read above). Reported separately so it can be attributed to loading.
    auto end_parse = std::chrono::high_resolution_clock::now();
    auto duration_parse = std::chrono::duration_cast<std::chrono::microseconds>(end_parse - end_load).count();
    std::cout << "Server Data Parse Time: " << duration_parse << " us" << std::endl;
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
  
    // Clear memory to force loading from disk on HandleRequest
    linpir_databases_.clear();
    linpir_servers_.clear();
    lwe_query_pad_.reset();
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
      // LinPIR server state (expanded pads / gadget digits) is not persisted,
      // so it is recomputed here. Report its cost separately.
      auto start_rehydrate = std::chrono::high_resolution_clock::now();
      for (int k = 0; k < rlwe_contexts_.size(); ++k) {
          linpir_servers_[k]->Preprocess().IgnoreError();
      }
      auto end_rehydrate = std::chrono::high_resolution_clock::now();
      std::cout << "Server LinPIR Rehydrate Time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_rehydrate - start_rehydrate).count()
                << " us" << std::endl;
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
