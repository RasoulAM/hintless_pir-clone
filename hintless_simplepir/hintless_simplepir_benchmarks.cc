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

#include <cstdint>
#include <cmath>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "hintless_simplepir/client.h"
#include "hintless_simplepir/database_hwy.h"
#include "hintless_simplepir/parameters.h"
#include "hintless_simplepir/server.h"
#include "linpir/parameters.h"
#include "shell_encryption/testing/status_testing.h"

ABSL_FLAG(int, num_rows, 1024, "Number of rows");
ABSL_FLAG(int, num_cols, 1024, "Number of cols");
ABSL_FLAG(int, db_record_bit_size, 8, "Record Size in bits");
ABSL_FLAG(bool, write_to_file, false, "Write preprocessing material to file");
ABSL_FLAG(bool, ssd_seed_state_in_memory, true,
          "SSD option 1: keep seed-derived state in memory, store only data "
          "used online.");
ABSL_FLAG(bool, ssd_compact_linpir, false,
          "SSD option 2: store the 32-bit hint once instead of BFV diagonals.");
ABSL_FLAG(bool, ssd_compact_linpir_coeff, false,
          "SSD option 2b: store BFV diagonals as int32 coefficients, NTT on "
          "load.");
ABSL_FLAG(bool, ssd_direct_read, false,
          "SSD option 3: read directly into destination buffers.");

namespace hintless_pir {
namespace hintless_simplepir {
namespace {

using RlweInteger = Parameters::RlweInteger;

const Parameters kParameters{
    .db_rows = 1024,
    .db_cols = 1024,
    .db_record_bit_size = 8,
    .lwe_secret_dim = 1400,
    .lwe_modulus_bit_size = 32,
    .lwe_plaintext_bit_size = 8,
    .lwe_error_variance = 8,
    .linpir_params =
        linpir::RlweParameters<RlweInteger>{
            .log_n = 12,
            .qs = {35184371884033ULL, 35184371703809ULL},  // 90 bits
            .ts = {2056193, 1990657},                      // 42 bits
            .gadget_log_bs = {16, 16},
            .error_variance = 8,
            .prng_type = rlwe::PRNG_TYPE_HKDF,
            .rows_per_block = 1024,
        },
    .prng_type = rlwe::PRNG_TYPE_HKDF,
    .write_to_file = false,
};

void BM_HintlessPirPreprocess(benchmark::State& state) {
  int64_t num_rows = absl::GetFlag(FLAGS_num_rows);
  int64_t num_cols = absl::GetFlag(FLAGS_num_cols);
  int64_t db_record_bit_size = absl::GetFlag(FLAGS_db_record_bit_size);

  // std::cout << "db_size_mb: " << num_rows * num_cols * db_record_bit_size / (8 * 1024 * 1024) << std::endl;

  // std::cout << "num_rows: " << num_rows << std::endl;
  // std::cout << "num_cols: " << num_cols << std::endl;

  Parameters params = kParameters;
  params.db_rows = num_rows;
  params.db_cols = num_cols;
  params.db_record_bit_size = db_record_bit_size;
  params.write_to_file = absl::GetFlag(FLAGS_write_to_file);
  params.ssd_seed_state_in_memory = absl::GetFlag(FLAGS_ssd_seed_state_in_memory);
  params.ssd_compact_linpir = absl::GetFlag(FLAGS_ssd_compact_linpir);
  params.ssd_direct_read = absl::GetFlag(FLAGS_ssd_direct_read);
  params.ssd_compact_linpir_coeff =
      absl::GetFlag(FLAGS_ssd_compact_linpir_coeff);

  // Create server and fill in random database records.
  auto server = Server::CreateWithRandomDatabaseRecords(params).value();
  const Database* database = server->GetDatabase();
  ASSERT_EQ(database->NumRecords(), num_rows * num_cols);

  // Preprocess the server and get public parameters.
  for (auto _ : state) {
    server->Preprocess();
  }

}

void BM_HintlessPirRlwe64(benchmark::State& state) {
  int64_t num_rows = absl::GetFlag(FLAGS_num_rows);
  int64_t num_cols = absl::GetFlag(FLAGS_num_cols);
  int64_t db_record_bit_size = absl::GetFlag(FLAGS_db_record_bit_size);

  // std::cout << "db_size_mb: " << num_rows * num_cols * db_record_bit_size / (8 * 1024 * 1024) << std::endl;

  // std::cout << "num_rows: " << num_rows << std::endl;
  // std::cout << "num_cols: " << num_cols << std::endl;

  Parameters params = kParameters;
  params.db_rows = num_rows;
  params.db_cols = num_cols;
  params.db_record_bit_size = db_record_bit_size;
  params.write_to_file = absl::GetFlag(FLAGS_write_to_file);
  params.ssd_seed_state_in_memory = absl::GetFlag(FLAGS_ssd_seed_state_in_memory);
  params.ssd_compact_linpir = absl::GetFlag(FLAGS_ssd_compact_linpir);
  params.ssd_direct_read = absl::GetFlag(FLAGS_ssd_direct_read);
  params.ssd_compact_linpir_coeff =
      absl::GetFlag(FLAGS_ssd_compact_linpir_coeff);

  std::cout << "num_rows: " << num_rows
            << ", num_cols: " << num_cols << std::endl;

  std::cout << "Database Size: "
            << (num_rows * num_cols * params.db_record_bit_size) / (8 * 1024 * 1024)
            << " MB" << std::endl;

  // Server Long term Storage calculation
  int64_t n = 1LL << params.linpir_params.log_n;
  int64_t N = params.lwe_secret_dim;
  double ks_bits = 0;
  for (size_t i = 0; i < params.linpir_params.qs.size(); ++i) {
    int64_t q_bits = std::ceil(std::log2(params.linpir_params.qs[i]));
    int64_t ell = std::ceil(static_cast<double>(q_bits) / params.linpir_params.gadget_log_bs[i]);
    ks_bits += n * N * (ell + 1) * q_bits;
  }
  double hint_matrix_bits = static_cast<double>(params.db_rows) * N * params.lwe_modulus_bit_size;
  double server_storage_bytes = (ks_bits + hint_matrix_bits) / 8.0;

  std::cout << "Server Preprocsesed Size Ideal (B): " << server_storage_bytes << std::endl;

  // Create server and fill in random database records.
  auto server = Server::CreateWithRandomDatabaseRecords(params).value();
  const Database* database = server->GetDatabase();
  ASSERT_EQ(database->NumRecords(), num_rows * num_cols);
  // Record the expected value before preprocessing: in SSD mode the in-memory
  // database is freed after it is written to disk and later reloaded, so this
  // also checks the write/read round trip.
  const std::string expected_before_ssd = database->Record(1).value();

  // Preprocess the server and get public parameters.
  ASSERT_OK(server->Preprocess());
  auto public_params = server->GetPublicParams();

  // Create a client and issue request.
  auto client = Client::Create(params, public_params).value();

  absl::Time gen_start = absl::Now();
  auto request = client->GenerateRequest(1).value();
  absl::Time gen_end = absl::Now();
  double gen_ms = absl::ToDoubleMilliseconds(gen_end - gen_start);
  std::cout << "Client Generate Request Time: " << gen_ms << " ms" << std::endl;

  for (auto _ : state) {
    auto response = server->HandleRequest(request);
    benchmark::DoNotOptimize(response);
  }

  // Sanity check on the correctness of the instantiation.
  auto response = server->HandleRequest(request).value();

  absl::Time rec_start = absl::Now();
  std::string record = client->RecoverRecord(response).value();
  absl::Time rec_end = absl::Now();
  double rec_ms = absl::ToDoubleMilliseconds(rec_end - rec_start);
  std::cout << "Client Extract Response Time: " << rec_ms << " ms" << std::endl;

  std::string expected = database->Record(1).value();
  ASSERT_EQ(record, expected);
  ASSERT_EQ(record, expected_before_ssd);
  std::cout << "Correctness check: PASSED (retrieved record matches the database)"
            << std::endl;
}

BENCHMARK(BM_HintlessPirPreprocess);
BENCHMARK(BM_HintlessPirRlwe64);

}  // namespace
}  // namespace hintless_simplepir
}  // namespace hintless_pir

// Declare benchmark_filter flag, which will be defined by benchmark library.
// Use it to check if any benchmarks were specified explicitly.
//
namespace benchmark {
extern std::string FLAGS_benchmark_filter;
}
using benchmark::FLAGS_benchmark_filter;

int main(int argc, char* argv[]) {
  FLAGS_benchmark_filter = "";
  benchmark::Initialize(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  int64_t num_rows = absl::GetFlag(FLAGS_num_rows);
  int64_t num_cols = absl::GetFlag(FLAGS_num_cols);
  int64_t db_record_bit_size = absl::GetFlag(FLAGS_db_record_bit_size);

  std::cout << "db_size_mb: " << num_rows * num_cols * db_record_bit_size / (8 * 1024 * 1024) << std::endl;

  if (!FLAGS_benchmark_filter.empty()) {
    benchmark::RunSpecifiedBenchmarks();
  }
  benchmark::Shutdown();
  return 0;
}
