/*
 * Copyright 2024 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HINTLESS_PIR_HINTLESS_SIMPLEPIR_PARAMETERS_H_
#define HINTLESS_PIR_HINTLESS_SIMPLEPIR_PARAMETERS_H_

#include <cstdint>

#include "linpir/parameters.h"
#include "lwe/types.h"
#include "shell_encryption/serialization.pb.h"

namespace hintless_pir {
namespace hintless_simplepir {

// Parameters of the hintless SimplePIR protocol.
struct Parameters {
  using LweInteger = lwe::Integer;
  using RlweInteger = linpir::Uint64;

  int64_t db_rows;
  int64_t db_cols;
  int db_record_bit_size;

  int lwe_secret_dim;
  int lwe_modulus_bit_size;  // 32 or 64
  int lwe_plaintext_bit_size;
  double lwe_error_variance;

  linpir::RlweParameters<RlweInteger> linpir_params;

  rlwe::PrngType prng_type;
  
  bool write_to_file;

  // SSD-mode options (only used when write_to_file is true).
  // (1) Keep the seed-derived, database-independent state (LWE query pad and
  //     the LinPIR server's expanded pads / gadget digits) resident in memory
  //     instead of storing it per database, and do not store the raw hint
  //     matrix, which is not used to answer queries.
  bool ssd_seed_state_in_memory = true;
  // (2) Store the LinPIR database compactly: the 32-bit hint matrix once,
  //     instead of the BFV-encoded diagonals (2 RNS limbs x 64 bits per slot,
  //     per plaintext modulus); the diagonals are re-encoded after loading.
  bool ssd_compact_linpir = false;
  // (2b) Alternative compact LinPIR storage: keep the BFV diagonals but store
  //     each one in coefficient form as small signed 32-bit integers (one
  //     value per coefficient instead of 2 RNS limbs x 64 bits); the NTT is
  //     redone after loading.
  bool ssd_compact_linpir_coeff = false;
  // (3) Read directly into the destination buffers instead of reading the
  //     whole file into a staging buffer and copying.
  bool ssd_direct_read = false;
};

}  // namespace hintless_simplepir
}  // namespace hintless_pir

#endif  // HINTLESS_PIR_HINTLESS_SIMPLEPIR_PARAMETERS_H_
