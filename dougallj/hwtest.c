#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "amx.h"
#include "simulator.h"

static int check_state(struct amx_state *sim_state, int flags) {
  __attribute__((aligned(0x80))) static struct amx_state state;

  store_amx_state(&state);
  if (memcmp(sim_state, &state, sizeof state)) {
    printf("state mismatch!\n");
    printf("real state:\n");
    diff_amx_state(&state, sim_state, flags | PF_SKIP_B);
    printf("simulated state:\n");
    diff_amx_state(&state, sim_state, flags | PF_SKIP_A);
    return 0;
  }
  return 1;
}

bool is_prime(uint64_t v) {
  if (v < 2) {
    return false;
  }

  for (uint64_t i = 2; i * i <= v; i++) {
    if (v % i == 0) {
      return false;
    }
  }

  return true;
}

static void init_test_f32s(float *data) {
  for (uint64_t v = 1000, o = 0; o < (8 + 8) * 16; v++) {
    if (is_prime(v)) {
      data[o++] = v;
    }
  }

  int i = 1;
  for (uint64_t o = (8 + 8) * 16; o < (8 + 8 + 64) * 16; o++) {
    data[o] = i++;
  }
}

static void test_start(struct amx_state *sim_state) {
  amx_state_zero(sim_state);
  AMX_START();
}

static void test_stop(struct amx_state *sim_state) { AMX_STOP(); }

// LDX/LDY/LDZ/LDZI

static void test_ldx(struct amx_state *sim_state, uint64_t op) {
  amx_state_ldx(sim_state, op);
  AMX_LDX(op);
  check_state(sim_state, PF_F32);
}

static void test_ldy(struct amx_state *sim_state, uint64_t op) {
  amx_state_ldy(sim_state, op);
  AMX_LDY(op);
  check_state(sim_state, PF_F32);
}

static void test_ldz(struct amx_state *sim_state, uint64_t op) {
  amx_state_ldz(sim_state, op);
  AMX_LDZ(op);
  check_state(sim_state, PF_F32);
}

static void test_ldzi(struct amx_state *sim_state, uint64_t op) {
  amx_state_ldzi(sim_state, op);
  AMX_LDZI(op);
  check_state(sim_state, PF_F32);
}

static void test_loads(void) {
  __attribute__((aligned(0x80))) static float test_data[(8 + 8 + 64) * 16];

  init_test_f32s(test_data);

  struct amx_state sim_state;

  // TODO: test alignment checks somehow

  for (int test_bit = 56 - 1; test_bit < 64; test_bit++) {
    uint64_t extra_bit = (test_bit == 55 ? 0 : (1ull << test_bit));

    for (uint64_t reg = 0; reg < 8; reg++) {
      test_start(&sim_state);
      test_ldx(&sim_state, (uint64_t)test_data | (reg << 56) | extra_bit);
      test_stop(&sim_state);
    }

    for (uint64_t reg = 0; reg < 8; reg++) {
      test_start(&sim_state);
      test_ldy(&sim_state, (uint64_t)test_data | (reg << 56) | extra_bit);
      test_stop(&sim_state);
    }

    for (uint64_t reg = 0; reg < 64; reg++) {
      test_start(&sim_state);
      test_ldz(&sim_state, (uint64_t)test_data | (reg << 56) | extra_bit);
      test_stop(&sim_state);
    }
    for (uint64_t reg = 0; reg < 64; reg++) {
      test_start(&sim_state);
      test_ldzi(&sim_state, (uint64_t)test_data | (reg << 56) | extra_bit);
      test_stop(&sim_state);
    }
  }
}

// STX/STY/STZ/STZI

static void test_stores(void) {
  struct amx_state sim_state;
  static float test_data[(8 + 8 + 64) * 16];
  init_test_f32s(test_data);

  __attribute__((aligned(0x80))) static float store_buffer1[0x100];
  __attribute__((aligned(0x80))) static float store_buffer2[0x100];

  test_start(&sim_state);
  memcpy(&sim_state, test_data, sizeof sim_state);
  load_amx_state(&sim_state);

  for (int test_bit = 56 - 1; test_bit < 64; test_bit++) {
    uint64_t extra_bit = (test_bit == 55 ? 0 : (1ull << test_bit));

    for (uint64_t reg = 0; reg < 8; reg++) {
      memset(store_buffer1, 0, sizeof store_buffer1);
      memset(store_buffer2, 0, sizeof store_buffer2);

      AMX_STX((uint64_t)store_buffer1 | (reg << 56) | extra_bit);
      amx_state_stx(&sim_state,
                    (uint64_t)store_buffer2 | (reg << 56) | extra_bit);

      if (memcmp(store_buffer1, store_buffer2, sizeof store_buffer1)) {
        printf("store test failed!\n");
      }
    }

    for (uint64_t reg = 0; reg < 8; reg++) {
      memset(store_buffer1, 0, sizeof store_buffer1);
      memset(store_buffer2, 0, sizeof store_buffer2);

      AMX_STY((uint64_t)store_buffer1 | (reg << 56) | extra_bit);
      amx_state_sty(&sim_state,
                    (uint64_t)store_buffer2 | (reg << 56) | extra_bit);

      if (memcmp(store_buffer1, store_buffer2, sizeof store_buffer1)) {
        printf("store test failed!\n");
      }
    }

    for (uint64_t reg = 0; reg < 64; reg++) {
      memset(store_buffer1, 0, sizeof store_buffer1);
      memset(store_buffer2, 0, sizeof store_buffer2);

      AMX_STZ((uint64_t)store_buffer1 | (reg << 56) | extra_bit);
      amx_state_stz(&sim_state,
                    (uint64_t)store_buffer2 | (reg << 56) | extra_bit);

      if (memcmp(store_buffer1, store_buffer2, sizeof store_buffer1)) {
        printf("store test failed!\n");
      }
    }

    for (uint64_t reg = 0; reg < 64; reg++) {
      memset(store_buffer1, 0, sizeof store_buffer1);
      memset(store_buffer2, 0, sizeof store_buffer2);

      AMX_STZI((uint64_t)store_buffer1 | (reg << 56) | extra_bit);
      amx_state_stzi(&sim_state,
                     (uint64_t)store_buffer2 | (reg << 56) | extra_bit);

      if (memcmp(store_buffer1, store_buffer2, sizeof store_buffer1)) {
        printf("store test failed! 1\n");
      }
    }
  }
  test_stop(&sim_state);
}

// EXTRX

static void extrx_test(void *initial_state, uint64_t operand) {
  struct amx_state sim_state;

  test_start(&sim_state);

  memcpy(&sim_state, initial_state, sizeof sim_state);
  load_amx_state(&sim_state);
  check_state(&sim_state, PF_F32);

  AMX_EXTRX(operand);
  amx_state_extrx(&sim_state, operand);
  check_state(&sim_state, PF_U32);

  test_stop(&sim_state);
}

void test_extrx(void) {
  uint32_t initial_state[(8 + 8 + 64) * 16];
  for (int i = 0; i < 8 * 16; i++) {
    initial_state[i] = 0xAA0000 | i;
  }
  for (int i = 8 * 16; i < (8 + 8) * 16; i++) {
    initial_state[i] = 0xBB00000 | i;
  }
  for (int i = (8 + 8) * 16; i < (8 + 8 + 64) * 16; i++) {
    initial_state[i] = 0xCC000000 | i;
  }

  // TODO: bit 26
  uint64_t mask = 0x800000001ff7fdc0 & ~(1 << 26);

  // for perf, skip some offsets
  mask &= ~(0x1FFull << 10);
  mask |= 0x147ull << 10;

  uint64_t operand = 0;
  do {
    extrx_test(initial_state, operand);

    // ryg's texture tiling and swizzling loop
    operand = (operand - mask) & mask;
  } while (operand);

#define __amx_op8_to_x(o) extrx_test(initial_state, o)

  // bit 26 test cases:
  __amx_op8_to_x(0x4000000);
  __amx_op8_to_x(0x4800000);
  __amx_op8_to_x(0x5000000);
  __amx_op8_to_x(0x5800000);
  __amx_op8_to_x(0x6000000);
  __amx_op8_to_x(0x6800000);
  __amx_op8_to_x(0x7000000);
  __amx_op8_to_x(0x7800000);

  // some failing bit 26 test cases:
  //__amx_op8_to_x(0x4110000);
  //__amx_op8_to_x(0x4910000);
  //__amx_op8_to_x(0x5110000);
  //__amx_op8_to_x(0x5910000);
  //__amx_op8_to_x(0x6110000);
  //__amx_op8_to_x(0x6910000);
  //__amx_op8_to_x(0x7110000);
  //__amx_op8_to_x(0x7910000);
  //__amx_op8_to_x(0x8000000004004500);
  //__amx_op8_to_x(0x8000000004204000);
  //__amx_op8_to_x(0x8000000004304000);
  //__amx_op8_to_x(0x8000000004604000);
  //__amx_op8_to_x(0x8000000004704000);
  //__amx_op8_to_x(0x8000000004804000);
  //__amx_op8_to_x(0x8000000004a04000);
  //__amx_op8_to_x(0x8000000004b04000);
  //__amx_op8_to_x(0x8000000004e04000);
  //__amx_op8_to_x(0x8000000004f04000);
  //__amx_op8_to_x(0x8000000005004040);
  //__amx_op8_to_x(0x8000000005204000);
  //__amx_op8_to_x(0x8000000005304000);
  //__amx_op8_to_x(0x8000000005604000);
  //__amx_op8_to_x(0x8000000005704000);
  //__amx_op8_to_x(0x8000000005804080);
  //__amx_op8_to_x(0x8000000005a04000);
  //__amx_op8_to_x(0x8000000005b04000);
  //__amx_op8_to_x(0x8000000005e04000);
  //__amx_op8_to_x(0x8000000005f04000);
  //__amx_op8_to_x(0x8000000006004540);
  //__amx_op8_to_x(0x8000000006204040);
  //__amx_op8_to_x(0x8000000006304040);
  //__amx_op8_to_x(0x8000000006604040);
  //__amx_op8_to_x(0x8000000006704040);
  //__amx_op8_to_x(0x8000000006804040);
  //__amx_op8_to_x(0x8000000006a04040);
  //__amx_op8_to_x(0x8000000006b04040);
  //__amx_op8_to_x(0x8000000006e04040);
  //__amx_op8_to_x(0x8000000006f04040);
  //__amx_op8_to_x(0x8000000007204040);
  //__amx_op8_to_x(0x8000000007304040);
  //__amx_op8_to_x(0x8000000007604040);
  //__amx_op8_to_x(0x8000000007704040);
  //__amx_op8_to_x(0x8000000007a04040);
  //__amx_op8_to_x(0x8000000007b04040);
  //__amx_op8_to_x(0x8000000007e04040);
  //__amx_op8_to_x(0x8000000007f04000);
  //__amx_op8_to_x(0x8000000007f04040);

  // TODO: commented out things (and the rest, as some probably interact with
  // other bits)
  for (int i = 0; i < 41; i++) {
    extrx_test(initial_state, (1ull << i));
  }
  // extrx_test(initial_state, (1ull << 41));
  // extrx_test(initial_state, (1ull << 42));
  // extrx_test(initial_state, (1ull << 43));
  // extrx_test(initial_state, (1ull << 44));
  // extrx_test(initial_state, (1ull << 45));
  // extrx_test(initial_state, (1ull << 46));
  for (int i = 47; i < 64; i++) {
    extrx_test(initial_state, (1ull << i));
  }
}

// EXTRY

static void extry_test(void *initial_state, uint64_t operand) {
  struct amx_state sim_state;

  test_start(&sim_state);

  memcpy(&sim_state, initial_state, sizeof sim_state);
  load_amx_state(&sim_state);
  check_state(&sim_state, PF_F32);

  AMX_EXTRY(operand);
  amx_state_extry(&sim_state, operand);
  check_state(&sim_state, PF_U32);

  test_stop(&sim_state);
}

void test_extry(void) {
  uint32_t initial_state[(8 + 8 + 64) * 16];
  for (int i = 0; i < 8 * 16; i++) {
    initial_state[i] = 0xAA0000 | i;
  }
  for (int i = 8 * 16; i < (8 + 8) * 16; i++) {
    initial_state[i] = 0xBB00000 | i;
  }
  for (int i = (8 + 8) * 16; i < (8 + 8 + 64) * 16; i++) {
    initial_state[i] = 0xCC000000 | i;
  }

  // still can't handle (1ull << 14) and some higher bits

  // uint64_t mask = (1ull << 63) | (1ull << 29) | (1ull << 28) | (1ull << 27) |
  // (1ull << 26) | (1ull << 13) | (1ull << 12) | (1ull << 11) | (1ull << 10) |
  // 0x40 | 0x20 | 1 | (63<<20);
  uint64_t mask = (0x800000003ff04dc0 & ~(1ull << 14)) | 1;

  uint64_t operand = 0;
  do {
    extry_test(initial_state, operand);
    operand = (operand - mask) & mask;
  } while (operand);

  // bit 14 tests that already pass:
  extry_test(initial_state, 0x8000000004004000);
  extry_test(initial_state, 0x8000000004104000);
  extry_test(initial_state, 0x8000000004104040);
  extry_test(initial_state, 0x8000000004204000);
  extry_test(initial_state, 0x8000000004304000);
  extry_test(initial_state, 0x8000000004404040);
  extry_test(initial_state, 0x8000000004404100);
  extry_test(initial_state, 0x8000000004504000);
  extry_test(initial_state, 0x8000000004604000);
  extry_test(initial_state, 0x8000000004704040);
  extry_test(initial_state, 0x8000000004804080);
  extry_test(initial_state, 0x8000000004804400);
  extry_test(initial_state, 0x8000000004c040c0);
  extry_test(initial_state, 0x8000000004c04500);
  extry_test(initial_state, 0x8000000005004040);
  extry_test(initial_state, 0x8000000005004100);
  extry_test(initial_state, 0x8000000005404140);
  extry_test(initial_state, 0x8000000005804180);
  extry_test(initial_state, 0x8000000005804440);
  extry_test(initial_state, 0x8000000005c041c0);
  extry_test(initial_state, 0x8000000005c04540);
  extry_test(initial_state, 0x8000000006004080);
  extry_test(initial_state, 0x8000000006004400);
  extry_test(initial_state, 0x8000000006404180);
  extry_test(initial_state, 0x8000000006404440);
  extry_test(initial_state, 0x8000000006804480);
  extry_test(initial_state, 0x8000000006c044c0);
  extry_test(initial_state, 0x8000000006c04580);
  extry_test(initial_state, 0x80000000070040c0);
  extry_test(initial_state, 0x8000000007004500);
  extry_test(initial_state, 0x80000000074041c0);
  extry_test(initial_state, 0x8000000007404540);
  extry_test(initial_state, 0x80000000078044c0);
  extry_test(initial_state, 0x8000000007804580);
  extry_test(initial_state, 0x8000000007c045c0);
}

// FMA32

static void fma32_test(void *initial_state, uint64_t operand) {
  struct amx_state sim_state;

  test_start(&sim_state);

  memcpy(&sim_state, initial_state, sizeof sim_state);
  load_amx_state(&sim_state);
  // check_state(&sim_state, PF_F32);

  AMX_FMA32(operand);
  amx_state_fma32(&sim_state, operand);
  check_state(&sim_state, PF_F32);

  test_stop(&sim_state);
}

static void fms32_test(void *initial_state, uint64_t operand) {
  struct amx_state sim_state;

  test_start(&sim_state);

  memcpy(&sim_state, initial_state, sizeof sim_state);
  load_amx_state(&sim_state);
  // check_state(&sim_state, PF_F32);

  AMX_FMS32(operand);
  amx_state_fms32(&sim_state, operand);
  check_state(&sim_state, PF_F32);

  test_stop(&sim_state);
}

static void test_fma32_fms32(void) {
  static float test_data[(8 + 8 + 64) * 16];
  init_test_f32s(test_data);

  // slow, but passes
  // TODO: fields at bit 32/41/60
  // uint64_t mask = 0xa000da4f3bf709f8 & ~((0x3Full << 32) | (0x3Full << 41) |
  // (0x3ull << 60));

  uint64_t offset_mask = 0x101; // optimised - thorough is 0x1FF
  uint64_t mask = (1ull << 63) | (1ull << 27) | (1ull << 28) | (1ull << 29) |
                  (63 << 20) | (offset_mask << 10) | offset_mask;
  uint64_t operand = 0;
  do {
    fma32_test(test_data, operand);

    // TODO: this is almost working, but when we clear we have problems with
    // negative zero
    fms32_test(test_data, operand & ~(1ull << 27));

    operand = (operand - mask) & mask;
  } while (operand);

  // TODO: commented out things (and the rest, as some probably interact with
  // other bits)
  for (int i = 0; i < 32; i++) {
    fma32_test(test_data, (1ull << i));
  }
  // fma32_test(test_data, (1ull << 32));
  // fma32_test(test_data, (1ull << 33));
  // fma32_test(test_data, (1ull << 34));
  // fma32_test(test_data, (1ull << 35));
  // fma32_test(test_data, (1ull << 36));
  // fma32_test(test_data, (1ull << 37));
  fma32_test(test_data, (1ull << 38));
  fma32_test(test_data, (1ull << 39));
  fma32_test(test_data, (1ull << 40));
  // fma32_test(test_data, (1ull << 41));
  // fma32_test(test_data, (1ull << 42));
  // fma32_test(test_data, (1ull << 43));
  // fma32_test(test_data, (1ull << 44));
  // fma32_test(test_data, (1ull << 45));
  // fma32_test(test_data, (1ull << 46));
  fma32_test(test_data, (1ull << 47));
  fma32_test(test_data, (1ull << 48));
  fma32_test(test_data, (1ull << 49));
  fma32_test(test_data, (1ull << 50));
  fma32_test(test_data, (1ull << 51));
  fma32_test(test_data, (1ull << 52));
  fma32_test(test_data, (1ull << 53));
  fma32_test(test_data, (1ull << 54));
  fma32_test(test_data, (1ull << 55));
  fma32_test(test_data, (1ull << 56));
  fma32_test(test_data, (1ull << 57));
  fma32_test(test_data, (1ull << 58));
  fma32_test(test_data, (1ull << 59));
  // fma32_test(test_data, (1ull << 60));
  // fma32_test(test_data, (1ull << 61));
  fma32_test(test_data, (1ull << 62));
  fma32_test(test_data, (1ull << 63));
}

// FMA64/FMS64

static void init_test_f64s(double *test) {

  for (uint64_t v = 1000, o = 0; o < (8 + 8) * 8; v++) {
    if (is_prime(v)) {
      test[o++] = v;
    }
  }

  for (uint64_t o = (8 + 8) * 8; o < (8 + 8 + 64) * 8; o++) {
    test[o] = 1.0f;
  }
}

static void fma64_test(void *initial_state, uint64_t operand) {
  struct amx_state sim_state;

  test_start(&sim_state);

  memcpy(&sim_state, initial_state, sizeof sim_state);
  load_amx_state(&sim_state);

  AMX_FMA64(operand);
  amx_state_fma64(&sim_state, operand);
  check_state(&sim_state, PF_F64);

  test_stop(&sim_state);
}

static void fms64_test(void *initial_state, uint64_t operand) {
  struct amx_state sim_state;

  test_start(&sim_state);

  memcpy(&sim_state, initial_state, sizeof sim_state);
  load_amx_state(&sim_state);

  AMX_FMS64(operand);
  amx_state_fms64(&sim_state, operand);
  check_state(&sim_state, PF_F64);

  test_stop(&sim_state);
}

static void test_fma64_fms64(void) {
  static double test_data[(8 + 8 + 64) * 8];

  init_test_f64s(test_data);

  fma64_test(test_data, 0);

  // slow, but passes
  // uint64_t mask = 0xa000da4f3bf709f8 & ~((0x3Full << 32) | (0x3Full << 41));

  uint64_t offset_mask = 0x101; // optimised - thorough is 0x1FF
  uint64_t mask = (1ull << 63) | (1ull << 27) | (1ull << 28) | (1ull << 29) |
                  (63 << 20) | (offset_mask << 10) | offset_mask;
  uint64_t operand = 0;
  do {
    fma64_test(test_data, operand);

    // TODO: this is almost working, but when we clear we have problems with
    // negative zero
    fms64_test(test_data, operand & ~(1ull << 27));

    // ryg's texture tiling and swizzling loop
    operand = (operand - mask) & mask;
  } while (operand);
}

// FMA16/FMS16

static void fma16_test(void *initial_state, uint64_t operand) {
  struct amx_state sim_state;

  test_start(&sim_state);

  memcpy(&sim_state, initial_state, sizeof sim_state);
  load_amx_state(&sim_state);

  AMX_FMA16(operand);
  amx_state_fma16(&sim_state, operand);
  if (!check_state(&sim_state, PF_U16)) {
    // check_state(&sim_state, PF_F16);
    check_state(&sim_state, PF_F32);
    printf("^ %llx\n\n", operand);
  }

  test_stop(&sim_state);
}

static void fms16_test(void *initial_state, uint64_t operand) {
  struct amx_state sim_state;

  test_start(&sim_state);

  memcpy(&sim_state, initial_state, sizeof sim_state);
  load_amx_state(&sim_state);

  AMX_FMS16(operand);
  amx_state_fms16(&sim_state, operand);
  if (!check_state(&sim_state, PF_U16)) {
    // check_state(&sim_state, PF_F16);
    check_state(&sim_state, PF_F32);
    printf("^ %llx\n\n", operand);
  }

  test_stop(&sim_state);
}

static void init_test_f16s(float16 *test) {

  for (uint64_t v = 5, o = 0; o < (8 + 8) * 32; v++) {
    if (is_prime(v)) {
      test[o++] = v;
    }
  }

  for (uint64_t o = (8 + 8) * 32; o < (8 + 8 + 64) * 32; o++) {
    test[o] = 1.0f;
  }
}

static void test_fma16_fms16(void) {
  // TODO: There's something wrong with NaN accuracy - probably in the others as
  // well but it shows up here if we let offset_mask be odd.

  static float16 test_data[(8 + 8 + 64) * 32];

  init_test_f16s(test_data);

  uint64_t offset_mask = 0x102; // optimised - thorough is 0x1FF
  uint64_t mask = (1ull << 63) | (1ull << 62) | (1ull << 27) | (1ull << 28) |
                  (1ull << 29) | (63 << 20) | (offset_mask << 10) | offset_mask;
  uint64_t operand = 0;
  do {
    fma16_test(test_data, operand);

    // TODO: this is almost working, but when we clear we have problems with
    // negative zero
    fms16_test(test_data, operand & ~(1ull << 27));

    // ryg's texture tiling and swizzling loop
    operand = (operand - mask) & mask;
  } while (operand);
}

// MAC16

static void mac16_test(void *initial_state, uint64_t operand) {
  struct amx_state sim_state;

  test_start(&sim_state);
  memcpy(&sim_state, initial_state, sizeof sim_state);
  load_amx_state(&sim_state);

  AMX_MAC16(operand);
  amx_state_mac16(&sim_state, operand);
  if (!check_state(&sim_state, PF_U16)) {
    printf("^ %llx\n\n", operand);
  }

  test_stop(&sim_state);
}

static void init_test_u16s(uint16_t *test) {

  for (uint64_t v = 0x8000 - 100, o = 0; o < (8 + 8) * 32; v++) {
    if (is_prime(v)) {
      test[o++] = v;
    }
  }

  for (uint64_t o = (8 + 8) * 32; o < (8 + 8 + 64) * 32; o++) {
    test[o] = 1;
  }
}

static void test_mac16(void) {
  static uint16_t test_data[(8 + 8 + 64) * 32];

  init_test_u16s(test_data);

  uint64_t offset_mask = 0x101; // optimised - thorough is 0x1FF
  uint64_t mask = (1ull << 63) | (1ull << 62) | (1ull << 27) | (1ull << 28) |
                  (1ull << 29) | (63 << 20) | (offset_mask << 10) | offset_mask;
  uint64_t operand = 0;
  do {
    mac16_test(test_data, operand);
    operand = (operand - mask) & mask;
  } while (operand);
}

int main(int argc, char **argv) {
  // TODO: test stores
  test_loads();
  test_stores();
  test_mac16();
  test_fma16_fms16();
  test_fma32_fms32();
  test_fma64_fms64();
  test_extrx();
  test_extry();

  return 0;
}
