#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "amx.h"

#ifdef __aarch64__

double fma64(double a, double b, double c) {
  double out;
  asm("fmadd %d[out], %d[a], %d[b], %d[c]"
      : [out] "=w"(out)
      : [a] "w"(a), [b] "w"(b), [c] "w"(c));
  return out;
}

float fma32(float a, float b, float c) {
  float out;
  asm("fmadd %s[out], %s[a], %s[b], %s[c]"
      : [out] "=w"(out)
      : [a] "w"(a), [b] "w"(b), [c] "w"(c));
  return out;
}

float fms32(float a, float b, float c) {
  float out;
  asm("fmsub %s[out], %s[a], %s[b], %s[c]"
      : [out] "=w"(out)
      : [a] "w"(a), [b] "w"(b), [c] "w"(c));
  return out;
}

float16 fma16(float16 a, float16 b, float16 c) {
  float16 out;
  asm("fmadd %h[out], %h[a], %h[b], %h[c]"
      : [out] "=w"(out)
      : [a] "w"(a), [b] "w"(b), [c] "w"(c));
  return out;
}

#else

#error "TODO: portable fma64/fma32/fma16 implementations"

float fma32(float a, float b, float c) {
#pragma clang fp contract(fast)
  return a * b + c;
}

double fma64(double a, double b, double c) {
#pragma clang fp contract(fast)
  return a * b + c;
}

#endif

uint16_t mac16(uint16_t a, uint16_t b, uint16_t c) {
  return (uint16_t)(((int64_t)a * (int64_t)b + (int64_t)c) & 0xFFFF);
}

void amx_state_zero(struct amx_state *state) {
  memset(state, 0, sizeof *state);
}

#define LDST_ADDRESS_MASK ((1ull << 56) - 1)
#define LDST_DOUBLE_WIDTH (1ull << 62)

static void amx_state_load_impl(union amx_row *rows, int mask,
                                uint64_t operand) {
  uint64_t double_width = (operand & LDST_DOUBLE_WIDTH);
  if (double_width && (operand & 0x7F) != 0) {
    // TODO: some way to test exceptions
    printf("error: bad alignment\n");
  }

  char *addr = (char *)(operand & LDST_ADDRESS_MASK);
  uint64_t reg = (operand >> 56) & mask;

  memcpy(&rows[reg], addr, 0x40);
  if (double_width) {
    memcpy(&rows[(reg + 1) & mask], addr + 0x40, 0x40);
  }
}

static void amx_state_store_impl(union amx_row *rows, int mask,
                                 uint64_t operand) {
  uint64_t double_width = (operand & LDST_DOUBLE_WIDTH);
  if (double_width && (operand & 0x7F) != 0) {
    // TODO: some way to test exceptions
    printf("error: bad alignment\n");
  }

  char *addr = (char *)(operand & LDST_ADDRESS_MASK);
  uint64_t reg = (operand >> 56) & mask;

  memcpy(addr, &rows[reg], 0x40);
  if (double_width) {
    memcpy(addr + 0x40, &rows[(reg + 1) & mask], 0x40);
  }
}

void amx_state_ldx(struct amx_state *state, uint64_t operand) {
  amx_state_load_impl(state->x, 7, operand);
}

void amx_state_ldy(struct amx_state *state, uint64_t operand) {
  amx_state_load_impl(state->y, 7, operand);
}

void amx_state_ldz(struct amx_state *state, uint64_t operand) {
  amx_state_load_impl(state->z, 0x3F, operand);
}

void amx_state_stx(struct amx_state *state, uint64_t operand) {
  amx_state_store_impl(state->x, 7, operand);
}

void amx_state_sty(struct amx_state *state, uint64_t operand) {
  amx_state_store_impl(state->y, 7, operand);
}

void amx_state_stz(struct amx_state *state, uint64_t operand) {
  amx_state_store_impl(state->z, 0x3F, operand);
}

void amx_state_ldzi(struct amx_state *state, uint64_t operand) {
  char *addr = (char *)(operand & LDST_ADDRESS_MASK);
  uint32_t row[16];
  memcpy(row, addr, sizeof row);

  uint64_t reg = (operand >> 56) & 0x3F;
  for (int i = 0; i < 16; i++) {
    state->z[(reg & ~1) + (i & 1)].u32[((reg & 1) << 3) + (i >> 1)] = row[i];
  }
}

void amx_state_stzi(struct amx_state *state, uint64_t operand) {
  uint64_t reg = (operand >> 56) & 0x3F;
  char *addr = (char *)(operand & LDST_ADDRESS_MASK);
  uint32_t row[16];
  for (int i = 0; i < 16; i++) {
    row[i] = state->z[(reg & ~1) + (i & 1)].u32[((reg & 1) << 3) + (i >> 1)];
  }
  memcpy(addr, row, sizeof row);
}

static void load_from_x(void *output, struct amx_state *state, size_t offset,
                        size_t size) {
  char *p = (char *)output;
  for (size_t i = 0; i < size; i++) {
    memcpy(p++, ((char *)&state->x) + ((offset + i) & 0x1FF), 1);
  }
}

static void load_from_y(void *output, struct amx_state *state, size_t offset,
                        size_t size) {
  char *p = (char *)output;
  for (size_t i = 0; i < size; i++) {
    memcpy(p++, ((char *)&state->y) + ((offset + i) & 0x1FF), 1);
  }
}

static void load_from_z(void *output, struct amx_state *state, size_t offset,
                        size_t size) {
  char *p = (char *)output;
  for (size_t i = 0; i < size; i++) {
    memcpy(p++, ((char *)&state->z) + ((offset + i) & 0xFFF), 1);
  }
}

static void store_to_x(void *output, struct amx_state *state, size_t offset,
                       size_t size) {
  char *p = (char *)output;
  for (size_t i = 0; i < size; i++) {
    memcpy(((char *)&state->x) + ((offset + i) & 0x1FF), p++, 1);
  }
}

static void store_to_y(void *output, struct amx_state *state, size_t offset,
                       size_t size) {
  char *p = (char *)output;
  for (size_t i = 0; i < size; i++) {
    memcpy(((char *)&state->y) + ((offset + i) & 0x1FF), p++, 1);
  }
}

#define FMA_SKIP_Z_INPUT (1ull << 27)
#define FMA_SKIP_Y_INPUT (1ull << 28)
#define FMA_SKIP_X_INPUT (1ull << 29)

static void amx_state_fmas32_impl(struct amx_state *state, uint64_t operand,
                                  bool sub) {
  float x[16];
  float y[16];

  // TODO: need to wrap around byte offsets
  uint64_t y_offset = operand & 0x1FF;
  uint64_t x_offset = (operand >> 10) & 0x1FF;
  uint64_t z_offset = (operand >> 20) & 63;

  assert((sizeof x) == 0x40);

  if ((operand & FMA_SKIP_Y_INPUT) && (operand & FMA_SKIP_X_INPUT)) {
    memset(&x, 0, sizeof x);
    memset(&y, 0, sizeof y);
  } else {
    if (operand & FMA_SKIP_X_INPUT) {
      for (int i = 0; i < 16; i++) {
        x[i] = 1.0;
      }
    } else {
      load_from_x(x, state, x_offset, sizeof x);
    }
    if (operand & FMA_SKIP_Y_INPUT) {
      for (int i = 0; i < 16; i++) {
        y[i] = 1.0;
      }
    } else {
      load_from_y(y, state, y_offset, sizeof y);
    }
  }

  float sub_mul = sub ? -1.0 : 1.0;

  if (operand & (1ull << 63)) {
    for (int i = 0; i < 16; i++) {
      float *z = &state->z[z_offset].f32[i];
      *z =
          fma32(sub_mul * x[i], y[i], (operand & FMA_SKIP_Z_INPUT) ? 0.0f : *z);
    }
  } else {
    z_offset &= 3;
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
        float *z = &state->z[(j * 4) + z_offset].f32[i];
        *z = fma32(sub_mul * x[i], y[j],
                   (operand & FMA_SKIP_Z_INPUT) ? 0.0f : *z);
      }
    }
  }
}

void amx_state_fma32(struct amx_state *state, uint64_t operand) {
  amx_state_fmas32_impl(state, operand, false);
}

void amx_state_fms32(struct amx_state *state, uint64_t operand) {
  amx_state_fmas32_impl(state, operand, true);
}

static void amx_state_fmas64_impl(struct amx_state *state, uint64_t operand,
                                  bool sub) {
  double x[8];
  double y[8];

  // TODO: need to wrap around byte offsets
  uint64_t y_offset = operand & 0x1FF;
  uint64_t x_offset = (operand >> 10) & 0x1FF;
  uint64_t z_offset = (operand >> 20) & 63;

  assert((sizeof x) == 0x40);

  if ((operand & FMA_SKIP_Y_INPUT) && (operand & FMA_SKIP_X_INPUT)) {
    memset(&x, 0, sizeof x);
    memset(&y, 0, sizeof y);
  } else {
    if (operand & FMA_SKIP_X_INPUT) {
      for (int i = 0; i < 8; i++) {
        x[i] = 1.0;
      }
    } else {
      load_from_x(x, state, x_offset, sizeof x);
    }
    if (operand & FMA_SKIP_Y_INPUT) {
      for (int i = 0; i < 8; i++) {
        y[i] = 1.0;
      }
    } else {
      load_from_y(y, state, y_offset, sizeof y);
    }
  }

  double sub_mul = sub ? -1.0 : 1.0;

  if (operand & (1ull << 63)) {
    for (int i = 0; i < 8; i++) {
      double *z = &state->z[z_offset].f64[i];
      *z =
          fma64(sub_mul * x[i], y[i], (operand & FMA_SKIP_Z_INPUT) ? 0.0f : *z);
    }
  } else {
    z_offset &= 7;
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        double *z = &state->z[(j * 8) + z_offset].f64[i];
        *z = fma64(sub_mul * x[i], y[j],
                   (operand & FMA_SKIP_Z_INPUT) ? 0.0f : *z);
      }
    }
  }
}

void amx_state_fma64(struct amx_state *state, uint64_t operand) {
  amx_state_fmas64_impl(state, operand, false);
}

void amx_state_fms64(struct amx_state *state, uint64_t operand) {
  amx_state_fmas64_impl(state, operand, true);
}

static void amx_state_fmas16_impl(struct amx_state *state, uint64_t operand,
                                  bool sub) {
  float16 x[32];
  float16 y[32];

  // TODO: need to wrap around byte offsets
  uint64_t y_offset = operand & 0x1FF;
  uint64_t x_offset = (operand >> 10) & 0x1FF;
  uint64_t z_offset = (operand >> 20) & 63;

  assert((sizeof x) == 0x40);

  if ((operand & FMA_SKIP_Y_INPUT) && (operand & FMA_SKIP_X_INPUT)) {
    memset(&x, 0, sizeof x);
    memset(&y, 0, sizeof y);
  } else {
    if (operand & FMA_SKIP_X_INPUT) {
      for (int i = 0; i < 32; i++) {
        x[i] = (float16)1.0;
      }
    } else {
      load_from_x(x, state, x_offset, sizeof x);
    }
    if (operand & FMA_SKIP_Y_INPUT) {
      for (int i = 0; i < 32; i++) {
        y[i] = (float16)1.0;
      }
    } else {
      load_from_y(y, state, y_offset, sizeof y);
    }
  }

  float16 sub_mul = sub ? -1.0 : 1.0;

  if (operand & (1ull << 63)) {
    for (int i = 0; i < 32; i++) {
      float16 *z = &state->z[z_offset].f16[i];
      *z = fma16(sub_mul * x[i], y[i],
                 (operand & FMA_SKIP_Z_INPUT) ? (float16)0.0f : *z);
    }
  } else {
    z_offset &= 1;
    for (int i = 0; i < 32; i++) {
      for (int j = 0; j < 32; j++) {
        if (operand & (1ull << 62)) {
          float *z = &state->z[(j * 2) + (i & 1)].f32[i >> 1];
          float acc = (operand & FMA_SKIP_Z_INPUT) ? 0 : *z;
          acc = fma32((float)sub_mul * (float)x[i], (float)y[j], acc);
          *z = acc;
        } else {
          float16 *z = &state->z[(j * 2) + z_offset].f16[i];
          *z = fma16(sub_mul * x[i], y[j],
                     (operand & FMA_SKIP_Z_INPUT) ? (float16)0.0f : *z);
        }
      }
    }
  }
}

void amx_state_fma16(struct amx_state *state, uint64_t operand) {
  amx_state_fmas16_impl(state, operand, false);
}

void amx_state_fms16(struct amx_state *state, uint64_t operand) {
  amx_state_fmas16_impl(state, operand, true);
}

void amx_state_mac16(struct amx_state *state, uint64_t operand) {
  uint16_t x[32];
  uint16_t y[32];

  // TODO: need to wrap around byte offsets
  uint64_t y_offset = operand & 0x1FF;
  uint64_t x_offset = (operand >> 10) & 0x1FF;
  uint64_t z_offset = (operand >> 20) & 63;

  assert((sizeof x) == 0x40);

  if ((operand & FMA_SKIP_Y_INPUT) && (operand & FMA_SKIP_X_INPUT)) {
    memset(&x, 0, sizeof x);
    memset(&y, 0, sizeof y);
  } else {
    if (operand & FMA_SKIP_X_INPUT) {
      for (int i = 0; i < 32; i++) {
        x[i] = (uint16_t)1;
      }
    } else {
      load_from_x(x, state, x_offset, sizeof x);
    }
    if (operand & FMA_SKIP_Y_INPUT) {
      for (int i = 0; i < 32; i++) {
        y[i] = (uint16_t)1;
      }
    } else {
      load_from_y(y, state, y_offset, sizeof y);
    }
  }

  if (operand & (1ull << 63)) {
    for (int i = 0; i < 32; i++) {
      uint16_t *z = &state->z[z_offset].u16[i];
      *z = mac16(x[i], y[i], (operand & FMA_SKIP_Z_INPUT) ? (uint16_t)0 : *z);
    }
  } else {
    z_offset &= 1;
    for (int i = 0; i < 32; i++) {
      for (int j = 0; j < 32; j++) {
        if (operand & (1ull << 62)) {
          uint32_t *z = &state->z[(j * 2) + (i & 1)].u32[i >> 1];
          uint32_t acc = (operand & FMA_SKIP_Z_INPUT) ? 0 : *z;
          acc += (uint32_t)((int64_t)(int16_t)x[i] * (int64_t)(int16_t)y[j]);
          *z = acc;
        } else {
          uint16_t *z = &state->z[(j * 2) + z_offset].u16[i];
          *z = mac16(x[i], y[j],
                     (operand & FMA_SKIP_Z_INPUT) ? (uint16_t)0 : *z);
        }
      }
    }
  }
}

void amx_state_extrx(struct amx_state *state, uint64_t operand) {
  // uint64_t y_offset = operand & 0x1FF;
  uint64_t x_offset = (operand >> 10) & 0x1FF;
  uint64_t z_offset = (operand >> 20) & 63;

  uint32_t buffer[16];

  if ((operand & (1ull << 27))) {
    x_offset &= ~0x3F;
    load_from_y(buffer, state, z_offset * 0x40, 0x40);
  } else {
    load_from_z(buffer, state, z_offset * 0x40, 0x40);
  }

  store_to_x(buffer, state, x_offset, 0x40);
}

void amx_state_extry(struct amx_state *state, uint64_t operand) {
  // I have misgivings about calling this "extry" as it does sometimes move to
  // "x"
  uint64_t y_offset = operand & 0x1FF;
  uint64_t z_offset = (operand >> 20) & 63;

  if (operand & (1ull << 26)) {
    // TODO: might be a good place to use the "union amx_row" type
    uint8_t buffer[64];
    uint64_t operation = ((operand >> 11) & 0xF) | ((operand >> 63) << 4);

    switch (operation) {
    case 0x00: {
      for (int i = 0; i < 64; i++) {
        buffer[i] = state->z[i].u8[z_offset & 63];
      }
    } break;

    case 0x0B: {
      // TODO: mishandles z_offset
      for (int i = 0; i < 64; i++) {
        buffer[i] = state->z[i].u8[z_offset & 63];
      }
    } break;

    case 0x0D: {
      // TODO: mishandles z_offset
      for (int i = 0; i < 64; i++) {
        buffer[i] = state->z[i].u8[z_offset & 63];
      }
    } break;

    case 0x09: {
      // TODO: mishandles z_offset
      uint16_t buffer1[32];
      for (int i = 0; i < 32; i++) {
        buffer1[i] =
            state->z[((i & ~1) << 1) + (z_offset & ((1 << 1) - 1)) + (i & 1)]
                .u16[(z_offset >> 1) & 31];
      }
      memcpy(buffer, buffer1, sizeof buffer);
    } break;

    case 0x0A: {
      // TODO: still mishandles z_offset
      uint16_t buffer1[32];
      for (int i = 0; i < 32; i++) {
        buffer1[i] =
            state->z[((i << 1) + (z_offset & ((1 << 1) - 1))) ^ (z_offset & 2)]
                .u16[((z_offset >> 2) & 31)];
      }
      memcpy(buffer, buffer1, sizeof buffer);
    } break;

    case 0x11: {
      uint64_t buffer1[8];
      for (int i = 0; i < 8; i++) {
        buffer1[i] = state->z[(i << 3) + (z_offset & ((1 << 3) - 1))]
                         .u64[(z_offset >> 3) & 7];
      }
      memcpy(buffer, buffer1, sizeof buffer);
    } break;

    // (does this have an undiscovered difference?)
    case 0x08:
    case 0x18: {
      uint32_t buffer1[16];
      for (int i = 0; i < 16; i++) {
        buffer1[i] = state->z[(i << 2) + (z_offset & ((1 << 2) - 1))]
                         .u32[(z_offset >> 2) & 15];
      }
      memcpy(buffer, buffer1, sizeof buffer);
    } break;

    default: {
      uint16_t buffer1[32];
      for (int i = 0; i < 32; i++) {
        buffer1[i] = state->z[(i << 1) + (z_offset & ((1 << 1) - 1))]
                         .u16[(z_offset >> 1) & 31];
      }
      memcpy(buffer, buffer1, sizeof buffer);
    } break;
    }

    if (operand & (1ull << 10)) {
      store_to_y(buffer, state, y_offset, 0x40);
    } else {
      store_to_x(buffer, state, y_offset, 0x40);
    }
  } else if ((operand & (1ull << 27))) {
    y_offset &= ~0x3F;

    uint32_t buffer[16];
    load_from_x(buffer, state, z_offset * 0x40, 0x40);
    store_to_y(buffer, state, y_offset, 0x40);
  } else {
    // TODO: rewrite as a switch on (operand >> 28) & 3?
    if ((operand & (1ull << 29)) && (operand & (1ull << 28))) {
      uint16_t buffer[32];
      load_from_y(buffer, state, y_offset, 0x40);
      for (int i = 0; i < 32; i++) {
        buffer[i] &= 0xFF00;
        buffer[i] |= state->z[(i << 1) + (z_offset & ((1 << 1) - 1))]
                         .u16[(z_offset >> 1) & 31] &
                     0xFF;
      }
      store_to_y(buffer, state, y_offset, 0x40);
    } else if (operand & (1ull << 29)) {
      uint16_t buffer[32];
      for (int i = 0; i < 32; i++) {
        buffer[i] = state->z[(i << 1) + (z_offset & ((1 << 1) - 1))]
                        .u16[(z_offset >> 1) & 31];
      }
      store_to_y(buffer, state, y_offset, 0x40);
    } else if (operand & (1ull << 28)) {
      uint32_t buffer[16];
      for (int i = 0; i < 16; i++) {
        buffer[i] = state->z[(i << 2) + (z_offset & ((1 << 2) - 1))]
                        .u32[(z_offset >> 2) & 15];
      }
      store_to_y(buffer, state, y_offset, 0x40);
    } else {
      uint64_t buffer[8];
      for (int i = 0; i < 8; i++) {
        buffer[i] = state->z[(i << 3) + (z_offset & ((1 << 3) - 1))]
                        .u64[(z_offset >> 3) & 7];
      }
      store_to_y(buffer, state, y_offset, 0x40);
    }
  }
}

// print flags
enum print_flags {
  PF_TYPE_MASK = 0x0F,
  PF_U32 = 0x00,
  PF_U16 = 0x01,
  PF_U8 = 0x02,
  PF_F16 = 0x03,
  PF_F32 = 0x04,
  PF_F64 = 0x05,
  PF_U64 = 0x06,

  PF_SKIP_X = 0x10,
  PF_SKIP_Y = 0x20,
  PF_SKIP_Z = 0x40,
  PF_SKIP_ZERO_ROWS = 0x80,

  // for diffs
  PF_SKIP_A = 0x10000,
  PF_SKIP_B = 0x20000,
};

void print_amx_row(char reg, int num, union amx_row *r, int flags) {
  bool should_print = true;

  if (flags & PF_SKIP_ZERO_ROWS) {
    should_print = false;
    for (uint64_t o = 0; o < 16; o++) {
      if (r->u32[o] != 0) {
        should_print = true;
        break;
      }
    }
  }

  if (should_print) {
    if (num < 10)
      printf(" ");
    printf(" %c%d:", reg, num);
    switch (flags & PF_TYPE_MASK) {
    case PF_U32:
      for (uint64_t o = 0; o < 16; o++) {
        printf(" %8x", r->u32[o]);
      }
      break;
    case PF_U16:
      for (uint64_t o = 0; o < 32; o++) {
        printf(" %4x", r->u16[o]);
      }
      break;
    case PF_U8:
      for (uint64_t o = 0; o < 64; o++) {
        printf(" %2x", r->u8[o]);
      }
      break;
    case PF_F16:
      for (uint64_t o = 0; o < 32; o++) {
        printf(" %16f", (float)r->f16[o]);
      }
      break;
    case PF_F32:
      for (uint64_t o = 0; o < 16; o++) {
        printf(" %16f", r->f32[o]);
      }
      break;
    case PF_F64:
      for (uint64_t o = 0; o < 8; o++) {
        printf(" %16lf", r->f64[o]);
      }
      break;
    case PF_U64:
      for (uint64_t o = 0; o < 8; o++) {
        printf(" %16llx", r->u64[o]);
      }
      break;
    default:
      assert(0 && "invalid print flags");
    }
    printf("\n");
  }
}

void print_amx_state(struct amx_state *state, int flags) {
  if (!(flags & PF_SKIP_X)) {
    for (int i = 0; i < 8; i++) {
      print_amx_row('x', i, &state->x[i], flags);
    }
  }
  if (!(flags & PF_SKIP_Y)) {
    for (int i = 0; i < 8; i++) {
      print_amx_row('y', i, &state->y[i], flags);
    }
  }
  if (!(flags & PF_SKIP_Z)) {
    for (int i = 0; i < 64; i++) {
      print_amx_row('z', i, &state->z[i], flags);
    }
  }
}

int diff_amx_state(struct amx_state *a, struct amx_state *b, int flags) {
  int same = 1;
  if (!(flags & PF_SKIP_X)) {
    for (int i = 0; i < 8; i++) {
      if (memcmp(&a->x[i], &b->x[i], sizeof(union amx_row))) {
        if (!(flags & PF_SKIP_A))
          print_amx_row('x', i, &a->x[i], flags);
        if (!(flags & PF_SKIP_B))
          print_amx_row('x', i, &b->x[i], flags);
        same = 0;
      }
    }
  }
  if (!(flags & PF_SKIP_Y)) {
    for (int i = 0; i < 8; i++) {
      if (memcmp(&a->y[i], &b->y[i], sizeof(union amx_row))) {
        if (!(flags & PF_SKIP_A))
          print_amx_row('y', i, &a->y[i], flags);
        if (!(flags & PF_SKIP_B))
          print_amx_row('y', i, &b->y[i], flags);
        same = 0;
      }
    }
  }
  if (!(flags & PF_SKIP_Z)) {
    for (int i = 0; i < 64; i++) {
      if (memcmp(&a->z[i], &b->z[i], sizeof(union amx_row))) {
        if (!(flags & PF_SKIP_A))
          print_amx_row('z', i, &a->z[i], flags);
        if (!(flags & PF_SKIP_B))
          print_amx_row('z', i, &b->z[i], flags);
        same = 0;
      }
    }
  }
  return same;
}
