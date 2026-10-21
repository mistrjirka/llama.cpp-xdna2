// Universal prefill core kernel (AIE2P), layout v2.  64-token tile x (16-row record) per step: expand the record to bfp16ebs8 in L1
// (w = a*q - b), bfp16 mmul against the resident bfp16 activation slab (64 tok x 256 k), fp32 C accumulate in the output object.
//
// Control (see genp.py): the first A object carries the header (64 B):
//   i32 0 nsb (superblocks per core), 1 nbk (K/256), 2 qtype (4,5,6,8), 3 G (groups per core), 4 gsb (groups per superblock)
// per slab (sb, kb): pf_slab_begin -> number of 1 KB W chunks of the slab; pf_chunk per chunk assembles records in wrec, processes complete records.
#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"
#include <stdint.h>

#ifndef PF_S
#define PF_S 1024
#endif
#define GSB 5
#define MT 64                       // tokens per tile
#define MB (MT / 8)                 // 8-token blocks
// per-record scale tables, overlaid by qtype (only one record is in flight):
//   Q4_K/Q5_K/Q4_0/Q4_1 : SVH/SVL[j][nb][64] bf16 (2 KB each: a = d*sc split in two bf16), CV[j][nb][64] f32 (4 KB: c = b - 128 a)
//                         j = 32-k sub-block, nb = 8-row half, 64 lanes = [n][k] replicated over k.  a and b must stay (nearly) exact: for a
//                         positive activation vector the weights' mean a*q + b cancels, so a bf16-rounded a or b becomes a coherent error (NMSE 8e-4,
//                         see test_mul_mat_pos).  CV is loaded straight into the accumulator: replicating c in the expand loop costs shuffles and
//                         registers (the j-loop spilled), so the expand is one rolled loop over kb that the scheduler overlaps.
//   Q6_K      : SV6[s][nb][64] bf16 (4 KB)                                    s = 16-k sub-block
//   Q8_0      : SVH[b][nb][64], SVL[b][nb][64] bf16 (2 KB each)               b = 32-k block (d split in two bf16)
// The staging vectors SCB/MNB/ABF live in the CV half (data memory is full: stack 6 KB, .bss + .data + .rodata <= 9332 B); fill_tables reads
// them before fill_cv overwrites them.  CSTAGE must not overlap CV: fill_cv reads it while writing CV (restrict pointers).
alignas(64) static uint8_t TBL[8192];
#define SV6 ((bfloat16 (*)[2][64])(TBL))
#define SVH ((bfloat16 (*)[2][64])(TBL))
#define SVL ((bfloat16 (*)[2][64])(TBL + 2048))
#define CV ((float (*)[2][64])(TBL + 4096))
#define SCB ((bfloat16 *)(TBL + 4096))      // 256 bf16
#define MNB ((bfloat16 *)(TBL + 4608))      // 128 bf16
#define ABF ((bfloat16 *)(TBL + 5120))      // 256 bf16
alignas(64) static float CSTAGE[8 * 16];     // [j][n] c

// fp16 (uint16 x16) -> f32 x16 (normal, zero, denormal)
static inline aie::vector<float, 16> h2f(const uint16_t *p) {
  aie::vector<uint16, 16> h = aie::load_v<16>(p);
  aie::vector<uint32, 16> h32 = aie::unpack(h);
  aie::mask<16> isneg = aie::neq(aie::bit_and(h32, aie::broadcast<uint32, 16>(0x8000)), aie::zeros<uint32, 16>());
  aie::vector<uint32, 16> em = aie::bit_and(h32, aie::broadcast<uint32, 16>(0x7FFF));
  aie::vector<uint32, 16> f = aie::add(aie::upshift(em, 13), aie::broadcast<uint32, 16>(0x38000000u));
  aie::mask<16> isz = aie::eq(em, aie::zeros<uint32, 16>());
  f = aie::select(f, aie::zeros<uint32, 16>(), isz);
  aie::vector<float, 16> r = f.cast_to<float>();
  aie::mask<16> isden = aie::lt(em, aie::broadcast<uint32, 16>(0x400));
  aie::vector<float, 16> den = aie::to_float<float>(em.cast_to<int32>(), 24);     // em * 2^-24 (a float*float mul is emulated: ~9 bf16 products)
  r = aie::select(r, den, isden);
  return aie::select(r, aie::neg(r), isneg);
}

// f = hi + lo in two bf16 pieces: exact for the 11-bit fp16 scales, 2^-17 relative for a 17-bit d*sc.  (A cheaper truncating split with
// bit masks is 300 B larger: the kernel is within ~1 KB of the 16 KB program memory, XAIE_INVALID_ELF beyond.)
static inline void split_hilo(aie::vector<float, 16> f, aie::vector<bfloat16, 16> &hi, aie::vector<bfloat16, 16> &lo) {
  aie::accum<accfloat, 16> t, u;
  t.from_vector(f); hi = t.to_vector<bfloat16>();
  u.from_vector(hi);
  t.from_vector(aie::sub(f, u.to_vector<float>()));
  lo = t.to_vector<bfloat16>();
}

// c[j] (16 floats, one per row n) replicated over the 8 k lanes of each row, as two 64-lane vectors (rows 0-7, 8-15).  No 32-bit transpose on
// AIE2P: three zip levels instead.
static inline void rep_c(const float *__restrict cs, aie::vector<float, 64> &r0, aie::vector<float, 64> &r1) {
  const aie::vector<float, 16> c = aie::load_v<16>(cs);
  auto [l0, h0] = aie::interleave_zip(c, c, 1);        // c0 c0 .. c7 c7 | c8 c8 .. c15 c15
  auto [l1, h1] = aie::interleave_zip(l0, l0, 1);      // c0 x4 .. c3 x4 | c4 x4 .. c7 x4
  auto [l2, h2] = aie::interleave_zip(h0, h0, 1);      // c8 x4 .. c11 x4 | c12 x4 .. c15 x4
  auto [a0, a1] = aie::interleave_zip(l1, l1, 1);
  auto [a2, a3] = aie::interleave_zip(h1, h1, 1);
  auto [b0, b1] = aie::interleave_zip(l2, l2, 1);
  auto [b2, b3] = aie::interleave_zip(h2, h2, 1);
  r0 = aie::concat(aie::concat(a0, a1), aie::concat(a2, a3));
  r1 = aie::concat(aie::concat(b0, b1), aie::concat(b2, b3));
}

// SVH/SVL[j][nb] (64 lanes, [n][k] replicated) from ABF (a hi), SCB (a lo), each [j][n]
// (restrict: the staging is in TBL too, but disjoint from SVH/SVL; without it the loop serializes, 2x slower.  External linkage: a static
// function gets its TBL arguments constant-propagated, which drops the restrict.)
__attribute__((noinline)) void pf_fill_tables(const bfloat16 *__restrict hi, const bfloat16 *__restrict lo, bfloat16 *__restrict th,
                                                    bfloat16 *__restrict tl) {
  AIE_LOOP_UNROLL(4)
  for (int i = 0; i < 16; i++) {           // i = j*2 + nb
    aie::vector<bfloat16, 8> h8 = aie::load_v<8>(hi + i * 8), l8 = aie::load_v<8>(lo + i * 8);
    aie::store_v(th + i * 64, aie::transpose(h8.grow_replicate<64>(), 8, 8));
    aie::store_v(tl + i * 64, aie::transpose(l8.grow_replicate<64>(), 8, 8));
  }
}
static inline void fill_tables() { pf_fill_tables(ABF, SCB, &SVH[0][0][0], &SVL[0][0][0]); }
// CV[j][nb] from CSTAGE (overwrites the staging vectors, see TBL)
__attribute__((noinline)) void pf_fill_cv(const float *__restrict cs, float *__restrict cv) {
  AIE_LOOP_NO_UNROLL
  for (int j = 0; j < 8; j++) {
    aie::vector<float, 64> r0, r1;
    rep_c(cs + j * 16, r0, r1);
    aie::store_v(cv + j * 128, r0); aie::store_v(cv + j * 128 + 64, r1);
  }
}
static inline void fill_cv() { pf_fill_cv(CSTAGE, &CV[0][0][0]); }

// ---- scale prep v6: a_j = d*sc_j (hi/lo bf16), c_j = -dmin*m_j - 128 (a_hi + a_lo) (f32); the expand forms c + x*a with x = 128 + q.
// 32 lanes = 2 sub-blocks j x 16 rows.  The split stays in the accumulator: a_hi = bf16(a), then acc - a_hi*1 is the exact remainder.
// One copy for Q4_K and Q5_K (offsets as pointers): two inlined template copies cost 1.4 KB of the 16 KB program memory.
__attribute__((noinline)) static void prep_scales_k(const uint8_t *__restrict scp, const uint16_t *__restrict dp, const uint16_t *__restrict dmp) {
  aie::vector<uint16, 32> Va = aie::unpack(aie::load_v<32>(scp)), Vb = aie::unpack(aie::load_v<32>(scp + 32)),
                          Vc = aie::unpack(aie::load_v<32>(scp + 64)), Vd = aie::unpack(aie::load_v<32>(scp + 96)),
                          Ve = aie::unpack(aie::load_v<32>(scp + 128)), Vf = aie::unpack(aie::load_v<32>(scp + 160));
  const aie::vector<uint16, 32> m63 = aie::broadcast<uint16, 32>(63), m15 = aie::broadcast<uint16, 32>(15),
                                mF0 = aie::broadcast<uint16, 32>(0xF0), mC0 = aie::broadcast<uint16, 32>(0xC0);
  aie::vector<int16, 32> scp2[4], mnp2[4];
  scp2[0] = aie::bit_and(Va, m63).cast_to<int16>();
  scp2[1] = aie::bit_and(Vb, m63).cast_to<int16>();
  scp2[2] = aie::add(aie::bit_and(Ve, m15), aie::downshift(aie::bit_and(Va, mC0), 2)).cast_to<int16>();
  scp2[3] = aie::add(aie::bit_and(Vf, m15), aie::downshift(aie::bit_and(Vb, mC0), 2)).cast_to<int16>();
  mnp2[0] = aie::bit_and(Vc, m63).cast_to<int16>();
  mnp2[1] = aie::bit_and(Vd, m63).cast_to<int16>();
  mnp2[2] = aie::add(aie::downshift(aie::bit_and(Ve, mF0), 4), aie::downshift(aie::bit_and(Vc, mC0), 2)).cast_to<int16>();
  mnp2[3] = aie::add(aie::downshift(aie::bit_and(Vf, mF0), 4), aie::downshift(aie::bit_and(Vd, mC0), 2)).cast_to<int16>();
  for (int jp = 0; jp < 4; jp++) {
    aie::store_v(SCB + jp * 32, aie::to_float<bfloat16>(scp2[jp]));
    aie::store_v(MNB + jp * 32, aie::to_float<bfloat16>(mnp2[jp]));
  }
  // d, -dmin as exact bf16 hi/lo pieces, replicated over the 2 sub-blocks of a 32-lane vector
  aie::vector<bfloat16, 16> d_hi, d_lo, n_hi, n_lo;
  split_hilo(h2f(dp), d_hi, d_lo);
  split_hilo(aie::neg(h2f(dmp)), n_hi, n_lo);
  const aie::vector<bfloat16, 32> dh = aie::concat(d_hi, d_hi), dl = aie::concat(d_lo, d_lo), nh = aie::concat(n_hi, n_hi), nl = aie::concat(n_lo, n_lo);
  const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f), m128 = aie::broadcast<bfloat16, 32>(-128.0f);
  AIE_LOOP_UNROLL_FULL
  for (int jp = 0; jp < 4; jp++) {            // sub-blocks j = 2jp, 2jp+1
    const aie::vector<bfloat16, 32> s = aie::load_v<32>(SCB + jp * 32), m = aie::load_v<32>(MNB + jp * 32);
    aie::accum<accfloat, 32> aa = aie::mul(dh, s);
    aa = aie::mac(aa, dl, s);                                                 // a = d*sc exact (17 bits)
    const aie::vector<bfloat16, 32> a_hi = aa.to_vector<bfloat16>();
    aa = aie::msc(aa, a_hi, one);                                             // exact remainder a - a_hi
    const aie::vector<bfloat16, 32> a_lo = aa.to_vector<bfloat16>();
    aie::store_v(ABF + jp * 32, a_hi); aie::store_v(SCB + jp * 32, a_lo);     // SCB: the scales are consumed
    aie::accum<accfloat, 32> bb = aie::mul(nh, m);                            // c = b - 128 (a_hi + a_lo)
    bb = aie::mac(bb, nl, m);
    bb = aie::mac(bb, a_hi, m128);
    bb = aie::mac(bb, a_lo, m128);
    aie::store_v(CSTAGE + jp * 32, bb.to_vector<float>());
  }
  fill_tables();
  fill_cv();
}

// ---- Q4_0 / Q4_1 scale prep: the Q4_K form w = a*q + b per 32-k block j with a = d, b = -8 d (Q4_0) or m (Q4_1)
__attribute__((noinline)) static void prep_q40(const int8_t *__restrict rec, bool q41) {
  const aie::vector<bfloat16, 16> m128 = aie::broadcast<bfloat16, 16>(-128.0f), m136 = aie::broadcast<bfloat16, 16>(-136.0f);
  AIE_LOOP_NO_UNROLL
  for (int j = 0; j < 8; j++) {
    const aie::vector<float, 16> a_f = h2f((const uint16_t *)(rec + 2048 + j * 32));
    aie::vector<bfloat16, 16> a_hi, a_lo;
    split_hilo(a_f, a_hi, a_lo);                                              // exact: a is fp16
    aie::store_v(ABF + j * 16, a_hi); aie::store_v(SCB + j * 16, a_lo);
    aie::accum<accfloat, 16> bb;                                              // c = b - 128 (a_hi + a_lo)
    if (q41) { bb.from_vector(h2f((const uint16_t *)(rec + 2304 + j * 32))); bb = aie::mac(bb, a_hi, m128); bb = aie::mac(bb, a_lo, m128); }
    else { bb = aie::mul(a_hi, m136); bb = aie::mac(bb, a_lo, m136); }        // Q4_0: b = -8 a, c = -136 a (bf16 macs: a float mul is emulated)
    aie::store_v(CSTAGE + j * 16, bb.to_vector<float>());
  }
  fill_tables();
  fill_cv();
}

// ---- expand v3: x = bf16(128+q) built by byte-zip with 0x43 (no int->float convert); w = c + x*a_hi + x*a_lo with c = b - 128 a from CV
static inline void expand_q4(const int8_t *__restrict rec, int8_t *__restrict Bt) {
  const aie::vector<uint8, 64> c43 = aie::broadcast<uint8, 64>(0x43);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bt);
  AIE_LOOP_NO_UNROLL                              // rolled: the scheduler overlaps iterations; unrolled (2) or a j x t nest spills
  for (int kb = 0; kb < 32; kb++) {               // kb = 4j + t
    const int j = kb >> 2;
    aie::vector<uint4, 128> q4 = aie::load_v<128>((const uint4 *)(rec + kb * 64));
    aie::vector<uint8, 128> q = q4.unpack();
    auto [e0, e1] = aie::interleave_unzip(q.extract<64>(0), q.extract<64>(1), 8);
    aie::vector<uint8, 64> t0 = aie::transpose(e0, 8, 8), t1 = aie::transpose(e1, 8, 8);
    auto [z00, z01] = aie::interleave_zip(t0, c43, 1);
    auto [z10, z11] = aie::interleave_zip(t1, c43, 1);
    aie::vector<bfloat16, 64> x0 = aie::concat(z00.cast_to<bfloat16>(), z01.cast_to<bfloat16>());
    aie::vector<bfloat16, 64> x1 = aie::concat(z10.cast_to<bfloat16>(), z11.cast_to<bfloat16>());
    aie::accum<accfloat, 64> acc0, acc1;
    acc0.from_vector(aie::load_v<64>(CV[j][0])); acc1.from_vector(aie::load_v<64>(CV[j][1]));
    acc0 = aie::mac(acc0, x0, aie::load_v<64>(SVH[j][0])); acc1 = aie::mac(acc1, x1, aie::load_v<64>(SVH[j][1]));
    acc0 = aie::mac(acc0, x0, aie::load_v<64>(SVL[j][0])); acc1 = aie::mac(acc1, x1, aie::load_v<64>(SVL[j][1]));
    out.push(aie::block_vector<bfp16ebs8, 64>(::to_v64bfp16ebs8(acc0)));
    out.push(aie::block_vector<bfp16ebs8, 64>(::to_v64bfp16ebs8(acc1)));
  }
}

// ================= Q5_K (REC 2816): lo nibbles B-tiles [0,2048), hi bits [2048,2560), scale planes 2560, d 2752, dmin 2784 =================
static inline void expand_q5(const int8_t *__restrict rec, int8_t *__restrict Bt) {
  const aie::vector<uint8, 64> c43 = aie::broadcast<uint8, 64>(0x43);
  const aie::vector<uint32, 16> m10 = aie::broadcast<uint32, 16>(0x10101010u);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bt);
  AIE_LOOP_NO_UNROLL
  for (int jh = 0; jh < 16; jh++) {                   // tile t = 2j + h: bit (c + 4h) of the hi byte, moved to bit 4 of the same byte
    const int j = jh >> 1, h = jh & 1;
    const aie::vector<uint32, 16> V = aie::downshift(aie::load_v<64>((const uint8 *)(rec + 2048 + j * 64)).cast_to<uint32>(), 4 * h);
    aie::vector<uint8, 64> T[4];
    AIE_LOOP_UNROLL_FULL
    for (int c = 0; c < 4; c++) {
      aie::vector<uint32, 16> hb = aie::bit_and(aie::upshift(V, 4 - c), m10);
      aie::vector<uint4, 64> l4 = aie::load_v<64>((const uint4 *)(rec + jh * 128 + c * 32));
      aie::vector<uint8, 64> w = aie::add(aie::vector<uint8, 64>(l4.unpack()), hb.cast_to<uint8>());
      T[c] = aie::transpose(w, 8, 8);
    }
    AIE_LOOP_UNROLL_FULL
    for (int oi = 0; oi < 4; oi++) {
      const int c = (oi == 1) ? 2 : (oi == 2) ? 1 : oi;
      const int nb = c >> 1;
      auto [z0, z1] = aie::interleave_zip(T[c], c43, 1);
      aie::vector<bfloat16, 64> x = aie::concat(z0.cast_to<bfloat16>(), z1.cast_to<bfloat16>());
      aie::accum<accfloat, 64> acc;
      acc.from_vector(aie::load_v<64>(CV[j][nb]));
      acc = aie::mac(acc, x, aie::load_v<64>(SVH[j][nb]));
      acc = aie::mac(acc, x, aie::load_v<64>(SVL[j][nb]));
      out.push(aie::block_vector<bfp16ebs8, 64>(::to_v64bfp16ebs8(acc)));
    }
  }
}

// ================= Q6_K (REC 3360): lo nibbles B-tiles [0,2048), hi2 [2048,3072), int8 scales [t][n] 3072, d 3328 =================
static inline void prep_q6(const int8_t *__restrict rec) {
  aie::vector<bfloat16, 16> d_hi, d_lo;
  split_hilo(h2f((const uint16_t *)(rec + 3328)), d_hi, d_lo);
  for (int sp = 0; sp < 8; sp++) {
    aie::vector<int16, 32> sc16 = aie::unpack(aie::load_v<32>(rec + 3072 + sp * 32));
    aie::store_v(SCB + sp * 32, aie::to_float<bfloat16>(sc16));
  }
  AIE_LOOP_NO_UNROLL
  for (int s = 0; s < 16; s++) {
    aie::vector<bfloat16, 16> sc = aie::load_v<16>(SCB + s * 16);
    aie::accum<accfloat, 16> aa = aie::mul(d_hi, sc);
    aa = aie::mac(aa, d_lo, sc);
    aie::store_v(ABF + s * 16, aa.to_vector<bfloat16>());
  }
  AIE_LOOP_NO_UNROLL
  for (int i = 0; i < 32; i++) {            // i = s*2 + nb
    aie::vector<bfloat16, 8> a8 = aie::load_v<8>(ABF + i * 8);
    aie::vector<bfloat16, 64> av = aie::transpose(a8.grow_replicate<64>(), 8, 8);
    aie::store_v(&SV6[0][0][0] + i * 64, av);
  }
}
static inline void expand_q6(const int8_t *__restrict rec, int8_t *__restrict Bt) {
  const aie::vector<uint8, 64> c43 = aie::broadcast<uint8, 64>(0x43);
  const aie::vector<uint32, 16> m30 = aie::broadcast<uint32, 16>(0x30303030u);
  const aie::vector<bfloat16, 64> m160 = aie::broadcast<bfloat16, 64>(-160.0f);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bt);
  AIE_LOOP_NO_UNROLL
  for (int s = 0; s < 16; s++) {
    aie::vector<uint32, 16> H = aie::load_v<64>((const uint8 *)(rec + 2048 + s * 64)).cast_to<uint32>();
    aie::vector<uint32, 16> hh[4];
    hh[0] = aie::bit_and(aie::upshift(H, 4), m30);
    hh[1] = aie::bit_and(aie::upshift(H, 2), m30);
    hh[2] = aie::bit_and(H, m30);
    hh[3] = aie::bit_and(aie::downshift(H, 2), m30);
    aie::vector<uint8, 64> T[4];
    AIE_LOOP_UNROLL_FULL
    for (int c = 0; c < 4; c++) {
      aie::vector<uint4, 64> l4 = aie::load_v<64>((const uint4 *)(rec + s * 128 + c * 32));
      aie::vector<uint8, 64> w = aie::add(aie::vector<uint8, 64>(l4.unpack()), hh[c].cast_to<uint8>());
      T[c] = aie::transpose(w, 8, 8);
    }
    AIE_LOOP_UNROLL_FULL
    for (int oi = 0; oi < 4; oi++) {
      const int c = (oi == 1) ? 2 : (oi == 2) ? 1 : oi;
      const int nb = c >> 1;
      auto [z0, z1] = aie::interleave_zip(T[c], c43, 1);
      aie::vector<bfloat16, 64> x = aie::concat(z0.cast_to<bfloat16>(), z1.cast_to<bfloat16>());
      const aie::vector<bfloat16, 64> sv = aie::load_v<64>(SV6[s][nb]);
      aie::accum<accfloat, 64> acc = aie::mul(m160, sv);
      acc = aie::mac(acc, x, sv);
      out.push(aie::block_vector<bfp16ebs8, 64>(::to_v64bfp16ebs8(acc)));
    }
  }
}

// ================= Q8_0 (REC 4352): int8 [b][kc][nh][k16][n8] (b = 32-k block) [0,4096), fp16 scales [b][n] at 4096 =================
static inline void prep_q8(const int8_t *__restrict rec) {
  AIE_LOOP_NO_UNROLL
  for (int b = 0; b < 8; b++) {
    aie::vector<bfloat16, 16> d_hi, d_lo;
    split_hilo(h2f((const uint16_t *)(rec + 4096 + b * 32)), d_hi, d_lo);
    aie::store_v(ABF + b * 16, d_hi);
    aie::store_v(SCB + b * 16, d_lo);
  }
  fill_tables();
}
static inline void expand_q8(const int8_t *__restrict rec, int8_t *__restrict Bt) {
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bt);
  AIE_LOOP_NO_UNROLL
  for (int b = 0; b < 8; b++) {
    AIE_LOOP_UNROLL_FULL
    for (int kc = 0; kc < 2; kc++) {
      AIE_LOOP_UNROLL_FULL
      for (int hf = 0; hf < 2; hf++) {           // kb32 = 4b + 2kc + hf
        AIE_LOOP_UNROLL_FULL
        for (int nb = 0; nb < 2; nb++) {
          aie::vector<int8, 64> q = aie::load_v<64>(rec + b * 512 + kc * 256 + nb * 128 + hf * 64);
          aie::vector<int8, 64> qt = aie::transpose(q, 8, 8);
          aie::vector<int16, 64> q16 = aie::unpack(qt);
          aie::vector<bfloat16, 64> x = aie::concat(aie::to_float<bfloat16>(q16.extract<32>(0)), aie::to_float<bfloat16>(q16.extract<32>(1)));
          aie::accum<accfloat, 64> acc = aie::mul(x, aie::load_v<64>(SVH[b][nb]));
          acc = aie::mac(acc, x, aie::load_v<64>(SVL[b][nb]));
          out.push(aie::block_vector<bfp16ebs8, 64>(::to_v64bfp16ebs8(acc)));
        }
      }
    }
  }
}

// ---- mmul: C[mb][nb] (8x8 f32 blocks) += A * B^T.  A stream layout [z 4][kb 32][2 mb] , B stream layout [kb 32][nb 2]
static inline void mm_blocks(const int8_t *__restrict A, const int8_t *__restrict Bt, float *__restrict C) {
  for (unsigned z = 0; z < MB; z += 2) {
    float *c0 = C + (z * 2) * 64, *c1 = C + ((z + 1) * 2) * 64;
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA((const bfp16ebs8 *)A), pB((const bfp16ebs8 *)Bt);
    pA.seek(z * 32);                    // z/2*64 blocks
    aie::accum<accfloat, 64> c00(aie::load_v<64>(c0)), c01(aie::load_v<64>(c0 + 64));
    aie::accum<accfloat, 64> c10(aie::load_v<64>(c1)), c11(aie::load_v<64>(c1 + 64));
    AIE_PREPARE_FOR_PIPELINING
    for (unsigned i = 0; i < 32; i += 4) {
      const auto A00 = pA.pop(); const auto A10 = pA.pop(); const auto B00 = pB.pop(); const auto B10 = pB.pop();
      const auto A01 = pA.pop(); const auto A11 = pA.pop(); const auto B01 = pB.pop(); const auto B11 = pB.pop();
      const auto A02 = pA.pop(); const auto A12 = pA.pop(); const auto B02 = pB.pop(); const auto B12 = pB.pop();
      const auto A03 = pA.pop(); const auto A13 = pA.pop(); const auto B03 = pB.pop(); const auto B13 = pB.pop();
      c00 = mac_8x8_8x8T(A00, B00, c00); c01 = mac_8x8_8x8T(A00, B10, c01); c10 = mac_8x8_8x8T(A10, B00, c10); c11 = mac_8x8_8x8T(A10, B10, c11);
      c00 = mac_8x8_8x8T(A01, B01, c00); c01 = mac_8x8_8x8T(A01, B11, c01); c10 = mac_8x8_8x8T(A11, B01, c10); c11 = mac_8x8_8x8T(A11, B11, c11);
      c00 = mac_8x8_8x8T(A02, B02, c00); c01 = mac_8x8_8x8T(A02, B12, c01); c10 = mac_8x8_8x8T(A12, B02, c10); c11 = mac_8x8_8x8T(A12, B12, c11);
      c00 = mac_8x8_8x8T(A03, B03, c00); c01 = mac_8x8_8x8T(A03, B13, c01); c10 = mac_8x8_8x8T(A13, B03, c10); c11 = mac_8x8_8x8T(A13, B13, c11);
    }
    aie::store_v(c0, c00.to_vector<float>()); aie::store_v(c0 + 64, c01.to_vector<float>());
    aie::store_v(c1, c10.to_vector<float>()); aie::store_v(c1 + 64, c11.to_vector<float>());
  }
}


static inline int rec_bytes(int qt) { return qt == 4 ? 2304 : qt == 5 ? 2816 : qt == 6 ? 3360 : qt == 40 ? 2304 : qt == 41 ? 2560 : 4352; }

static int st_kb = 0, st_sb = 0, st_gl = 0, st_gc = 0, st_fill = 0;
static int st_rec = 0;                       // record bytes of hdr[2]: one switch per dispatch (each switch is a 148 B jump table in data memory)

extern "C" void pf_hdr(const int8_t *__restrict a, int32_t *__restrict hdr) {
  aie::store_v(hdr, aie::load_v<16>((const int32_t *)a));
  st_kb = -1; st_sb = 0; st_fill = 0; st_gl = 0; st_rec = rec_bytes(((const int32_t *)a)[2]);
}
extern "C" void pf_zero(float *__restrict c) {
  for (int i = 0; i < GSB * 1024; i += 16) aie::store_v(c + i, aie::zeros<float, 16>());
}
// called at the start of each slab: chunk count into ctl[0]
extern "C" void pf_slab_begin(const int32_t *__restrict hdr, int32_t *__restrict ctl) {
  const int G = hdr[3];
  if (++st_kb == hdr[1]) { st_kb = 0; st_sb++; }
  const int gsb = hdr[4];              // groups per superblock (layout::Geom::gsb <= GSB, the C capacity)
  int rem = G - st_sb * gsb;
  st_gc = rem < gsb ? rem : gsb;
  st_gl = 0; st_fill = 0;
  ctl[0] = (st_gc * st_rec + PF_S - 1) / PF_S;
  // the chunks of the first record are consumed (and the record expanded) before the A slab is acquired, so the A transfer overlaps that work
  ctl[1] = st_gc > 0 ? (st_rec + PF_S - 1) / PF_S : 0;
  if (ctl[1] > ctl[0]) ctl[1] = ctl[0];
  ctl[2] = ctl[0] - ctl[1];
}
static inline void copy_bytes64(int8_t *__restrict dst, const int8_t *__restrict src, int n) {
  // the pipeliner is off for this kernel: unrolled by hand so the loads issue back to back
  int i = 0;
  for (; i + 512 <= n; i += 512) {
    const auto v0 = aie::load_v<64>(src + i),       v1 = aie::load_v<64>(src + i + 64),  v2 = aie::load_v<64>(src + i + 128), v3 = aie::load_v<64>(src + i + 192);
    const auto v4 = aie::load_v<64>(src + i + 256), v5 = aie::load_v<64>(src + i + 320), v6 = aie::load_v<64>(src + i + 384), v7 = aie::load_v<64>(src + i + 448);
    aie::store_v(dst + i, v0);       aie::store_v(dst + i + 64, v1);  aie::store_v(dst + i + 128, v2); aie::store_v(dst + i + 192, v3);
    aie::store_v(dst + i + 256, v4); aie::store_v(dst + i + 320, v5); aie::store_v(dst + i + 384, v6); aie::store_v(dst + i + 448, v7);
  }
  for (; i < n; i += 64) aie::store_v(dst + i, aie::load_v<64>(src + i));
}
static inline void copy_bytes32(int8_t *__restrict dst, const int8_t *__restrict src, int n) {
  for (int i = 0; i < n; i += 32) aie::store_v(dst + i, aie::load_v<32>(src + i));
}
// one noinline function per format: the dispatch costs a call per record, and each body is register-allocated on its own
__attribute__((noinline)) static void rec_q4(const int8_t *__restrict rec, int8_t *__restrict Bt, int qt) {
  if (qt == 4) prep_scales_k((const uint8_t *)(rec + 2048), (const uint16_t *)(rec + 2240), (const uint16_t *)(rec + 2272));
  else prep_q40(rec, qt == 41);
  expand_q4(rec, Bt);
}
__attribute__((noinline)) static void rec_q5(const int8_t *__restrict rec, int8_t *__restrict Bt) {
  prep_scales_k((const uint8_t *)(rec + 2560), (const uint16_t *)(rec + 2752), (const uint16_t *)(rec + 2784));
  expand_q5(rec, Bt);
}
__attribute__((noinline)) static void rec_q6(const int8_t *__restrict rec, int8_t *__restrict Bt) { prep_q6(rec); expand_q6(rec, Bt); }
__attribute__((noinline)) static void rec_q8(const int8_t *__restrict rec, int8_t *__restrict Bt) { prep_q8(rec); expand_q8(rec, Bt); }
static inline void expand_rec(const int8_t *__restrict rec, int8_t *__restrict Bt, int qt) {
  if (qt == 4 || qt == 40 || qt == 41) rec_q4(rec, Bt, qt);
  else if (qt == 5) rec_q5(rec, Bt);
  else if (qt == 6) rec_q6(rec, Bt);
  else rec_q8(rec, Bt);
}
static inline void process(const int8_t *__restrict rec, const int8_t *__restrict A, float *__restrict C, int8_t *__restrict Bt, int qt) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  expand_rec(rec, Bt, qt);
  mm_blocks(A, Bt, C);
  aie::set_rounding(saved);
}
static int st_pend = -1;                     // record expanded before the A slab was acquired, waiting for its mmul
extern "C" void pf_chunk(const int8_t *__restrict chunk, const int8_t *__restrict A, float *__restrict C, int8_t *__restrict wrec,
                         int8_t *__restrict Bt, const int32_t *__restrict hdr) {
  const int qt = hdr[2];
  const int REC = st_rec;
  int pos = 0;
  while (pos < PF_S && st_gl < st_gc) {
    int n = REC - st_fill; if (n > PF_S - pos) n = PF_S - pos;
    if (((st_fill | pos | n) & 63) == 0) copy_bytes64(wrec + st_fill, chunk + pos, n); else copy_bytes32(wrec + st_fill, chunk + pos, n);
    st_fill += n; pos += n;
    if (st_fill == REC) {
      st_fill = 0;
      process(wrec, A, C + st_gl * 1024, Bt, qt);
      st_gl++;
    }
  }
}

// phase 1 of a slab: like pf_chunk, but the record that completes is only expanded (its mmul waits for the A slab: pf_flush)
extern "C" void pf_chunk_pre(const int8_t *__restrict chunk, int8_t *__restrict wrec, int8_t *__restrict Bt, const int32_t *__restrict hdr) {
  const int qt = hdr[2];
  const int REC = st_rec;
  int pos = 0;
  while (pos < PF_S && st_gl < st_gc) {
    int n = REC - st_fill; if (n > PF_S - pos) n = PF_S - pos;
    if (((st_fill | pos | n) & 63) == 0) copy_bytes64(wrec + st_fill, chunk + pos, n); else copy_bytes32(wrec + st_fill, chunk + pos, n);
    st_fill += n; pos += n;
    if (st_fill == REC) {
      st_fill = 0;
      const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
      expand_rec(wrec, Bt, qt);
      aie::set_rounding(saved);
      st_pend = st_gl;
      st_gl++;
    }
  }
}
// the A slab is here: mmul of the record expanded in phase 1
extern "C" void pf_flush(const int8_t *__restrict A, float *__restrict C, int8_t *__restrict Bt) {
  if (st_pend < 0) return;
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  mm_blocks(A, Bt, C + st_pend * 1024);
  aie::set_rounding(saved);
  st_pend = -1;
}
