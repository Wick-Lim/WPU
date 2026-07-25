`timescale 1ns/1ps
`include "glm_fp.vh"
`include "q4k.vh"         // fp16_to_fp32, u7_to_fp32, s8_to_fp32, q4k_scale_min (leaf prims)
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_matmul_q4k.v  --  GLM-5.2 Q4_K-NATIVE GEMM datapath (local-device target)
//                       a DROP-IN sibling of glm_matmul_pipe.v (prior FP8 twin
//                       glm_matmul_fp8 preserved on tag 'fp8-verified-baseline').
//----------------------------------------------------------------------------
// FUNCTION
//   C[M,N] = A[M,K] x W[K,N], computed in the OFFICIAL GGML Q4_K numerics so the
//   Q4_K-typed weights run with NO re-quantization -- bit-exact to the ggml Q4_K
//   reference `dequantize_row_q4_K` (tools/q4k_ref.py).  (The dynamic UD-Q4_K_XL
//   mix also keeps Q6_K/Q8_0/F16 tensors not yet consumed by this Q4_K-only path.)
//
//   * Weights W arrive as GGML Q4_K: per output column pj a super-block carries
//       - fp16  d[pj]      (super-block scale)
//       - fp16  dmin[pj]   (super-block min)
//       - 96b   scales[pj] (8x 6-bit block-scales + 8x 6-bit block-mins, packed)
//       - per K-beat, a 4-bit quant code w_q[pj] (the weight for row k, col pj).
//     The K-beat's sub-block b = k/32 selects (sc_b, m_b) via get_scale_min_k4;
//     the weight dequantizes EXACTLY to  w = (d*sc_b)*q - (dmin*m_b)  (fp32).
//   * Activations A arrive bf16 (same interface as glm_matmul_pipe).
//
// CONTRACT (bit-exact to tools/q4k_ref.py `matmul_q4k_col`):
//   out[pi][pj] = bf16( SUM_k fp32(a[pi][k]) * w_deq[k][pj] ), the fp32 products
//   sequentially fp32-accumulated in K (streaming) order, rounded to bf16 (RNE).
//   Same accumulate structure as the proven bf16 glm_matmul_pipe -- only the
//   weight source changes (Q4_K dequant instead of a bf16 weight).  All fp32 ops
//   are glm_fp.vh's IEEE fp32_mul / fp32_add (the datapath's canonical numerics).
//
// This is the CORRECT-FIRST reference core: the per-weight dequant + fp32 MAC are
//   combinational, one K-beat per cycle, the accumulator registered each beat (a
//   single-cycle sequential accumulate -- no hazard).  KMAX <= 256 = one Q4_K
//   super-block along K (the caller tiles larger K, as with glm_matmul_pipe).
//
// CONVENTIONS: synchronous ACTIVE-HIGH reset; NO latch; NO combinational loop
//   (the accumulator feedback rides the per-beat register).
//----------------------------------------------------------------------------
// HANDSHAKE  (mirrors glm_matmul_pipe; activation interface identical)
//   start     : 1-cycle pulse; latches k_len + the per-column super-block params.
//   k_len     : number of K beats this tile (<= KMAX <= 256).
//   in_valid  : a K-beat is presented (a_col + w_q).
//   a_col[pi] : bf16 A[pi][k]  (PE_M packed, 16b each).
//   w_q[pj]   : 4-bit Q4_K code W[k][pj] (PE_N packed, 4b each).
//   w_d/w_dmin: fp16 super-block scale/min per column (PE_N packed, 16b), at start.
//   w_scales  : 96b packed 6-bit scales/mins per column (PE_N packed), at start.
//   out_valid : C tile (PE_M x PE_N bf16) valid for 1 cycle.  busy high in flight.
//============================================================================
module glm_matmul_q4k #(
    parameter integer PE_M = 4,       // array rows (== tile M)
    parameter integer PE_N = 4,       // array cols (== tile N)
    parameter integer KMAX = 256      // max K per tile (one Q4_K super-block)
) (
    input  wire                       clk,
    input  wire                       rst,        // sync, active-high

    input  wire                       start,      // begin a tile
    input  wire [$clog2(KMAX+1)-1:0]  k_len,      // number of K beats this tile

    // per-column Q4_K super-block params (latched at start).  A column spans
    // NSB = ceil(KMAX/256) super-blocks along K; super-block sb of column pj is at
    // index (pj*NSB + sb).  (KMAX=256 -> NSB=1, one super-block per column.)
    input  wire [16*PE_N*((KMAX+255)/256)-1:0] w_d,      // fp16 d    per (col, super-block)
    input  wire [16*PE_N*((KMAX+255)/256)-1:0] w_dmin,   // fp16 dmin per (col, super-block)
    input  wire [96*PE_N*((KMAX+255)/256)-1:0] w_scales, // 96b scales per (col, super-block)

    input  wire                       in_valid,   // a K-beat is presented
    input  wire [16*PE_M-1:0]         a_col,      // bf16 A[*][k], PE_M packed
    input  wire [ 4*PE_N-1:0]         w_q,        // 4-bit Q4_K codes W[k][*], PE_N packed

    output reg                        busy,
    output reg                        out_valid,  // C tile valid (1 cycle)
    output reg  [16*PE_M*PE_N-1:0]    c_out,      // bf16 C[pi][pj] packed

    // ---- ADDED: mixed-type (Q6_K/Q8_0/F16) selector + high-precision buses ----
    //   All four are OPTIONAL: a Q4_K-only caller may leave them unconnected.  The
    //   per-column w_type is LATCHED at start (off the per-beat MAC critical path);
    //   a tile is one type so the loader broadcasts one type to all columns.  The
    //   decode `case` below routes Q6_K/Q8_0/F16 to their primitive but keeps Q4_K
    //   as the DEFAULT branch -- which matches w_type==2'b00 AND any undriven (x/z)
    //   w_type -- so an unconnected w_type reads ONLY the pre-existing Q4_K buses
    //   above and yields the byte-identical proven Q4_K result (zero regression).
    input  wire [ 2*PE_N-1:0]                   w_type,  // per col: 0=Q4_K 1=Q6_K 2=Q8_0 3=F16
    input  wire [16*PE_N-1:0]                   w_hp,    // per beat: Q6_K[5:0]/Q8_0[7:0]/F16[15:0]
    input  wire [128*PE_N*((KMAX+255)/256)-1:0] w_q6_sc, // Q6_K 16xint8 scales / (col, super-block)
    input  wire [16*PE_N*((KMAX+31)/32)-1:0]    w_q8_d   // Q8_0 fp16 d / (col, 32-weight block)
);
    localparam integer KW  = $clog2(KMAX+1);
    localparam integer NSB = (KMAX + 255) / 256;   // super-blocks along K
    localparam integer NB8 = (KMAX + 31)  / 32;    // Q8_0 32-weight blocks along K (== 8*NSB)
    localparam [1:0] WT_Q4K = 2'd0, WT_Q6K = 2'd1, WT_Q80 = 2'd2, WT_F16 = 2'd3;

    // ---- latched tile params ----
    reg [KW-1:0]              k_cnt;      // beats consumed
    reg [KW-1:0]              k_len_r;
    reg [16*PE_N*NSB-1:0]     d_r, dmin_r;
    reg [96*PE_N*NSB-1:0]     scales_r;
    reg [2*PE_N-1:0]          wtype_r;    // per-column weight type (latched at start)
    reg [128*PE_N*NSB-1:0]    q6sc_r;     // Q6_K 16xint8 scales / (col, super-block)
    reg [16*PE_N*NB8-1:0]     q8d_r;      // Q8_0 fp16 d / (col, 32-weight block)

    // ---- accumulators (fp32) ----
    reg [31:0] acc [0:PE_M*PE_N-1];

    //========================================================================
    // PER-BEAT PIPELINE  (REPIPELINED FOR FMAX -- bit-exact)
    //
    //   The correct-first core evaluated the whole per-beat chain -- header
    //   muls, code mul, min subtract, activation mul, accumulate -- in ONE
    //   cycle (5 serial fp32 ops, the chip's worst cone after the act/norm
    //   fixes).  It is now PLAT=4 register stages + the accumulate:
    //
    //     P1  header selects for THIS beat's k (sb/sub/sidx/blk) + the header
    //         multiplies: Q4_K d1 = d*sc_b and m1 = dmin*m_b (parallel pair),
    //         Q6_K d1 = d*sc16[sidx]; Q8_0/F16 forward their raw halves.
    //     P2  the code multiply: Q4_K t = d1*u7(q), Q6_K wdeq = d1*s8(code-32),
    //         Q8_0 wdeq = f32(d)*s8(qs), F16 wdeq = widen(raw16).
    //     P3  Q4_K wdeq = t - m1 (add with sign-flipped m1); others forward.
    //     P4  aprod[pi] = f32(a[pi]) * wdeq   (PE_M parallel multiplies).
    //     ACC acc += aprod on the final tap -- ONE registered fp32 add: the
    //         loop-carried accumulate stays single-cycle, everything feeding
    //         it is pipelined.
    //
    //   Beats enter in K order and the pipe is in-order, so the sequential
    //   fp32-accumulate ORDER -- the bit-exactness contract -- is unchanged;
    //   every op is the same glm_fp.vh/q4k.vh call in the same grouping.
    //   out_valid now fires once the pipe DRAINS after the last beat (busy
    //   stays high); every consumer waits on out_valid, not a beat count.
    //========================================================================
    localparam integer PLAT = 4;
    reg [PLAT-1:0]        vp;
    reg                   draining;

    // P1 registers (per column): selected/multiplied header values
    reg [31:0] p1_d1  [0:PE_N-1];   // Q4_K d*sc | Q6_K d*sc16
    reg [31:0] p1_m1  [0:PE_N-1];   // Q4_K dmin*m
    reg [15:0] p1_h16 [0:PE_N-1];   // raw w_hp half (Q6 code / Q8 qs / F16 raw)
    reg [3:0]  p1_q4  [0:PE_N-1];   // raw Q4_K code
    reg [15:0] p1_q8d [0:PE_N-1];   // selected Q8_0 fp16 d for this beat
    // P2 / P3 registers
    reg [31:0] p2_t   [0:PE_N-1];   // Q4_K d1*q | others: final wdeq
    reg [31:0] p2_m1  [0:PE_N-1];
    reg [31:0] p3_w   [0:PE_N-1];   // final wdeq (all types)
    // activation delay line (raw bf16; widen at P4 -- a pure function)
    reg [16*PE_M-1:0] a_d1, a_d2, a_d3;
    // P4 products
    reg [31:0] p4_p   [0:PE_M*PE_N-1];

    integer pi, pj, idx, col, blk;
    reg [KW-1:0] sb;                // super-block = k / 256
    reg [2:0]  sub;                 // sub-block within super-block = (k%256)/32
    reg [3:0]  sidx;                // Q6_K scale index = (k%256)/16
    reg [11:0] sm;                  // {min6, scale6}

    wire accept = busy && !draining && in_valid;

    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0; out_valid <= 1'b0; k_cnt <= 0; k_len_r <= 0;
            vp <= {PLAT{1'b0}}; draining <= 1'b0;
            a_d1 <= 0; a_d2 <= 0; a_d3 <= 0;
            for (idx = 0; idx < PE_M*PE_N; idx = idx + 1) begin
                acc[idx] <= 32'd0; p4_p[idx] <= 32'd0;
            end
            for (pj = 0; pj < PE_N; pj = pj + 1) begin
                p1_d1[pj] <= 32'd0; p1_m1[pj] <= 32'd0; p1_h16[pj] <= 16'd0;
                p1_q4[pj] <= 4'd0;  p1_q8d[pj] <= 16'd0;
                p2_t[pj] <= 32'd0;  p2_m1[pj] <= 32'd0; p3_w[pj] <= 32'd0;
            end
        end else begin
            out_valid <= 1'b0;
            vp <= {vp[PLAT-2:0], accept};

            if (start) begin
                busy    <= 1'b1;
                k_cnt   <= 0;
                k_len_r <= k_len;
                d_r     <= w_d;
                dmin_r  <= w_dmin;
                scales_r<= w_scales;
                wtype_r <= w_type;     // per-tile type select (off the per-beat path)
                q6sc_r  <= w_q6_sc;    // Q6_K scales (unread when the tile is Q4_K)
                q8d_r   <= w_q8_d;     // Q8_0 d      (unread when the tile is Q4_K)
                vp      <= {PLAT{1'b0}};
                draining<= 1'b0;
                for (idx = 0; idx < PE_M*PE_N; idx = idx + 1) acc[idx] <= 32'd0;
            end else begin
                // ---- P1: header select + header multiplies (on accept) ----
                if (accept) begin
                    sb  = k_cnt >> 8;                      // k / 256 (super-block)
                    // SHIFT forms (not part-selects) stay in-range for narrow
                    // k_cnt (KMAX<128) -- see the original narrow-K note.
                    sub = k_cnt >> 5;                      // (k%256) / 32
                    for (pj = 0; pj < PE_N; pj = pj + 1) begin
                        col = pj*NSB + sb;
                        case (wtype_r[2*pj +: 2])
                            WT_Q6K: begin
                                sidx = k_cnt >> 4;         // (k%256)/16
                                p1_d1[pj] <= fp32_mul(fp16_to_fp32(d_r[16*col +: 16]),
                                                      s8_to_fp32(q6sc_r[128*col + 8*sidx +: 8]));
                                p1_m1[pj] <= 32'd0;
                            end
                            WT_Q80: begin
                                blk = pj*NB8 + (k_cnt >> 5);
                                p1_q8d[pj] <= q8d_r[16*blk +: 16];
                                p1_d1[pj]  <= 32'd0;
                                p1_m1[pj]  <= 32'd0;
                            end
                            WT_F16: begin
                                p1_d1[pj] <= 32'd0;
                                p1_m1[pj] <= 32'd0;
                            end
                            default: begin                 // Q4_K (and undriven)
                                sm = q4k_scale_min({1'b0, sub}, scales_r[96*col +: 96]);
                                p1_d1[pj] <= fp32_mul(fp16_to_fp32(d_r[16*col +: 16]),
                                                      u7_to_fp32({1'b0, sm[5:0]}));
                                p1_m1[pj] <= fp32_mul(fp16_to_fp32(dmin_r[16*col +: 16]),
                                                      u7_to_fp32({1'b0, sm[11:6]}));
                            end
                        endcase
                        p1_h16[pj] <= w_hp[16*pj +: 16];
                        p1_q4[pj]  <= w_q[4*pj +: 4];
                    end
                    a_d1  <= a_col;
                    k_cnt <= k_cnt + 1'b1;
                    if (k_cnt + 1'b1 == k_len_r) draining <= 1'b1;
                end
                // ---- P2: the code multiply ----
                if (vp[0]) begin
                    for (pj = 0; pj < PE_N; pj = pj + 1) begin
                        case (wtype_r[2*pj +: 2])
                            WT_Q6K: p2_t[pj] <= fp32_mul(p1_d1[pj],
                                        s8_to_fp32({2'b00, p1_h16[pj][5:0]} - 8'd32));
                            WT_Q80: p2_t[pj] <= fp32_mul(fp16_to_fp32(p1_q8d[pj]),
                                        s8_to_fp32(p1_h16[pj][7:0]));
                            WT_F16: p2_t[pj] <= fp16_to_fp32(p1_h16[pj]);
                            default: p2_t[pj] <= fp32_mul(p1_d1[pj],
                                        u7_to_fp32({3'd0, p1_q4[pj]}));
                        endcase
                        p2_m1[pj] <= p1_m1[pj];
                    end
                    a_d2 <= a_d1;
                end
                // ---- P3: Q4_K min subtract; others forward ----
                if (vp[1]) begin
                    for (pj = 0; pj < PE_N; pj = pj + 1) begin
                        // case-with-default so an UNDRIVEN (x/z) w_type still
                        // takes the Q4_K arm -- the original decode's exact
                        // semantics (an `if (== WT_Q4K)` would evaluate x as
                        // false and silently skip the min subtract).
                        case (wtype_r[2*pj +: 2])
                            WT_Q6K, WT_Q80, WT_F16: p3_w[pj] <= p2_t[pj];
                            default: p3_w[pj] <= fp32_add(p2_t[pj],
                                         {~p2_m1[pj][31], p2_m1[pj][30:0]});
                        endcase
                    end
                    a_d3 <= a_d2;
                end
                // ---- P4: activation multiplies (PE_M x PE_N parallel) ----
                if (vp[2]) begin
                    for (pj = 0; pj < PE_N; pj = pj + 1)
                        for (pi = 0; pi < PE_M; pi = pi + 1)
                            p4_p[pi*PE_N + pj] <= fp32_mul(
                                bf16_to_fp32(a_d3[16*pi +: 16]), p3_w[pj]);
                end
                // ---- ACC: the (single-cycle) sequential fp32 accumulate ----
                if (vp[3])
                    for (idx = 0; idx < PE_M*PE_N; idx = idx + 1)
                        acc[idx] <= fp32_add(acc[idx], p4_p[idx]);
                // ---- drain complete -> the tile result is final ----
                if (draining && vp == {PLAT{1'b0}}) begin
                    busy      <= 1'b0;
                    draining  <= 1'b0;
                    out_valid <= 1'b1;
                end
            end
        end
    end

    // ---- output: round each fp32 accumulator to bf16 ----
    // out_valid is asserted the cycle the last beat's accumulate is registered; the
    // combinational read of acc[] below reflects the final sums on that same edge.
    always @(*) begin
        for (idx = 0; idx < PE_M*PE_N; idx = idx + 1)
            c_out[16*idx +: 16] = fp32_to_bf16(acc[idx]);
    end
endmodule
/* verilator lint_on DECLFILENAME */
