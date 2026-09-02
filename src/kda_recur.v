//============================================================================
// kda_recur.v -- Kimi Delta Attention recurrent state update, ONE decode token.
//
// This is the machine GLM-5.3-Flash needs that this repo did not have: 34 of its
// 45 layers are KDA linear attention, and they carry a FIXED-SIZE recurrent
// state instead of a growing KV cache.  Golden: tools/glm53_flash_ref.py
// `kda_step`, a transcription of `recurrent_kimi_delta_attention` (seq_len == 1).
//
// PER HEAD h, with state S[DK][DV]:
//     qn = l2norm(q) * DK^-0.5      kn = l2norm(k)      gi = exp(g)
//     S[d][e] *= gi[d]                          <- decay BEFORE the kv read
//     kv[e]    = SUM_d S[d][e]*kn[d]
//     delta[e] = (v[e] - kv[e]) * beta
//     S[d][e] += kn[d]*delta[e]                 <- delta rule, not a plain write
//     out[e]   = SUM_d S[d][e]*qn[d]
//
// ACCURACY CONTRACT -- do not call this bit-exact without reading this.
//   EXACT=1 : q/k arrive already l2-normed and g arrives already exponentiated,
//             so everything this module does is fp32 mul/add, which src/glm_fp.vh
//             fp32_mul / fp32_add implement EXACTLY.  This leg is bit-exact to
//             the golden.
//   EXACT=0 : the module computes l2norm and exp itself, via fp32_rsqrt (Quake
//             inverse sqrt, 2 Newton iterations) and a Horner exp.  Both are
//             APPROXIMATIONS, so this leg is checkable only to a TOLERANCE --
//             the same status swiglu_expert_q4k has.  Two legs, two honest claims.
//
//   REDUCTION ORDER is part of the contract: both SUM_d run in ASCENDING d,
//   sequentially.  numpy's pairwise .sum() differs from a sequential accumulate
//   in >half of random length-8 fp32 cases (measured), so the golden pins the
//   sequential order too (tools/glm53_flash_ref.py _seq_sum).  torch/FLA reduce
//   in their own blocked order; whole-runtime equality with them is out of
//   contract, exactly as it is for llama.cpp.
//
// STATE TRAFFIC NOTE (the microarchitecture point).  The decayed state is needed
// twice -- once to form kv, once to be updated.  Storing it back after the decay
// costs 2 reads + 2 writes of S per token; RECOMPUTING the decay in the second
// pass costs 2 reads + 1 write and one extra multiply.  At the real shape the
// state is 4.19 MB per layer and 285 MB/token of traffic across 34 layers
// (docs/HARDWARE_LADDER.md), so the write is the expensive half: this module
// takes the recompute path.  RECOMPUTE=0 selects the store-back variant, kept
// because it is the obvious reading and the two must agree bit-exactly -- which
// is a gate leg, not an assumption.
//============================================================================
`ifndef KDA_RECUR_V
`define KDA_RECUR_V
`include "glm_fp.vh"

module kda_recur #(
    parameter integer H     = 3,    // heads
    parameter integer DK    = 8,    // key dim per head
    parameter integer DV    = 8,    // value dim per head
    parameter integer EXACT = 1,    // 1 = pre-normed q/k + pre-exp g (bit-exact leg)
    parameter integer RECOMPUTE = 1, // 1 = recompute decay in pass B (1 write)
    // fp32 bit pattern of DK^-0.5.  A PARAMETER, not something this module
    // computes: src/glm_fp.vh fp32_rsqrt is the Quake approximation, and deriving
    // the q scale from it would put an approximation inside the leg that claims to
    // be bit-exact.  The caller passes the exactly-rounded constant (the golden's
    // np.float32(1/sqrt(DK))).  Default is DK=8: 1/sqrt(8) = 0x3EB504F3.
    parameter [31:0] INV_SQRT_DK = 32'h3EB504F3
)(
    input  wire                    clk,
    input  wire                    rst,        // sync, active-high
    input  wire                    start,      // 1-cycle: operands are valid
    output reg                     busy,
    output reg                     done,       // 1-cycle pulse; out_v valid

    // per-token operands, fp32, head-major
    input  wire [32*H*DK-1:0]      q_in,       // raw q      (EXACT=0) or l2-normed q (EXACT=1)
    input  wire [32*H*DK-1:0]      k_in,       // raw k      (EXACT=0) or l2-normed k (EXACT=1)
    input  wire [32*H*DK-1:0]      g_in,       // log-decay  (EXACT=0) or exp(g)      (EXACT=1)
    input  wire [32*H*DV-1:0]      v_in,
    input  wire [32*H-1:0]         beta_in,

    // state, carried by the caller (a real layer holds this in BRAM/DDR)
    input  wire [32*H*DK*DV-1:0]   s_in,
    output reg  [32*H*DK*DV-1:0]   s_out,
    output reg  [32*H*DV-1:0]      out_v
);
    localparam integer SN = H*DK*DV;

    integer h, d, e;
    reg [31:0] qn   [0:H*DK-1];
    reg [31:0] kn   [0:H*DK-1];
    reg [31:0] ge   [0:H*DK-1];
    reg [31:0] st   [0:SN-1];
    reg [31:0] kv   [0:DV-1];
    reg [31:0] dl   [0:DV-1];
    reg [31:0] acc, prod, nrm;

    localparam [2:0] S_IDLE=3'd0, S_PREP=3'd1, S_HEAD=3'd2, S_DONE=3'd3;
    reg [2:0]  state;
    reg [$clog2(H+1)-1:0] hi;


    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0; hi <= 0;
        end else begin
            done <= 1'b0;
            case (state)
            S_IDLE: if (start) begin
                busy <= 1'b1; hi <= 0; state <= S_PREP;
                for (d = 0; d < SN; d = d + 1) st[d] <= s_in[32*d +: 32];
            end

            // ---- operand prep: l2norm + exp, or pass-through on the exact leg
            S_PREP: begin
                for (h = 0; h < H; h = h + 1) begin
                    if (EXACT != 0) begin
                        for (d = 0; d < DK; d = d + 1) begin
                            qn[h*DK+d] <= fp32_mul(q_in[32*(h*DK+d) +: 32], INV_SQRT_DK);
                            kn[h*DK+d] <= k_in[32*(h*DK+d) +: 32];
                            ge[h*DK+d] <= g_in[32*(h*DK+d) +: 32];
                        end
                    end else begin
                        // l2norm: eps INSIDE the sqrt, sequential sum of squares
                        nrm = 32'h00000000;
                        for (d = 0; d < DK; d = d + 1)
                            nrm = fp32_add(nrm, fp32_mul(q_in[32*(h*DK+d) +: 32],
                                                         q_in[32*(h*DK+d) +: 32]));
                        nrm = fp32_rsqrt(fp32_add(nrm, 32'h358637BD));   // +1e-6
                        for (d = 0; d < DK; d = d + 1)
                            qn[h*DK+d] <= fp32_mul(fp32_mul(q_in[32*(h*DK+d) +: 32], nrm),
                                                   INV_SQRT_DK);
                        nrm = 32'h00000000;
                        for (d = 0; d < DK; d = d + 1)
                            nrm = fp32_add(nrm, fp32_mul(k_in[32*(h*DK+d) +: 32],
                                                         k_in[32*(h*DK+d) +: 32]));
                        nrm = fp32_rsqrt(fp32_add(nrm, 32'h358637BD));
                        for (d = 0; d < DK; d = d + 1)
                            kn[h*DK+d] <= fp32_mul(k_in[32*(h*DK+d) +: 32], nrm);
                        // EXACT=0 expects the caller to still supply exp(g): a
                        // Horner exp belongs in fp32_exp_pipe, not inlined here.
                        for (d = 0; d < DK; d = d + 1)
                            ge[h*DK+d] <= g_in[32*(h*DK+d) +: 32];
                    end
                end
                state <= S_HEAD;
            end

            // ---- one head per cycle (the slice is small; a real layer pipelines
            //      the DK x DV loop and streams the state from memory)
            S_HEAD: begin
                // pass A: decay the state, reduce kv in ASCENDING d
                for (e = 0; e < DV; e = e + 1) begin
                    acc = 32'h00000000;
                    for (d = 0; d < DK; d = d + 1) begin
                        prod = fp32_mul(st[hi*DK*DV + d*DV + e], ge[hi*DK+d]);
                        if (RECOMPUTE == 0) st[hi*DK*DV + d*DV + e] <= prod;
                        acc = fp32_add(acc, fp32_mul(prod, kn[hi*DK+d]));
                    end
                    kv[e] = acc;
                    // delta = (v - kv) * beta   -- sign flip is the fp32 negate
                    dl[e] = fp32_mul(fp32_add(v_in[32*(hi*DV+e) +: 32],
                                              {~kv[e][31], kv[e][30:0]}),
                                     beta_in[32*hi +: 32]);
                end
                // pass B: update with the outer product, reduce out in ASCENDING d
                for (e = 0; e < DV; e = e + 1) begin
                    acc = 32'h00000000;
                    for (d = 0; d < DK; d = d + 1) begin
                        // RECOMPUTE: re-apply the decay instead of having stored it
                        // INJECTION (never a normal build): drop the pass-B decay.
                        // The decay is applied BEFORE the kv read, so pass B must
                        // see the decayed state too; omitting it leaves the state
                        // un-decayed by exactly one step -- a small, compounding
                        // error that a loose tolerance would swallow.
`ifdef INJ_KDA_NODECAY
                        prod = st[hi*DK*DV + d*DV + e];
`else
                        prod = (RECOMPUTE != 0)
                             ? fp32_mul(st[hi*DK*DV + d*DV + e], ge[hi*DK+d])
                             : st[hi*DK*DV + d*DV + e];
`endif
                        prod = fp32_add(prod, fp32_mul(kn[hi*DK+d], dl[e]));
                        st[hi*DK*DV + d*DV + e] <= prod;
                        acc = fp32_add(acc, fp32_mul(prod, qn[hi*DK+d]));
                    end
                    out_v[32*(hi*DV+e) +: 32] <= acc;
                end
                if (hi == H[$clog2(H+1)-1:0] - 1'b1) state <= S_DONE;
                else hi <= hi + 1'b1;
            end

            S_DONE: begin
                for (d = 0; d < SN; d = d + 1) s_out[32*d +: 32] <= st[d];
                done <= 1'b1; busy <= 1'b0; state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // KDA_RECUR_V
