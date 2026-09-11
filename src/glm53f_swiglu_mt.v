//============================================================================
// glm53f_swiglu_mt.v -- GLM-5.3-Flash's clamped SwiGLU over MIXED-TYPE weights,
// streamed off glm_matmul_q4k.  The dense FFN and the MoE experts are the same
// datapath; only the weight TYPE differs, and it differs per tensor AND per block.
//
//   gate = Wg @ x        up = Wu @ x        (Q8_0, K = HIDDEN)
//   act  = silu(min(gate, +limit)) * clip(up, -limit, +limit)
//   y    = Wd @ act                          (Q8_0, K = INTER)
//
// WHY A SIBLING RATHER THAN swiglu_expert_q4k.  That unit's `w_q` port is FOUR
// BITS PER LANE, and the census says this FFN is Q8_0: blk.N.ffn_{gate,up,down}
// are [12288,4096] Q8_0 on the three dense-front blocks [scan].  It cannot carry
// these weights at all -- not "less accurately", at all.  It also carries PE_M
// batching, a DOWN/GATE/UP w_sel encoding and a GU_CONC second engine that this
// path does not need, and it is netlist-pinned.  So: a sibling, the same choice
// made for every other GLM-5.3-Flash module.
//
// THE CLAMP IS THE POINT, AND IT IS ASYMMETRIC.  swiglu_limit = 10.0 on every
// block; GLM-5.2 has no clamp at all, so an unclamped SwiGLU here is numerically
// WRONG, not approximate (tools/glm53_flash_ref.py: clamped_swiglu):
//     gate.clamp(min=None, max=+limit)     upper bound ONLY
//     up  .clamp(min=-limit, max=+limit)   both bounds
// Clamping the gate symmetrically is the plausible wrong reading -- both operands
// look like they deserve the same treatment -- and it only moves large-NEGATIVE
// gates, where silu is already near zero.  INJ_SWQ8_SYMCLAMP exists to reproduce
// that, but it is NOT a must-fail here and must not be added to one: measured, the
// difference reaches 0.25 against a tolerance of 2.9, and that tolerance is forced
// by glm_act's polynomial silu (the 0.02*mag term in the generator), for which no
// exact Python model exists.  A must-fail entry that cannot fail is worse than
// none.  The ASYMMETRY is gated on a tighter slice by `make swiglu`'s own
// INJ_SWIGLU_SYMCLAMP leg; what THIS gate checks is that the clamp is there at all
// -- INJ_SWQ8_NOCLAMP must fail, and it does by a wide margin, because unclamped
// gates reach +/-30 where silu is ~30 rather than ~10.
//
// WHY THE TYPE IS A RUNTIME INPUT AND NOT A PARAMETER.  The census gives the
// dense FFN as Q8_0, but the MoE experts as a MIX that varies per tensor and per
// BLOCK: ffn_gate_exps / ffn_up_exps are Q4_K on 42 blocks and Q5_K on one,
// ffn_down_exps is Q5_K on 40 and Q6_K on 3, and the shared expert is Q8_0. One
// instantiation therefore cannot carry a compile-time type. Hence wt_gate /
// wt_up / wt_down, and the FULL weight-response bundle -- exactly the shape
// weight_loader_q4k already emits, so the system side needs nothing new.
//   The engine reads a DIFFERENT bus per type (Q4_K's 4-bit code on w_q; Q5_K,
// Q6_K, Q8_0 and F16 on w_hp; headers on w_d/w_dmin/w_scales, w_q6_sc, w_q8_d),
// so all of them are passed through rather than tied off. w_type is always driven
// explicitly: an UNDRIVEN w_type reads as Q4_K, which makes "forgot to drive it"
// and "drove it wrong" look identical downstream.
//
// TN AND THE TAIL: output rows are produced TN at a time and there is no partial
// final group, so INTER and HIDDEN must both be multiples of TN. Checked at
// elaboration rather than assumed (12288 and 4096 are, for any sane TN).
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_SWIGLU_MT_V
`define GLM53F_SWIGLU_MT_V
`include "glm_fp.vh"

module glm53f_swiglu_mt #(
    parameter integer HIDDEN = 16,
    parameter integer INTER  = 32,
    parameter integer TN     = 2,
    parameter integer KMAX   = 32,          // >= max(HIDDEN, INTER)
    parameter [15:0]  LIM    = 16'h4120,    // 10.0 bf16 = swiglu_limit
    parameter integer ACT_HW = 0
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    start,
    output reg                     busy,
    output reg                     done,

    input  wire [16*HIDDEN-1:0]    x_in,     // bf16
    output wire                    w_req,
    output wire [1:0]              w_sel,    // 0 = GATE, 1 = UP, 2 = DOWN
    output wire [$clog2((INTER>HIDDEN?INTER:HIDDEN)/TN+1)-1:0] w_grp,
    output wire [$clog2(KMAX+1)-1:0] w_k,
    // per-pass weight type: 0=Q4_K 1=Q6_K 2=Q8_0 3=F16 4=Q5_K
    input  wire [2:0]              wt_gate,
    input  wire [2:0]              wt_up,
    input  wire [2:0]              wt_down,
    // the loader's full response bundle -- which bus is read depends on the type
    input  wire [4*TN-1:0]         w_q,      // Q4_K 4-bit codes
    input  wire [16*TN-1:0]        w_hp,     // Q5_K/Q6_K/Q8_0/F16 code lane
    input  wire [16*TN*((KMAX+255)/256)-1:0] w_d,
    input  wire [16*TN*((KMAX+255)/256)-1:0] w_dmin,
    input  wire [96*TN*((KMAX+255)/256)-1:0] w_scales,
    input  wire [128*TN*((KMAX+255)/256)-1:0] w_q6_sc,
    input  wire [16*TN*((KMAX+31)/32)-1:0]    w_q8_d,

    output reg  [16*HIDDEN-1:0]    y_out
);
    localparam integer NSB = (KMAX + 255) / 256;
    localparam integer KW  = $clog2(KMAX + 1);
    localparam integer GW  = $clog2((INTER > HIDDEN ? INTER : HIDDEN)/TN + 1);

`ifndef YOSYS
    initial begin
        if (INTER % TN != 0 || HIDDEN % TN != 0)
            $fatal(1, "glm53f_swiglu_q8: INTER and HIDDEN must be multiples of TN; there is no partial-group tail path");
        if (INTER > KMAX || HIDDEN > KMAX)
            $fatal(1, "glm53f_swiglu_q8: KMAX must cover both K = HIDDEN and K = INTER");
    end
`endif

    reg [15:0] gate_r [0:INTER-1];
    reg [15:0] up_r   [0:INTER-1];
    reg [15:0] act_r  [0:INTER-1];

    localparam [2:0] S_IDLE=3'd0, S_PREP=3'd1, S_STREAM=3'd2, S_WAIT=3'd3,
                     S_ACT=3'd4,  S_FIN=3'd5;
    reg [2:0]  st;
    reg [1:0]  pass;                       // 0 GATE, 1 UP, 2 DOWN
    reg [KW-1:0] kcnt, klen;
    reg [GW-1:0] grp, ngrp;

    reg           mm_start;
    reg  [KW-1:0] mm_k_len;
    wire          mm_busy, mm_ov;
    wire [16*TN-1:0] mm_c;

    wire stream = (st == S_STREAM);
    assign w_req = stream;
    assign w_sel = pass;
    assign w_grp = grp;
    assign w_k   = kcnt;

    // GATE/UP read x; DOWN reads the activation
    wire [15:0] a_col = (pass == 2'd2) ? act_r[kcnt] : x_in[16*kcnt +: 16];

`ifdef INJ_SWMT_PASS_TYPE
    // must FAIL: every pass uses the GATE's type, i.e. the per-pass mux collapsed.
    // This is a strictly narrower claim than INJ_SWQ8_Q4K_TYPE: the type is still
    // a runtime input and still reaches the engine, only the PASS SELECT is wrong.
    // On the checkpoint's (Q4_K,Q4_K,Q5_K) expert that is invisible on two of the
    // three passes -- which is exactly why the mixed-type corpus has to contain a
    // combo whose down type differs from its gate type. Both of ours do.
    wire [2:0] wt_now = wt_gate;
`else
    wire [2:0] wt_now = (pass == 2'd0) ? wt_gate : (pass == 2'd1) ? wt_up : wt_down;
`endif
`ifdef INJ_SWQ8_Q4K_TYPE
    // must FAIL: w_type forced to Q4_K regardless of what the caller asked for --
    // which is also what an UNDRIVEN w_type reads as.
    wire [3*TN-1:0] w_type_b = {TN{3'd0}};
`else
    wire [3*TN-1:0] w_type_b = {TN{wt_now}};
`endif

    glm_matmul_q4k #(.PE_M(1), .PE_N(TN), .KMAX(KMAX)) u_mm (
        .clk(clk), .rst(rst), .start(mm_start), .k_len(mm_k_len),
        .w_d(w_d), .w_dmin(w_dmin), .w_scales(w_scales),
        .in_valid(stream), .a_col(a_col), .w_q(w_q),
        .busy(mm_busy), .out_valid(mm_ov), .c_out(mm_c),
        .w_type(w_type_b), .w_hp(w_hp),
        .w_q6_sc(w_q6_sc), .w_q8_d(w_q8_d));

    // ---- the asymmetric clamp, on wires at the two consumption points --------
    //   bf16 is fp32's top 16 bits, so for finite values an unsigned compare of
    //   the low 15 bits orders magnitudes exactly. NaN/Inf saturate to +/-limit
    //   here where torch.clamp would propagate NaN -- recorded, not papered over;
    //   same policy as swiglu_expert_q4k.
    function automatic [15:0] bf16_clamp_hi(input [15:0] x, input [15:0] lim);
        bf16_clamp_hi = (!x[15] && (x[14:0] > lim[14:0])) ? {1'b0, lim[14:0]} : x;
    endfunction
    function automatic [15:0] bf16_clamp_sym(input [15:0] x, input [15:0] lim);
        bf16_clamp_sym = (x[14:0] > lim[14:0]) ? {x[15], lim[14:0]} : x;
    endfunction

    // AW must cover INTER itself, not INTER-1: the `ao == INTER-1` terminator and
    // the `ai < INTER` guard both compare against the full count.  These were
    // `reg [7:0]` with `INTER[7:0]` comparisons until 2026-09-11, which TRUNCATES
    // TO ZERO at INTER = 256 -- `ai < 0` is never true, act_iv never fires, and
    // S_ACT hangs forever.  `make swiglu-q8` runs INTER = 32 and could not see it;
    // the real dense FFN is INTER = 12288, which truncates to 0 as well, so this
    // was a latent defect on the actual model, not a test-slice artifact.
    localparam integer AW = $clog2(INTER + 1);
    reg  [AW-1:0] ai, ao;
    reg        act_iv;
    reg  [15:0] act_x;
    wire        act_ov;
    wire [15:0] act_y;
    glm_act #(.MODE(1), .LANES(1), .HW_LANES(ACT_HW)) u_silu (
        .clk(clk), .rst(rst), .in_valid(act_iv), .x_in(act_x),
        .out_valid(act_ov), .y_out(act_y));

    // The clamped operands are functions of an EXPLICIT index and are latched
    // together with the valid they belong to: driving act_x combinationally off
    // `ai` while `ai` advances puts the operand one lane ahead of its own
    // in_valid, which is the kind of off-by-one a tolerance gate would absorb.
    function automatic [15:0] gate_eff(input [15:0] g);
        begin
`ifdef INJ_SWQ8_NOCLAMP
            gate_eff = g;
`elsif INJ_SWQ8_SYMCLAMP
            gate_eff = bf16_clamp_sym(g, LIM);   // must FAIL: gate clamped both ways
`else
            gate_eff = bf16_clamp_hi(g, LIM);
`endif
        end
    endfunction
    function automatic [15:0] up_eff(input [15:0] u);
        begin
`ifdef INJ_SWQ8_NOCLAMP
            up_eff = u;
`else
            up_eff = bf16_clamp_sym(u, LIM);
`endif
        end
    endfunction

    integer gi;
    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0; mm_start <= 1'b0;
            act_iv <= 1'b0; grp <= 0; kcnt <= 0; pass <= 2'd0; ai <= 0; ao <= 0;
        end else begin
            done <= 1'b0; mm_start <= 1'b0; act_iv <= 1'b0;
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1'b1; pass <= 2'd0; grp <= 0;
                    klen <= HIDDEN[KW-1:0]; mm_k_len <= HIDDEN[KW-1:0];
                    ngrp <= (INTER/TN);
                    mm_start <= 1'b1; st <= S_PREP;
                end
                S_PREP: begin kcnt <= 0; st <= S_STREAM; end
                S_STREAM: begin
                    if (kcnt == klen - 1'b1) st <= S_WAIT;
                    kcnt <= kcnt + 1'b1;
                end
                S_WAIT: if (mm_ov) begin
                    for (gi = 0; gi < TN; gi = gi + 1) begin
                        if (pass == 2'd0)      gate_r[grp*TN + gi] <= mm_c[16*gi +: 16];
                        else if (pass == 2'd1) up_r  [grp*TN + gi] <= mm_c[16*gi +: 16];
                        else                   y_out[16*(grp*TN + gi) +: 16] <= mm_c[16*gi +: 16];
                    end
                    if (grp == ngrp - 1'b1) begin
                        grp <= 0;
                        if (pass == 2'd0) begin
                            pass <= 2'd1; mm_start <= 1'b1; st <= S_PREP;
                        end else if (pass == 2'd1) begin
                            ai <= 0; ao <= 0; st <= S_ACT;
                        end else begin
                            st <= S_FIN;
                        end
                    end else begin
                        grp <= grp + 1'b1; mm_start <= 1'b1; st <= S_PREP;
                    end
                end

                // silu(clamped gate) streamed one lane per cycle, multiplied by
                // the clamped up as each result emerges
                S_ACT: begin
                    if (ai < INTER[AW-1:0]) begin
                        act_iv <= 1'b1;
                        act_x  <= gate_eff(gate_r[ai]);
                        ai     <= ai + 1'b1;
                    end
                    if (act_ov) begin
                        act_r[ao] <= bf16_mul(act_y, up_eff(up_r[ao]));
                        ao <= ao + 1'b1;
                        if (ao == INTER[AW-1:0] - 1'b1) begin
                            pass <= 2'd2; grp <= 0;
                            klen <= INTER[KW-1:0]; mm_k_len <= INTER[KW-1:0];
                            ngrp <= (HIDDEN/TN);
                            mm_start <= 1'b1; st <= S_PREP;
                        end
                    end
                end

                S_FIN: begin done <= 1'b1; busy <= 1'b0; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // GLM53F_SWIGLU_MT_V
