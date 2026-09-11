//============================================================================
// glm53f_moe_ffn.v -- GLM-5.3-Flash's MoE FFN: route, run the chosen experts,
// add the always-on shared expert.
//
//     idx, w = router(x)                       glm53f_moe_router  (4.3t)
//     y      = SUM_i w[i] * expert_idx[i](x)  +  shared(x)
//
// where each expert is a clamped SwiGLU -- glm53f_swiglu_mt (4.3s), ONE instance
// reused serially across all TOPK+1 evaluations.  That matches what
// glm_decoder_block_q4k already does for GLM-5.2: one expert's weights are
// fetched per evaluation and shared by every row.
//
// WHY A SIBLING.  The GLM-5.2 MoE path is fused inside glm_decoder_block_q4k with
// its own router, its own Q4_K-only expert, no exp_probs_b and no clamp.  None of
// those four things holds here, and its `w_q` port is four bits per lane.
//
// ---- THE SHARED EXPERT IS ADDED WITH WEIGHT 1 ----
// `expert_weights_scale` = 2.5 scales the ROUTED weights only.  This is not a
// guess: glm_decoder_block_q4k's header already transcribes the same rule for
// GLM-5.2 (`combine = SUM_e gate_e * y_e + y_shared`), and glm53_flash_ref's
// self-test pins it by zeroing every routed expert and requiring y == shared(x)
// EXACTLY.  INJ_MOE_SHARED_WEIGHTED is the must-fail leg for it.
//
// ---- THE SHARED EXPERT HAS ITS OWN WEIGHT TYPES ----
// [scan] the routed experts are Q4_K/Q5_K/Q6_K and the shared expert is Q8_0, so
// `wt_sh_*` is a SEPARATE input triple from `wt_*`, not the same one reused.  The
// gate drives two different types across that boundary for exactly this reason.
//
// ---- ACCUMULATION ORDER: ASCENDING EXPERT INDEX, SHARED LAST -- AND NOT GATED ----
// The router emits its TOPK in SCORE-DESCENDING order (topk_select's), which is
// not the order glm53_flash_ref.moe_ffn accumulates in.  Rather than build an
// 8-element sorter, the FSM SCANS ecur = 0 .. E-1 and evaluates an expert when it
// is in the selected set: ascending by construction, and the E scan cycles are
// nothing against TOPK+1 expert evaluations.
//   There is DELIBERATELY NO must-fail leg for this.  fp add does not associate,
// so the order is observable in principle -- but MEASURED at the bf16 output it
// is not: ascending and score-descending accumulation gave BITWISE IDENTICAL
// results on 16/16 draws across four configurations (E=8/16/32, TOPK=3/8,
// INTER=64/128), worst relative difference exactly 0.  The final fp32->bf16
// rounding swallows the reordering.  An injection here would pass and prove
// nothing, so the order is pinned for DETERMINISM and the generator's self-test
// carries a tripwire that fires if the two orders ever stop agreeing.
//
// Accumulation is fp32 (`glm_fp.vh`), one element per cycle in S_ACC; each
// expert's own output is bf16, because that is the port glm53f_swiglu_mt has.
// glm53_flash_ref.moe_ffn(expert_out_bf16=True) models that rounding exactly.
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_MOE_FFN_V
`define GLM53F_MOE_FFN_V
`include "glm_fp.vh"

module glm53f_moe_ffn #(
    parameter integer HIDDEN = 16,
    parameter integer INTER  = 32,              // moe_intermediate_size (real 2048)
    parameter integer E      = 8,               // expert_count (real 288)
    parameter integer TOPK   = 3,               // expert_used_count (real 8)
    parameter integer TN     = 2,               // expert matmul lanes
    parameter integer KMAX   = 32,
    parameter [31:0]  SCALE  = 32'h40200000,    // expert_weights_scale = 2.5
    parameter [15:0]  LIMIT  = 16'h4120,        // swiglu_clamp_* = 10.0 (bf16)
    parameter integer RECIP_ITERS = 4,
    parameter integer IDXW   = (E <= 1) ? 1 : $clog2(E)
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    start,
    output reg                     busy,
    output reg                     done,

    input  wire [16*HIDDEN-1:0]    x_in,

    // ---- router weight stream (ffn_gate_inp, F32) ----
    output wire                    rw_req,
    output wire [$clog2(HIDDEN+1)-1:0] rw_k,
    input  wire [32*E-1:0]         rw_row,
    input  wire [32*E-1:0]         bias_in,     // exp_probs_b, F32

    // ---- expert weight stream: WHICH expert, then the usual bundle ----
    output wire                    fw_req,
    output reg                     fw_shared,   // 1 = the always-on shared expert
    output reg  [IDXW-1:0]         fw_eidx,
    output wire [1:0]              fw_sel,      // 0 GATE, 1 UP, 2 DOWN
    output wire [$clog2(((INTER>HIDDEN?INTER:HIDDEN)/TN)+1)-1:0] fw_grp,
    output wire [$clog2(KMAX+1)-1:0] fw_k,

    input  wire [2:0]              wt_gate, wt_up, wt_down,        // routed
    input  wire [2:0]              wt_sh_gate, wt_sh_up, wt_sh_down, // shared

    input  wire [4*TN-1:0]         w_q,
    input  wire [16*TN-1:0]        w_hp,
    input  wire [16*TN*((KMAX+255)/256)-1:0]  w_d, w_dmin,
    input  wire [96*TN*((KMAX+255)/256)-1:0]  w_scales,
    input  wire [128*TN*((KMAX+255)/256)-1:0] w_q6_sc,
    input  wire [16*TN*((KMAX+31)/32)-1:0]    w_q8_d,

    output reg  [16*HIDDEN-1:0]    y_out,
    output reg  [TOPK*IDXW-1:0]    dbg_sel_idx,   // observability for the TB
    output reg  [TOPK*16-1:0]      dbg_sel_weight
);
    localparam integer EW  = (E <= 1) ? 1 : $clog2(E + 1);
    localparam integer HW  = $clog2(HIDDEN + 1);
    localparam [31:0]  F32_ONE = 32'h3F800000;

    // ---------------- router ----------------
    reg         rt_start;
    wire        rt_busy, rt_done;
    wire [TOPK*IDXW-1:0] rt_idx;
    wire [TOPK*16-1:0]   rt_w;

    glm53f_moe_router #(.HIDDEN(HIDDEN), .E(E), .TOPK(TOPK), .SCALE(SCALE),
                        .RECIP_ITERS(RECIP_ITERS), .IDXW(IDXW)) u_rt (
        .clk(clk), .rst(rst), .start(rt_start), .busy(rt_busy), .done(rt_done),
        .x_in(x_in), .w_req(rw_req), .w_k(rw_k), .w_row(rw_row), .bias_in(bias_in),
        .sel_idx(rt_idx), .sel_weight(rt_w));

    // ---------------- one reused expert ----------------
    reg         ex_start;
    wire        ex_busy, ex_done;
    wire [16*HIDDEN-1:0] ex_y;
    reg  [2:0]  ex_wg, ex_wu, ex_wd;

    glm53f_swiglu_mt #(.HIDDEN(HIDDEN), .INTER(INTER), .TN(TN), .KMAX(KMAX),
                       .LIM(LIMIT)) u_ex (
        .clk(clk), .rst(rst), .start(ex_start), .busy(ex_busy), .done(ex_done),
        .x_in(x_in), .wt_gate(ex_wg), .wt_up(ex_wu), .wt_down(ex_wd),
        .w_req(fw_req), .w_sel(fw_sel), .w_grp(fw_grp), .w_k(fw_k),
        .w_q(w_q), .w_hp(w_hp), .w_d(w_d), .w_dmin(w_dmin), .w_scales(w_scales),
        .w_q6_sc(w_q6_sc), .w_q8_d(w_q8_d), .y_out(ex_y));

    // ---------------- state ----------------
    localparam [2:0] S_IDLE=3'd0, S_ROUTE=3'd1, S_SCAN=3'd2, S_RUN=3'd3,
                     S_ACC=3'd4,  S_SHARED=3'd5, S_OUT=3'd6, S_FIN=3'd7;
    reg [2:0]  st;
    reg [EW-1:0]  ecur;                 // ascending scan cursor over all E
    reg [HW-1:0]  ai;                   // accumulate cursor
    reg [31:0]    acc [0:HIDDEN-1];
    reg [31:0]    cur_w;                // fp32 weight for the evaluation in flight
    reg [TOPK*IDXW-1:0] sidx;
    reg [TOPK*16-1:0]   sw;
    integer t, h;

    // is ecur one of the selected experts, and if so with what weight?
    //
    // The must-fail legs live here and at the shared-expert arm.  Each was
    // MEASURED in Python against the golden before being written, on 3 draws x 32
    // outputs: WEIGHT_ROTATE 96/96 outputs differ, SHARED_TYPE 96/96,
    // WRONG_SELECT 96/96, SHARED_WEIGHTED 94/96.  Writing an injection first and
    // hoping it fires is how INJ_MOE_ROUTER_ORDER nearly got in -- see the header.
    reg               hit;
    reg [15:0]        hit_w;
    always @* begin
        hit = 1'b0; hit_w = 16'd0;
`ifdef INJ_MOE_WRONG_SELECT
        // must FAIL: ignore the router and run experts 0 .. TOPK-1 with the
        // router's weights in emission order.  The weights are right, the
        // EXPERTS are wrong.
        if (ecur < TOPK[EW-1:0]) begin
            hit   = 1'b1;
            hit_w = sw[16*ecur[$clog2(TOPK>1?TOPK:2)-1:0] +: 16];
        end
`else
        for (t = 0; t < TOPK; t = t + 1)
            if (sidx[IDXW*t +: IDXW] == ecur[IDXW-1:0]) begin
                hit   = 1'b1;
`ifdef INJ_MOE_WEIGHT_ROTATE
                // must FAIL: right experts, right weights, WRONG PAIRING -- each
                // selected expert gets the next one's weight.  This is the leg
                // that says the weight follows its own index rather than the
                // scan position.
                hit_w = sw[16*((t + 1) % TOPK) +: 16];
`else
                hit_w = sw[16*t +: 16];
`endif
            end
`endif
    end

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            rt_start <= 1'b0; ex_start <= 1'b0;
            fw_shared <= 1'b0; fw_eidx <= {IDXW{1'b0}};
            ecur <= 0; ai <= 0; cur_w <= F32_ONE;
        end else begin
            done <= 1'b0; rt_start <= 1'b0; ex_start <= 1'b0;
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1'b1; rt_start <= 1'b1; st <= S_ROUTE;
                    for (h = 0; h < HIDDEN; h = h + 1) acc[h] <= 32'd0;
                end

                S_ROUTE: if (rt_done) begin
                    sidx <= rt_idx; sw <= rt_w;
                    dbg_sel_idx <= rt_idx; dbg_sel_weight <= rt_w;
                    ecur <= 0; st <= S_SCAN;
                end

                // ascending scan: evaluate ecur if it was selected, else step on
                S_SCAN: begin
                    if (ecur == E[EW-1:0]) begin
                        fw_shared <= 1'b1;
                        fw_eidx   <= {IDXW{1'b0}};
`ifdef INJ_MOE_SHARED_TYPE
                        // must FAIL: the shared expert decoded with the ROUTED
                        // types.  [scan] says routed are Q4_K/Q5_K/Q6_K and the
                        // shared one is Q8_0, so these two triples are genuinely
                        // separate inputs -- this leg is what says so.
                        ex_wg <= wt_gate; ex_wu <= wt_up; ex_wd <= wt_down;
`else
                        ex_wg <= wt_sh_gate; ex_wu <= wt_sh_up; ex_wd <= wt_sh_down;
`endif
`ifdef INJ_MOE_SHARED_WEIGHTED
                        // must FAIL: the shared expert scaled by a routed weight
                        // instead of 1.  expert_weights_scale applies to the
                        // ROUTED weights only.
                        cur_w <= bf16_to_fp32(sw[15:0]);
`else
                        cur_w <= F32_ONE;                  // weight 1, see header
`endif
                        ex_start <= 1'b1; st <= S_SHARED;
                    end else if (hit) begin
                        fw_eidx <= ecur[IDXW-1:0];
                        fw_shared <= 1'b0;
                        ex_wg <= wt_gate; ex_wu <= wt_up; ex_wd <= wt_down;
                        cur_w <= bf16_to_fp32(hit_w);
                        ex_start <= 1'b1; st <= S_RUN;
                    end else begin
                        ecur <= ecur + 1'b1;
                    end
                end

                S_RUN:    if (ex_done) begin ai <= 0; st <= S_ACC; end
                S_SHARED: if (ex_done) begin ai <= 0; st <= S_ACC; end

                // acc[i] += cur_w * expert_y[i], one element per cycle, fp32
                S_ACC: begin
                    acc[ai] <= fp32_add(acc[ai],
                                        fp32_mul(cur_w, bf16_to_fp32(ex_y[16*ai +: 16])));
                    if (ai == HIDDEN[HW-1:0] - 1'b1) begin
                        ai <= 0;
                        if (fw_shared) st <= S_OUT;        // shared was last
                        else begin ecur <= ecur + 1'b1; st <= S_SCAN; end
                    end else ai <= ai + 1'b1;
                end

                S_OUT: begin
                    for (h = 0; h < HIDDEN; h = h + 1)
                        y_out[16*h +: 16] <= fp32_to_bf16(acc[h]);
                    st <= S_FIN;
                end

                S_FIN: begin done <= 1'b1; busy <= 1'b0; fw_shared <= 1'b0;
                             st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // GLM53F_MOE_FFN_V
