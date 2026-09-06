//============================================================================
// glm53f_kda_layer.v -- ONE Kimi Delta Attention layer: the sublayer that sits
// where MLA+DSA sits in the other 11 blocks, for 34 of GLM-5.3-Flash's 45.
//
// It composes the four proven non-GEMV units and OWNS the two pieces of state
// that make KDA not a drop-in for mla_attn_q4k:
//     s_state    [H, DK, DV] fp32   the delta-rule recurrence -- fixed size, NOT
//                                   a growing KV cache (this is why only 11 of 45
//                                   layers page KV at all)
//     conv_hist  [3*H*DK, K-1] fp32 the short causal conv's history
//
// SEQUENCE (pinned from tools/glm53_flash_ref.py, not inferred):
//     q,k,v      = attn_{q,k,v} @ x
//     qkv        = SiLU(depthwise causal conv over the CONCATENATION of q,k,v)
//                  -- ONE conv, C = 3*H*DK, per the reference's causal_conv_step
//     beta       = sigma( ssm_beta @ x )                       [H], one per head
//     f          = ssm_f_b( ssm_f_a( x ) ) + dt_bias           [H, DK]
//     g_gate     = ssm_g_b( ssm_g_a( x ) )                     [H, DV]
//     g, beta    = forget_gate(f, dt_bias, decay=exp(A_log))   decay is a PER-LAYER
//                  constant, so exp(A_log) is precomputed by the host and arrives
//                  on `decay_in` -- kda_gate_step already expects it that way
//     out, s'    = kda_recur(q, k, v, g, beta, s)
//     y          = kda_onorm_step(out, ssm_norm, g_gate)
//     y_out      = attn_output @ y
//
// WHY THE PROJECTIONS ARE A HANDSHAKE, AND WHAT THAT COSTS.  The nine GEMVs are
// requested through proj_req/proj_sel and answered on proj_out -- the same shape
// swiglu_expert_q4k uses against the shared glm_matmul_q4k, and the reason this
// module is testable standalone. It is ALSO the reason this module alone does not
// justify defining GLM53F_KDA_RTL_PRESENT: unlike glm53f_hc_block, whose sublayers
// are genuinely separate machines, these projections are this layer's OWN weights.
// A layer that cannot fetch them is incomplete. Driving glm_matmul_q4k directly
// (w_type = 2, code on w_hp, fp16 d on w_q8_d -- all already engine inputs, all
// already emitted by weight_loader_q4k) is the follow-on, and it is mechanical.
//
// NO SHARED RTL CHANGES.  docs/GLM53_FLASH_PORT.md 4.3g used to call the Q8_0
// fan-out and the state ownership "gaps in shared, baseline-pinned RTL". That was
// wrong: it assumed GLM-5.3-Flash would reuse glm_decoder_block_q4k. It does not
// -- this is a sibling, and a sibling declares its own port widths, exactly as
// mhc_block_site declares the four residual streams.
//
// PRECISION BOUNDARIES, each one deliberate:
//   * projections arrive fp32 and the conv consumes fp32;
//   * kda_conv_step emits bf16 (silu'd) and kda_recur wants fp32, so there is a
//     widen between them -- the rounding is the conv unit's, already gated;
//   * kda_recur's output is bf16-valued by the reference's own contract, which is
//     why kda_onorm_step takes a bf16 x port;
//   * the gate path is fp32 throughout, and runs on glm_act's bf16 sigmoid today
//     -- ~1-3 % per layer, measured (4.3e). Retrofitting fp32_sigmoid_pipe is a
//     separate change to those units, not to this layer.
//   * kda_recur runs EXACT=0, so it does its own l2norm of q and k (Quake rsqrt)
//     -- the tolerance leg it is gated on. It does NOT exponentiate: g_in is
//     exp(g) on BOTH legs, so this layer feeds kda_gate_step's `ge_out`. The
//     unit's header said otherwise until 2026-09-06 and that is what this layer's
//     first build got wrong.
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_KDA_LAYER_V
`define GLM53F_KDA_LAYER_V
`include "glm_fp.vh"

module glm53f_kda_layer #(
    parameter integer MODEL_DIM = 16,
    parameter integer H         = 2,      // KDA heads (real: 64)
    parameter integer DK        = 4,      // key dim   (real: 128)
    parameter integer DV        = 4,      // value dim (real: 128, == DK)
    parameter integer RANK      = 4,      // f_a/g_a low rank (real: 128)
    parameter integer CONV_K    = 4,      // ssm.conv_kernel
    parameter [31:0]  EPS       = 32'h3727C5AC,
    // 1/sqrt(DK), supplied rather than derived.  It is NOT optional and it does
    // NOT track DK on its own: kda_recur defaults it to 1/sqrt(8), so a slice with
    // any other DK silently scales q by the wrong constant -- q feeds `out` but
    // not the state update, so the recurrence's STATE still matches the golden
    // while its OUTPUT is off by sqrt(8/DK). That is exactly how this was found.
    // It is a parameter rather than a computation because 1/sqrt(128) is not
    // representable exactly and deriving it from the approximate fp32_rsqrt would
    // put an approximation inside the bit-exact leg.
    //   DK=4 -> 0x3F000000 (0.5)   DK=8 -> 0x3EB504F3   DK=128 -> 0x3DB504F3
    parameter [31:0]  INV_SQRT_DK = 32'h3F000000,
    // Derived, but declared HERE because the port list below uses them and a
    // localparam in the body is not visible to it.
    parameter integer PMAX_IN   = (MODEL_DIM > H*DV) ? MODEL_DIM : H*DV,
    parameter integer PMAX_OUT  = (H*DK > MODEL_DIM) ? H*DK : MODEL_DIM
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          start,
    output reg                           busy,
    output reg                           done,

    input  wire [16*MODEL_DIM-1:0]       x_in,        // bf16, post-norm layer input

    // ---- projection service: "compute projection SEL on this vector" ----
    output reg                           proj_req,
    output reg  [3:0]                    proj_sel,    // 0..8, see PSEL_* below
    output reg  [32*PMAX_IN-1:0]         proj_in,     // fp32 operand
    input  wire                          proj_done,
    input  wire [32*PMAX_OUT-1:0]        proj_out,    // fp32 result

    // ---- per-layer constants ----
    input  wire [32*H-1:0]               decay_in,    // exp(ssm_a), precomputed
    input  wire [32*H*DK-1:0]            dt_bias_in,
    input  wire [32*3*H*DK*CONV_K-1:0]   conv_w_in,   // channel-major, oldest..newest
    input  wire [16*DV-1:0]              onorm_w_in,  // ssm_norm, bf16

    // ---- recurrent state, owned by the caller across tokens ----
    input  wire [32*H*DK*DV-1:0]         s_in,
    output reg  [32*H*DK*DV-1:0]         s_out,
    input  wire [32*3*H*DK*(CONV_K-1)-1:0] hist_in,
    output reg  [32*3*H*DK*(CONV_K-1)-1:0] hist_out,

    output reg  [32*MODEL_DIM-1:0]       y_out
);
    localparam integer C = 3*H*DK;                        // conv channels: q|k|v

    localparam [3:0] PSEL_Q = 4'd0, PSEL_K = 4'd1, PSEL_V = 4'd2, PSEL_B = 4'd3,
                     PSEL_FA = 4'd4, PSEL_FB = 4'd5, PSEL_GA = 4'd6, PSEL_GB = 4'd7,
                     PSEL_O  = 4'd8;

`ifndef YOSYS
    initial begin
        if (DK != DV)
            $fatal(1, "glm53f_kda_layer: the reference concatenates q,k,v into ONE conv, which needs DK == DV");
    end
`endif

    integer i;

    reg [32*H*DK-1:0] q_r, k_r, f_r;
    reg [32*H*DV-1:0] v_r, ggate_r;
    reg [32*H-1:0]    b_r;
    reg [32*RANK-1:0] lr_r;                               // f_a / g_a result

    // ---- conv over the concatenation ----
    reg               cv_start;
    wire              cv_busy, cv_done;
    reg  [32*C-1:0]   cv_x;
    wire [32*C*(CONV_K-1)-1:0] cv_sout;
    wire [16*C-1:0]   cv_y;
    wire [16*C-1:0]   cv_pre;
    kda_conv_step #(.C(C), .K(CONV_K)) u_conv (
        .clk(clk), .rst(rst), .start(cv_start), .busy(cv_busy), .done(cv_done),
        .x_in(cv_x), .w_in(conv_w_in), .s_in(hist_in), .s_out(cv_sout),
        .conv_out(cv_pre), .y_out(cv_y));
    reg [16*C-1:0] cv_y_r;

    // ---- forget gate ----
    reg               gt_start;
    wire              gt_busy, gt_done;
    wire [32*H*DK-1:0] gt_g, gt_ge;
    wire [32*H-1:0]    gt_beta;
    kda_gate_step #(.H(H), .DK(DK)) u_gate (
        .clk(clk), .rst(rst), .start(gt_start), .busy(gt_busy), .done(gt_done),
        .decay_in(decay_in), .b_in(b_r), .f_in(f_r), .dt_bias_in(dt_bias_in),
        .g_out(gt_g), .ge_out(gt_ge), .beta_out(gt_beta));
    reg [32*H*DK-1:0] g_r;
    reg [32*H-1:0]    beta_r;

    // ---- the delta-rule recurrence ----
    reg               rc_start;
    wire              rc_busy, rc_done;
    reg  [32*H*DK-1:0] rc_q, rc_k, rc_g;
    reg  [32*H*DV-1:0] rc_v;
    wire [32*H*DK*DV-1:0] rc_sout;
    wire [32*H*DV-1:0]    rc_out;
    kda_recur #(.H(H), .DK(DK), .DV(DV), .EXACT(0), .INV_SQRT_DK(INV_SQRT_DK)) u_rec (
        .clk(clk), .rst(rst), .start(rc_start), .busy(rc_busy), .done(rc_done),
        .q_in(rc_q), .k_in(rc_k), .g_in(rc_g), .v_in(rc_v), .beta_in(beta_r),
        .s_in(s_in), .s_out(rc_sout), .out_v(rc_out));
    reg [32*H*DV-1:0] rec_out_r;

    // ---- gated output norm ----
    reg               on_start;
    wire              on_busy, on_done;
    reg  [16*H*DV-1:0] on_x;
    wire [16*H*DV-1:0] on_y;
    kda_onorm_step #(.H(H), .DV(DV), .EPS(EPS)) u_onorm (
        .clk(clk), .rst(rst), .start(on_start), .busy(on_busy), .done(on_done),
        .x_in(on_x), .gate_in(ggate_r), .weight_in(onorm_w_in), .y_out(on_y));

    localparam [3:0] S_IDLE=4'd0,  S_PQ=4'd1,  S_PK=4'd2,  S_PV=4'd3,  S_CONV=4'd4,
                     S_PB=4'd5,    S_PFA=4'd6, S_PFB=4'd7, S_PGA=4'd8, S_PGB=4'd9,
                     S_GATE=4'd10, S_REC=4'd11, S_ON=4'd12, S_PO=4'd13, S_FIN=4'd14;
    reg [3:0] st;

    // one place that issues a projection request
    task issue(input [3:0] sel, input [32*PMAX_IN-1:0] vec);
        begin proj_sel <= sel; proj_in <= vec; proj_req <= 1'b1; end
    endtask

    wire [32*PMAX_IN-1:0] x_f32;                       // x widened to fp32 once
    genvar gx;
    generate
        for (gx = 0; gx < PMAX_IN; gx = gx + 1) begin : g_xf
            assign x_f32[32*gx +: 32] = (gx < MODEL_DIM)
                 ? bf16_to_fp32(x_in[16*gx +: 16]) : 32'd0;
        end
    endgenerate

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0; proj_req <= 1'b0;
            cv_start <= 1'b0; gt_start <= 1'b0; rc_start <= 1'b0; on_start <= 1'b0;
        end else begin
            done <= 1'b0; proj_req <= 1'b0;
            cv_start <= 1'b0; gt_start <= 1'b0; rc_start <= 1'b0; on_start <= 1'b0;

            case (st)
                S_IDLE: if (start) begin busy <= 1'b1; issue(PSEL_Q, x_f32); st <= S_PQ; end

                S_PQ: if (proj_done) begin
                    q_r <= proj_out[32*H*DK-1:0]; issue(PSEL_K, x_f32); st <= S_PK;
                end
                S_PK: if (proj_done) begin
                    k_r <= proj_out[32*H*DK-1:0]; issue(PSEL_V, x_f32); st <= S_PV;
                end
                S_PV: if (proj_done) begin
                    v_r <= proj_out[32*H*DV-1:0];
                    // ONE conv over the concatenation q|k|v, per the reference
                    cv_x <= {proj_out[32*H*DV-1:0], k_r, q_r};
                    cv_start <= 1'b1; st <= S_CONV;
                end

                S_CONV: if (cv_done) begin
                    cv_y_r <= cv_y;
`ifdef INJ_KDAL_CONV_NOHIST
                    hist_out <= hist_in;      // must FAIL: history never advances
`else
                    hist_out <= cv_sout;
`endif
                    issue(PSEL_B, x_f32); st <= S_PB;
                end

                S_PB: if (proj_done) begin
                    b_r <= proj_out[32*H-1:0]; issue(PSEL_FA, x_f32); st <= S_PFA;
                end
                S_PFA: if (proj_done) begin
                    lr_r <= proj_out[32*RANK-1:0];
                    issue(PSEL_FB, {{(32*(PMAX_IN-RANK)){1'b0}}, proj_out[32*RANK-1:0]});
                    st <= S_PFB;
                end
                S_PFB: if (proj_done) begin
                    f_r <= proj_out[32*H*DK-1:0]; issue(PSEL_GA, x_f32); st <= S_PGA;
                end
                S_PGA: if (proj_done) begin
                    lr_r <= proj_out[32*RANK-1:0];
                    issue(PSEL_GB, {{(32*(PMAX_IN-RANK)){1'b0}}, proj_out[32*RANK-1:0]});
                    st <= S_PGB;
                end
                S_PGB: if (proj_done) begin
`ifdef INJ_KDAL_GATE_ORDER
                    // must FAIL: the two low-rank paths mis-routed. f drives the
                    // DECAY, g_gate drives the OUTPUT norm; they are not the same
                    // tensor even though both are a rank-RANK pair.
                    ggate_r <= f_r;
`else
                    ggate_r <= proj_out[32*H*DV-1:0];
`endif
                    gt_start <= 1'b1; st <= S_GATE;
                end

                S_GATE: if (gt_done) begin
                    g_r <= gt_g; beta_r <= gt_beta;
                    // conv output is bf16 (silu'd); the recurrence wants fp32
                    for (i = 0; i < H*DK; i = i + 1) begin
`ifdef INJ_KDAL_QK_SWAP
                        rc_q[32*i +: 32] <= bf16_to_fp32(cv_y_r[16*(H*DK + i) +: 16]);
                        rc_k[32*i +: 32] <= bf16_to_fp32(cv_y_r[16*i +: 16]);
`else
                        rc_q[32*i +: 32] <= bf16_to_fp32(cv_y_r[16*i +: 16]);
                        rc_k[32*i +: 32] <= bf16_to_fp32(cv_y_r[16*(H*DK + i) +: 16]);
`endif
                    end
                    for (i = 0; i < H*DV; i = i + 1)
                        rc_v[32*i +: 32] <= bf16_to_fp32(cv_y_r[16*(2*H*DK + i) +: 16]);
                    // ge_out, NOT g_out: kda_recur wants exp(g) on BOTH legs -- its
                    // header used to claim EXACT=0 exponentiates internally; it does
                    // not, and passing the raw log-decay silently multiplies the state
                    // by g instead of exp(g). Corrected there too.
                    rc_g <= gt_ge;
                    rc_start <= 1'b1; st <= S_REC;
                end

                S_REC: if (rc_done) begin
`ifdef INJ_KDAL_NO_STATE
                    s_out <= s_in;            // must FAIL: the recurrence never persists
`else
                    s_out <= rc_sout;
`endif
                    rec_out_r <= rc_out;
                    for (i = 0; i < H*DV; i = i + 1)
                        on_x[16*i +: 16] <= fp32_to_bf16(rc_out[32*i +: 32]);
                    on_start <= 1'b1; st <= S_ON;
                end

                // o_proj takes the LAYER's own output, not x, so it is built
                // element-wise here rather than through `issue`.
                S_ON: if (on_done) begin
                    proj_sel <= PSEL_O; proj_req <= 1'b1;
                    for (i = 0; i < PMAX_IN; i = i + 1)
                        proj_in[32*i +: 32] <= (i < H*DV)
                            ? bf16_to_fp32(on_y[16*i +: 16]) : 32'd0;
                    st <= S_PO;
                end

                S_PO: if (proj_done) begin
                    y_out <= proj_out[32*MODEL_DIM-1:0];
                    st <= S_FIN;
                end

                S_FIN: begin done <= 1'b1; busy <= 1'b0; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // GLM53F_KDA_LAYER_V
