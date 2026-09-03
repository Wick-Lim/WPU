//============================================================================
// kda_onorm_step.v -- KDA output norm (Glm5NextTextRMSNormGated), ONE token,
// H heads over head_dim DV.
//
//     y[h][i] = weight[i] * ( x[h][i] * rsqrt( mean_i x[h]^2 + eps ) ) * sigmoid(gate[h][i])
//     out     = bf16(y)                                    ONE rounding, at the end
//   x    : the recurrence output -- ALREADY bf16-valued at o_norm entry in the
//          reference (recurrent_kimi_delta_attention returns .to(bf16)), so a
//          bf16 x port here is faithful, not a shortcut.
//   eps  : rms_norm_eps = 1e-5 (rmsnorm_unit's default).   weight: ssm_norm.weight.
//
// COMPOSITION -- why the gate is folded into gamma.  The proven rmsnorm_unit is
// bf16-in / fp32-reduce / bf16-out and applies gamma INSIDE its normalize pass.
// Multiplying sigmoid(gate) onto its bf16 OUTPUT would round twice where the
// reference rounds once.  So per head this module computes
//     gamma_eff[i] = bf16( weight[i] * sigmoid(gate[h][i]) )
// and streams (x, gamma_eff) through rmsnorm_unit UNMODIFIED, keeping the single
// final rounding.  The cost is rounding gamma_eff to bf16 where the reference
// keeps weight*sigmoid in fp32 -- measured by tools/kda_onorm_gen.py, stated.
//
// ACCURACY: three approximation sources -- the Quake fp32_rsqrt inside
// rmsnorm_unit, glm_act's bf16 polynomial sigmoid, and the bf16 gamma_eff -- so
// this is a TOLERANCE leg only.  (The bf16 rounding of x is not a divergence.)
//
// INJECTION.  -DINJ_ONORM_GATE_FIRST gates x BEFORE the norm (x*sigmoid into the
// reduce, plain weight as gamma).  That is the plausible misreading -- "gated
// norm" read as "norm of the gated input" -- and it changes the variance, so
// `make kda-onorm` must fail with it.  The generator self-test shows the two
// never coincide on its corpus, so the injection is live.
//
// HANDSHAKE.  rmsnorm_unit pulls: it raises in_req / g_req, the producer answers
// with x_valid / g_valid the NEXT cycle from a registered beat, indexing beats
// with a counter -- exactly the idiom glm_decoder_block_q4k uses.  LANES=1 here
// (the slice is small); a real layer widens LANES.
//============================================================================
`timescale 1ns/1ps
`ifndef KDA_ONORM_STEP_V
`define KDA_ONORM_STEP_V
`include "glm_fp.vh"

module kda_onorm_step #(
    parameter integer H  = 3,
    parameter integer DV = 16,
    parameter [31:0]  EPS = 32'h3727C5AC,   // 1e-5 fp32 = rms_norm_eps
    parameter integer ACT_HW = 0
)(
    input  wire                  clk,
    input  wire                  rst,          // sync, active-high
    input  wire                  start,
    output reg                   busy,
    output reg                   done,         // y_out valid for all heads

    input  wire [16*H*DV-1:0]    x_in,         // recurrence output, bf16-valued
    input  wire [32*H*DV-1:0]    gate_in,      // g_b(g_a(h)), fp32, pre-sigmoid
    input  wire [16*DV-1:0]      weight_in,    // ssm_norm.weight, bf16 lane
    output reg  [16*H*DV-1:0]    y_out
);
    localparam integer IW = (DV > 1) ? $clog2(DV) : 1;
    integer i;

    // ---- per-head sigmoid of the gate (bf16 in/out) ----
    reg              sg_v;
    reg  [16*DV-1:0] gate_bf;
    wire             sg_ov;
    wire [16*DV-1:0] sig_bf;
    glm_act #(.MODE(0), .LANES(DV), .HW_LANES(ACT_HW)) u_sig (
        .clk(clk), .rst(rst), .in_valid(sg_v), .x_in(gate_bf), .out_valid(sg_ov), .y_out(sig_bf));

    // ---- the proven norm, LANES=1, pulled one element per beat ----
    reg          rn_start, rn_xv, rn_gv;
    wire         rn_inreq, rn_greq, rn_yv, rn_busy, rn_done;
    reg  [15:0]  rn_x, rn_g;
    wire [15:0]  rn_y;
    rmsnorm_unit #(.LEN(DV), .LANES(1), .EPS(EPS)) u_rn (
        .clk(clk), .rst(rst), .start(rn_start),
        .in_req(rn_inreq), .x_in(rn_x), .x_valid(rn_xv),
        .g_req(rn_greq),  .gamma_in(rn_g), .g_valid(rn_gv),
        .y_valid(rn_yv), .y_out(rn_y), .busy(rn_busy), .done(rn_done));

    reg [16*DV-1:0] geff;          // gamma_eff for the current head
    reg [16*DV-1:0] xfeed;         // x actually streamed (x, or gated x under injection)
    reg [IW-1:0]    xidx, gidx, yidx;
    reg [$clog2(H+1)-1:0] hi;
    // Index arithmetic on 32-bit zero-EXTENDED copies of the narrow counters.
    // The linter flags `hi*DV + yidx` (WIDTHEXPAND) because the narrow regs get
    // widened implicitly inside a 32-bit expression; making the extension
    // explicit is the silent, unambiguous form -- not a lint pragma.  (And a
    // comment line must not START with the tool's own name: it reads that as a
    // pragma and errors -- the same trap test/glm53f_dims_wrap.v records.)
    wire [31:0] hi_w   = {{(32-$clog2(H+1)){1'b0}}, hi};
    wire [31:0] yidx_w = {{(32-IW){1'b0}}, yidx};
    wire [31:0] y_at   = hi_w * DV + yidx_w;            // element index into y_out
    wire [31:0] g_next = (hi_w + 32'd1) * DV;           // base of the NEXT head's gate

    localparam [2:0] S_IDLE=3'd0, S_SIG=3'd1, S_GEFF=3'd2, S_NORM=3'd3, S_DONE=3'd4;
    reg [2:0] state;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0; sg_v <= 1'b0;
            rn_start <= 1'b0; rn_xv <= 1'b0; rn_gv <= 1'b0; hi <= 0;
        end else begin
            done <= 1'b0; sg_v <= 1'b0; rn_start <= 1'b0;
            rn_xv <= 1'b0; rn_gv <= 1'b0;          // decoder-block idiom: default-low, one-cycle answers
            case (state)
            S_IDLE: if (start) begin
                busy <= 1'b1; hi <= 0; state <= S_SIG;
                for (i = 0; i < DV; i = i + 1) gate_bf[16*i +: 16] <= fp32_to_bf16(gate_in[32*(0*DV+i) +: 32]);
                sg_v <= 1'b1;
            end
            S_SIG: if (sg_ov) begin
                for (i = 0; i < DV; i = i + 1) begin
`ifdef INJ_ONORM_GATE_FIRST
                    // INJECTION (never a normal build): gate the INPUT, norm the gated
                    // input, plain weight as gamma -- "norm of the gated x".
                    xfeed[16*i +: 16] <= fp32_to_bf16(fp32_mul(bf16_to_fp32(x_in[16*(hi*DV+i) +: 16]),
                                                               bf16_to_fp32(sig_bf[16*i +: 16])));
                    geff[16*i +: 16]  <= weight_in[16*i +: 16];
`else
                    xfeed[16*i +: 16] <= x_in[16*(hi*DV+i) +: 16];
                    geff[16*i +: 16]  <= fp32_to_bf16(fp32_mul(bf16_to_fp32(weight_in[16*i +: 16]),
                                                               bf16_to_fp32(sig_bf[16*i +: 16])));
`endif
                end
                xidx <= 0; gidx <= 0; yidx <= 0;
                rn_start <= 1'b1; state <= S_GEFF;
            end
            S_GEFF: state <= S_NORM;                // one cycle for rn_start to land
            S_NORM: begin
                if (rn_inreq) begin rn_x <= xfeed[16*xidx +: 16]; rn_xv <= 1'b1; xidx <= xidx + 1'b1; end
                if (rn_greq)  begin rn_g <= geff [16*gidx +: 16]; rn_gv <= 1'b1; gidx <= gidx + 1'b1; end
                if (rn_yv)    begin y_out[16*y_at +: 16] <= rn_y; yidx <= yidx + 1'b1; end
                if (rn_done) begin
                    if (hi == H[$clog2(H+1)-1:0] - 1'b1) state <= S_DONE;
                    else begin
                        hi <= hi + 1'b1;
                        for (i = 0; i < DV; i = i + 1)
                            gate_bf[16*i +: 16] <= fp32_to_bf16(gate_in[32*(g_next+i) +: 32]);
                        sg_v <= 1'b1; state <= S_SIG;
                    end
                end
            end
            S_DONE: begin done <= 1'b1; busy <= 1'b0; state <= S_IDLE; end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // KDA_ONORM_STEP_V
