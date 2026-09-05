//============================================================================
// glm53f_hc_block.v -- the GLM-5.3-Flash block's RESIDUAL SKELETON: two mHC
// sites (attention, FFN) wrapped around two sublayers it does not own.
//
// This is the thing glm_decoder_block_q4k.v cannot express.  That block computes
//     h = x + attn(rmsnorm(x));   y = h + FFN(rmsnorm(h))
// -- ONE residual, added.  A GLM-5.3-Flash block carries FOUR residual streams
// and replaces each `+` with a hyper-connection:
//     collapsed = sum_h pre[h]*streams[h]          (mHC, per site)
//     sub_out   = SUBLAYER( rmsnorm(collapsed) )   (unchanged sublayer)
//     streams   = comb @ streams + post (x) sub_out
// run twice per block, with hc_attn_* then hc_ffn_*.
//
// A SIBLING, NOT AN EDIT.  The repo already keeps glm_decoder_block.v (bf16) and
// glm_decoder_block_q4k.v (Q4_K) as siblings; this is the GLM-5.3-Flash one.
// Editing the Q4_K block instead would re-pin every netlist baseline that depends
// on it for a change no GLM-5.2 build wants.
//
// ONE SITE INSTANCE, RUN TWICE.  mhc_block_site owns the streams, so running it
// twice with the weights muxed keeps ONE [H,D] buffer for the block rather than
// two plus a copy between them.  The second pass sees the streams the first pass
// wrote -- which is the whole point, and is what INJ_HCB_STALE_STREAMS checks.
//
// WHERE fp32 STOPS.  mHC's gating math is fp32 because its eps-floored maps
// demand it (docs/GLM53_FLASH_PORT.md 4.3i-4.3l).  The SUBLAYER path is bf16,
// exactly like every other activation in this repo, and that is faithful rather
// than a concession: `collapsed` is converted to bf16 for the block's own
// rmsnorm, the sublayer works in bf16 throughout, and its output is widened back
// to fp32 only to re-enter the mix.  So the four bf16 collisions this port hit
// were all INSIDE mHC's gating, not in the main activation path -- worth stating,
// because "GLM-5.3-Flash needs fp32" would be the wrong lesson to draw.
//
// The two sublayers are reached by ONE muxed handshake (`sub_is_ffn` says which),
// so this module is testable against stubs and drops around mla_attn_q4k and the
// MoE/dense FFN without knowing anything about them.
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_HC_BLOCK_V
`define GLM53F_HC_BLOCK_V
`include "glm_fp.vh"

module glm53f_hc_block #(
    parameter integer H           = 4,
    parameter integer D           = 64,
    parameter integer QK          = 32,
    parameter [31:0]  RMS_EPS     = 32'h3727C5AC,
    parameter [31:0]  EPS         = 32'h358637BD,
    parameter integer ITERS       = 20,
    parameter integer RECIP_ITERS = 4,
    parameter integer DLANES      = 1
)(
    input  wire                        clk,
    input  wire                        rst,
    input  wire                        start,
    output reg                         busy,
    output reg                         done,

    input  wire                        streams_load,
    input  wire [32*H*D-1:0]           streams_init,
    output wire [32*H*D-1:0]           streams_cur,

    // mHC weights, per site
    input  wire [8*((2+H)*H)*H*D-1:0]  a_w_q,
    input  wire [16*((2+H)*H)*(H*D/32)-1:0] a_w_d,
    input  wire [32*((2+H)*H)-1:0]     a_base,
    input  wire [31:0]                 a_s0, a_s1, a_s2,
    input  wire [8*((2+H)*H)*H*D-1:0]  f_w_q,
    input  wire [16*((2+H)*H)*(H*D/32)-1:0] f_w_d,
    input  wire [32*((2+H)*H)-1:0]     f_base,
    input  wire [31:0]                 f_s0, f_s1, f_s2,

    // the block's own learned norms (bf16 gamma), between collapsed and sublayer
    input  wire [16*D-1:0]             attn_norm_w,
    input  wire [16*D-1:0]             ffn_norm_w,

    // ONE muxed sublayer handshake
    output reg                         sub_start,
    output reg                         sub_is_ffn,
    output reg  [16*D-1:0]             sub_vec,     // bf16, post-norm
    input  wire                        sub_done,
    input  wire [16*D-1:0]             sub_out      // bf16
);
    localparam integer ROWS = (2+H)*H;
    localparam integer IW   = (D <= 1) ? 1 : $clog2(D);

    reg site_sel;                                   // 0 = attention, 1 = FFN
    reg inj_reload;                                 // see INJ_HCB_STALE_STREAMS

    // ---- must-fail injections (test/glm53f_hc_block_tb.v); none in a normal build ----
    // These target the block's JOB -- routing two sites' weights and norms, and
    // threading the streams from the first into the second. The mHC numerics are
    // already gated by mhc-site; what is new here is the wiring.
`ifdef INJ_HCB_SAME_WEIGHTS
    wire wsel = 1'b0;                               // must FAIL: FFN site runs attn weights
`else
    wire wsel = site_sel;
`endif
`ifdef INJ_HCB_NORM_SWAP
    wire nsel = ~site_sel;                          // must FAIL: norms swapped
`else
    wire nsel = site_sel;
`endif

    // ---- the mHC site (one instance, weights muxed) ----
    reg               st_start;
    wire              st_busy, st_done, st_sub_start;
    wire [32*D-1:0]   st_coll;
    reg               st_sub_done;
    reg  [32*D-1:0]   st_sub_in;

    mhc_block_site #(.H(H), .D(D), .QK(QK), .RMS_EPS(RMS_EPS), .EPS(EPS),
                     .ITERS(ITERS), .RECIP_ITERS(RECIP_ITERS), .DLANES(DLANES)) u_site (
        .clk(clk), .rst(rst), .start(st_start), .busy(st_busy), .done(st_done),
        .streams_load(streams_load | inj_reload), .streams_init(streams_init), .streams_cur(streams_cur),
        .w_q   (wsel ? f_w_q  : a_w_q),
        .w_d   (wsel ? f_w_d  : a_w_d),
        .base_in(wsel ? f_base : a_base),
        .scale0(wsel ? f_s0 : a_s0),
        .scale1(wsel ? f_s1 : a_s1),
        .scale2(wsel ? f_s2 : a_s2),
        .sub_start(st_sub_start), .collapsed_out(st_coll),
        .sub_done(st_sub_done), .sub_in(st_sub_in));

    // ---- the block's own RMSNorm on `collapsed`, bf16 like every activation ----
    reg              rn_start, rn_xv, rn_gv;
    reg  [15:0]      rn_x, rn_g;
    wire             rn_inreq, rn_greq, rn_yv, rn_busy, rn_done;
    wire [15:0]      rn_y;
    rmsnorm_unit #(.LEN(D), .LANES(1), .EPS(RMS_EPS)) u_rn (
        .clk(clk), .rst(rst), .start(rn_start),
        .in_req(rn_inreq), .x_in(rn_x), .x_valid(rn_xv),
        .g_req(rn_greq),  .gamma_in(rn_g), .g_valid(rn_gv),
        .y_valid(rn_yv), .y_out(rn_y), .busy(rn_busy), .done(rn_done));

    reg [16*D-1:0] xbf;                             // collapsed, converted to bf16
    reg [IW:0]     xidx, gidx, yidx;
    integer        i;

    localparam [2:0] B_IDLE=3'd0, B_SITE=3'd1, B_NORM=3'd2, B_SUB=3'd3,
                     B_WAIT=3'd4, B_NEXT=3'd5;
    reg [2:0] bst;

    wire [16*D-1:0] gam = nsel ? ffn_norm_w : attn_norm_w;

    always @(posedge clk) begin
        if (rst) begin
            bst <= B_IDLE; busy <= 1'b0; done <= 1'b0;
            st_start <= 1'b0; st_sub_done <= 1'b0; sub_start <= 1'b0;
            inj_reload <= 1'b0;
            rn_start <= 1'b0; rn_xv <= 1'b0; rn_gv <= 1'b0;
            site_sel <= 1'b0; sub_is_ffn <= 1'b0;
        end else begin
            done <= 1'b0; st_start <= 1'b0; st_sub_done <= 1'b0;
            sub_start <= 1'b0; rn_start <= 1'b0; rn_xv <= 1'b0; rn_gv <= 1'b0;
            inj_reload <= 1'b0;

            case (bst)
                B_IDLE: if (start) begin
                    busy <= 1'b1; site_sel <= 1'b0; sub_is_ffn <= 1'b0;
                    st_start <= 1'b1; bst <= B_SITE;
                end

                // the site raises st_sub_start once `collapsed` is ready
                B_SITE: if (st_sub_start) begin
                    for (i = 0; i < D; i = i + 1)
                        xbf[16*i +: 16] <= fp32_to_bf16(st_coll[32*i +: 32]);
                    xidx <= 0; gidx <= 0; yidx <= 0;
                    rn_start <= 1'b1; bst <= B_NORM;
                end

                B_NORM: begin
`ifdef INJ_HCB_SKIP_NORM
                    // must FAIL: the sublayer fed raw `collapsed`, unnormalised.
                    // attn_norm/ffn_norm exist on all 46 blocks [scan]; mHC wraps
                    // the norm, it does not replace it.
                    sub_vec <= xbf; sub_start <= 1'b1; bst <= B_SUB;
`else
                    if (rn_inreq) begin rn_x <= xbf[16*xidx +: 16]; rn_xv <= 1'b1; xidx <= xidx + 1'b1; end
                    if (rn_greq)  begin rn_g <= gam[16*gidx +: 16]; rn_gv <= 1'b1; gidx <= gidx + 1'b1; end
                    if (rn_yv)    begin sub_vec[16*yidx +: 16] <= rn_y; yidx <= yidx + 1'b1; end
                    if (rn_done) begin sub_start <= 1'b1; bst <= B_SUB; end
`endif
                end

                // hand the normed vector to whichever sublayer this site wraps
                B_SUB: if (sub_done) begin
                    for (i = 0; i < D; i = i + 1)
                        st_sub_in[32*i +: 32] <= bf16_to_fp32(sub_out[16*i +: 16]);
                    st_sub_done <= 1'b1;
                    bst <= B_WAIT;
                end

                B_WAIT: if (st_done) bst <= B_NEXT;

                B_NEXT: begin
                    if (!site_sel) begin
                        // second site: the SAME instance, FFN weights, and the
                        // streams the attention site just wrote
                        site_sel <= 1'b1; sub_is_ffn <= 1'b1;
`ifdef INJ_HCB_STALE_STREAMS
                        // must FAIL: reload the ORIGINAL streams, so the FFN site
                        // never sees what the attention site wrote.
                        inj_reload <= 1'b1;
`endif
                        st_start <= 1'b1; bst <= B_SITE;
                    end else begin
                        done <= 1'b1; busy <= 1'b0; bst <= B_IDLE;
                    end
                end

                default: bst <= B_IDLE;
            endcase
        end
    end
endmodule
`endif // GLM53F_HC_BLOCK_V
