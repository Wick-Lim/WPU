//============================================================================
// mhc_block_site.v -- ONE mHC site: the whole hyper-connection residual path
// around one sublayer.  A GLM-5.3-Flash block has TWO of these (attention and
// FFN), each with its own hc_{attn,ffn}_{fn,base,scale}.
//
//   mixed        = fn @ (streams.flat / rms)              mhc_fn_gemv
//   pre,post,comb= map(mixed, base, scale)                mhc_map_step
//   collapsed    = sum_h pre[h]*streams[h]                mhc_stream_ops (COLLAPSE)
//   sub_out      = SUBLAYER( block_norm(collapsed) )      <- outside this module
//   streams'     = comb @ streams + post (x) sub_out      mhc_stream_ops (MIX)
//
// WHAT THIS MODULE OWNS, AND WHAT IT DELIBERATELY DOES NOT.
//   It owns the FOUR residual streams -- the structural change that a
// GLM-5.3-Flash block carries [H,D] where a GLM-5.2 block carried [D].  They live
// here, fp32, one running buffer (64 KB at H=4, D=4096 -- bf16 would halve that
// and cost 5.9e-3 RMS after 90 sites, so fp32; see mhc_stream_ops.v).
//   It does NOT own the sublayer, and it does NOT normalise `collapsed`.  The
// GGUF census settles that: attn_norm[4096] and ffn_norm[4096] exist on all 46
// blocks ALONGSIDE the hc_* tensors, so the block's own learned norm still
// applies between `collapsed` and the sublayer.  The sublayer therefore keeps its
// [D]-in / [D]-out contract exactly as before -- which is why hyper-connections
// and the KDA layer wrapper do not block each other, and why attention and FFN
// need no change to sit inside this.
//   The sublayer is reached by a handshake (sub_start / sub_done), so this module
// is testable standalone against a stub and drops into a decoder block without
// knowing what the sublayer is.
//
// NOT BITWISE, and the reason is inherited rather than introduced here:
// mhc_map_step runs fp32_exp_pipe, whose polynomial is 2.3e-4 (`make fp-sigmoid`).
// The generator therefore bounds this composition by perturbing pre/post/comb
// within the map's OWN GATED bounds and pushing that through collapse and mix --
// so the bound comes from the units' published contracts, not from what this DUT
// happens to print.
//============================================================================
`timescale 1ns/1ps
`ifndef MHC_BLOCK_SITE_V
`define MHC_BLOCK_SITE_V
`include "glm_fp.vh"

module mhc_block_site #(
    parameter integer H           = 4,
    parameter integer D           = 64,
    parameter integer QK          = 32,
    parameter [31:0]  RMS_EPS     = 32'h3727C5AC,   // 1e-5
    parameter [31:0]  EPS         = 32'h358637BD,   // 1e-6
    parameter integer ITERS       = 20,
    parameter integer RECIP_ITERS = 4,
    parameter integer DLANES      = 1
)(
    input  wire                        clk,
    input  wire                        rst,
    input  wire                        start,
    output reg                         busy,
    output reg                         done,

    // the four residual streams: loaded once, then carried across sites
    input  wire                        streams_load,
    input  wire [32*H*D-1:0]           streams_init,
    output wire [32*H*D-1:0]           streams_cur,

    // this site's mHC weights
    input  wire [8*((2+H)*H)*H*D-1:0]  w_q,
    input  wire [16*((2+H)*H)*(H*D/32)-1:0] w_d,
    input  wire [32*((2+H)*H)-1:0]     base_in,
    input  wire [31:0]                 scale0,
    input  wire [31:0]                 scale1,
    input  wire [31:0]                 scale2,

    // sublayer handshake -- the block's own norm sits between these two
    output reg                         sub_start,
    output wire [32*D-1:0]             collapsed_out,
    input  wire                        sub_done,
    input  wire [32*D-1:0]             sub_in
);
    localparam integer ROWS = (2+H)*H;
    localparam integer NS   = H*D;

    localparam [2:0] S_IDLE = 3'd0, S_GEMV = 3'd1, S_MAP = 3'd2, S_COLL = 3'd3,
                     S_SUB  = 3'd4, S_MIX  = 3'd5, S_FIN = 3'd6;

    reg [2:0]  st;
    reg [32*NS-1:0] streams;
    reg [32*D-1:0]  sub_r;

    assign streams_cur = streams;

    // ---- fn GEMV ----
    reg              gv_start;
    wire             gv_busy, gv_done;
    wire [32*ROWS-1:0] gv_mixed;
    mhc_fn_gemv #(.H(H), .D(D), .QK(QK), .RMS_EPS(RMS_EPS)) u_gemv (
        .clk(clk), .rst(rst), .start(gv_start), .busy(gv_busy), .done(gv_done),
        .x_in(streams), .w_q(w_q), .w_d(w_d), .mixed_out(gv_mixed));

    reg [32*ROWS-1:0] mixed_r;

    // ---- the map ----
    reg              mp_start;
    wire             mp_busy, mp_done;
    wire [32*H-1:0]  mp_pre, mp_post;
    wire [32*H*H-1:0] mp_comb;
    mhc_map_step #(.H(H), .EPS(EPS), .ITERS(ITERS), .RECIP_ITERS(RECIP_ITERS)) u_map (
        .clk(clk), .rst(rst), .start(mp_start), .busy(mp_busy), .done(mp_done),
        .mixed_in(mixed_r), .base_in(base_in),
        .scale0(scale0), .scale1(scale1), .scale2(scale2),
        .pre_out(mp_pre), .post_out(mp_post), .comb_out(mp_comb));

    reg [32*H-1:0]   pre_r, post_r;
    reg [32*H*H-1:0] comb_r;

    // ---- collapse / mix (one instance, used in both modes) ----
    reg              op_start, op_mode;
    wire             op_busy, op_done;
    wire [32*D-1:0]  op_coll;
    wire [32*NS-1:0] op_strm;
    mhc_stream_ops #(.H(H), .D(D), .DLANES(DLANES)) u_ops (
        .clk(clk), .rst(rst), .start(op_start), .mode(op_mode),
        .busy(op_busy), .done(op_done),
        .streams_in(streams), .pre_in(pre_r), .comb_in(comb_r),
        .post_in(post_r), .sub_in(sub_r),
        .collapsed_out(op_coll), .streams_out(op_strm));

    reg [32*D-1:0] coll_r;
    assign collapsed_out = coll_r;

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            gv_start <= 1'b0; mp_start <= 1'b0; op_start <= 1'b0; sub_start <= 1'b0;
            op_mode <= 1'b0;
        end else begin
            done <= 1'b0; gv_start <= 1'b0; mp_start <= 1'b0;
            op_start <= 1'b0; sub_start <= 1'b0;

            if (streams_load) streams <= streams_init;

            case (st)
                S_IDLE: if (start) begin
                    busy <= 1'b1; gv_start <= 1'b1; st <= S_GEMV;
                end

                S_GEMV: if (gv_done) begin
                    mixed_r <= gv_mixed; mp_start <= 1'b1; st <= S_MAP;
                end

                S_MAP: if (mp_done) begin
`ifdef INJ_SITE_PRE_FOR_POST
                    // must FAIL: the map's three outputs mis-routed. pre collapses
                    // the streams IN, post places the sublayer OUT; they are not
                    // interchangeable even though both are H-wide sigmoid products.
                    pre_r <= mp_pre; post_r <= mp_pre; comb_r <= mp_comb;
`else
                    pre_r <= mp_pre; post_r <= mp_post; comb_r <= mp_comb;
`endif
                    op_mode <= 1'b0; op_start <= 1'b1; st <= S_COLL;
                end

                S_COLL: if (op_done) begin
                    coll_r <= op_coll;
                    sub_start <= 1'b1;
                    st <= S_SUB;
                end

                // the block's own norm and the sublayer live outside this module
                S_SUB: if (sub_done) begin
`ifdef INJ_SITE_IGNORE_SUB
                    sub_r <= {(32*D){1'b0}};        // must FAIL: sublayer output dropped
`else
                    sub_r <= sub_in;
`endif
                    op_mode <= 1'b1; op_start <= 1'b1; st <= S_MIX;
                end

                S_MIX: if (op_done) begin
`ifdef INJ_SITE_NO_UPDATE
                    streams <= streams;             // must FAIL: streams never advance
`else
                    streams <= op_strm;
`endif
                    st <= S_FIN;
                end

                S_FIN: begin done <= 1'b1; busy <= 1'b0; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // MHC_BLOCK_SITE_V
