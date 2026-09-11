//============================================================================
// glm53f_decoder_block.v -- the first GLM-5.3-Flash module where a real sublayer
// lives INSIDE the hyper-connection: KDA attention wired into the attention site
// of a two-site mHC block.
//
//   streams [4,D]
//     -> mHC attention site: collapse -> attn_norm -> glm53f_kda_attn -> mix
//     -> mHC FFN site:       collapse -> ffn_norm  -> <FFN handshake>  -> mix
//     -> streams'
//
// BOTH SIDES ARE NOW WIRED, so this is a COMPLETE GLM-5.3-Flash decoder layer for
// blocks 0-2: glm53f_kda_attn in the attention site (fetching its own nine Q8_0
// projections, threading its own recurrence and conv history) and
// glm53f_swiglu_q8 in the FFN site (gate/up/down off Q8_0, asymmetric clamp).
// Those three blocks are the dense front (GLM53F_N_DENSE = 3) and they are KDA,
// since the first MLA block is 3.
//
// WHY THE FFN IS A SIBLING AND NOT swiglu_expert_q4k: that unit's w_q port is
// FOUR BITS PER LANE and the census says this FFN is Q8_0
// (`blk.N.ffn_{gate,up,down}` [12288,4096] x3 [scan]). It cannot carry these
// weights at all -- not "less accurately", at all.
//
// THE TWO FFN ARMS.  FFN_KIND = 0 is the dense Q8_0 SwiGLU of blocks 0-2;
// FFN_KIND = 1 is the MoE of blocks 3-44 -- glm53f_moe_ffn, i.e. the router, the
// Q4_K/Q5_K/Q6_K routed experts and the always-on Q8_0 shared expert.  Both
// present the same start/busy/done handshake and the same [D] bf16 in / out, so
// the site takes either without changing.
//
// WHAT IS STILL PARAMETERISED BUT NOT BUILT: the 11 MLA blocks' attention site,
// which takes mla_attn_q4k.  ATTN_KIND still $fatal's on that arm -- elaborating
// a block whose site is empty would be a top that lies.
//
// The mHC sublayer contract is what makes this a wiring job rather than a
// redesign: attn_norm/ffn_norm sit between `collapsed` and the sublayer on all 46
// blocks [scan], so the sublayer still sees [D] bf16 in and [D] bf16 out -- the
// same shape mla_attn_q4k and the FFN already present.
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_DECODER_BLOCK_V
`define GLM53F_DECODER_BLOCK_V
`include "glm_fp.vh"

module glm53f_decoder_block #(
    parameter integer MODEL_DIM   = 16,
    parameter integer H           = 4,      // hc_mult: residual streams
    parameter integer KH          = 2,      // KDA heads
    parameter integer DK          = 4,
    parameter integer DV          = 4,
    parameter integer RANK        = 4,
    parameter integer CONV_K      = 4,
    parameter integer TN          = 2,
    parameter integer KMAX        = 32,
    parameter integer INTER       = 32,     // dense FFN inter size
    parameter integer ATTN_KIND   = 0,      // 0 = KDA (34/45). 1 = MLA: not built.
    parameter integer FFN_KIND    = 0,      // 0 = dense Q8_0 SwiGLU (blocks 0-2).
                                            // 1 = MoE (blocks 3-44): glm53f_moe_ffn.
    parameter integer N_EXPERT    = 8,      // expert_count (real 288)
    parameter integer TOPK        = 3,      // expert_used_count (real 8)
    parameter [31:0]  EXPERT_SCALE = 32'h40200000,   // expert_weights_scale = 2.5
    parameter integer EIDXW       = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT),
    parameter [15:0]  SWIGLU_LIM  = 16'h4120,  // 10.0 bf16 = swiglu_limit
    parameter [31:0]  EPS         = 32'h358637BD,
    parameter [31:0]  RMS_EPS     = 32'h3727C5AC,
    parameter [31:0]  INV_SQRT_DK = 32'h3F000000,
    parameter integer ITERS       = 20,
    parameter integer RECIP_ITERS = 4,
    parameter integer DLANES      = 1,
    parameter integer PMAX_OUT    = (KH*DK > MODEL_DIM) ? KH*DK : MODEL_DIM
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          start,
    output wire                          busy,
    output wire                          done,

    // the four residual streams
    input  wire                          streams_load,
    input  wire [32*H*MODEL_DIM-1:0]     streams_init,
    output wire [32*H*MODEL_DIM-1:0]     streams_cur,

    // mHC weights, per site
    input  wire [8*((2+H)*H)*H*MODEL_DIM-1:0]      a_w_q,
    input  wire [16*((2+H)*H)*(H*MODEL_DIM/32)-1:0] a_w_d,
    input  wire [32*((2+H)*H)-1:0]       a_base,
    input  wire [31:0]                   a_s0, a_s1, a_s2,
    input  wire [8*((2+H)*H)*H*MODEL_DIM-1:0]      f_w_q,
    input  wire [16*((2+H)*H)*(H*MODEL_DIM/32)-1:0] f_w_d,
    input  wire [32*((2+H)*H)-1:0]       f_base,
    input  wire [31:0]                   f_s0, f_s1, f_s2,

    // the block's own learned norms
    input  wire [16*MODEL_DIM-1:0]       attn_norm_w,
    input  wire [16*MODEL_DIM-1:0]       ffn_norm_w,

    // ---- KDA sublayer: its own Q8_0 weight pull, constants and state ----
    output wire                          kda_w_req,
    output wire [3:0]                    kda_w_sel,
    output wire [$clog2(PMAX_OUT/TN+1)-1:0] kda_w_grp,
    output wire [$clog2(KMAX+1)-1:0]     kda_w_k,
    input  wire [16*TN-1:0]              kda_w_hp,
    input  wire [16*TN*((KMAX+31)/32)-1:0] kda_w_q8_d,
    input  wire [32*KH-1:0]              decay_in,
    input  wire [32*KH*DK-1:0]           dt_bias_in,
    input  wire [32*3*KH*DK*CONV_K-1:0]  conv_w_in,
    input  wire [16*DV-1:0]              onorm_w_in,
    input  wire [32*KH*DK*DV-1:0]        kda_s_in,
    output wire [32*KH*DK*DV-1:0]        kda_s_out,
    input  wire [32*3*KH*DK*(CONV_K-1)-1:0] kda_hist_in,
    output wire [32*3*KH*DK*(CONV_K-1)-1:0] kda_hist_out,

    // ---- FFN sublayer: its own Q8_0 weight pull ----
    output wire                          ffn_w_req,
    output wire [1:0]                    ffn_w_sel,
    output wire [$clog2((INTER>MODEL_DIM?INTER:MODEL_DIM)/TN+1)-1:0] ffn_w_grp,
    output wire [$clog2(KMAX+1)-1:0]     ffn_w_k,
    input  wire [16*TN-1:0]              ffn_w_hp,
    input  wire [16*TN*((KMAX+31)/32)-1:0] ffn_w_q8_d,

    // ---- MoE arm (FFN_KIND = 1 only; tied off and inert at FFN_KIND = 0) ----
    // These exist unconditionally because Verilog cannot declare a port
    // conditionally.  At FFN_KIND = 0 the generate arm drives every output to 0
    // and reads none of the inputs, so a dense block is byte-identical to what it
    // was before the MoE arm existed -- `make dec-block` is the check on that.
    output wire                          moe_rw_req,
    output wire [$clog2(MODEL_DIM+1)-1:0] moe_rw_k,
    input  wire [32*N_EXPERT-1:0]        moe_rw_row,   // ffn_gate_inp, F32
    input  wire [32*N_EXPERT-1:0]        moe_bias,     // exp_probs_b, F32
    output wire                          moe_fw_shared,
    output wire [EIDXW-1:0]              moe_fw_eidx,
    input  wire [2:0]                    moe_wt_gate, moe_wt_up, moe_wt_down,
    input  wire [2:0]                    moe_wt_sh_gate, moe_wt_sh_up, moe_wt_sh_down,
    input  wire [4*TN-1:0]               ffn_w_q,
    input  wire [16*TN*((KMAX+255)/256)-1:0]  ffn_w_d, ffn_w_dmin,
    input  wire [96*TN*((KMAX+255)/256)-1:0]  ffn_w_scales,
    input  wire [128*TN*((KMAX+255)/256)-1:0] ffn_w_q6_sc
);
`ifndef YOSYS
    initial begin
        if (ATTN_KIND != 0)
            $fatal(1, "glm53f_decoder_block: ATTN_KIND=1 (MLA) is not built -- mla_attn_q4k is not wired into the mHC attention site yet, and elaborating as if it were would be a top that lies");
        if (FFN_KIND > 1)
            $fatal(1, "glm53f_decoder_block: FFN_KIND must be 0 (dense Q8_0 SwiGLU, blocks 0-2) or 1 (MoE, blocks 3-44)");
    end
`endif

    // ---- the two-site hyper-connection ----
    wire              hc_sub_start, hc_sub_is_ffn;
    wire [16*MODEL_DIM-1:0] hc_sub_vec;
    reg               hc_sub_done;
    reg  [16*MODEL_DIM-1:0] hc_sub_out;

    glm53f_hc_block #(.H(H), .D(MODEL_DIM), .QK(32), .RMS_EPS(RMS_EPS), .EPS(EPS),
                      .ITERS(ITERS), .RECIP_ITERS(RECIP_ITERS), .DLANES(DLANES)) u_hc (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .streams_load(streams_load), .streams_init(streams_init), .streams_cur(streams_cur),
        .a_w_q(a_w_q), .a_w_d(a_w_d), .a_base(a_base), .a_s0(a_s0), .a_s1(a_s1), .a_s2(a_s2),
        .f_w_q(f_w_q), .f_w_d(f_w_d), .f_base(f_base), .f_s0(f_s0), .f_s1(f_s1), .f_s2(f_s2),
        .attn_norm_w(attn_norm_w), .ffn_norm_w(ffn_norm_w),
        .sub_start(hc_sub_start), .sub_is_ffn(hc_sub_is_ffn), .sub_vec(hc_sub_vec),
        .sub_done(hc_sub_done), .sub_out(hc_sub_out));

    // ---- the KDA sublayer, sitting in the attention site ----
    reg                     kda_start;
    reg  [16*MODEL_DIM-1:0] kda_x;
    wire                    kda_busy, kda_done;
    wire [32*MODEL_DIM-1:0] kda_y;

    glm53f_kda_attn #(.MODEL_DIM(MODEL_DIM), .H(KH), .DK(DK), .DV(DV), .RANK(RANK),
                      .CONV_K(CONV_K), .TN(TN), .KMAX(KMAX), .EPS(RMS_EPS),
                      .INV_SQRT_DK(INV_SQRT_DK)) u_kda (
        .clk(clk), .rst(rst), .start(kda_start), .busy(kda_busy), .done(kda_done),
        .x_in(kda_x),
        .w_req(kda_w_req), .w_sel(kda_w_sel), .w_grp(kda_w_grp), .w_k(kda_w_k),
        .w_hp(kda_w_hp), .w_q8_d(kda_w_q8_d),
        .decay_in(decay_in), .dt_bias_in(dt_bias_in), .conv_w_in(conv_w_in),
        .onorm_w_in(onorm_w_in),
        .s_in(kda_s_in), .s_out(kda_s_out),
        .hist_in(kda_hist_in), .hist_out(kda_hist_out),
        .y_out(kda_y));

    // ---- the dense FFN, sitting in the FFN site ----
    reg                     ffn_start;
    reg  [16*MODEL_DIM-1:0] ffn_x;
    wire                    ffn_busy, ffn_done;
    wire [16*MODEL_DIM-1:0] ffn_y;

    // Both arms present the SAME handshake to the mHC FFN site -- start/busy/done
    // and one bf16 vector in, one out -- which is why swapping them is a generate
    // and not a redesign of the site.
    localparam integer FNSB = (KMAX + 255) / 256;
    generate
    if (FFN_KIND == 0) begin : g_dense
        // The dense front is Q8_0 on all three tensors [scan], so the Q4_K/Q6_K
        // header buses are tied off HERE rather than inside the SwiGLU -- the unit
        // itself is type-generic now, because the MoE experts are a Q4_K/Q5_K/Q6_K
        // mix. The MoE arm below drives them.
        glm53f_swiglu_mt #(.HIDDEN(MODEL_DIM), .INTER(INTER), .TN(TN), .KMAX(KMAX),
                           .LIM(SWIGLU_LIM)) u_ffn (
            .clk(clk), .rst(rst), .start(ffn_start), .busy(ffn_busy), .done(ffn_done),
            .x_in(ffn_x),
            .wt_gate(3'd2), .wt_up(3'd2), .wt_down(3'd2),      // Q8_0
            .w_req(ffn_w_req), .w_sel(ffn_w_sel), .w_grp(ffn_w_grp), .w_k(ffn_w_k),
            .w_q({4*TN{1'b0}}), .w_hp(ffn_w_hp),
            .w_d({16*TN*FNSB{1'b0}}), .w_dmin({16*TN*FNSB{1'b0}}),
            .w_scales({96*TN*FNSB{1'b0}}), .w_q6_sc({128*TN*FNSB{1'b0}}),
            .w_q8_d(ffn_w_q8_d),
            .y_out(ffn_y));
        assign moe_rw_req    = 1'b0;
        assign moe_rw_k      = {$clog2(MODEL_DIM+1){1'b0}};
        assign moe_fw_shared = 1'b0;
        assign moe_fw_eidx   = {EIDXW{1'b0}};
    end else begin : g_moe
        // blocks 3-44.  The routed experts and the shared expert carry DIFFERENT
        // weight types ([scan]: routed Q4_K/Q5_K/Q6_K, shared Q8_0), so the two
        // type triples come in separately and are passed straight through.
        glm53f_moe_ffn #(.HIDDEN(MODEL_DIM), .INTER(INTER), .E(N_EXPERT),
                         .TOPK(TOPK), .TN(TN), .KMAX(KMAX), .SCALE(EXPERT_SCALE),
                         .LIMIT(SWIGLU_LIM), .RECIP_ITERS(RECIP_ITERS),
                         .IDXW(EIDXW)) u_ffn (
            .clk(clk), .rst(rst), .start(ffn_start), .busy(ffn_busy), .done(ffn_done),
            .x_in(ffn_x),
            .rw_req(moe_rw_req), .rw_k(moe_rw_k), .rw_row(moe_rw_row),
            .bias_in(moe_bias),
            .fw_req(ffn_w_req), .fw_shared(moe_fw_shared), .fw_eidx(moe_fw_eidx),
            .fw_sel(ffn_w_sel), .fw_grp(ffn_w_grp), .fw_k(ffn_w_k),
            .wt_gate(moe_wt_gate), .wt_up(moe_wt_up), .wt_down(moe_wt_down),
            .wt_sh_gate(moe_wt_sh_gate), .wt_sh_up(moe_wt_sh_up),
            .wt_sh_down(moe_wt_sh_down),
            .w_q(ffn_w_q), .w_hp(ffn_w_hp),
            .w_d(ffn_w_d), .w_dmin(ffn_w_dmin), .w_scales(ffn_w_scales),
            .w_q6_sc(ffn_w_q6_sc), .w_q8_d(ffn_w_q8_d),
            .y_out(ffn_y), .dbg_sel_idx(), .dbg_sel_weight());
    end
    endgenerate

    integer i;
    localparam [1:0] R_IDLE = 2'd0, R_ATTN = 2'd1, R_FFN = 2'd2;
    reg [1:0] rst_st;

    always @(posedge clk) begin
        if (rst) begin
            rst_st <= R_IDLE; hc_sub_done <= 1'b0;
            kda_start <= 1'b0; ffn_start <= 1'b0;
        end else begin
            hc_sub_done <= 1'b0; kda_start <= 1'b0; ffn_start <= 1'b0;

            case (rst_st)
                // route the mHC site's request to whichever sublayer it names
                R_IDLE: if (hc_sub_start) begin
`ifdef INJ_DBLK_SITE_SWAP
                    // must FAIL: the two sites' sublayers exchanged. The site the
                    // mHC block is asking for is carried on sub_is_ffn; ignoring
                    // it sends the FFN's input through KDA and vice versa.
                    if (hc_sub_is_ffn) begin
                        kda_x <= hc_sub_vec; kda_start <= 1'b1; rst_st <= R_ATTN;
                    end else begin
                        ffn_x <= hc_sub_vec; ffn_start <= 1'b1; rst_st <= R_FFN;
                    end
`else
`ifdef INJ_DBLK_NO_KDA
                    // must FAIL: BOTH sites routed to the FFN, i.e. the
                    // attention sublayer never runs. The residual streams still
                    // move and the block still completes -- which is exactly why
                    // this needs to be checked rather than assumed.
                    ffn_x <= hc_sub_vec; ffn_start <= 1'b1; rst_st <= R_FFN;
`else
                    if (hc_sub_is_ffn) begin
                        ffn_x <= hc_sub_vec; ffn_start <= 1'b1; rst_st <= R_FFN;
                    end else begin
                        kda_x <= hc_sub_vec; kda_start <= 1'b1; rst_st <= R_ATTN;
                    end
`endif
`endif
                end

                // KDA's y_out is fp32-typed but bf16-VALUED (it comes off
                // glm_matmul_q4k's bf16 c_out), so narrowing here is lossless.
                R_ATTN: if (kda_done) begin
                    for (i = 0; i < MODEL_DIM; i = i + 1)
                        hc_sub_out[16*i +: 16] <= fp32_to_bf16(kda_y[32*i +: 32]);
                    hc_sub_done <= 1'b1; rst_st <= R_IDLE;
                end

                R_FFN: if (ffn_done) begin
                    hc_sub_out <= ffn_y;
                    hc_sub_done <= 1'b1; rst_st <= R_IDLE;
                end

                default: rst_st <= R_IDLE;
            endcase
        end
    end
endmodule
`endif // GLM53F_DECODER_BLOCK_V
