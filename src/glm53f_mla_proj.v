//============================================================================
// glm53f_mla_proj.v -- the MLA projection front for GLM-5.3-Flash, streamed off
// the shared glm_matmul_q4k with Q8_0 weights.
//
// It turns one token into the two things glm53f_mla_score consumes:
//
//     c_q   = W_dq  @ x                       [Q_LORA]
//     q     = W_uq  @ rmsnorm(c_q)            [H*QK]
//     qa[h] = q[h]  @ W_uk[h]                 [KV_LORA]   <- the FOLD
//     c_kv  = W_dkv @ x                       [KV_LORA]   <- RAW, see below
//
// THERE IS NO W_kr AND NO ROTATION, and that is the whole difference from
// mla_attn_q4k rather than a simplification of it: GLM-5.3-Flash is NoPE
// ([gguf] rope.dimension_count = 0, [cfg] qk_rope_head_dim = 0,
// mla_use_nope = true). The structural consequence the config guard already
// asserts is `attention.key_length == kv_lora_rank` (both 512), where GLM-5.2's
// key was the latent PLUS a 64-wide rotary tail. A zero-width tail is not
// expressible in Verilog, which is why this is a sibling.
//
// THE FOLD IS THE POINT.  qa[h] = q[h] @ W_uk[h] is done ONCE per token, here,
// instead of expanding a key per CACHED token downstream: H*QK MACs over KV_LORA
// for every cached token (64*256*512 = 8.4 M at the real shapes) against
// H*KV_LORA = 32 K for a dot against the latent. It is also a GEMV like the
// others -- K = QK, KV_LORA output columns, one pass per head -- so it needs
// nothing new from the engine, only a head index on the weight request.
//
// THE LATENT IS CACHED RAW.  rmsnorm belongs on the READ side, once, inside
// glm53f_mla_score. Normalising here as well would apply it twice; rmsnorm is
// nearly idempotent so the error is small (measured 2.8e-06 relative) and every
// shape check still passes, which is exactly how it slipped into the reference
// before being caught. tools/glm53f_mla_ref.py now asserts the cached latent's
// RMS is NOT 1.
//
// Q8_0 NEEDS NOTHING NEW FROM THE ENGINE ([scan]: the attention tensors are Q8_0):
// w_type = 2, the code on w_hp[7:0], the fp16 block scale on w_q8_d -- all already
// inputs, and weight_loader_q4k already emits them. The Q4_K header buses are tied
// off because the default arm never reads them under Q8_0.
//
// TN AND THE TAIL: output rows are processed TN at a time and a partial final
// group is NOT handled, so every projection's row count must be a multiple of TN.
// Checked at elaboration rather than assumed. At the real shapes the row counts
// are 1536 / 16384 / 512 / 512, so TN up to 512 is legal.
//============================================================================
`timescale 1ns/1ps
`ifndef GLM53F_MLA_PROJ_V
`define GLM53F_MLA_PROJ_V
`include "glm_fp.vh"

module glm53f_mla_proj #(
    parameter integer MODEL_DIM = 16,
    parameter integer H         = 2,      // heads (real 64)
    parameter integer QK        = 8,      // qk_nope_head_dim (real 256)
    parameter integer QLORA     = 8,      // q_lora_rank (real 1536)
    parameter integer KVL       = 8,      // kv_lora_rank (real 512)
    parameter integer TN        = 2,      // output columns per pass (= matmul PE_N)
    parameter integer KMAX      = 32,     // >= max reduction length
    parameter [31:0]  RMS_EPS   = 32'h3727C5AC,
    // derived, but PARAMETERS rather than localparams: the port list below uses
    // them, and a localparam declared after the ports cannot be bound there.
    parameter integer PMAX_OUT  = (H*QK > QLORA)
                                ? ((H*QK > KVL) ? H*QK : KVL)
                                : ((QLORA > KVL) ? QLORA : KVL),
    parameter integer PMAX_IN   = (MODEL_DIM > QLORA)
                                ? ((MODEL_DIM > QK) ? MODEL_DIM : QK)
                                : ((QLORA > QK) ? QLORA : QK)
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    start,
    output reg                     busy,
    output reg                     done,

    input  wire [16*MODEL_DIM-1:0] x_in,

    // ---- the weight pull, shaped like glm53f_kda_gemv's ----
    output wire                    w_req,
    output wire [1:0]              w_sel,    // 0 W_dq, 1 W_uq, 2 W_uk, 3 W_dkv
    output wire [$clog2(H)-1:0]    w_head,   // which head's W_uk slice (w_sel==2)
    output wire [$clog2(PMAX_OUT/TN+1)-1:0] w_grp,
    output wire [$clog2(KMAX+1)-1:0]        w_k,
    input  wire [16*TN-1:0]        w_hp,     // Q8_0 code in [7:0] per column
    input  wire [16*TN*((KMAX+31)/32)-1:0] w_q8_d,

    output reg  [16*H*KVL-1:0]     qa_out,   // q folded through W_uk
    output reg  [16*KVL-1:0]       ckv_out   // RAW latent for the cache
);
    localparam integer NSB = (KMAX + 255) / 256;
    localparam integer NB8 = (KMAX + 31) / 32;
    localparam integer KW  = $clog2(KMAX + 1);
    localparam integer GW  = $clog2(PMAX_OUT/TN + 1);
    localparam integer HW  = (H <= 1) ? 1 : $clog2(H);

`ifndef YOSYS
    initial begin
        if ((QLORA % TN) || ((H*QK) % TN) || (KVL % TN))
            $fatal(1, "glm53f_mla_proj: every projection's row count must be a multiple of TN (no partial final group)");
        if (KMAX < MODEL_DIM || KMAX < QLORA || KMAX < QK)
            $fatal(1, "glm53f_mla_proj: KMAX must cover the longest reduction (MODEL_DIM, QLORA, QK)");
    end
`endif

    // per-pass shape: rows written, reduction length
    function integer prows(input [1:0] s);
        begin
            case (s)
                2'd0:    prows = QLORA;
                2'd1:    prows = H*QK;
                2'd2:    prows = KVL;      // one head's fold
                default: prows = KVL;
            endcase
        end
    endfunction
    function integer pk(input [1:0] s);
        begin
            case (s)
                2'd0:    pk = MODEL_DIM;
                2'd1:    pk = QLORA;
                2'd2:    pk = QK;
                default: pk = MODEL_DIM;
            endcase
        end
    endfunction

    reg [1:0]  sel;
    reg [HW-1:0] hd;
    reg [KW-1:0] kcnt, klen;
    reg [GW-1:0] grp, ngrp;
    reg [16*PMAX_IN-1:0]  in_r;      // this pass's activation vector, bf16
    reg [16*QLORA-1:0]    cq;        // W_dq output
    reg [16*H*QK-1:0]     qv;        // W_uq output

    localparam [3:0] S_IDLE=4'd0, S_PREP=4'd1, S_STREAM=4'd2, S_WAIT=4'd3,
                     S_RN=4'd4,     S_NEXT=4'd5, S_FIN=4'd6;
    reg [3:0] st;

    reg          mm_start;
    reg  [KW-1:0] mm_k_len;
    wire          mm_busy, mm_ov;
    wire [16*TN-1:0] mm_c;
    wire          stream = (st == S_STREAM);
    wire [3*TN-1:0] w_type_q8 = {TN{3'd2}};     // Q8_0

    assign w_req  = stream;
    assign w_sel  = sel;
    assign w_head = hd;
    assign w_grp  = grp;
    assign w_k    = kcnt;
    wire [15:0] a_col = in_r[16*kcnt +: 16];

    glm_matmul_q4k #(.PE_M(1), .PE_N(TN), .KMAX(KMAX)) u_mm (
        .clk(clk), .rst(rst), .start(mm_start), .k_len(mm_k_len),
        .w_d({16*TN*NSB{1'b0}}), .w_dmin({16*TN*NSB{1'b0}}),
        .w_scales({96*TN*NSB{1'b0}}),
        .in_valid(stream), .a_col(a_col), .w_q({4*TN{1'b0}}),
        .busy(mm_busy), .out_valid(mm_ov), .c_out(mm_c),
        .w_type(w_type_q8), .w_hp(w_hp),
        .w_q6_sc({128*TN*NSB{1'b0}}), .w_q8_d(w_q8_d));

    // ---- the repo's own gated rmsnorm, gamma = 1, over c_q ----
    reg         rn_start, rn_xv, rn_gv;
    reg  [15:0] rn_x, rn_g;
    wire        rn_inreq, rn_greq, rn_yv, rn_busy, rn_done;
    wire [15:0] rn_y;
    reg  [KW-1:0] xidx, gidx, yidx;
    rmsnorm_unit #(.LEN(QLORA), .LANES(1), .EPS(RMS_EPS)) u_rn (
        .clk(clk), .rst(rst), .start(rn_start),
        .in_req(rn_inreq), .x_in(rn_x), .x_valid(rn_xv),
        .g_req(rn_greq),  .gamma_in(rn_g), .g_valid(rn_gv),
        .y_valid(rn_yv), .y_out(rn_y), .busy(rn_busy), .done(rn_done));

    integer gi;
    // NOTE: the results of pk()/prows() go through locals -- a part-select on a
    // function CALL is not legal Verilog and iverilog rejects it.
    task setup_pass(input [1:0] s);
        reg [31:0] kk, rr;
        begin
            kk = pk(s);
            rr = prows(s) / TN;
            klen     <= kk[KW-1:0];
            mm_k_len <= kk[KW-1:0];
            ngrp     <= rr[GW-1:0];
            grp      <= 0;
            mm_start <= 1'b1;
            st       <= S_PREP;
        end
    endtask

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0; mm_start <= 1'b0;
            rn_start <= 1'b0; rn_xv <= 1'b0; rn_gv <= 1'b0;
            sel <= 2'd0; hd <= 0; grp <= 0; kcnt <= 0;
            xidx <= 0; gidx <= 0; yidx <= 0;
        end else begin
            done <= 1'b0; mm_start <= 1'b0;
            rn_start <= 1'b0; rn_xv <= 1'b0; rn_gv <= 1'b0;
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1'b1; sel <= 2'd0; hd <= 0;
                    in_r <= {{(16*(PMAX_IN-MODEL_DIM)){1'b0}}, x_in};
                    setup_pass(2'd0);
                end

                S_PREP:   begin kcnt <= 0; st <= S_STREAM; end
                S_STREAM: begin
                    if (kcnt == klen - 1'b1) st <= S_WAIT;
                    kcnt <= kcnt + 1'b1;
                end

                S_WAIT: if (mm_ov) begin
                    for (gi = 0; gi < TN; gi = gi + 1) begin
                        case (sel)
                            2'd0: cq[16*(grp*TN + gi) +: 16] <= mm_c[16*gi +: 16];
                            2'd1: qv[16*(grp*TN + gi) +: 16] <= mm_c[16*gi +: 16];
                            2'd2: qa_out[16*(hd*KVL + grp*TN + gi) +: 16] <= mm_c[16*gi +: 16];
                            default: ckv_out[16*(grp*TN + gi) +: 16] <= mm_c[16*gi +: 16];
                        endcase
                    end
                    if (grp == ngrp - 1'b1) st <= S_NEXT;
                    else begin grp <= grp + 1'b1; mm_start <= 1'b1; st <= S_PREP; end
                end

                // c_q -> rmsnorm -> the activation for W_uq
                S_RN: begin
                    if (rn_inreq) begin rn_x <= cq[16*xidx +: 16]; rn_xv <= 1'b1; xidx <= xidx + 1'b1; end
                    if (rn_greq)  begin rn_g <= 16'h3F80;          rn_gv <= 1'b1; gidx <= gidx + 1'b1; end
                    if (rn_yv)    begin in_r[16*yidx +: 16] <= rn_y;              yidx <= yidx + 1'b1; end
                    if (rn_done) begin sel <= 2'd1; setup_pass(2'd1); end
                end

                S_NEXT: begin
                    case (sel)
`ifdef INJ_MLAP_NO_QNORM
                        // must FAIL: hand c_q to W_uq unnormalised. The generator's
                        // self-test pins that the norm is live (q differs with and
                        // without it), so this is not a hopeful injection.
                        2'd0: begin
                            for (gi = 0; gi < QLORA; gi = gi + 1)
                                in_r[16*gi +: 16] <= cq[16*gi +: 16];
                            sel <= 2'd1; setup_pass(2'd1);
                        end
`else
                        2'd0: begin                       // c_q done -> normalise it
                            xidx <= 0; gidx <= 0; yidx <= 0;
                            rn_start <= 1'b1; st <= S_RN;
                        end
`endif
                        2'd1: begin                       // q done -> fold head 0
                            hd <= 0; sel <= 2'd2;
                            for (gi = 0; gi < QK; gi = gi + 1)
                                in_r[16*gi +: 16] <= qv[16*gi +: 16];
                            setup_pass(2'd2);
                        end
                        2'd2: begin
                            if (hd == H[HW-1:0] - 1'b1) begin
                                sel <= 2'd3;
                                in_r <= {{(16*(PMAX_IN-MODEL_DIM)){1'b0}}, x_in};
                                setup_pass(2'd3);
                            end else begin
`ifdef INJ_MLAP_FOLD_HEAD0
                                // must FAIL: fold every head with HEAD 0's q. Right
                                // shape, right magnitudes, heads mixed. The
                                // generator pins that heads do not mix normally.
                                for (gi = 0; gi < QK; gi = gi + 1)
                                    in_r[16*gi +: 16] <= qv[16*gi +: 16];
`else
                                for (gi = 0; gi < QK; gi = gi + 1)
                                    in_r[16*gi +: 16] <= qv[16*((hd+1'b1)*QK + gi) +: 16];
`endif
                                hd <= hd + 1'b1;
                                setup_pass(2'd2);
                            end
                        end
                        default: st <= S_FIN;
                    endcase
                end

`ifdef INJ_MLAP_NORM_CKV
                // must FAIL: normalise the latent HERE as well. rmsnorm belongs on
                // the READ side, once, inside glm53f_mla_score -- doing it here too
                // applies it twice. Downstream that is only ~2.8e-06 relative and
                // no shape check notices, which is why it needs its own leg; at
                // THIS port it is loud, because a raw latent's RMS is not 1.
                S_FIN: begin
                    for (gi = 0; gi < KVL; gi = gi + 1)
                        ckv_out[16*gi +: 16] <= 16'h3F80;
                    done <= 1'b1; busy <= 1'b0; st <= S_IDLE;
                end
`else
                S_FIN: begin done <= 1'b1; busy <= 1'b0; st <= S_IDLE; end
`endif
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
`endif // GLM53F_MLA_PROJ_V
