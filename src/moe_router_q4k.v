`timescale 1ns/1ps
`include "glm_fp.vh"
`include "q4k.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// moe_router_q4k.v  --  GLM-5.2 / DeepSeek-v3 MoE ROUTER, Q4_K-NATIVE GATE GEMV
//                       (the Q4_K sibling of moe_router.v)       (§5,§6)
//----------------------------------------------------------------------------
// FUNCTION  (IDENTICAL math/ordering to moe_router.v; only the GATE GEMV numerics
//   change to the official GGML Q4_K weight format)
//
//       logits = x @ W_g          (W_g : HIDDEN x N_EXPERT, GEMV, K=HIDDEN)
//       gate   = sigmoid(logits)  (elementwise, N_EXPERT)
//       (idx, g) = TOP-K(gate)    (K=TOPK, lower-index tie-break)
//       s      = Σ_{j in TopK} gate_j
//       w_j    = (gate_j / s) * SCALE      for each selected j
//       OUTPUT : {idx_j, bf16(w_j)}  for the TOPK routed experts.
//
//   THE Q4_K SPLIT (this module's whole reason to exist):
//     * The router gate GEMV (logits = x @ W_g) runs through glm_matmul_q4k --
//       the official GGML Q4_K numerics: W_g is a GGML Q4_K-typed weight -- per
//       output column a super-block of 4-bit codes q with fp16 d/dmin + packed
//       6-bit block scales/mins, dequantized EXACTLY to fp32
//       w = (d*sc)*q - (dmin*m)  with NO
//       re-quantization; the activation x is bf16 fed DIRECT (no a_shift, no
//       activation quant); products are fp32 MACs accumulated in K order, then
//       rounded to bf16.  SAME drop-in glm_matmul_q4k wiring as swiglu_expert_q4k.
//     * The "tail" stays bf16, EXACTLY as moe_router.v (these are NOT weight
//       matmuls, so they are NOT quantized): the sigmoid (glm_act), the top-K
//       (topk_select), and the renormalize-then-x2.5 (the K gate weights sum to
//       SCALE=2.5).  Reuses glm_act + topk_select + glm_fp(_pipe) UNCHANGED.
//
//   This preserves the EXACT GLM/DeepSeek-v3 order: sigmoid-gate -> top-k ->
//   RENORMALIZE-THEN-SCALE (NOT scale-then-renormalize, NOT softmax).  The
//   shared expert is always-on and handled OUTSIDE this unit (§5).
//
//----------------------------------------------------------------------------
// PE_M BATCHING (B token rows share ONE gate-weight fetch)        (ULTRA_PERF#2)
//   PE_M (default 1 == byte-identical to the original single-token router) is the
//   number of token ROWS routed through the SAME gate matrix W_g in one pass.
//   glm_matmul_q4k is already PE_M-ready: it streams PE_M bf16 activation lanes
//   (a_col[16*PE_M]) against ONE weight tile (w_q[4*N_EXPERT] Q4_K codes,
//   shared) and emits PE_M*N_EXPERT results, time-sharing the weight stream and
//   the dequant.  So widening PE_M here costs activation-lane area
//   and a per-row TAIL (sigmoid/topk/renorm) but adds ZERO extra weight
//   bandwidth: the w_req / w_k request stream and the w_q / w_scale response
//   are IDENTICAL to PE_M=1 -- ONE fetch of W_g feeds all B rows.
//
//   Each row streams its OWN bf16 activation (that row's own token) DIRECT --
//   no per-row scale -- exactly as a PE_M=1 run on that row would, and glm_matmul_q4k
//   accumulates every (row,expert) independently -> row r's logits are
//   BIT-IDENTICAL to the PE_M=1 router run on row r's activation.  The bf16 tail
//   is REPLICATED per row (one sigmoid lane-group, one topk_select, one renorm
//   add-tree / rsqrt / fold per row), all marching LOCKSTEP against the one
//   shared weight stream -> row r's selected indices and routed weights are
//   exactly the PE_M=1 result for row r.  topk_select has data-INDEPENDENT
//   latency, so the PE_M instances pull scores and finish on the SAME cycles;
//   the FSM drives them with ONE shared score-address / valid stream (row 0 is
//   the representative for handshakes).  At PE_M=1 every PE_M-indexed construct
//   constant-folds to the original single-row datapath (identical ports, so the
//   committed TB instantiates it unchanged).
//
//----------------------------------------------------------------------------
// ACTIVATIONS (bf16, fed DIRECT -- Q4_K quantizes ONLY the weights)
//   Unlike the prior FP8 path there is NO activation quant and NO a_shift: each token
//   row's bf16 x vector streams straight into glm_matmul_q4k, which multiplies it
//   against the fp32-dequantized weights.  The ONLY quantized operand is W_g
//   (a GGML Q4_K-typed weight, dequantized exactly per super-block -- no re-quant).
//
//----------------------------------------------------------------------------
// DATAFLOW / FSM  (deterministic, handshake-driven off the matmul out_valid;
//   NO hardcoded matmul latency.  IDENTICAL FSM to moe_router.v; only the
//   data-bearing lanes / tail fan out with PE_M.)
//   S_IDLE : wait `start` (x_vec latched per-row into xbuf; per-row xsh latched).
//   S_MMP  : prime -- glm_matmul_q4k.start asserted the prior cycle.
//   S_MM   : stream K=HIDDEN beats into the Q4_K GEMV.  Each beat presents
//              a_col = {x_r[k] : r in 0..PE_M-1}  and the WEIGHT-PULL request
//              (w_req/w_k) so the system answers w_q = Q4_K column k of W_g
//              combinationally that cycle -- ONE column shared by all PE_M rows.
//   S_MMW  : wait the GEMV drain; on out_valid fire one sigmoid beat covering all
//            PE_M*N_EXPERT logit lanes.
//   S_ACT  : wait sigmoid; latch the PE_M*N_EXPERT bf16 gates, pulse the topk
//            starts.
//   S_TKL  : feed the topk score-pull (1/cycle, shared address); answer each
//            row's gate_f[idx].  On topk.done capture per-row indices + gates.
//   S_SUM  : per-row renorm add-tree over the TOPK selected gates -> s_r.
//   S_RCP  : per-row reciprocal r = 1/s and rs = r*SCALE = SCALE/s.
//   S_MUL  : per-row, per selected gate, w_j = gate_j * (SCALE/s); narrow to bf16.
//   S_DONE : pulse `done`; outputs {sel_idx, sel_weight} held until next start.
//
//----------------------------------------------------------------------------
// STYLE
//   synchronous ACTIVE-HIGH reset; NO latch (every reg assigned on every path);
//   NO combinational loop (all FP feedback rides pipeline registers).  Reuses
//   glm_matmul_q4k + glm_act + topk_select + glm_fp(_pipe) UNCHANGED.
//============================================================================
module moe_router_q4k #(
    // ---- HDR_LATE : L3 header path (the rw headers are stub-fed) ----
    parameter integer HDR_LATE= 0,
    parameter integer HIDDEN  = 128,           // model hidden size (scales to 6144)
    parameter integer N_EXPERT= 8,             // routed experts (real 256)
    parameter integer TOPK    = 2,             // experts per token (real 8)
    parameter [31:0]  SCALE   = 32'h40200000,  // routed_scaling_factor 2.5 (fp32)
    parameter integer KMAX    = 16384,         // >= HIDDEN (matmul K counter)
    parameter integer PE_M    = 1,             // token ROWS (batch B) sharing one W_g fetch
    parameter integer ACT_HW  = 0,             // sigmoid HW lanes (0 = full) -- result-invariant resource knob (glm_act HW_LANES)
    // IDXW DERIVED ($clog2(N_EXPERT)); exposed only to size the index port.
    // Do NOT override -- always leave default.
    parameter integer IDXW    = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT)
)(
    input  wire                       clk,
    input  wire                       rst,        // sync, active-high

    // ---- control handshake ----
    input  wire                       start,      // begin one route (PE_M token rows)
    output reg                        busy,
    output reg                        done,       // 1-cycle pulse, outputs valid

    // ---- token input (PE_M rows, row-major packed) ----
    //   row r element k = x_vec[16*(HIDDEN*r + k) +: 16]
    input  wire [16*HIDDEN*PE_M-1:0]  x_vec,      // PE_M bf16 tokens, packed

    // ---- gate-weight pull (mirror swiglu_expert_q4k's w_req/w_k/w_q + scales) ---
    //   The router drives, every K-beat, a request for column `w_k` of W_g; the
    //   system answers COMBINATIONALLY the same cycle with the N_EXPERT Q4_K 4-bit
    //   code lanes:  w_q[4*e +: 4] = W_g[ w_k , e ]   (e = expert 0..N_EXPERT-1).
    //   This request/response is INDEPENDENT of PE_M -- the B rows share it.
    output wire                       w_req,      // need a W_g column this cycle
    output wire [$clog2(KMAX+1)-1:0]  w_k,        // reduction index k (= row of W_g)
    input  wire [4*N_EXPERT-1:0]      w_q,        // N_EXPERT Q4_K 4-bit codes = W_g[k,*]
    // per-(expert col, super-block) Q4_K scales, latched by the matmul at start:
    input  wire [16*N_EXPERT*((KMAX+255)/256)-1:0] w_d,      // fp16 d
    input  wire [16*N_EXPERT*((KMAX+255)/256)-1:0] w_dmin,   // fp16 dmin
    input  wire [96*N_EXPERT*((KMAX+255)/256)-1:0] w_scales, // 6-bit scales

    // ---- routed result (PE_M rows; held from done until next start) ----
    //   row r slot t index  = sel_idx[IDXW*(TOPK*r + t) +: IDXW]  (slot 0 = top)
    //   row r slot t weight = sel_weight[16*(TOPK*r + t) +: 16]   (bf16)
    output reg  [TOPK*IDXW*PE_M-1:0]  sel_idx,    // PE_M x TOPK expert indices
    output reg  [TOPK*16*PE_M-1:0]    sel_weight  // PE_M x TOPK bf16 routed weights
);
    `include "glm_fp.vh"

    // ---------------- derived sizes ----
    localparam integer KW     = $clog2(KMAX+1);
    localparam integer MN     = N_EXPERT * PE_M;          // sigmoid lanes (PE_M rows x N_EXPERT)

    // renorm add-tree depth: ceil(log2 TOPK) levels of fp32_add_pipe.  TOPK is a
    // small power-of-two in practice (2 / 8); we build a CONSTANT-bounded tree.
    localparam integer SUMLEV = (TOPK <= 1) ? 1 : $clog2(TOPK);
    localparam integer NPOW   = (1 << SUMLEV);            // padded leaf count >= TOPK

    // ===================================================================
    //  Token buffer (latched at start; streamed into the GEMV).  PE_M rows.
    // ===================================================================
    reg [15:0] xbuf [0:PE_M-1][0:HIDDEN-1];
    integer xr, xk;
    always @(posedge clk) begin
        if (start) begin
            for (xr = 0; xr < PE_M; xr = xr + 1)
                for (xk = 0; xk < HIDDEN; xk = xk + 1)
                    xbuf[xr][xk] <= x_vec[16*(HIDDEN*xr + xk) +: 16];
        end
    end


    // ===================================================================
    //  FSM state
    // ===================================================================
    localparam [3:0] S_IDLE = 4'd0,
                     S_MMP  = 4'd1,   // prime: let matmul `streaming` go live
                     S_MM   = 4'd2,   // stream K=HIDDEN beats into the GEMV
                     S_MMW  = 4'd3,   // wait GEMV drain -> sigmoid launch
                     S_ACT  = 4'd4,   // wait sigmoid -> gates -> topk start
                     S_TKL  = 4'd5,   // feed topk score-pull, wait done
                     S_SUM  = 4'd6,   // renorm add-tree (Σ selected gates), per row
                     S_RCP  = 4'd7,   // reciprocal + (1/s)*SCALE precompute, per row
                     S_MUL  = 4'd8,   // per-gate * (SCALE/s) -> bf16 weights, per row
                     S_DONE = 4'd9;
    reg [3:0]  state;
    reg [KW-1:0] kcnt;               // K beat counter (GEMV stream)

    // ===================================================================
    //  (1) Q4_K GEMV : logits = x @ W_g   (PE_M token rows, PE_N=N_EXPERT logits)
    //      One shared W_g column stream; PE_M activation lanes fan out.
    // ===================================================================
    reg          mm_start;
    reg  [KW-1:0] mm_k_len;
    wire             mm_ov;
    wire [16*MN-1:0] mm_c;            // PE_M x N_EXPERT bf16 logits tile (row-major)

    wire stream_mm = (state == S_MM);
    localparam integer XIW = (HIDDEN   > 1) ? $clog2(HIDDEN)   : 1;
    localparam integer EIW = (N_EXPERT > 1) ? $clog2(N_EXPERT) : 1;
    wire [XIW-1:0]         x_idx = kcnt[XIW-1:0];

    // PE_M activation lanes: each row r presents its own x[k] this beat, all
    // multiplied against the SAME shared weight column inside the matmul.
    wire [16*PE_M-1:0] a_col;
    genvar ar;
    generate
    for (ar = 0; ar < PE_M; ar = ar + 1) begin : ACOL
        assign a_col[16*ar +: 16] = stream_mm ? xbuf[ar][x_idx] : 16'b0;
    end
    endgenerate

    wire                  mm_in_valid = stream_mm;
    assign w_req = stream_mm;
    assign w_k   = kcnt;

    // mm_busy is unused (FSM gates on out_valid + deterministic drain). Waived.
    /* verilator lint_off UNUSEDSIGNAL */
    wire             mm_busy;
    glm_matmul_q4k #(.PE_M(PE_M), .PE_N(N_EXPERT), .KMAX(KMAX), .HDR_LATE(HDR_LATE)) u_gemv (
        .clk(clk), .rst(rst),
        .start(mm_start), .k_len(mm_k_len),
        .w_d(w_d), .w_dmin(w_dmin), .w_scales(w_scales),
        .in_valid(mm_in_valid), .a_col(a_col), .w_q(w_q),
        .busy(mm_busy), .out_valid(mm_ov), .c_out(mm_c)
    );
    /* verilator lint_on UNUSEDSIGNAL */

    // ===================================================================
    //  (2) SIGMOID gating : gate = sigmoid(logits)  (PE_M*N_EXPERT lanes, 1 beat).
    //      glm_act is LANES-independent, so ONE instance covers all rows; lane
    //      l = r*N_EXPERT + e matches mm_c packing (logit[r][e] at 16*(r*N_EXPERT+e)).
    // ===================================================================
    reg                act_in_valid;
    reg  [16*MN-1:0]   act_x_in;     // logits fed to sigmoid
    wire               act_ov;
    wire [16*MN-1:0]   act_y;        // sigmoid(logits), bf16
    glm_act #(.MODE(0), .LANES(MN), .HW_LANES(ACT_HW)) u_sigmoid (  // MODE=0 => SIGMOID
        .clk(clk), .rst(rst),
        .in_valid(act_in_valid), .x_in(act_x_in),
        .out_valid(act_ov), .y_out(act_y)
    );

    // bf16 gates captured at act_ov, per row.  We store the 16-bit bf16 and apply
    // the EXACT bf16->fp32 widen at the topk feed point (bit-identical ordering).
    reg [15:0] gate_bf [0:PE_M-1][0:N_EXPERT-1];

    // ===================================================================
    //  (3) TOP-K : pick the TOPK largest gates per row -> indices + selected gates.
    //      ONE topk_select per row.  topk has DATA-INDEPENDENT latency, so all
    //      PE_M instances pull scores and finish on the SAME cycles; the FSM
    //      drives them with one shared score-address / score_valid (row 0 is the
    //      representative for the load_req / done handshakes).
    // ===================================================================
    reg               tk_start;
    reg               tk_score_valid;
    reg  [31:0]       tk_score_in [0:PE_M-1];   // per-row score answered this beat
    wire [TOPK*IDXW-1:0]  tk_sel_idx_r   [0:PE_M-1];
    wire [TOPK*32-1:0]    tk_sel_score_r [0:PE_M-1];
    // load_req/done are read only for row 0 (lockstep); the others + sel_valid/
    // mask/busy are unused.  Waived.
    /* verilator lint_off UNUSEDSIGNAL */
    wire                  tk_load_req_r  [0:PE_M-1];
    wire                  tk_done_r      [0:PE_M-1];
    wire [TOPK-1:0]       tk_sel_valid_r [0:PE_M-1];
    wire [N_EXPERT-1:0]   tk_mask_r      [0:PE_M-1];
    wire                  tk_busy_r      [0:PE_M-1];
    /* verilator lint_on UNUSEDSIGNAL */
    genvar tr;
    generate
    for (tr = 0; tr < PE_M; tr = tr + 1) begin : TOPK_R
        topk_select #(.N(N_EXPERT), .K(TOPK), .SCORE_W(32), .LANES_IN(1)) u_topk (
            .clk(clk), .rst(rst), .start(tk_start),
            .load_req(tk_load_req_r[tr]), .score_in(tk_score_in[tr]),
            .score_valid(tk_score_valid),
            .sel_idx_o(tk_sel_idx_r[tr]), .sel_score_o(tk_sel_score_r[tr]),
            .sel_valid_o(tk_sel_valid_r[tr]), .mask_o(tk_mask_r[tr]),
            .busy(tk_busy_r[tr]), .done(tk_done_r[tr])
        );
    end
    endgenerate
    wire tk_load_req = tk_load_req_r[0];   // representative (all rows lockstep)
    wire tk_done     = tk_done_r[0];

    // score-load address counter (which gate we hand topk this beat).  EIW+1 bits
    // so it counts 0..N_EXPERT cleanly (one spare bit for the == N).  SHARED.
    reg [EIW:0] tk_addr;
    // captured TOPK winner gates (fp32) per row at topk.done; winner indices are
    // written straight into sel_idx.
    reg [31:0]  win_gate [0:PE_M-1][0:TOPK-1];

    // ===================================================================
    //  (4) RENORM add-tree : s_r = Σ_{j in TopK} win_gate[r][j]  (fp32_add_pipe).
    //      ONE balanced binary tree of SUMLEV levels PER ROW (lockstep valids).
    // ===================================================================
    reg sum_go;                      // 1-cycle pulse: launch all per-row add-trees

    wire [31:0] sum_y_r [0:PE_M-1];  // s_r = Σ selected gates (fp32)
    wire        sum_v_r [0:PE_M-1];  // tree result valid (row r)
    genvar sgr, gl, gi;
    generate
    for (sgr = 0; sgr < PE_M; sgr = sgr + 1) begin : SUMTREE
        wire [31:0] tnode_y [0:SUMLEV][0:NPOW-1];
        wire        tnode_v [0:SUMLEV][0:NPOW-1];
        // level 0 leaves: win_gate padded with +0.0; valid = sum_go pulse.
        for (gi = 0; gi < NPOW; gi = gi + 1) begin : LEAF
            assign tnode_y[0][gi] = (gi < TOPK) ? win_gate[sgr][gi] : 32'h0000_0000;
            assign tnode_v[0][gi] = sum_go;
        end
        // internal levels: each node = add of its two children (one fp32_add_pipe).
        for (gl = 0; gl < SUMLEV; gl = gl + 1) begin : LVL
            for (gi = 0; gi < (NPOW >> (gl+1)); gi = gi + 1) begin : NODE
                /* verilator lint_off UNUSEDSIGNAL */
                wire nv;             // representative valid (siblings lockstep)
                /* verilator lint_on UNUSEDSIGNAL */
                fp32_add_pipe u_add (
                    .clk(clk), .rst(rst), .valid_in(tnode_v[gl][2*gi]),
                    .a(tnode_y[gl][2*gi]), .b(tnode_y[gl][2*gi+1]),
                    .valid_out(nv), .result(tnode_y[gl+1][gi])
                );
                assign tnode_v[gl+1][gi] = nv;
            end
            for (gi = (NPOW >> (gl+1)); gi < NPOW; gi = gi + 1) begin : PAD
                assign tnode_y[gl+1][gi] = 32'h0000_0000;
                assign tnode_v[gl+1][gi] = 1'b0;
            end
        end
        assign sum_y_r[sgr] = tnode_y[SUMLEV][0];
        assign sum_v_r[sgr] = tnode_v[SUMLEV][0];
    end
    endgenerate
    reg [31:0] s_reg [0:PE_M-1];     // latched s_r

    // ===================================================================
    //  (5) RECIPROCAL : r = 1/s_r via rsqrt(s_r)^2 ; rs_r = r*SCALE = SCALE/s_r.
    //      ONE rsqrt + fold PER ROW (lockstep valids).
    // ===================================================================
    reg          rcp_go;
    wire [31:0]  rsq_y_r [0:PE_M-1];          // rsqrt(s_r)
    /* verilator lint_off UNUSEDSIGNAL */
    wire         rsq_v_r [0:PE_M-1];          // only row 0 read (lockstep)
    /* verilator lint_on UNUSEDSIGNAL */
    wire [31:0]  c_rs_r  [0:PE_M-1];          // SCALE/s_r combinational (registered at rsq_v)
    genvar rr;
    generate
    for (rr = 0; rr < PE_M; rr = rr + 1) begin : RCP_R
        fp32_rsqrt_pipe u_rsqrt (
            .clk(clk), .rst(rst), .valid_in(rcp_go),
            .x(s_reg[rr]), .valid_out(rsq_v_r[rr]), .result(rsq_y_r[rr])
        );
        // r = rsqrt(s)^2 = 1/s ; rs = r * SCALE = SCALE/s.  Feed-forward, no loop.
        assign c_rs_r[rr] = fp32_mul(fp32_mul(rsq_y_r[rr], rsq_y_r[rr]), SCALE);
    end
    endgenerate
    wire rsq_v = rsq_v_r[0];                   // representative (all rows lockstep)
    reg [31:0] rs_reg [0:PE_M-1];             // SCALE / s_r (the single fold factor)

    // ===================================================================
    //  Main FSM + datapath
    // ===================================================================
    integer fr, fe, ft;
    always @(posedge clk) begin
        if (rst) begin
            state          <= S_IDLE;
            busy           <= 1'b0;
            done           <= 1'b0;
            kcnt           <= {KW{1'b0}};
            mm_start       <= 1'b0;
            mm_k_len       <= {KW{1'b0}};
            act_in_valid   <= 1'b0;
            act_x_in       <= {16*MN{1'b0}};
            tk_start       <= 1'b0;
            tk_score_valid <= 1'b0;
            tk_addr        <= {(EIW+1){1'b0}};
            sum_go         <= 1'b0;
            rcp_go         <= 1'b0;
            sel_idx        <= {TOPK*IDXW*PE_M{1'b0}};
            sel_weight     <= {TOPK*16*PE_M{1'b0}};
            for (fr = 0; fr < PE_M; fr = fr + 1) begin
                s_reg[fr]       <= 32'b0;
                rs_reg[fr]      <= 32'b0;
                tk_score_in[fr] <= 32'b0;
                for (fe = 0; fe < N_EXPERT; fe = fe + 1) gate_bf[fr][fe] <= 16'b0;
                for (ft = 0; ft < TOPK;     ft = ft + 1) win_gate[fr][ft] <= 32'b0;
            end
        end else begin
            // ---- defaults (deassert pulses) ----
            done           <= 1'b0;
            mm_start       <= 1'b0;
            act_in_valid   <= 1'b0;
            tk_start       <= 1'b0;
            tk_score_valid <= 1'b0;
            sum_go         <= 1'b0;
            rcp_go         <= 1'b0;

            case (state)
            // ----------------------------------------------------------
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy     <= 1'b1;
                    kcnt     <= {KW{1'b0}};
                    mm_start <= 1'b1;                 // start the Q4_K GEMV
                    mm_k_len <= HIDDEN[KW-1:0];
                    state    <= S_MMP;
                end
            end

            // ---- prime: matmul `streaming` goes live this cycle ----
            S_MMP: begin
                kcnt  <= {KW{1'b0}};
                state <= S_MM;
            end

            // ---- stream K=HIDDEN beats into the GEMV (operands driven comb) ----
            S_MM: begin
                if (kcnt == HIDDEN[KW-1:0] - 1'b1)
                    state <= S_MMW;
                kcnt <= kcnt + 1'b1;
            end

            // ---- wait GEMV drain; on out_valid launch sigmoid (all rows) ----
            S_MMW: begin
                if (mm_ov) begin
                    act_in_valid <= 1'b1;
                    act_x_in     <= mm_c;             // PE_M x N_EXPERT logits -> sigmoid
                    state        <= S_ACT;
                end
            end

            // ---- wait sigmoid; latch per-row gates, start topk ----
            S_ACT: begin
                if (act_ov) begin
                    for (fr = 0; fr < PE_M; fr = fr + 1)
                        for (fe = 0; fe < N_EXPERT; fe = fe + 1)
                            gate_bf[fr][fe] <= act_y[16*(fr*N_EXPERT + fe) +: 16];
                    tk_start <= 1'b1;                 // begin top-K (all rows)
                    tk_addr  <= {(EIW+1){1'b0}};
                    state    <= S_TKL;
                end
            end

            // ---- feed topk score-pull (1 score/cycle, shared addr), wait done ----
            S_TKL: begin
                if (tk_load_req) begin
                    tk_score_valid <= 1'b1;
                    for (fr = 0; fr < PE_M; fr = fr + 1)
                        tk_score_in[fr] <= bf16_to_fp32(gate_bf[fr][tk_addr[EIW-1:0]]);
                    tk_addr <= tk_addr + 1'b1;
                end
                if (tk_done) begin
                    for (fr = 0; fr < PE_M; fr = fr + 1)
                        for (ft = 0; ft < TOPK; ft = ft + 1) begin
                            sel_idx[IDXW*(TOPK*fr + ft) +: IDXW]
                                <= tk_sel_idx_r[fr][IDXW*ft +: IDXW];
                            win_gate[fr][ft] <= tk_sel_score_r[fr][32*ft +: 32];
                        end
                    sum_go <= 1'b1;                   // launch per-row renorm add-trees
                    state  <= S_SUM;
                end
            end

            // ---- wait add-trees -> s_r ; then launch per-row reciprocal ----
            S_SUM: begin
                if (sum_v_r[0]) begin
                    for (fr = 0; fr < PE_M; fr = fr + 1)
                        s_reg[fr] <= sum_y_r[fr];     // s_r = Σ selected gates
                    rcp_go <= 1'b1;                   // launch rsqrt(s_r) (all rows)
                    state  <= S_RCP;
                end
            end

            // ---- wait rsqrt; compute rs_r = SCALE/s_r ; then per-gate multiply ----
            S_RCP: begin
                if (rsq_v) begin
                    for (fr = 0; fr < PE_M; fr = fr + 1)
                        rs_reg[fr] <= c_rs_r[fr];     // SCALE / s_r (one fold factor)
                    state  <= S_MUL;
                end
            end

            // ---- per row, per selected gate: w_j = gate_j * (SCALE/s) -> bf16 ----
            S_MUL: begin
                for (fr = 0; fr < PE_M; fr = fr + 1)
                    for (ft = 0; ft < TOPK; ft = ft + 1)
                        sel_weight[16*(TOPK*fr + ft) +: 16] <=
                            fp32_to_bf16(fp32_mul(win_gate[fr][ft], rs_reg[fr]));
                state <= S_DONE;
            end

            // ---- done ----
            S_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
