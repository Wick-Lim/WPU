`timescale 1ns/1ps
// glm_matmul_mixed_tb.v -- BIT-EXACT verification of glm_matmul_q4k's MIXED-type
// front-end: a GEMM tile whose PE_N output columns each carry a DIFFERENT w_type
// (Q4_K / Q6_K / Q8_0 / F16), driven from the golden weights emitted by
// tools/q4k_mixed_gen.py (build/q4k_mixed_vec.txt), asserting every bf16 c_out
// equals the q4k_ref.py matmul_q4k_col golden EXACTLY (32-bit-then-bf16 compare).
//
// This is the integration sibling of glm_matmul_q4k_tb.v.  Where that TB proves the
// Q4_K path alone, this one exercises the w_type mux + ALL FOUR decoders in a single
// tile (columns rotated so every column slot sees every type across the 4 tiles),
// the high-precision buses (w_hp / w_q6_sc / w_q8_d), and accumulator reset between
// tiles.  The golden is the SAME q4k_ref.matmul_q4k_col contract used by the proven
// Q4_K TB -- only the per-column weight source differs -- so a green result proves
// the assembled mixed-type GEMM is bit-exact to the ggml-dequant reference.
//
// Vector layout (tools/q4k_mixed_gen.emit_mixed_gemm), col-outer / sb-inner:
//   NTEST PE_M PE_N
//   per tile:  K NSB NB8
//              wtype   : PE_N            (0=Q4_K 1=Q6_K 2=Q8_0 3=F16, per column)
//              w_d     : PE_N*NSB   4hex (Q4_K & Q6_K super-block d; 0 else)
//              w_dmin  : PE_N*NSB   4hex (Q4_K only)
//              w_scales: PE_N*NSB  24hex (Q4_K only)
//              w_q6_sc : PE_N*NSB*16 2hex(Q6_K 16 int8 scales/(col,sb); 0 else)
//              w_q8_d  : PE_N*NB8   4hex (Q8_0 fp16 d/(col,32blk); 0 else)
//              per beat k: a[pi](PE_M 4hex bf16)  then per col: w_q(1hex) w_hp(4hex)
//              C       : PE_M*PE_N  4hex bf16
module glm_matmul_mixed_tb;
    localparam integer PE_M = 2;
    localparam integer PE_N = 4;
    localparam integer KMAX = 512;                  // largest tile in the vector set
    localparam integer NSB  = (KMAX + 255) / 256;   // 2  (super-blocks along K)
    localparam integer NB8  = (KMAX + 31)  / 32;    // 16 (Q8_0 32-weight blocks)

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1;

    reg                        start = 0;
    reg  [$clog2(KMAX+1)-1:0]  k_len = 0;
    // UNCHANGED Q4_K buses (latched at start)
    reg  [16*PE_N*NSB-1:0]     w_d = 0, w_dmin = 0;
    reg  [96*PE_N*NSB-1:0]     w_scales = 0;
    // ADDED mixed-type buses
    reg  [ 3*PE_N-1:0]         w_type  = 0;         // per-column type (latched)
    reg  [16*PE_N-1:0]         w_hp    = 0;         // per-beat Q6_K/Q8_0/F16 code lane
    reg  [128*PE_N*NSB-1:0]    w_q6_sc = 0;         // Q6_K 16xint8 scales / (col,sb)
    reg  [16*PE_N*NB8-1:0]     w_q8_d  = 0;         // Q8_0 fp16 d / (col,32blk)

    reg                   in_valid = 0;
    reg  [16*PE_M-1:0]    a_col = 0;
    reg  [ 4*PE_N-1:0]    w_q = 0;
    wire                  busy, out_valid;
    wire [16*PE_M*PE_N-1:0] c_out;

    glm_matmul_q4k #(.PE_M(PE_M), .PE_N(PE_N), .KMAX(KMAX)) dut (
        .clk(clk), .rst(rst), .start(start), .k_len(k_len),
        .w_d(w_d), .w_dmin(w_dmin), .w_scales(w_scales),
        .in_valid(in_valid), .a_col(a_col), .w_q(w_q),
        .busy(busy), .out_valid(out_valid), .c_out(c_out),
        // mixed-type ports
        .w_type(w_type), .w_hp(w_hp), .w_q6_sc(w_q6_sc), .w_q8_d(w_q8_d)
    );

    integer fd, ntest, pm, pn, t, k, K, pi, pj, code, errors, checks;
    integer nsb_t, nb8_t, sb, b, i, tyv, tycol;
    integer n_q4k, n_q6k, n_q8, n_f16, n_q5k, seen;
    reg [15:0] a_beat  [0:PE_M-1];
    reg [3:0]  q_beat  [0:PE_N-1];
    reg [15:0] hp_beat [0:PE_N-1];
    reg [15:0] exp_c   [0:PE_M*PE_N-1];
    reg [15:0] dtmp, got, wqtmp;
    reg [95:0] stmp;
    reg [7:0]  sctmp8;

    initial begin
        errors = 0; checks = 0;
        n_q4k = 0; n_q6k = 0; n_q8 = 0; n_f16 = 0; n_q5k = 0; seen = 0;
        fd = $fopen("build/q4k_mixed_vec.txt", "r");
        if (fd == 0) begin $display("[glm_matmul_mixed] FAIL: cannot open build/q4k_mixed_vec.txt"); $finish; end
        code = $fscanf(fd, "%d %d %d", ntest, pm, pn);
        if (pm != PE_M || pn != PE_N) begin
            $display("[glm_matmul_mixed] FAIL: vector PE_M/PE_N (%0d,%0d) != TB (%0d,%0d)", pm, pn, PE_M, PE_N);
            $finish;
        end
        @(negedge clk); rst = 0;

        for (t = 0; t < ntest; t = t + 1) begin
            code = $fscanf(fd, "%d %d %d", K, nsb_t, nb8_t);

            // ---- per-column type selector (latched) ----
            w_type = 0;
            for (pj = 0; pj < PE_N; pj = pj + 1) begin
                code = $fscanf(fd, "%d", tyv);
                w_type[3*pj +: 3] = tyv[2:0];
                seen = seen | (1 << tyv);            // coverage: which types appeared
            end

            // ---- five latched header buses; slot = pj*NSB(+sb) / pj*NB8(+b),
            //      matching the RTL's compile-time NSB/NB8 stride (col-outer) ----
            w_d = 0; w_dmin = 0; w_scales = 0; w_q6_sc = 0; w_q8_d = 0;
            for (pj = 0; pj < PE_N; pj = pj + 1) for (sb = 0; sb < nsb_t; sb = sb + 1) begin
                code = $fscanf(fd, "%h", dtmp); w_d[16*(pj*NSB + sb) +: 16] = dtmp; end
            for (pj = 0; pj < PE_N; pj = pj + 1) for (sb = 0; sb < nsb_t; sb = sb + 1) begin
                code = $fscanf(fd, "%h", dtmp); w_dmin[16*(pj*NSB + sb) +: 16] = dtmp; end
            for (pj = 0; pj < PE_N; pj = pj + 1) for (sb = 0; sb < nsb_t; sb = sb + 1) begin
                code = $fscanf(fd, "%h", stmp); w_scales[96*(pj*NSB + sb) +: 96] = stmp; end
            for (pj = 0; pj < PE_N; pj = pj + 1) for (sb = 0; sb < nsb_t; sb = sb + 1)
                for (i = 0; i < 16; i = i + 1) begin
                    code = $fscanf(fd, "%h", sctmp8);
                    w_q6_sc[128*(pj*NSB + sb) + 8*i +: 8] = sctmp8; end
            for (pj = 0; pj < PE_N; pj = pj + 1) for (b = 0; b < nb8_t; b = b + 1) begin
                code = $fscanf(fd, "%h", dtmp); w_q8_d[16*(pj*NB8 + b) +: 16] = dtmp; end

            // ---- start pulse (latches k_len + all header/type buses) ----
            @(negedge clk); start = 1; k_len = K[$clog2(KMAX+1)-1:0]; in_valid = 0;
            @(negedge clk); start = 0;

            // ---- stream K beats: a[pi] + per-column {w_q, w_hp} ----
            for (k = 0; k < K; k = k + 1) begin
                for (pi = 0; pi < PE_M; pi = pi + 1) code = $fscanf(fd, "%h", a_beat[pi]);
                for (pj = 0; pj < PE_N; pj = pj + 1) begin
                    code = $fscanf(fd, "%h", wqtmp);  q_beat[pj]  = wqtmp[3:0];
                    code = $fscanf(fd, "%h", hp_beat[pj]);
                end
                for (pi = 0; pi < PE_M; pi = pi + 1) a_col[16*pi +: 16] = a_beat[pi];
                for (pj = 0; pj < PE_N; pj = pj + 1) begin
                    w_q [ 4*pj +:  4] = q_beat[pj];
                    w_hp[16*pj +: 16] = hp_beat[pj];
                end
                in_valid = 1;
                @(negedge clk);
            end
            in_valid = 0;

            // ---- expected bf16 outputs (pi-outer, pj-inner) ----
            for (pi = 0; pi < PE_M*PE_N; pi = pi + 1) begin code = $fscanf(fd, "%h", exp_c[pi]); end

            // out_valid pulses at the last beat's posedge -> already high at this
            // (loop-exit) negedge; poll a few cycles for robustness, then compare.
            k = 0;
            while (out_valid !== 1'b1 && k < 40) begin @(negedge clk); k = k + 1; end
            if (out_valid !== 1'b1) begin
                $display("FAIL tile %0d: out_valid never asserted", t); errors = errors + 1;
            end
            for (pi = 0; pi < PE_M; pi = pi + 1)
                for (pj = 0; pj < PE_N; pj = pj + 1) begin
                    got    = c_out[16*(pi*PE_N + pj) +: 16];
                    tycol  = w_type[3*pj +: 3];
                    checks = checks + 1;
                    case (tycol)
                        0: n_q4k = n_q4k + 1;
                        1: n_q6k = n_q6k + 1;
                        2: n_q8  = n_q8  + 1;
                        3: n_f16 = n_f16 + 1;
                        4: n_q5k = n_q5k + 1;
                    endcase
                    if (^got === 1'bx) begin
                        $display("FAIL tile %0d [%0d,%0d] type=%0d: X in output", t, pi, pj, tycol);
                        errors = errors + 1;
                    end else if (got !== exp_c[pi*PE_N + pj]) begin
                        $display("FAIL tile %0d [%0d,%0d] type=%0d K=%0d: got %h exp %h",
                                 t, pi, pj, tycol, K, got, exp_c[pi*PE_N + pj]);
                        errors = errors + 1;
                    end
                end
            @(negedge clk);
        end
        $fclose(fd);

        // every one of the FIVE types must have been exercised (guards against a
        // trivially-passing all-Q4_K file / a mis-parsed w_type column).  With
        // PE_N=4 no single tile can show all five, so this is a cross-tile
        // coverage assertion -- and it is what proves the Q5_K column is really
        // reaching the DUT rather than being silently dropped by the emitter.
        if (seen !== 5'b11111) begin
            $display("[glm_matmul_mixed] FAIL: not all 5 w_types exercised (seen mask %05b)", seen[4:0]);
            errors = errors + 1;
        end

        if (errors == 0)
            $display("[glm_matmul_mixed] ALL %0d TESTS PASSED (%0d tiles, mixed cols Q4_K=%0d Q5_K=%0d Q6_K=%0d Q8_0=%0d F16=%0d, bit-exact vs q4k_ref matmul)",
                     checks, ntest, n_q4k, n_q5k, n_q6k, n_q8, n_f16);
        else
            $display("[glm_matmul_mixed] %0d/%0d FAILURES", errors, checks);
        $finish;
    end

    initial begin #4000000; $display("[glm_matmul_mixed] TIMEOUT"); $finish; end
endmodule
