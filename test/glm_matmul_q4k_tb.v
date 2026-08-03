`timescale 1ns/1ps
// glm_matmul_q4k_tb.v -- bit-exact verification of glm_matmul_q4k vs the ggml Q4_K
// golden (tools/q4k_ref.py, vectors from tools/q4k_matmul_gen.py -> build/q4k_vec.txt).
// Proves the Q4_K-typed weights run with NO re-quantization, bit-exact to the
// ggml Q4_K reference (tools/q4k_ref.py).
// Real-dims sweep overrides (docs/SCALE_FUNCTIONAL.md item 2, `make scale-ops`):
//   -DTB_KMAX=6144 -DTB_VEC='"build/q4k_real_vec.txt"' -DTB_TIMEOUT_NS=...
//   runs the SAME bit-exact contract at the real projection K (NSB=24).
//   Defaults reproduce the committed slice run byte-identically.
`ifndef TB_KMAX
    `define TB_KMAX 1024
`endif
`ifndef TB_VEC
    `define TB_VEC "build/q4k_vec.txt"
`endif
`ifndef TB_TIMEOUT_NS
    `define TB_TIMEOUT_NS 2000000
`endif
`ifndef TB_HDR_LATE
`define TB_HDR_LATE 0
`endif
module glm_matmul_q4k_tb;
    localparam integer PE_M = 2;
    localparam integer PE_N = 2;
    localparam integer KMAX = `TB_KMAX;
    localparam integer NSB  = (KMAX + 255) / 256;   // 4 at the slice KMAX=1024

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1;

    reg                        start = 0;
    reg  [$clog2(KMAX+1)-1:0]  k_len = 0;
    reg  [16*PE_N*NSB-1:0]     w_d = 0, w_dmin = 0;
    reg  [16*PE_N*NSB-1:0]     saved_d, saved_dmin;
    reg  [96*PE_N*NSB-1:0]     saved_scales;
    reg  [96*PE_N*NSB-1:0]     w_scales = 0;
    reg                   in_valid = 0;
    reg  [16*PE_M-1:0]    a_col = 0;
    reg  [ 4*PE_N-1:0]    w_q = 0;
    wire                  busy, out_valid;
    wire [16*PE_M*PE_N-1:0] c_out;

`ifndef TB_REG_COUT
    `define TB_REG_COUT 0
`endif
    // REG_COUT=1 registers the C bus + delays out_valid with it (rank 7).  The TB
    // waits on out_valid, so the SAME golden must pass in BOTH settings -- that is
    // exactly the property that makes the pipeline stage safe to enable.
    glm_matmul_q4k #(.PE_M(PE_M), .PE_N(PE_N), .KMAX(KMAX), .REG_COUT(`TB_REG_COUT), .HDR_LATE(`TB_HDR_LATE)) dut (
        .clk(clk), .rst(rst), .start(start), .k_len(k_len),
        .w_d(w_d), .w_dmin(w_dmin), .w_scales(w_scales),
        .in_valid(in_valid), .a_col(a_col), .w_q(w_q),
        .busy(busy), .out_valid(out_valid), .c_out(c_out)
    );

    integer fd, ntest, pm, pn, t, k, K, pi, pj, code, errors, checks, nsb_t, sb;
    reg [15:0] a_beat [0:PE_M-1];
    reg [3:0]  q_beat [0:PE_N-1];
    reg [15:0] exp_c  [0:PE_M*PE_N-1];
    reg [15:0] dtmp, got;
    reg [95:0] stmp;

    initial begin
        errors = 0; checks = 0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[glm_matmul_q4k] FAIL: cannot open %s", `TB_VEC); $finish; end
        code = $fscanf(fd, "%d %d %d", ntest, pm, pn);
        if (pm != PE_M || pn != PE_N) begin
            $display("[glm_matmul_q4k] FAIL: vector PE_M/PE_N (%0d,%0d) != TB (%0d,%0d)", pm, pn, PE_M, PE_N);
            $finish;
        end
        @(negedge clk); rst = 0;

        for (t = 0; t < ntest; t = t + 1) begin
            code = $fscanf(fd, "%d %d", K, nsb_t);
            w_d = 0; w_dmin = 0; w_scales = 0;
            for (pj = 0; pj < PE_N; pj = pj + 1) for (sb = 0; sb < nsb_t; sb = sb + 1) begin
                code = $fscanf(fd, "%h", dtmp); w_d   [16*(pj*NSB + sb) +: 16] = dtmp; end
            for (pj = 0; pj < PE_N; pj = pj + 1) for (sb = 0; sb < nsb_t; sb = sb + 1) begin
                code = $fscanf(fd, "%h", dtmp); w_dmin[16*(pj*NSB + sb) +: 16] = dtmp; end
            for (pj = 0; pj < PE_N; pj = pj + 1) for (sb = 0; sb < nsb_t; sb = sb + 1) begin
                code = $fscanf(fd, "%h", stmp); w_scales[96*(pj*NSB + sb) +: 96] = stmp; end

            // start pulse (latches params)
`ifdef TB_HDR_STAGE
            // LATE-HEADER stimulus (the L3 BRAM-store timing): GARBAGE on the
            //   header buses during the start cycle, the real values from the
            //   first stream cycle on.  HDR_LATE=1 must be immune (it reads the
            //   wires at accept time); HDR_LATE=0 MUST fail (its start-latch
            //   grabs the garbage) -- that failure is what proves this stimulus
            //   actually poisons the early latch, so the =1 pass is not vacuous.
            saved_d = w_d; saved_dmin = w_dmin; saved_scales = w_scales;
            w_d = {16*PE_N*NSB{1'b1}}; w_dmin = {16*PE_N*NSB{1'b1}};
            w_scales = {96*PE_N*NSB{1'b1}};
            @(negedge clk); start = 1; k_len = K[$clog2(KMAX+1)-1:0]; in_valid = 0;
            @(negedge clk); start = 0;
            w_d = saved_d; w_dmin = saved_dmin; w_scales = saved_scales;
`else
            @(negedge clk); start = 1; k_len = K[$clog2(KMAX+1)-1:0]; in_valid = 0;
            @(negedge clk); start = 0;
`endif

            // stream K beats
            for (k = 0; k < K; k = k + 1) begin
                for (pi = 0; pi < PE_M; pi = pi + 1) code = $fscanf(fd, "%h", a_beat[pi]);
                for (pj = 0; pj < PE_N; pj = pj + 1) code = $fscanf(fd, "%h", q_beat[pj]);
                for (pi = 0; pi < PE_M; pi = pi + 1) a_col[16*pi +: 16] = a_beat[pi];
                for (pj = 0; pj < PE_N; pj = pj + 1) w_q [ 4*pj +:  4] = q_beat[pj][3:0];
                in_valid = 1;
                @(negedge clk);
            end
            in_valid = 0;

            // expected outputs
            for (pi = 0; pi < PE_M*PE_N; pi = pi + 1) begin code = $fscanf(fd, "%h", exp_c[pi]); end

            // out_valid pulses at the last beat's posedge -> it is already high at this
            // (loop-exit) negedge; poll a few cycles in case of off-by-one, then compare.
            k = 0;
            while (out_valid !== 1'b1 && k < 40) begin @(negedge clk); k = k + 1; end
            if (out_valid !== 1'b1) begin
                $display("FAIL test %0d: out_valid never asserted", t); errors = errors + 1;
            end
            for (pi = 0; pi < PE_M; pi = pi + 1)
                for (pj = 0; pj < PE_N; pj = pj + 1) begin
                    got = c_out[16*(pi*PE_N + pj) +: 16];
                    checks = checks + 1;
                    if (^got === 1'bx) begin
                        $display("FAIL test %0d [%0d,%0d]: X in output", t, pi, pj); errors = errors + 1;
                    end else if (got !== exp_c[pi*PE_N + pj]) begin
                        $display("FAIL test %0d [%0d,%0d] K=%0d: got %h exp %h", t, pi, pj, K, got, exp_c[pi*PE_N + pj]);
                        errors = errors + 1;
                    end
                end
            @(negedge clk);
        end
        $fclose(fd);
        if (errors == 0)
            $display("[glm_matmul_q4k] ALL %0d TESTS PASSED (%0d tiles, bit-exact vs ggml Q4_K golden)", checks, ntest);
        else
            $display("[glm_matmul_q4k] %0d/%0d FAILURES", errors, checks);
        $finish;
    end

    initial begin #`TB_TIMEOUT_NS; $display("[glm_matmul_q4k] TIMEOUT"); $finish; end
endmodule
