`timescale 1ns/1ps
//============================================================================
// weight_loader_q4k_mixed_tb.v -- BIT-EXACT loader->GEMM check for the MIXED-type
//   (Q6_K / Q8_0 / F16) feed of weight_loader_q4k.  The sibling of
//   test/weight_loader_q4k_tb.v (which proves the Q4_K feed): this one closes the
//   gap that the loader's mixed-type DMA path (Q6_K {d,sc128} header word, Q8_0
//   8-co-packed-fp16-d header word, F16 ns==0 no-header, mm_w_type broadcast, the
//   desc_wtype geometry select) was previously verified only by elaboration.
//
//   For EACH w_type (Q4_K/Q6_K/Q8_0/F16) the TB builds a weight-memory image of a
//   tile in EXACTLY the loader's expected per-type word layout, sets desc_wtype +
//   the tile descriptor, lets weight_loader_q4k PULL the header + code stream and
//   DRIVE glm_matmul_q4k, and compares the streamed bf16 C against the ggml golden
//   ref.matmul_q4k_col (materialised by tools/q4k_mixed_gen.emit_wlmixed into
//   build/wlmixed_vec.txt) -- 32-bit-then-bf16 exact (!==), X-aware (any X fails).
//   Because glm_matmul_q4k is already proven bit-exact, a mismatch localises to the
//   loader's byte-layout/geometry for that type.  The file is a MIXED SEQUENCE:
//   consecutive tiles of DIFFERENT w_type flow through the SAME loader+GEMM (as the
//   real dynamic UD-Q4_K_XL checkpoint interleaves types), each type at NSB=1/2/3.
//   Emits "ALL <N> TESTS PASSED"; $fatal on mismatch; asserts all 4 types covered.
//
//   LOADER WORD-MEMORY LAYOUT built per tile (word-addressed, from `base`):
//     HEADER region (none for F16): address base + (pj*NSB + sb), pj-outer/sb-inner
//       Q4_K: word[15:0]=d  word[31:16]=dmin  word[127:32]=scales(96b)
//       Q6_K: word[15:0]=d  word[16+8*i +: 8]=sc[i] (16 int8, i=0..15)
//       Q8_0: word[16*j +: 16]=d(8*sb+j) (8 fp16 d co-packed / super-block, j=0..7)
//     CODE region: base + ns + k  (ns = F16 ? 0 : NSB*PE_N), k=0..K-1
//       Q4_K: word[4*pj +: 4]  = 4-bit code   (loader drives mm_w_q)
//       else: word[16*pj +: 16]= code lane    (loader drives mm_w_hp)
//============================================================================
module weight_loader_q4k_mixed_tb;
    localparam integer PE_M   = 2;                  // must match the vector file
    // PE_N / DATA_W are `parameter` (iverilog -P overridable) so this same TB proves the
    // loader->GEMM bit-exact at scaled lane counts (see `make weight-loader-lanes`), not
    // only the default.  DATA_W must be >= 16*PE_N (the loader guards this).  VEC_FILE
    // lets the scaled gate point at its own generated golden without racing the default.
    parameter integer PE_N   = 4;
    localparam integer KMAX   = 768;                // one..three super-blocks along K
    localparam integer NSB    = (KMAX + 255) / 256; // 3
    localparam integer NB8    = (KMAX + 31)  / 32;  // 24 (== 8*NSB)
    localparam integer ADDR_W = 16;
    parameter integer DATA_W = 256;
    parameter VEC_FILE = "build/wlmixed_vec.txt";
    localparam integer KW     = $clog2(KMAX+1);
    localparam integer SBW    = $clog2(NSB+1);

    // weight-type enum (matches weight_loader_q4k / glm_matmul_q4k desc_wtype)
    localparam [1:0] WT_Q4K = 2'd0, WT_Q6K = 2'd1, WT_Q80 = 2'd2, WT_F16 = 2'd3;

    integer errors = 0;
    integer checks = 0;
    integer n_q4k = 0, n_q6k = 0, n_q8 = 0, n_f16 = 0;
    reg [3:0] seen = 4'b0000;                       // per-type coverage mask

    // ---------------- shared clock / reset ----------------
    reg clk = 1'b0;
    reg rst;
    always #5 clk = ~clk;

    // ---------------- tile storage (logical fields, declared before use) --------
    reg [15:0] a_a  [0:PE_M-1][0:KMAX-1];       // bf16 activations A[pi][k]
    reg [ 3:0] q_a  [0:PE_N-1][0:KMAX-1];       // Q4_K 4-bit codes W[k][pj]
    reg [15:0] hp_a [0:PE_N-1][0:KMAX-1];       // Q6_K/Q8_0/F16 16-bit code lane
    reg [15:0] d_a  [0:PE_N-1][0:NSB-1];        // fp16 super-block d   (Q4_K & Q6_K)
    reg [15:0] dm_a [0:PE_N-1][0:NSB-1];        // fp16 super-block dmin(Q4_K)
    reg [95:0] sc_a [0:PE_N-1][0:NSB-1];        // 96b packed scales   (Q4_K)
    reg [ 7:0] q6_a [0:PE_N-1][0:NSB-1][0:15];  // Q6_K 16 int8 scales /(col,sb)
    reg [15:0] q8_a [0:PE_N-1][0:NB8-1];        // Q8_0 fp16 d /(col,32blk)
    reg [15:0] exp_c[0:PE_M*PE_N-1];            // golden bf16 output

    // ====================================================================
    // DUT PATH : weight_loader_q4k reads the image and DRIVES glm_matmul_q4k.
    // ====================================================================
    reg                       load = 1'b0;
    reg  [ADDR_W-1:0]         desc_base;
    reg  [KW-1:0]             desc_klen;
    reg  [SBW-1:0]            desc_nsblk;
    reg  [2:0]                desc_wtype;

    wire                      mem_en;
    wire [ADDR_W-1:0]         mem_addr;
    reg  [DATA_W-1:0]         mem_data;

    wire                      mm_start;
    wire [KW-1:0]             mm_k_len;
    wire [ 4*PE_N-1:0]        mm_w_q;
    wire [16*PE_N*NSB-1:0]    mm_w_d;
    wire [16*PE_N*NSB-1:0]    mm_w_dmin;
    wire [96*PE_N*NSB-1:0]    mm_w_scales;
    wire                      mm_in_valid;
    wire [ 3*PE_N-1:0]        mm_w_type;
    wire [16*PE_N-1:0]        mm_w_hp;
    wire [128*PE_N*NSB-1:0]   mm_w_q6_sc;
    wire [16*PE_N*NB8-1:0]    mm_w_q8_d;
    wire                      ld_busy;
    wire                      ld_done;

    weight_loader_q4k #(
        .PE_N(PE_N), .KMAX(KMAX), .ADDR_W(ADDR_W), .DATA_W(DATA_W)
    ) u_ld (
        .clk(clk), .rst(rst),
        .load(load), .desc_base(desc_base), .desc_klen(desc_klen),
        .desc_nsblk(desc_nsblk), .desc_wtype(desc_wtype),
        .mem_en(mem_en), .mem_addr(mem_addr), .mem_data(mem_data),
        .mm_start(mm_start), .mm_k_len(mm_k_len), .mm_w_q(mm_w_q),
        .mm_w_d(mm_w_d), .mm_w_dmin(mm_w_dmin), .mm_w_scales(mm_w_scales),
        .mm_in_valid(mm_in_valid),
        .mm_w_type(mm_w_type), .mm_w_hp(mm_w_hp),
        .mm_w_q6_sc(mm_w_q6_sc), .mm_w_q8_d(mm_w_q8_d),
        .busy(ld_busy), .done(ld_done)
    );

    // activation side for the DUT: the TB owns it (the loader is the beat master).
    reg  [KW:0]              dut_beat;
    integer                  dut_K;
    reg  [16*PE_M-1:0]       acol_d;
    wire                     dut_busy;
    wire                     dut_ov;
    wire [16*PE_M*PE_N-1:0]  dut_cout;

    glm_matmul_q4k #(.PE_M(PE_M), .PE_N(PE_N), .KMAX(KMAX)) u_dut (
        .clk(clk), .rst(rst), .start(mm_start), .k_len(mm_k_len),
        .w_d(mm_w_d), .w_dmin(mm_w_dmin), .w_scales(mm_w_scales),
        .in_valid(mm_in_valid), .a_col(acol_d), .w_q(mm_w_q),
        .busy(dut_busy), .out_valid(dut_ov), .c_out(dut_cout),
        // mixed-type ports driven straight from the loader
        .w_type(mm_w_type), .w_hp(mm_w_hp),
        .w_q6_sc(mm_w_q6_sc), .w_q8_d(mm_w_q8_d)
    );

    // beat counter: reset at mm_start, ++ per streamed weight beat -> on the cycle
    // mm_in_valid is high for beat k, dut_beat == k.
    always @(posedge clk) begin
        if (rst)              dut_beat <= {(KW+1){1'b0}};
        else if (mm_start)    dut_beat <= {(KW+1){1'b0}};
        else if (mm_in_valid) dut_beat <= dut_beat + 1'b1;
    end

    // combinational activation feed -> A[*][dut_beat] on the in_valid cycle.
    integer ai;
    always @* begin
        acol_d = {(16*PE_M){1'b0}};
        for (ai = 0; ai < PE_M; ai = ai + 1)
            acol_d[16*ai +: 16] = (dut_beat < dut_K) ? a_a[ai][dut_beat] : 16'h0000;
    end

    // ---------------- latency-1 read memory (DDR5/Flash stub) ----------------
    localparam integer MEM_WORDS = 2048;
    reg [DATA_W-1:0] mem [0:MEM_WORDS-1];
    always @(posedge clk) begin
        if (mem_en) mem_data <= mem[mem_addr];
    end

    // ====================================================================
    // Build the loader's per-type word-memory image for one tile.
    // ====================================================================
    task build_mem(input [1:0] ty, input [ADDR_W-1:0] base,
                   input integer K, input integer nsb, input integer nb8);
        integer i, k, pj, sb, j, li, ns, cbase;
        reg [DATA_W-1:0] word;
        begin
            for (i = 0; i < MEM_WORDS; i = i + 1) mem[i] = {DATA_W{1'b0}};
            ns = (ty == WT_F16) ? 0 : nsb*PE_N;      // header words (F16: none)
            // ---- HEADER region : address base + (pj*NSB + sb), pj-outer/sb-inner ----
            if (ty != WT_F16) begin
                for (pj = 0; pj < PE_N; pj = pj + 1)
                    for (sb = 0; sb < nsb; sb = sb + 1) begin
                        li   = pj*nsb + sb;          // loader header addr: base + pj*nsblk + sb
                        word = {DATA_W{1'b0}};
                        case (ty)
                            WT_Q6K: begin
                                word[15:0] = d_a[pj][sb];
                                for (j = 0; j < 16; j = j + 1)
                                    word[16 + 8*j +: 8] = q6_a[pj][sb][j];
                            end
                            WT_Q80: begin
                                for (j = 0; j < 8; j = j + 1)
                                    word[16*j +: 16] = q8_a[pj][8*sb + j];
                            end
                            default: begin           // Q4_K
                                word[15:0]   = d_a [pj][sb];
                                word[31:16]  = dm_a[pj][sb];
                                word[127:32] = sc_a[pj][sb];
                            end
                        endcase
                        mem[base + li] = word;
                    end
            end
            // ---- CODE region : base + ns + k ----
            cbase = base + ns;
            for (k = 0; k < K; k = k + 1) begin
                word = {DATA_W{1'b0}};
                for (pj = 0; pj < PE_N; pj = pj + 1) begin
                    if (ty == WT_Q4K) word[4*pj  +:  4] = q_a [pj][k];
                    else              word[16*pj +: 16] = hp_a[pj][k];
                end
                mem[cbase + k] = word;
            end
        end
    endtask

    // drive one tile through the loader+GEMM.
    task run_dut(input [1:0] ty, input [ADDR_W-1:0] base,
                 input integer K, input integer nsb, input integer nb8);
        begin
            build_mem(ty, base, K, nsb, nb8);
            dut_K      = K;
            desc_wtype = ty;
            desc_base  = base;
            desc_klen  = K[KW-1:0];
            desc_nsblk = nsb[SBW-1:0];
            @(negedge clk);
            load = 1'b1;
            @(negedge clk);
            load = 1'b0;
            do @(negedge clk); while (dut_ov !== 1'b1);
        end
    endtask

    // bit-exact compare (X-aware) of the loader-streamed GEMM vs golden.
    task compare(input integer t, input [1:0] ty, input integer K, input integer nsb);
        integer pi, pj, e;
        reg [15:0] g, d;
        begin
            for (pi = 0; pi < PE_M; pi = pi + 1)
                for (pj = 0; pj < PE_N; pj = pj + 1) begin
                    e = pi*PE_N + pj;
                    checks = checks + 1;
                    case (ty)
                        WT_Q4K: n_q4k = n_q4k + 1;
                        WT_Q6K: n_q6k = n_q6k + 1;
                        WT_Q80: n_q8  = n_q8  + 1;
                        WT_F16: n_f16 = n_f16 + 1;
                    endcase
                    d = dut_cout[16*e +: 16];
                    g = exp_c[e];
                    if (^d === 1'bx) begin
                        errors = errors + 1;
                        $display("  FAIL test %0d type=%0d [%0d,%0d]: loader/GEMM has X (%b)",
                                 t, ty, pi, pj, d);
                    end else if (d !== g) begin
                        errors = errors + 1;
                        $display("  FAIL test %0d type=%0d [%0d,%0d] (K=%0d nsb=%0d): loader=%h golden=%h",
                                 t, ty, pi, pj, K, nsb, d, g);
                    end
                end
        end
    endtask

    // ====================================================================
    // Stimulus : read build/wlmixed_vec.txt, run each tile through loader+GEMM.
    // ====================================================================
    integer fd, code, ntest, pm, pn, t, K, nsb, nb8, pi, pj, sb, k, j, tyi;
    reg [15:0] tmp16;
    reg [95:0] tmp96;
    reg [ 7:0] tmp8;
    reg [ 3:0] tmpq;
    reg [ADDR_W-1:0] base;

    initial begin
        fd = $fopen(VEC_FILE, "r");
        if (fd == 0) begin
            $display("[weight_loader_q4k_mixed] FAIL: cannot open %0s (run: python3 tools/q4k_mixed_gen.py)", VEC_FILE);
            $fatal(1, "missing vectors");
        end
        code = $fscanf(fd, "%d %d %d", ntest, pm, pn);
        if (pm != PE_M || pn != PE_N) begin
            $display("[weight_loader_q4k_mixed] FAIL: vector PE_M/PE_N (%0d,%0d) != TB (%0d,%0d)",
                     pm, pn, PE_M, PE_N);
            $fatal(1, "dim mismatch");
        end

        rst = 1'b1;
        repeat (5) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        for (t = 0; t < ntest; t = t + 1) begin
            code = $fscanf(fd, "%d %d %d %d", tyi, K, nsb, nb8);
            seen = seen | (4'd1 << tyi[1:0]);        // per-type coverage

            // ---- header logical fields (col-outer / sb-inner) ----
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (sb = 0; sb < nsb; sb = sb + 1) begin
                    code = $fscanf(fd, "%h", tmp16); d_a[pj][sb] = tmp16; end
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (sb = 0; sb < nsb; sb = sb + 1) begin
                    code = $fscanf(fd, "%h", tmp16); dm_a[pj][sb] = tmp16; end
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (sb = 0; sb < nsb; sb = sb + 1) begin
                    code = $fscanf(fd, "%h", tmp96); sc_a[pj][sb] = tmp96; end
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (sb = 0; sb < nsb; sb = sb + 1)
                    for (j = 0; j < 16; j = j + 1) begin
                        code = $fscanf(fd, "%h", tmp8); q6_a[pj][sb][j] = tmp8; end
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (j = 0; j < nb8; j = j + 1) begin
                    code = $fscanf(fd, "%h", tmp16); q8_a[pj][j] = tmp16; end

            // ---- per-beat activations + per-column {wq, hp} ----
            for (k = 0; k < K; k = k + 1) begin
                for (pi = 0; pi < PE_M; pi = pi + 1) begin
                    code = $fscanf(fd, "%h", tmp16); a_a[pi][k] = tmp16; end
                for (pj = 0; pj < PE_N; pj = pj + 1) begin
                    code = $fscanf(fd, "%h", tmpq);  q_a[pj][k]  = tmpq;
                    code = $fscanf(fd, "%h", tmp16); hp_a[pj][k] = tmp16; end
            end
            // ---- golden outputs ----
            for (pi = 0; pi < PE_M*PE_N; pi = pi + 1) begin
                code = $fscanf(fd, "%h", tmp16); exp_c[pi] = tmp16; end

            // vary base to exercise a non-trivial tile origin too.
            base = (t & 1) ? 16'd1024 : 16'd0;
            run_dut(tyi[1:0], base, K, nsb, nb8);
            compare(t, tyi[1:0], K, nsb);
            repeat (2) @(negedge clk);
        end
        $fclose(fd);

        // every one of the four types must have flowed through the same loader.
        if (seen !== 4'b1111) begin
            $display("[weight_loader_q4k_mixed] FAIL: not all 4 w_types exercised (seen %04b)", seen);
            errors = errors + 1;
        end

        if (errors == 0)
            $display("[weight_loader_q4k_mixed] ALL %0d TESTS PASSED (%0d tiles, loader->GEMM bit-exact vs ggml golden; Q4_K=%0d Q6_K=%0d Q8_0=%0d F16=%0d checks)",
                     checks, ntest, n_q4k, n_q6k, n_q8, n_f16);
        else begin
            $display("[weight_loader_q4k_mixed] %0d/%0d CHECKS FAILED", errors, checks);
            $fatal(1, "weight_loader_q4k_mixed binding check FAILED");
        end
        $finish;
    end

    // safety timeout
    initial begin
        #20000000;
        $display("[weight_loader_q4k_mixed] TIMEOUT");
        $fatal(1, "weight_loader_q4k_mixed binding check TIMEOUT");
    end
endmodule
