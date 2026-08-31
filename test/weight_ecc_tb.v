`timescale 1ns/1ps
//============================================================================
// weight_ecc_tb.v  --  SECDED read-path fault-injection proof for
//                      weight_loader_q4k (WEIGHT_ECC=1).   USAGE_GAPS §B / #32
//----------------------------------------------------------------------------
// The resident ~467 GB weights have NO bit-error protection.  weight_loader_q4k
// gains a DEFAULT-OFF SECDED ECC read path: with WEIGHT_ECC=1 every read word is
// modelled as ECC_LANE_W-wide SECDED lanes (the weight memory stores no check
// bits, so the codec is a DECODE STAGE on the read data with an injectable fault
// port).  This TB proves the three obligations against that model, driving the
// REAL loader FSM over a real tile and observing the streamed weight codes:
//
//   (0) DEFAULT-OFF IDENTITY : a WEIGHT_ECC=0 twin and the WEIGHT_ECC=1 DUT are
//       fed the SAME memory and, with NO injected fault, stream BYTE-IDENTICAL
//       weight codes -- the ECC layer is transparent when clean.
//   (a) CLEAN     (inject=0)         : ECC=1 stream == the clean codes written
//       into memory; ecc_corr_count stays 0; ecc_uncorrectable stays 0.
//   (b) SBU  (1 bit flipped, lane 0) : the streamed codes STILL match the clean
//       codes (SECDED corrected the bit); ecc_corr_count INCREMENTS; no derr.
//   (c) DBU  (2 bits flipped, lane 0): ecc_uncorrectable ASSERTS (detected,
//       uncorrectable) and stays sticky.
//
// The fault is injected persistently across the whole tile load; the observable
// is mm_w_q / mm_w_hp (== the ECC-corrected read word's low lane during the code
// stream).  X-aware: any X on a checked code is a hard failure.  A test that
// cannot fail is a failure: (a) fails if a real correction is silently wrong,
// (b) fails if the counter never moves, (c) fails if a double error is missed --
// each is separately provable by construction.
//
// Emits "[weight_ecc] ALL <N> TESTS PASSED"; $fatal on any mismatch.
//============================================================================
module weight_ecc_tb;
    // ---- geometry (must match the DUT instances) ----
    localparam integer PE_N       = 4;
    localparam integer KMAX       = 256;
    localparam integer ADDR_W     = 16;
    localparam integer DATA_W     = 256;
    localparam integer ECC_LANE_W = 64;
    localparam integer ECC_CNT_W  = 32;
    localparam integer KW         = $clog2(KMAX + 1);
    localparam integer NSB        = (KMAX + 255) / 256;   // 1
    localparam integer SBW        = $clog2(NSB + 1);

    // SECDED lane geometry (independently derived; matches src/ecc_secded.v).
    function integer calc_p;
        input integer dw; integer p;
        begin p = 0; while ((1 << p) < (dw + p + 1)) p = p + 1; calc_p = p; end
    endfunction
    localparam integer ECC_P     = calc_p(ECC_LANE_W);        // 7
    localparam integer ECC_LCODE = ECC_LANE_W + ECC_P + 1;    // 72
    localparam integer ECC_NLANE = DATA_W / ECC_LANE_W;       // 4
    localparam integer ECC_CTOT  = ECC_NLANE * ECC_LCODE;     // 288

    integer tests  = 0;
    integer errors = 0;

    // ---------------- clock / reset ----------------
    reg clk = 1'b0;
    reg rst;
    always #5 clk = ~clk;

    // ---------------- descriptor / shared drive ----------------
    reg                  load = 1'b0;
    reg [ADDR_W-1:0]     desc_base;
    reg [KW-1:0]         desc_klen;
    reg [SBW-1:0]        desc_nsblk;

    // fault-injection mask (codeword domain) -- driven ONLY into the ECC=1 DUT.
    reg [ECC_CTOT-1:0]   inj;

    // ---------------- tile contents (known clean weight codes) ----------------
    localparam integer K = 12;                 // 12 code beats (single super-block)
    reg [3:0]  q_a [0:PE_N-1][0:K-1];          // clean 4-bit Q4_K codes W[k][pj]
    reg [15:0] d_a  [0:PE_N-1][0:NSB-1];
    reg [15:0] dm_a [0:PE_N-1][0:NSB-1];
    reg [95:0] sc_a [0:PE_N-1][0:NSB-1];

    // ============================================================
    // DUT 0 : WEIGHT_ECC=0 (the shipped default path -- reference)
    // DUT 1 : WEIGHT_ECC=1 (the SECDED read path -- under test)
    // Both share clk/rst/load/descriptor and the SAME memory image; only DUT1
    // sees the fault mask.  Because ECC never touches mem_en/mem_addr the two
    // FSMs run in lockstep (asserted below), so one memory feeds both.
    // ============================================================
    wire                 en0, en1;
    wire [ADDR_W-1:0]    ad0, ad1;
    reg  [DATA_W-1:0]    mdat;                 // shared latency-1 read data

    wire [4*PE_N-1:0]    wq0,  wq1;
    wire [16*PE_N-1:0]   whp0, whp1;
    wire                 iv0,  iv1;
    wire                 busy0, busy1, done0, done1;
    wire [ECC_CNT_W-1:0] cc1;
    wire                 unc1;

    weight_loader_q4k #(
        .PE_N(PE_N), .KMAX(KMAX), .ADDR_W(ADDR_W), .DATA_W(DATA_W),
        .WEIGHT_ECC(0)
    ) u_off (
        .clk(clk), .rst(rst), .load(load),
        .desc_base(desc_base), .desc_klen(desc_klen), .desc_nsblk(desc_nsblk),
        .desc_wtype(3'd0),
        .mem_en(en0), .mem_addr(ad0), .mem_data(mdat),
        .mm_start(), .mm_k_len(), .mm_w_q(wq0),
        .mm_w_d(), .mm_w_dmin(), .mm_w_scales(), .mm_in_valid(iv0),
        .mm_w_type(), .mm_w_hp(whp0), .mm_w_q6_sc(), .mm_w_q8_d(),
        .busy(busy0), .done(done0),
        .ecc_err_inject({ECC_CTOT{1'b0}}), .ecc_corr_count(), .ecc_uncorrectable()
    );

    weight_loader_q4k #(
        .PE_N(PE_N), .KMAX(KMAX), .ADDR_W(ADDR_W), .DATA_W(DATA_W),
        .WEIGHT_ECC(1), .ECC_LANE_W(ECC_LANE_W), .ECC_CNT_W(ECC_CNT_W)
    ) u_on (
        .clk(clk), .rst(rst), .load(load),
        .desc_base(desc_base), .desc_klen(desc_klen), .desc_nsblk(desc_nsblk),
        .desc_wtype(3'd0),
        .mem_en(en1), .mem_addr(ad1), .mem_data(mdat),
        .mm_start(), .mm_k_len(), .mm_w_q(wq1),
        .mm_w_d(), .mm_w_dmin(), .mm_w_scales(), .mm_in_valid(iv1),
        .mm_w_type(), .mm_w_hp(whp1), .mm_w_q6_sc(), .mm_w_q8_d(),
        .busy(busy1), .done(done1),
        .ecc_err_inject(inj), .ecc_corr_count(cc1), .ecc_uncorrectable(unc1)
    );

    // lockstep sanity: ECC must not perturb the request bus.
    always @(posedge clk) begin
        if (!rst && (en0 !== en1 || (en0 && (ad0 !== ad1)))) begin
            $display("  FAIL: ECC perturbed the request bus (en0=%b en1=%b ad0=%h ad1=%h)",
                     en0, en1, ad0, ad1);
            errors = errors + 1;
        end
    end

    // ---------------- shared latency-1 read memory ----------------
    localparam integer MEM_WORDS = 256;
    reg [DATA_W-1:0] mem [0:MEM_WORDS-1];
    always @(posedge clk) begin
        if (en1) mdat <= mem[ad1];             // en0==en1, ad0==ad1 (asserted above)
    end

    // ---------------- build the memory image (loader storage layout) ----------
    //   HEADER : base + (pj*nsblk + sb),  word[15:0]=d,[31:16]=dmin,[127:32]=sc
    //   CODE   : base + nsblk*PE_N + k,    word[4*pj +: 4] = q_a[pj][k]
    task build_mem(input integer nsblk);
        integer i, k, pj, sb, li;
        reg [DATA_W-1:0] word;
        begin
            for (i = 0; i < MEM_WORDS; i = i + 1) mem[i] = {DATA_W{1'b0}};
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (sb = 0; sb < nsblk; sb = sb + 1) begin
                    li   = pj*nsblk + sb;
                    word = {DATA_W{1'b0}};
                    word[15:0]   = d_a [pj][sb];
                    word[31:16]  = dm_a[pj][sb];
                    word[127:32] = sc_a[pj][sb];
                    mem[li] = word;
                end
            for (k = 0; k < K; k = k + 1) begin
                word = {DATA_W{1'b0}};
                for (pj = 0; pj < PE_N; pj = pj + 1) word[4*pj +: 4] = q_a[pj][k];
                mem[nsblk*PE_N + k] = word;
            end
        end
    endtask

    // expected clean code-row (low 4*PE_N bits of the code word for beat k).
    function [4*PE_N-1:0] clean_row(input integer k);
        integer pj; reg [4*PE_N-1:0] r;
        begin
            r = {(4*PE_N){1'b0}};
            for (pj = 0; pj < PE_N; pj = pj + 1) r[4*pj +: 4] = q_a[pj][k];
            clean_row = r;
        end
    endfunction

    // ---------------- per-beat capture of the streamed code rows ----------------
    // On every in_valid cycle, record which beat and the code each DUT streamed.
    reg [4*PE_N-1:0] seen_off [0:K-1];
    reg [4*PE_N-1:0] seen_on  [0:K-1];
    reg              got_off  [0:K-1];
    reg              got_on   [0:K-1];
    integer          bo, bn;

    task clear_capture;
        integer i;
        begin
            for (i = 0; i < K; i = i + 1) begin
                got_off[i] = 1'b0; got_on[i] = 1'b0;
                seen_off[i] = {(4*PE_N){1'bx}}; seen_on[i] = {(4*PE_N){1'bx}};
            end
            bo = 0; bn = 0;
        end
    endtask

    always @(posedge clk) begin
        if (!rst) begin
            if (iv0 && bo < K) begin seen_off[bo] = wq0; got_off[bo] = 1'b1; bo = bo + 1; end
            if (iv1 && bn < K) begin seen_on [bn] = wq1; got_on [bn] = 1'b1; bn = bn + 1; end
        end
    end

    // ---------------- run one tile through both loaders ----------------
    task run_tile;
        begin
            clear_capture;
            desc_base  = 16'd0;
            desc_klen  = K[KW-1:0];
            desc_nsblk = NSB[SBW-1:0];
            @(negedge clk);
            load = 1'b1;
            @(negedge clk);
            load = 1'b0;
            // wait until BOTH loaders signal done.
            do @(negedge clk); while (!(done0 && done1));
            repeat (2) @(negedge clk);
        end
    endtask

    // ---------------- assertions ----------------
    task chk(input cond, input [1023:0] msg);
        begin
            tests = tests + 1;
            if (cond !== 1'b1) begin
                errors = errors + 1;
                $display("  FAIL: %0s", msg);
            end
        end
    endtask

    // every beat 0..K-1 was streamed and matches the clean code (X-aware).
    task check_stream_clean(input on, input [1023:0] tag);
        integer k;
        reg [4*PE_N-1:0] v; reg g;
        begin
            for (k = 0; k < K; k = k + 1) begin
                g = on ? got_on[k]  : got_off[k];
                v = on ? seen_on[k] : seen_off[k];
                tests = tests + 1;
                if (!g) begin
                    errors = errors + 1;
                    $display("  FAIL[%0s]: beat %0d never streamed", tag, k);
                end else if (^v === 1'bx) begin
                    errors = errors + 1;
                    $display("  FAIL[%0s]: beat %0d has X (%b)", tag, k, v);
                end else if (v !== clean_row(k)) begin
                    errors = errors + 1;
                    $display("  FAIL[%0s]: beat %0d code=%h != clean %h", tag, k, v, clean_row(k));
                end
            end
        end
    endtask

    integer i, pj, sb, k;
    reg [ECC_CNT_W-1:0] cc_before;

    initial begin
        // deterministic, nonzero, distinct weight codes + plausible headers.
        for (pj = 0; pj < PE_N; pj = pj + 1) begin
            for (k = 0; k < K; k = k + 1)
                q_a[pj][k] = (4'(pj) + 4'(k) + 4'd1) & 4'hF;   // varied nibble codes
            for (sb = 0; sb < NSB; sb = sb + 1) begin
                d_a [pj][sb] = 16'h3C00 + pj;   // ~1.0 fp16 + jitter
                dm_a[pj][sb] = 16'h0000;
                sc_a[pj][sb] = {8{12'(pj*7 + sb + 3)}};
            end
        end
        build_mem(NSB);

        rst = 1'b1; inj = {ECC_CTOT{1'b0}};
        repeat (5) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        // ---- (0) DEFAULT-OFF IDENTITY + (a) CLEAN: no fault injected ----
        inj = {ECC_CTOT{1'b0}};
        run_tile;
        // both DUTs streamed all K beats matching the clean codes ...
        check_stream_clean(1'b0, "clean/off");
        check_stream_clean(1'b1, "clean/on");
        // ... and (0) they are byte-identical to each other, beat by beat.
        for (k = 0; k < K; k = k + 1)
            chk(seen_off[k] === seen_on[k], "clean: ECC=1 code != ECC=0 code (not transparent)");
        // clean telemetry: no corrections, not uncorrectable.
        chk(cc1  === {ECC_CNT_W{1'b0}}, "clean: ecc_corr_count nonzero");
        chk(unc1 === 1'b0,              "clean: ecc_uncorrectable set on clean read");

        // ---- (b) SBU: flip ONE bit in lane-0's stored codeword (a data bit) ----
        //   codeword bit index 2 == Hamming position 3 == the first DATA bit of
        //   lane 0 -> a genuine data-bit flip the decoder must CORRECT.
        cc_before = cc1;
        inj = {ECC_CTOT{1'b0}};
        inj[2] = 1'b1;                          // lane 0, one data bit
        run_tile;
        check_stream_clean(1'b1, "sbu/on");     // STILL clean -> corrected
        chk(cc1 !== cc_before, "SBU: ecc_corr_count did not increment (no correction counted)");
        chk((cc1 > cc_before), "SBU: ecc_corr_count went backwards");
        chk(unc1 === 1'b0,     "SBU: single-bit error wrongly flagged uncorrectable");
        // sanity: the OFF twin, with NO ECC and NO fault (inj is ECC=1-only), is
        // still clean -- confirms the fault path is ECC-exclusive.
        check_stream_clean(1'b0, "sbu/off-unaffected");

        // ---- (c) DBU: flip TWO bits in lane-0's codeword (two data bits) ----
        //   codeword bits 2 and 4 == Hamming positions 3 and 5 == two DATA bits
        //   of lane 0 -> syndrome!=0, even parity -> DETECTED, uncorrectable.
        inj = {ECC_CTOT{1'b0}};
        inj[2] = 1'b1;
        inj[4] = 1'b1;
        run_tile;
        chk(unc1 === 1'b1, "DBU: ecc_uncorrectable did NOT assert on a double-bit error");

        // ---- sticky-clear: a fresh clean load must clear ecc_uncorrectable ----
        inj = {ECC_CTOT{1'b0}};
        run_tile;
        chk(unc1 === 1'b0, "sticky: ecc_uncorrectable not cleared by a clean tile load");

        if (errors == 0)
            $display("[weight_ecc] ALL %0d TESTS PASSED (SECDED read path: default-off transparent; SBU corrected; DBU flagged)", tests);
        else begin
            $display("[weight_ecc] %0d/%0d CHECKS FAILED", errors, tests);
            $fatal(1, "weight_ecc ECC read-path proof FAILED");
        end
        $finish;
    end

    // safety timeout
    initial begin
        #2000000;
        $display("[weight_ecc] TIMEOUT");
        $fatal(1, "weight_ecc TIMEOUT");
    end
endmodule
