`timescale 1ns/1ps
//============================================================================
// test/packer_rtl_crosscheck_tb.v -- the PACKER's file, consumed by the RTL
//   (`make packer-rtl-crosscheck`)
//
// WHAT IS PROVEN
//   tools/ckpt_pack_q4k.py emitted HEADER words sb-outer while weight_loader_q4k
//   reads them at memory offset pj*n_sblk+sb (col-outer).  The orders coincide at
//   nb==1 -- every sim geometry this repo ever ran -- and silently diverge at real
//   K>256.  Silently: the wrong d/dmin/scales are plausible fp16 values, so the
//   model would have looked DEGRADED on the board, not broken.
//
//   This gate runs the REAL weight_loader_q4k over an image the REAL packer wrote
//   (build/pk_cross_img.hex, from tools/packer_rtl_crosscheck.py) at nb=3, and
//   compares every mm_w_d / mm_w_dmin / mm_w_scales slot and every code nibble
//   against expectations recorded from the SOURCE arrays before packing.  The
//   file the packer writes is the file the RTL reads: a cross-TOOL gate.
//
// INJECTION (`make packer-rtl-crosscheck` requires FAILURE)
//   -DINJ_PKX_OLDORDER permutes each tile's header region back to the OLD
//   sb-outer order after loading -- recreating the exact bug this gate exists to
//   catch.  At nb=3 the header checks MUST fail.  If they do not, the gate
//   cannot tell the two orders apart and constrains nothing.
//============================================================================
module packer_rtl_crosscheck_tb;

    localparam integer PE_N   = 4;
    localparam integer KMAX   = 768;
    localparam integer NSB    = 3;          // KMAX/256
    localparam integer ADDR_W = 24;
    localparam integer DATA_W = 256;
    localparam integer IMG_WORDS = 2048;

    localparam integer KW  = $clog2(KMAX+1);
    localparam integer SBW = $clog2(NSB+1);

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst;

    reg                   load;
    reg  [ADDR_W-1:0]     desc_base;
    reg  [KW-1:0]         desc_klen;
    reg  [SBW-1:0]        desc_nsblk;
    wire                  mem_en;
    wire [ADDR_W-1:0]     mem_addr;
    reg  [DATA_W-1:0]     mem_data;
    wire                  mm_start, mm_in_valid, busy, done;
    wire [KW-1:0]         mm_k_len;
    wire [4*PE_N-1:0]     mm_w_q;
    wire [16*PE_N*NSB-1:0] mm_w_d, mm_w_dmin;
    wire [96*PE_N*NSB-1:0] mm_w_scales;

    weight_loader_q4k #(.PE_N(PE_N), .KMAX(KMAX), .ADDR_W(ADDR_W), .DATA_W(DATA_W))
    dut (.clk(clk), .rst(rst), .load(load),
         .desc_base(desc_base), .desc_klen(desc_klen), .desc_nsblk(desc_nsblk),
         .desc_wtype(2'd0),
         .mem_en(mem_en), .mem_addr(mem_addr), .mem_data(mem_data),
         .mm_start(mm_start), .mm_k_len(mm_k_len), .mm_w_q(mm_w_q),
         .mm_w_d(mm_w_d), .mm_w_dmin(mm_w_dmin), .mm_w_scales(mm_w_scales),
         .mm_in_valid(mm_in_valid),
         /* verilator lint_off PINCONNECTEMPTY */
         .mm_w_type(), .mm_w_hp(), .mm_w_q6_sc(), .mm_w_q8_d(),
         .busy(busy), .done(done),
         .ecc_err_inject({1'b0}), .ecc_corr_count(), .ecc_uncorrectable());

    // ---- the packer's image, served latency-1 (the documented wl_mem contract) --
    reg [DATA_W-1:0] img [0:IMG_WORDS-1];
    always @(posedge clk) mem_data <= img[mem_addr[10:0]];

    // ---- expectations (from the SOURCE arrays, independent of pack order) ------
    integer fd, code, n_tiles, e_pen, e_nsb, e_k;
    reg [15:0]  e_d   [0:1][0:PE_N-1][0:NSB-1];
    reg [15:0]  e_dm  [0:1][0:PE_N-1][0:NSB-1];
    reg [95:0]  e_sc  [0:1][0:PE_N-1][0:NSB-1];
    reg [15:0]  e_nib [0:1][0:KMAX-1];
    integer     e_base[0:1];
    integer     tmp_i;
    reg [15:0]  tmp16a, tmp16b;
    reg [95:0]  tmp96;

    integer errors, tests, t, pj, sb, k, kcnt;
    task chk(input cond, input [8*96-1:0] name);
        begin tests = tests + 1;
              if (!cond) begin errors = errors + 1; $display("FAIL: %0s", name); end end
    endtask

    initial begin
        errors = 0; tests = 0;
        load = 1'b0; desc_base = 0; desc_klen = 0; desc_nsblk = 0;

        $readmemh("build/pk_cross_img.hex", img);
`ifdef INJ_PKX_OLDORDER
        // INJECTION: permute each tile's header region back to the OLD sb-outer
        //   order -- new[sb*PE_N+pj] = written[pj*NSB+sb].  Recreates the exact
        //   packer bug at nb=3; the header comparisons MUST fail.
        begin : inj
            reg [DATA_W-1:0] tmp [0:NSB*PE_N-1];
            integer ti, ii, jj;
            for (ti = 0; ti < 2; ti = ti + 1) begin
                for (ii = 0; ii < PE_N; ii = ii + 1)
                    for (jj = 0; jj < NSB; jj = jj + 1)
                        tmp[jj*PE_N + ii] = img[ti*(NSB*PE_N+768) + ii*NSB + jj];
                for (ii = 0; ii < NSB*PE_N; ii = ii + 1)
                    img[ti*(NSB*PE_N+768) + ii] = tmp[ii];
            end
        end
`endif

        fd = $fopen("build/pk_cross_exp.txt", "r");
        if (fd == 0) $fatal(1, "missing build/pk_cross_exp.txt (run tools/packer_rtl_crosscheck.py)");
        code = $fscanf(fd, "%d %d %d %d", n_tiles, e_pen, e_nsb, e_k);
        if (n_tiles != 2 || e_pen != PE_N || e_nsb != NSB || e_k != KMAX)
            $fatal(1, "expectation geometry mismatch");
        // iverilog cannot $fscanf into a multi-dimensional array element (it
        // treats the indexed element as a constant in system-task args), so scan
        // into scalars first -- same idiom as test/weight_loader_q4k_tb.v.
        for (t = 0; t < 2; t = t + 1) begin
            code = $fscanf(fd, "%d", tmp_i);  e_base[t] = tmp_i;
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (sb = 0; sb < NSB; sb = sb + 1) begin
                    code = $fscanf(fd, "%h %h %h", tmp16a, tmp16b, tmp96);
                    e_d[t][pj][sb]  = tmp16a;
                    e_dm[t][pj][sb] = tmp16b;
                    e_sc[t][pj][sb] = tmp96;
                end
            for (k = 0; k < KMAX; k = k + 1) begin
                code = $fscanf(fd, "%h", tmp16a);  e_nib[t][k] = tmp16a;
            end
        end
        $fclose(fd);

        rst = 1'b1; repeat (5) @(negedge clk); rst = 1'b0;

        for (t = 0; t < 2; t = t + 1) begin
            @(negedge clk);
            desc_base  = e_base[t][ADDR_W-1:0];
            desc_klen  = KMAX[KW-1:0];
            desc_nsblk = NSB[SBW-1:0];
            load = 1'b1; @(negedge clk); load = 1'b0;

            wait (mm_start);
            // headers: bus slot (pj*NSB + sb) must hold the SOURCE values
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (sb = 0; sb < NSB; sb = sb + 1) begin
                    chk(mm_w_d[16*(pj*NSB+sb) +: 16] === e_d[t][pj][sb],
                        "header d matches the source (col-outer slot)");
                    chk(mm_w_dmin[16*(pj*NSB+sb) +: 16] === e_dm[t][pj][sb],
                        "header dmin matches the source");
                    chk(mm_w_scales[96*(pj*NSB+sb) +: 96] === e_sc[t][pj][sb],
                        "header scales match the source");
                end
            // codes: every beat's PE_N nibbles
            kcnt = 0;
            while (kcnt < KMAX) begin
                @(negedge clk);
                if (mm_in_valid) begin
                    tests = tests + 1;
                    if (mm_w_q !== e_nib[t][kcnt][4*PE_N-1:0]) begin
                        if (errors < 6)
                            $display("FAIL: tile %0d code beat %0d = %h, source says %h",
                                     t, kcnt, mm_w_q, e_nib[t][kcnt][4*PE_N-1:0]);
                        errors = errors + 1;
                    end
                    kcnt = kcnt + 1;
                end
            end
            wait (done);
        end

        if (errors != 0) begin
            $display("FAILED: %0d error(s) across %0d checks", errors, tests);
            $fatal(1, "packer_rtl_crosscheck_tb had mismatches");
        end
        $display("ALL %0d TESTS PASSED  (packer-built image consumed by the real weight_loader_q4k at nb=%0d: every header slot and every code nibble matches the pre-pack source)",
                 tests, NSB);
        $finish;
    end

    initial begin
        #4000000;
        $display("FAIL: global timeout");
        $fatal(1, "timeout");
    end
endmodule
