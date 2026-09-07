//============================================================================
// q5k_loader_tb.v -- the packer-built Q5_K image consumed by the REAL
// weight_loader_q4k (image + expectations from tools/q5k_loader_crosscheck.py).
//
// glm_matmul_q4k has consumed Q5_K since the GEMM arm landed (`make mixedtype`).
// What did not exist was a way to GET a Q5_K tile to it: the loader $fatal'ed on
// a Q5_K descriptor. That guard's premise turned out to be wrong -- a Q5_K tile
// needs nothing new from the loader, because its header fields ARE Q4_K's and its
// 5-bit code rides the same mm_w_hp lane Q6_K and Q8_0 already use. The missing
// half was the PACKER. This is the cross-TOOL gate for that: the file the packer
// writes is the file the RTL reads, and the expectations come from the SOURCE
// arrays before packing, so they are independent of the layout under test.
//
// nb=3 on purpose, for the same reason packer_rtl_crosscheck uses it: the
// col-outer / super-block-inner header order coincides with sb-outer only at
// nb==1, which is every naive sim geometry.
//
// Must FAIL: an image packed by `q5k_loader_crosscheck.py --inj-noqh`, which
// drops the fifth bit while leaving the header and the expectations intact -- a
// byte-plausible Q5_K image that is really Q4_K data.
//============================================================================
`timescale 1ns/1ps
module q5k_loader_tb;
    localparam integer PE_N   = 4;
    localparam integer KMAX   = 768;
    localparam integer NSB    = 3;
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
    wire [16*PE_N*NSB-1:0] mm_w_d, mm_w_dmin;
    wire [96*PE_N*NSB-1:0] mm_w_scales;
    wire [3*PE_N-1:0]      mm_w_type;
    wire [16*PE_N-1:0]     mm_w_hp;

    weight_loader_q4k #(.PE_N(PE_N), .KMAX(KMAX), .ADDR_W(ADDR_W), .DATA_W(DATA_W))
    dut (.clk(clk), .rst(rst), .load(load),
         .desc_base(desc_base), .desc_klen(desc_klen), .desc_nsblk(desc_nsblk),
         .desc_wtype(3'd4),                       // WT_Q5K
         .mem_en(mem_en), .mem_addr(mem_addr), .mem_data(mem_data),
         .mm_start(mm_start), .mm_k_len(mm_k_len),
         /* verilator lint_off PINCONNECTEMPTY */
         .mm_w_q(),
         .mm_w_d(mm_w_d), .mm_w_dmin(mm_w_dmin), .mm_w_scales(mm_w_scales),
         .mm_in_valid(mm_in_valid),
         .mm_w_type(mm_w_type), .mm_w_hp(mm_w_hp), .mm_w_q6_sc(), .mm_w_q8_d(),
         .busy(busy), .done(done),
         .ecc_err_inject({1'b0}), .ecc_corr_count(), .ecc_uncorrectable());

    reg [DATA_W-1:0] img [0:IMG_WORDS-1];
    always @(posedge clk) mem_data <= img[mem_addr[10:0]];

    integer fd, code, n_tiles, e_pen, e_nsb, e_k;
    reg [15:0]  e_d  [0:1][0:PE_N-1][0:NSB-1];
    reg [15:0]  e_dm [0:1][0:PE_N-1][0:NSB-1];
    reg [95:0]  e_sc [0:1][0:PE_N-1][0:NSB-1];
    reg [63:0]  e_hp [0:1][0:KMAX-1];
    integer     e_base[0:1];
    integer     tmp_i;
    reg [15:0]  tmp16a, tmp16b;
    reg [95:0]  tmp96;
    reg [63:0]  tmp64;

    integer errors, tests, t, pj, sb, k, kcnt;
    task chk(input cond, input [8*96-1:0] name);
        begin tests = tests + 1;
              if (!cond) begin errors = errors + 1;
                  if (errors < 8) $display("FAIL: %0s", name); end end
    endtask

    initial begin
        errors = 0; tests = 0;
        load = 1'b0; desc_base = 0; desc_klen = 0; desc_nsblk = 0;

        $readmemh("build/q5k_cross_img.hex", img);
        fd = $fopen("build/q5k_cross_exp.txt", "r");
        if (fd == 0) $fatal(1, "missing build/q5k_cross_exp.txt (run tools/q5k_loader_crosscheck.py)");
        code = $fscanf(fd, "%d %d %d %d", n_tiles, e_pen, e_nsb, e_k);
        if (n_tiles != 2 || e_pen != PE_N || e_nsb != NSB || e_k != KMAX)
            $fatal(1, "expectation geometry mismatch");
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
                code = $fscanf(fd, "%h", tmp64);  e_hp[t][k] = tmp64;
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
            // the tile's type must reach every column, or the engine takes the
            // Q4_K default arm and reads the low nibble of a 5-bit code
            chk(mm_w_type === {PE_N{3'd4}}, "mm_w_type is WT_Q5K on every column");
            for (pj = 0; pj < PE_N; pj = pj + 1)
                for (sb = 0; sb < NSB; sb = sb + 1) begin
                    chk(mm_w_d[16*(pj*NSB+sb) +: 16] === e_d[t][pj][sb],
                        "header d matches the source (Q5_K shares Q4_K's header)");
                    chk(mm_w_dmin[16*(pj*NSB+sb) +: 16] === e_dm[t][pj][sb],
                        "header dmin matches the source");
                    chk(mm_w_scales[96*(pj*NSB+sb) +: 96] === e_sc[t][pj][sb],
                        "header scales match the source");
                end
            kcnt = 0;
            while (kcnt < KMAX) begin
                @(negedge clk);
                if (mm_in_valid) begin
                    tests = tests + 1;
                    if (mm_w_hp !== e_hp[t][kcnt][16*PE_N-1:0]) begin
                        if (errors < 8)
                            $display("FAIL: tile %0d code beat %0d hp = %h, source says %h",
                                     t, kcnt, mm_w_hp, e_hp[t][kcnt][16*PE_N-1:0]);
                        errors = errors + 1;
                    end
                    kcnt = kcnt + 1;
                end
            end
            wait (done);
        end

        if (errors != 0) begin
            $display("FAILED: %0d error(s) across %0d checks", errors, tests);
            $fatal(1, "q5k_loader_tb had mismatches");
        end
        $display("ALL %0d TESTS PASSED  (packer-built Q5_K image consumed by the real weight_loader_q4k at nb=%0d: WT_Q5K on every column, every header slot from the Q4_K-shared decode, and every pre-assembled 5-bit code on mm_w_hp matches the pre-pack source)",
                 tests, NSB);
        $finish;
    end
    initial begin #4000000; $display("FAIL: global timeout"); $fatal(1, "timeout"); end
endmodule
