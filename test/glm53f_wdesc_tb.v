//============================================================================
// glm53f_wdesc_tb.v -- src/glm53f_wdesc.v against tools/glm53f_wdesc_gen.py.
//
// Every (kind, layer, expert) triple in the table is resolved and compared
// EXACTLY: base, klen, nsblk and type. The table itself arrives as a GENERATED
// include (`build/glm53f_wdesc_params.vh`) rather than hand-copied literals, so
// the TB and the golden cannot drift apart silently.
//
// The generator asserts the corpus actually exercises what the legs below claim:
// a kind bumped on SOME layers and not others (so both the default and the
// exception are observed), a kind with a non-zero expert stride, more than one
// expert and more than one layer. Without those the injections are decorative.
//
// Must FAIL:
//   INJ_WDESC_NO_EXC   ignore the UD bumps. blk.{11,12,44}.ffn_down_exps then
//                      streams Q5_K geometry over Q6_K bytes -- same widths,
//                      wrong decode, and nothing anywhere reports an error.
//   INJ_WDESC_NO_ESTR  drop the expert stride. Every expert then reads expert 0's
//                      bytes: a well-formed model with one expert 288 times.
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/glm53f_wdesc_vec.txt"
`endif
`ifndef TB_PARAMS
    `define TB_PARAMS "build/glm53f_wdesc_params.vh"
`endif

module glm53f_wdesc_tb;
    `include `TB_PARAMS
    localparam integer AW = 32, KW = 16, SBW = 8;
    localparam integer KIDW = (G_NKIND  <= 1) ? 1 : $clog2(G_NKIND);
    localparam integer LIDW = (G_NLAYER <= 1) ? 1 : $clog2(G_NLAYER);
    localparam integer EIDW = (G_NEXP   <= 1) ? 1 : $clog2(G_NEXP);

    reg  [KIDW-1:0] kind;
    reg  [LIDW-1:0] layer;
    reg  [EIDW-1:0] expert;
    wire [AW-1:0]   d_base;
    wire [KW-1:0]   d_klen;
    wire [SBW-1:0]  d_nsblk;
    wire [2:0]      d_wtype;

    glm53f_wdesc #(.NKIND(G_NKIND), .NLAYER(G_NLAYER), .NEXC(G_NEXC), .NEXP(G_NEXP),
                   .ADDR_W(AW), .KW(KW), .SBW(SBW),
                   .K_BASE(G_K_BASE), .K_LSTR(G_K_LSTR), .K_ESTR(G_K_ESTR),
                   .K_KLEN(G_K_KLEN), .K_NSBLK(G_K_NSBLK), .K_WTYPE(G_K_WTYPE),
                   .X_KIND(G_X_KIND), .X_LAYER(G_X_LAYER), .X_WTYPE(G_X_WTYPE),
                   .X_VALID(G_X_VALID)) dut (
        .kind(kind), .layer(layer), .expert(expert),
        .desc_base(d_base), .desc_klen(d_klen), .desc_nsblk(d_nsblk),
        .desc_wtype(d_wtype));

    integer fd, code, i, ntrip, nk, nl, ne, nx, errors, checks;
    integer e_k, e_l, e_e, e_wt;
    reg [AW-1:0]  e_base;
    reg [KW-1:0]  e_klen;
    reg [SBW-1:0] e_nsb;
    reg [255:0]   nm;
    integer       w_dummy;
    reg [1023:0]  hexv;

    initial begin
        errors = 0; checks = 0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[wdesc] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d %d", nk, nl, ne, nx);
        if (nk != G_NKIND || nl != G_NLAYER || ne != G_NEXP || nx != G_NEXC) begin
            $display("[wdesc] FAIL: the vector file and the generated params disagree");
            $finish;
        end
        // skip the parameter lines: they are already in via the include, and
        // re-parsing them here would be a second source of truth
        for (i = 0; i < 10; i = i + 1) code = $fscanf(fd, "%s %d %h", nm, w_dummy, hexv);

        ntrip = nk * nl * ne;
        for (i = 0; i < ntrip; i = i + 1) begin
            code = $fscanf(fd, "%d %d %d %h %h %h %d",
                           e_k, e_l, e_e, e_base, e_klen, e_nsb, e_wt);
            kind = e_k[KIDW-1:0]; layer = e_l[LIDW-1:0]; expert = e_e[EIDW-1:0];
            #1;
            checks = checks + 1;
            if (d_base !== e_base) begin
                if (errors < 8) $display("FAIL kind%0d layer%0d exp%0d: base %h exp %h",
                                         e_k, e_l, e_e, d_base, e_base);
                errors = errors + 1;
            end
            checks = checks + 1;
            if (d_klen !== e_klen || d_nsblk !== e_nsb) begin
                if (errors < 8) $display("FAIL kind%0d: klen/nsblk %h/%h exp %h/%h",
                                         e_k, d_klen, d_nsblk, e_klen, e_nsb);
                errors = errors + 1;
            end
            checks = checks + 1;
            if (d_wtype !== e_wt[2:0]) begin
                if (errors < 8) $display("FAIL kind%0d layer%0d: wtype %0d exp %0d",
                                         e_k, e_l, d_wtype, e_wt);
                errors = errors + 1;
            end
        end
        $fclose(fd);
        if (errors == 0)
            $display("[wdesc] ALL %0d TESTS PASSED (%0d (kind,layer,expert) triples: base composes the layer and expert strides, klen/nsblk come from the kind, and the UD-bump exceptions override the kind's type on their own layers and only there)",
                     checks, ntrip);
        else
            $display("[wdesc] %0d/%0d FAILED", errors, checks);
        $finish;
    end
endmodule
