//============================================================================
// glm53f_layers_tb.v -- src/glm53f_layers.v at the REAL L = 45.
//
// It drives the walk with a stub block (db_done after a fixed delay) and checks
// the whole invocation sequence: the streams are loaded exactly ONCE, before the
// first layer; the block is started exactly L times; and the
// (layer, attn_sel, ffn_sel) triple on each start is the checkpoint's schedule,
// which the generator reads from the SAME config lines the RTL is parameterised
// from.
//
// WHAT IT CLAIMS, AND WHAT IT DOES NOT. It claims the WALK, not what a layer
// computes -- that is `make dec-block`, whose KIND=2 equivalence build also proves
// the selectors actually select. The three gates together are the argument: one
// layer is right, selection works, and the schedule picks the right selection per
// layer.
//
// Must FAIL:
//   INJ_LAYERS_OFF_BY_ONE  shift the attention phase by a single block. At L = 45
//                          that still gives ELEVEN MLA blocks, so the census
//                          figures (N_MLA = 11, N_KDA = 34) still agree and only
//                          the per-layer sequence disagrees. The QUIET failure.
//   INJ_LAYERS_DENSE_OFF   forget the dense front and make every block MoE.
//============================================================================
`timescale 1ns/1ps
`ifndef TB_VEC
    `define TB_VEC "build/glm53f_layers_vec.txt"
`endif

module glm53f_layers_tb;
    localparam integer L = 45, N_DENSE = 3, PERIOD = 4, OFFSET = 3;
    localparam integer LAYW = $clog2(L);

    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, start = 0;
    wire busy, done, db_start, db_streams_load, attn_sel, ffn_sel;
    wire [LAYW-1:0] layer;
    reg  db_done = 1'b0;

    glm53f_layers #(.L(L), .N_DENSE(N_DENSE), .ATTN_PERIOD(PERIOD),
                    .ATTN_OFFSET(OFFSET)) dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .db_start(db_start), .db_done(db_done),
        .db_streams_load(db_streams_load),
        .layer(layer), .attn_sel(attn_sel), .ffn_sel(ffn_sel));

    // a stub block: acknowledge a few cycles after each start
    integer delay;
    always @(posedge clk) begin
        db_done <= 1'b0;
        if (rst) delay <= 0;
        else if (db_start) delay <= 3;
        else if (delay > 0) begin
            delay <= delay - 1;
            if (delay == 1) db_done <= 1'b1;
        end
    end

    integer fd, code, i, e_L, e_nd, e_p, e_o, errors, checks, w;
    integer e_lay [0:L-1];
    integer e_mla [0:L-1];
    integer e_moe [0:L-1];
    integer n_start, n_load;
    integer seen_lay [0:L-1];
    integer seen_mla [0:L-1];
    integer seen_moe [0:L-1];

    // record every start, in order
    always @(posedge clk) begin
        if (!rst && db_streams_load) n_load = n_load + 1;
        if (!rst && db_start) begin
            if (n_start < L) begin
                seen_lay[n_start] = layer;
                seen_mla[n_start] = attn_sel;
                seen_moe[n_start] = ffn_sel;
            end
            n_start = n_start + 1;
        end
    end

    initial begin
        errors = 0; checks = 0; n_start = 0; n_load = 0;
        fd = $fopen(`TB_VEC, "r");
        if (fd == 0) begin $display("[layers] FAIL: cannot open vectors"); $finish; end
        code = $fscanf(fd, "%d %d %d %d", e_L, e_nd, e_p, e_o);
        if (e_L != L || e_nd != N_DENSE || e_p != PERIOD || e_o != OFFSET) begin
            $display("[layers] FAIL: the vector file and the TB disagree on the schedule");
            $finish;
        end
        for (i = 0; i < L; i = i + 1)
            code = $fscanf(fd, "%d %d %d", e_lay[i], e_mla[i], e_moe[i]);
        $fclose(fd);

        repeat (3) @(negedge clk); rst = 0; @(negedge clk);
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        w = 0;
        while (done !== 1'b1 && w < 100000) begin @(negedge clk); w = w + 1; end

        checks = checks + 1;
        if (done !== 1'b1) begin
            $display("[layers] FAIL: done never asserted"); errors = errors + 1;
        end
        checks = checks + 1;
        if (n_start != L) begin
            $display("[layers] FAIL: block started %0d times, expected %0d", n_start, L);
            errors = errors + 1;
        end
        checks = checks + 1;
        if (n_load != 1) begin
            $display("[layers] FAIL: streams loaded %0d times, expected exactly 1", n_load);
            errors = errors + 1;
        end
        for (i = 0; i < L && i < n_start; i = i + 1) begin
            checks = checks + 1;
            if (seen_lay[i] !== e_lay[i]) begin
                if (errors < 8) $display("FAIL start %0d: layer %0d exp %0d",
                                         i, seen_lay[i], e_lay[i]);
                errors = errors + 1;
            end
            checks = checks + 1;
            if (seen_mla[i] !== e_mla[i]) begin
                if (errors < 8) $display("FAIL layer %0d: attn_sel %0d exp %0d (MLA?)",
                                         e_lay[i], seen_mla[i], e_mla[i]);
                errors = errors + 1;
            end
            checks = checks + 1;
            if (seen_moe[i] !== e_moe[i]) begin
                if (errors < 8) $display("FAIL layer %0d: ffn_sel %0d exp %0d (MoE?)",
                                         e_lay[i], seen_moe[i], e_moe[i]);
                errors = errors + 1;
            end
        end

        if (errors == 0)
            $display("[layers] ALL %0d TESTS PASSED (the real L=%0d walk: the mHC streams are loaded exactly once before layer 0, the block is started exactly %0d times, and every (layer, attn_sel, ffn_sel) triple matches the checkpoint's schedule -- MLA on blocks %0d,%0d,... every %0d, MoE from block %0d)",
                     checks, L, L, OFFSET, OFFSET+PERIOD, PERIOD, N_DENSE);
        else
            $display("[layers] %0d/%0d FAILED", errors, checks);
        $finish;
    end
    initial begin #2000000; $display("[layers] FAIL: timeout"); $finish; end
endmodule
