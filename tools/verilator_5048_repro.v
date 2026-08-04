// Minimal cross-simulator divergence repro, kept as evidence for why
// make l3-e2e runs on iverilog: v5.048 --binary --timing computes vA/vC
// as bf18 where iverilog (and the direct-literal call) give 3ec3.  The
// integer-arg wrapper (vB) agrees across simulators in THIS repro but the
// full l3_e2e_tb still diverged, so the E2E gate pins iverilog only.
`timescale 1ns/1ps
module eminimal2;
    localparam integer MODEL_DIM = 16;
    function automatic integer f_h; input integer seed; begin
        f_h = (seed*2654435761)^(seed<<13)^(seed*40503);
    end endfunction
    function automatic [15:0] gen_bf16; input integer seed;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h = f_h(seed); s = h[3]; e = 8'd124 + {6'b0,h[5:4]}; m = h[12:6];
        gen_bf16 = {s,e,m};
    end endfunction
    function automatic [15:0] fx_em; input integer t; input integer i; begin
        fx_em = gen_bf16(t*MODEL_DIM + i + 7001);
    end endfunction
    reg  [3:0] tok, idx;
    wire [15:0] vA = gen_bf16(32'(tok)*MODEL_DIM + 32'(idx) + 7001);
    wire [15:0] vB = fx_em(tok, idx);
    wire [15:0] vC = gen_bf16({28'd0,tok}*MODEL_DIM + {28'd0,idx} + 7001);
    initial begin
        tok = 4'd7; idx = 4'd0; #1;
        $display("A(cast)=%h B(intarg)=%h C(concat)=%h want=3ec3", vA, vB, vC);
        $finish;
    end
endmodule
