`timescale 1ns/1ps
//============================================================================
// test/uart_host_tb.v -- uart_host_bridge gate  (`make uart-host`)
//
// WHAT IS PROVEN
//   Over REAL 8N1 waveforms (a TB bit-banged serial line at the parameterised
//   divisor, not a shortcut into the parser):
//     U1  a 6-byte 'T' frame produces exactly one start pulse with the right
//         {prompt_tok, start_pos, s_len};
//     U2  garbage bytes before the frame are ignored (resync on 'T');
//     U3  each tok_valid from the (faked) die returns a 3-byte 'K' frame whose
//         payload is the token, decoded from the TX line by an independent
//         TB-side 8N1 receiver;
//     U4  back-to-back tokens all arrive (the TX FIFO's push/pop accounting
//         survives coincident push and pop).
//
// INJECTION (`make uart-host` requires FAILURE)
//   -DINJ_UART_EDGE re-arms the RX bit counter with HALF a period, so every
//   sample lands on a bit EDGE instead of mid-bit.  Framing collapses and U1
//   MUST fail -- proving the gate actually constrains sampling position.
//============================================================================
module uart_host_tb;

    localparam integer CLK_DIV = 16;         // small divisor: fast sim, same logic
    localparam integer BITNS   = CLK_DIV*10; // one bit time in ns (clk = 10 ns)

    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst;

    reg  rx_line;
    wire tx_line;
    wire        start;
    wire [15:0] prompt_tok;
    wire [15:0] start_pos;
    wire [7:0]  s_len;
    reg         tok_valid;
    reg  [15:0] next_tok;

    uart_host_bridge #(.CLK_DIV(CLK_DIV)) dut (
        .clk(clk), .rst(rst),
        .uart_rx(rx_line), .uart_tx(tx_line),
        .start(start), .prompt_tok(prompt_tok), .start_pos(start_pos),
        .s_len(s_len), .tok_valid(tok_valid), .next_tok(next_tok));

    integer errors, tests;
    task chk(input cond, input [8*72-1:0] name);
        begin tests = tests + 1;
              if (!cond) begin errors = errors + 1; $display("FAIL: %0s", name); end end
    endtask

    // ---- TB-side bit-banged 8N1 sender (real waveforms) ---------------------
    task send_byte(input [7:0] b); integer i; begin
        rx_line = 1'b0;  #(BITNS);                 // start bit
        for (i = 0; i < 8; i = i + 1) begin
            rx_line = b[i];  #(BITNS);             // LSB first
        end
        rx_line = 1'b1;  #(BITNS);                 // stop bit
        #(BITNS/2);                                // inter-byte gap
    end endtask

    // ---- start-pulse capture -------------------------------------------------
    integer n_starts;
    reg [15:0] got_tok, got_pos;
    reg [7:0]  got_slen;
    always @(posedge clk) if (!rst && start) begin
        n_starts = n_starts + 1;
        got_tok  = prompt_tok; got_pos = start_pos; got_slen = s_len;
    end

    // ---- TB-side independent 8N1 receiver on the TX line ---------------------
    integer rx_n;
    reg [7:0] rx_bytes [0:31];
    reg [7:0] rb;
    integer bi;
    initial begin
        rx_n = 0;
        forever begin
            @(negedge tx_line);                    // start edge
            #(BITNS/2);                            // to mid start bit
            if (tx_line == 1'b0) begin
                for (bi = 0; bi < 8; bi = bi + 1) begin
                    #(BITNS);
                    rb[bi] = tx_line;              // mid-bit samples
                end
                #(BITNS);                          // stop bit
                rx_bytes[rx_n[4:0]] = rb;
                rx_n = rx_n + 1;
            end
        end
    end

    integer k;
    initial begin
        errors = 0; tests = 0; n_starts = 0;
        rx_line = 1'b1; tok_valid = 1'b0; next_tok = 16'h0;
        rst = 1'b1; repeat (5) @(negedge clk); rst = 1'b0;
        #(BITNS*2);

        // U2: garbage first -- must be ignored
        send_byte(8'h00); send_byte("Z"); send_byte(8'hFF);
        chk(n_starts == 0, "U2: garbage bytes produced no start pulse");

        // U1: a real frame
        send_byte("T");
        send_byte(8'h12); send_byte(8'h34);        // tok  = 0x1234
        send_byte(8'h00); send_byte(8'h07);        // pos  = 0x0007
        send_byte(8'h03);                          // slen = 3
        #(BITNS*2);
        chk(n_starts == 1,        "U1: exactly one start pulse");
        chk(got_tok  == 16'h1234, "U1: prompt_tok decoded");
        chk(got_pos  == 16'h0007, "U1: start_pos decoded");
        chk(got_slen == 8'h03,    "U1: s_len decoded");

        // U3/U4: three back-to-back tokens from the die -> three 'K' frames.
        //   The second token is timed to coincide with a frame boundary, so the
        //   FIFO's coincident push/pop path is actually exercised.
        for (k = 0; k < 3; k = k + 1) begin
            @(negedge clk);
            next_tok  = 16'hBEE0 + k[15:0];
            tok_valid = 1'b1; @(negedge clk); tok_valid = 1'b0;
            #(BITNS*15);                           // ~half a frame apart
        end
        #(BITNS*100);                              // drain: 3 frames = 90 bit
        //   times of TX; the old 40 ended mid-final-frame and lost its last byte

        chk(rx_n == 9, "U3: nine TX bytes (three 'K' frames) decoded");
        for (k = 0; k < 3; k = k + 1) begin
            chk(rx_bytes[3*k]   == "K",              "U3: frame marker");
            chk(rx_bytes[3*k+1] == 8'hBE,            "U3: token high byte");
            chk(rx_bytes[3*k+2] == (8'hE0 + k[7:0]), "U4: token low byte in order");
        end

        if (errors != 0) begin
            $display("FAILED: %0d error(s) across %0d checks", errors, tests);
            $fatal(1, "uart_host_tb had mismatches");
        end
        $display("ALL %0d TESTS PASSED  (uart_host_bridge: 'T' frame -> one exact start pulse over real 8N1 waveforms; three tokens returned as 'K' frames and independently decoded off the TX line)",
                 tests);
        $finish;
    end

    initial begin
        #8000000;
        $display("FAIL: global timeout");
        $fatal(1, "timeout");
    end
endmodule
