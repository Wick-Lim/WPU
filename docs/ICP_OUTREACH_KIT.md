# ICP outreach kit — landing the first legal design partner

*Turns the [`ICP.md`](ICP.md) beachhead (confidentiality-bound legal) into a copy-paste BD motion whose
single goal is **Gate 3**: one signed design-partner LOI. Companion to the customer's-side of the
[`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) / [`USBC_PRODUCT_PLAN.md`](USBC_PRODUCT_PLAN.md).*

> **Honest stance — read first.** There is **no shippable box yet** (P1.1 fidelity + board bring-up/demo
> are open gates; the **FPGA fit itself is measured** — routed on XCKU3P at 46.5 MHz,
> [`../fpga/`](../fpga/README.md)). So this is **not** a sales motion for a finished product. It is a **design-partner /
> problem-validation** motion: we're building it, and we want a small number of legal firms that
> *actually have the can't-cloud constraint* to (a) validate the pain, (b) shape the product, and (c)
> sign a **non-binding LOI to pilot when the demo lands**. Never imply a product you can ship today —
> the credibility is the whole asset. Sell the **vision + the design-partner relationship**, backed by
> the measured evidence (verified RTL, formally-proven memory controllers, and Q4_K compute kernels
> bit-exact to the ggml reference), not a demo you don't have.

---

## 0. This kit runs for **any** can't-cloud wedge (legal is the worked example)

The ICP is a **constraint** (frontier quality that runs **offline / air-gapped** — on data that can't
touch a cloud, and in places a connection can't reach), not a sector ([`ICP.md`](ICP.md)). The sharpest
form of that constraint is binary and testable: *does it still work with the ethernet cable unplugged?*
That bar categorically excludes **every** cloud option — including "secured cloud" (in-VPC / tenant
deployment, zero-retention APIs, TEE / confidential-computing enclaves), which all need a live
connection and fail it. This kit is written out for **legal** because you close by instantiating *one*
concrete motion — but the structure (pain script, discovery flow, LOI, objection logic) is identical
for every wedge. To run a second wedge in parallel (recommended: **quant/prop-trading**, the co-lead),
swap only the **buyer persona + vocabulary**:

| | Legal (worked example) | Quant / prop-trading (parallel swap) | Defense / OT / field-edge (offline swap) |
|---|---|---|---|
| Champion | Director of Innovation / KM / Practice Tech | Head of Research Eng / CTO / a PM running an internal LLM | Program / mission-systems lead; OT or ICS architect |
| Sponsor | practice-group partner | a portfolio manager / desk head | program office / ops commander / plant manager |
| Approver | CISO / GC / risk | CISO / Head of Infosec (already FPGA/colo-savvy) | accreditation / ATO authority (ISSM); OT security lead |
| The pain line | "privileged & client-confidential matters can't touch cloud AI" | "your signals, positions, and research can't leak to any cloud" | "the network is air-gapped by mandate — there is no cloud path, and the site may have no link at all" |
| Cold-email subject | "on-prem frontier AI for [Firm] — nothing leaves your network" | "frontier AI on your research — nothing leaves your infra" | "frontier AI that runs air-gapped — works with the ethernet unplugged" |
| Reference value | sells the next law firm | sells the next fund | sells the next program / site |

Everything below reads as legal; mentally substitute a column above to fire the quant (or
defense/OT/edge) motion at the same time. **Test legal + quant in parallel for ~2 weeks; let
speed-to-LOI pick the beachhead.** Defense / OT / field-edge is a first-class, high-value third where
air-gap is *mandatory* — but procurement runs slower, so work it as an expansion track, not the 2-week
speed test.

> **The quant wedge is written out in full** — its own personas, cold emails, discovery script, and
> quant-specific objection handling ("we already have GPUs", "is it fast enough", "does it give alpha") —
> in [`ICP_OUTREACH_KIT_QUANT.md`](ICP_OUTREACH_KIT_QUANT.md). Use this doc for legal, that one for quant,
> and run both in parallel.

## 1. Target — the account and the three people

**Account fit (all three, or don't bother):**
- **Frontier quality genuinely needed** — the work is hard (M&A, litigation, complex drafting), not
  form-filling a 70B model already handles.
- **A real can't-cloud constraint** — a firm/practice with clients who contractually forbid cloud AI,
  or a GC/risk posture that already blocked ChatGPT/Copilot for client matters. The bar is *runs
  disconnected*: "secured cloud" (in-VPC / zero-retention / TEE) doesn't clear it, because it still
  needs a connection.
- **AI-forward enough to move** — they've *tried* private AI (Harvey, Copilot, an internal pilot) and
  hit the data-boundary wall. A firm that's never thought about AI is too early.

**Where to find them:** AmLaw 100–200 innovation teams; firms publicly piloting legal AI; legal-tech
conference speaker lists (ILTACON, Legalweek); "Director of Innovation / Knowledge Management /
Practice Technology" on LinkedIn; boutiques in IP / M&A / regulated-industry practices.

**The three personas (you need all three to close a design partner):**

| Persona | Title patterns | What they care about | Their role |
|---|---|---|---|
| **Champion** | Director of Innovation / KM / Practice Technology; "Legal AI" lead | Being first to a defensible AI edge without a data-breach headline | **Entry point** — start here |
| **Sponsor** | A practice-group partner (litigation / M&A / IP) | Leverage on the confidential work they can't currently use AI on | **Pain owner** — makes it real |
| **Approver** | CISO / Head of IT Security / GC / risk | "Nothing leaves the building," auditability | **Unblocker** — loves "provably no path out — it works with the ethernet unplugged" |

---

## 2. The message (positioning, translated for legal — not the investor story)

Lead with *their* pain and the *capability* it unlocks, not the quantization format or the RTL. Three
lines you can say in an elevator:

> **"Your attorneys are locked out of the best AI on exactly their most valuable work — the privileged,
> client-confidential matters — because every frontier model lives in the cloud. We're building an
> appliance that runs a full frontier-scale model **completely offline**: it works with the ethernet
> cable unplugged. That's the whole promise — your data *can't* leave because there is no path out, and
> the audit is literally 'does it still work disconnected?' It does. So the work you can't put in the
> cloud finally gets frontier AI — and you own the box outright."**

Note the frame: **lead with the capability** (frontier AI on the matters you're locked out of, owned
outright), and let non-egress be the *proof* — the unplugged-ethernet test — not a defensive "avoid the
breach" pitch (that just competes with "don't use AI at all").

**Don't oversell "offline" alone** — a 70B laptop model is offline too. The moat is the *combination*:
offline + a **full frontier (753B)** model + an **appliance / per-seat** price. (A laptop model fails on
quality; an 8×H100 rack fails on price and form factor; "secured cloud" fails the unplugged test.) Be
honest on provisioning, too: the ~467 GB of weights (UD-Q4_K_XL) load **once** (itself doable offline,
in a secure facility), and model updates are a **physical re-provision** — expected and fine for air-gap
buyers, but say so.

Keep two proof-points in your back pocket for the technical/security persona (these are *measured in
gated simulation / formally proven*, not marketing): the core compute kernels are **bit-exact to the
ggml Q4_K reference** (the open quantization the model ships in), and the memory controllers are
**formally verified** (BMC, several lifted to unbounded k-induction) — i.e. "provably offline **and** the
datapath provably matches the reference." Be honest about scope: this is verified at the **kernel /
controller** level; **whole-model fidelity against a real checkpoint is still an open gate (P1.1)**, so
don't imply the assembled model has been validated end-to-end. Don't lead with any of this; deploy it
when the approver asks "how do I trust it."

---

## 3. Cold outreach

### Email A — to the Champion (primary)
> **Subject: frontier AI for [Firm] that runs offline — nothing leaves your network**
>
> Hi [Name],
>
> I'm building WPU — an appliance that runs a full **frontier-scale** AI model **completely offline**:
> it works with the ethernet cable unplugged, so no data can leave the building — no cloud API, no
> per-use fees. It's aimed squarely at the work your attorneys *can't* put into ChatGPT or Copilot
> today — privileged and client-confidential matters.
>
> We're selecting a small number of **design partners** in legal to shape it before release. No cost and
> no commitment beyond a few conversations — you'd get early access and direct input into what we build;
> we'd get to build it for a firm that actually lives with the constraint.
>
> Worth a 20-minute call? I mainly want to understand how [Firm] handles AI on confidential matters
> today, and whether *provably local* would change that.
>
> [Name] · [link to a 1-page brief]

### Email B — to a Sponsor partner (pain-led, if you have a warm angle)
> **Subject: the AI you can't use on your [practice] matters**
>
> Hi [Name] — quick one. On your most confidential [M&A / litigation / IP] matters, is your team
> effectively locked out of the good AI tools because the data can't go to a cloud? I'm building an
> on-prem appliance that fixes exactly that — a frontier model that runs entirely inside the firm.
> We're picking a few design-partner firms in legal. Would 20 minutes be worth it to pressure-test
> whether this is real for your practice?

### LinkedIn DM (short)
> Hi [Name] — I'm building an on-prem appliance that runs a full frontier AI model **inside** a firm's
> network (nothing leaves the building), for the privileged/confidential work that's off-limits to cloud
> AI. Selecting a few legal design partners. Open to a 20-min call to see if it's relevant to [Firm]?

### Warm-intro ask (send to a mutual connection)
> Would you introduce me to [Name] at [Firm]? I'm building on-prem frontier AI for legal — runs entirely
> inside the firm's network for privileged matters — and I'm looking for a design partner who actually
> has the can't-cloud constraint. One line from you would mean a lot. Happy to send a blurb you can
> forward.

### Follow-ups (space 4–6 days; stop after two)
- **Nudge 1:** *"Bumping this up — one line is enough: is 'frontier AI that never leaves your network'
  a real need for [Firm]'s confidential matters, or not a fit? Either answer helps me."*
- **Nudge 2 (value, then close the loop):** *"Last note — [one concrete proof: 'the Q4_K compute path
  now reproduces the ggml reference kernels bit-for-bit in simulation, and the memory controllers are
  formally verified']. If the timing's wrong I'll stop here; if it's worth a look, grab any 20 min:
  [link]."*

---

## 4. Discovery call — script (20–30 min)

**Frame (first 60s):** *"Thanks for the time. I'm not selling anything — there's no box to buy yet.
I'm building this and I want to learn from firms who actually have the constraint, and find one or two
design partners. So mostly I want to ask about how you work today. That ok?"* → disarms, earns candor.

**Qualify — ask, then listen (don't pitch yet):**
1. "Where do your attorneys use AI today — and where are they *not allowed* to?"
2. "On privileged / client-confidential matters, what's the actual rule — firm policy, client contract,
   GC call? How binding?" *(This is the whole thesis — probe hard.)*
3. "Have you piloted anything (Harvey, Copilot, internal)? Where did it stop?"
4. "If there were a tool that ran a frontier model **entirely inside your network**, what would it
   unlock — and who'd have to bless it (IT/security, GC, the partners)?"
5. "How do you buy tools like this — per seat? Who signs? What's a normal per-seat number here?"

**Green flags** (pursue): a named client contract or GC policy that blocks cloud AI · a partner who
already asked for this · a security lead who says "on-prem changes everything" · existing per-seat tool
budget. **Red flags** (deprioritize): "we just use ChatGPT, it's fine" · no confidential-data constraint
· "call us in two years" · wants a finished product now.

**Then — and only then — the vision (2 min):** the one-box/one-seat appliance, full 753B locally and
**fully offline / air-gapped** (it works with the ethernet unplugged — nothing leaves because there's no
path out) — **provably local and auditable** (the non-egress is the unplugged test; the compute path is
verified against the ggml Q4_K reference, with whole-model fidelity the open gate). Show the 1-page
brief / evidence. Be explicit about stage: *"here's what's proven today, here's the two gates left, and
here's why I want a design partner **now** — so we build the thing your firm would actually deploy."*

**The ask (close):** *"Would [Firm] be one of our design partners? Concretely: a few working sessions
to shape it, and a short, non-binding letter that says you'd pilot it when the on-prem demo is ready.
No cost. In return you get first access and you shape the product around your constraints."*

### Objection handling (legal-specific)

| They say | You say (honest) |
|---|---|
| "You don't even have a product." | "Correct — that's *why* I want you now. Design partners shape it and get first access; the hard tech risk is unusually retired (compute kernels bit-exact to the ggml Q4_K reference in sim, memory controllers formally verified, and the whole design placed-and-routed on a real FPGA at a measured 46.5 MHz). The two open gates are whole-model fidelity and board bring-up — exactly what a design partner helps us close. I'm asking for input + an LOI, not a purchase." |
| "How is this different from Harvey / Copilot?" | "Those still send your text to a cloud. This runs the whole model **offline, inside your network** — it works with the ethernet unplugged, so nothing can leave. That's the exact line your confidential matters can't cross." |
| "Our GC will never approve cloud AI." | "Right — that's the point. There's no cloud, and no connection: the box runs with the ethernet unplugged. Your security team can audit that nothing egresses — because there's no path out — while the compute path itself is verified against the open Q4_K reference kernels." |
| "Why not a private cloud — in-VPC / tenant deployment, a zero-retention API, or a TEE / confidential-computing enclave?" | "Those are all still the cloud: they need a live connection, so they can't pass the test your constraint actually sets — *does it work with the ethernet unplugged?* A VPC, a zero-retention promise, and a TEE each fail it (no link, no service). This is the only option that runs disconnected, which is what ends the 'secured cloud' debate for good." |
| "Is a local model good enough?" | "The small models that fit a laptop aren't — that's the gap. This runs the *full* 753B frontier model locally and offline, not a shrunk one. Offline alone is table-stakes (a laptop model is offline too); the moat is the combination — offline **and** full-frontier, at a per-seat price." |
| "What will it cost?" | "TBD until board quotes close — the FPGA fit is already measured (a KU3P-class part), which sets the silicon class behind the BOM; we're targeting a **per-seat** price, in the range of the premium tools you already buy, not a datacenter build. Design partners get preferential terms." |
| "When is it real?" | "Two gates: real-model fidelity (a GPU run) and a running-board demo (the FPGA fit/Fmax is already measured — what's left is bring-up on a physical board). I'll show you the demo the moment it exists — the LOI just means you're first in line, non-binding." |

---

## 5. The design-partner offer

Make it a clear, low-friction, mutual deal:

**They give:** 3–4 working sessions (pain, workflow, security requirements) · a signed non-binding LOI ·
a named security/IT contact to define the "works-unplugged / nothing-leaves-the-network" audit ·
willingness to pilot when the demo lands. **They get:** direct product influence · first access / priority allocation ·
design-partner pricing · a demonstrable head-start on defensible, compliant AI. **No cost** at this
stage. **Timeline:** LOI now → design sessions over the next weeks → pilot when Gate 2 (demo) closes.

---

## 6. LOI template (non-binding — keep it one page)

> **Letter of Intent — WPU legal design partnership**
>
> This non-binding letter records the mutual intent of **[Firm]** and **WPU** to collaborate on an
> on-premise, **offline / air-gapped**, single-user frontier-AI appliance for confidential legal work.
>
> **[Firm] intends to:** (1) participate as a design partner through a small number of working sessions;
> (2) provide input on requirements, workflow, and the "no data leaves the network" security posture;
> (3) **evaluate a pilot** of the appliance on [Firm]'s premises once a working demonstration is
> available, subject to a separate pilot agreement and [Firm]'s security review.
>
> **WPU intends to:** (1) give [Firm] early/priority access and design-partner terms; (2) build toward
> the on-prem, provably-local guarantees discussed; (3) keep [Firm]'s shared information confidential.
>
> This letter is **non-binding**, creates no purchase obligation or exclusivity, and either party may
> end the collaboration at any time. It signals genuine intent to work together toward a pilot.
>
> [Firm] ______________________  ·  WPU ______________________  ·  Date __________

*(Have a lawyer review before sending — you're pitching lawyers; a clean, correct LOI is itself a
credibility signal. Keep it non-binding to lower the bar to signature.)*

---

## 7. Cadence & the one metric

- **Target list:** ~30–50 qualified accounts (fit criteria in §1). Quality over volume.
- **Outreach:** ~5–8 personalized touches/week (Champion first; Sponsor/warm-intro where available).
- **Goal:** book ~5–8 discovery calls → **1 signed design-partner LOI.** That single LOI, plus the
  measured FPGA demo, is the pre-seed package ([`ICP.md`](ICP.md) §"investable levers").
- **Disqualify fast.** A "no constraint / call us in two years" is a *gift* — it saves weeks. Log it and
  move on.
