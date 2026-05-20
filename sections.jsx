// All main sections for the Stratos landing page
const { useState: useStateS, useEffect: useEffectS, useRef: useRefS } = React;

/* ---------- Hero with animated agent runner ---------- */
function Hero({ onBook, headline, tagline }) {
  const [activeStep, setActive] = useStateS(2);

  useEffectS(() => {
    const id = setInterval(() => {
      setActive((s) => (s + 1) % 5);
    }, 1800);
    return () => clearInterval(id);
  }, []);

  const steps = [
    { label: "Read invoice from Gmail", meta: "0.4s", icon: <IconMail size={11} /> },
    { label: "Match line items to PO #4129", meta: "1.1s", icon: <IconDoc size={11} /> },
    { label: "Categorize against GL accounts", meta: "0.8s", icon: <IconCard size={11} /> },
    { label: "Post draft entry to NetSuite", meta: "0.3s", icon: <IconLink size={11} /> },
    { label: "Queue for human approval", meta: "—", icon: <IconCheck size={11} /> },
  ];

  return (
    <section className="hero">
      <div className="hero-blob" />
      <div className="container">
        <div className="hero-grid">
          <div className="hero-copy">
            <span className="eyebrow">Agentic AI · for growing teams</span>
            <h1>{headline}</h1>
            <p className="lede">{tagline}</p>
            <div className="hero-cta-row">
              <button className="btn btn-primary" onClick={onBook}>
                Book a call <span className="arrow"><IconArrow size={14} /></span>
              </button>
              <a href="#how" className="btn btn-ghost">
                <IconPlay size={12} /> See agents in action
              </a>
            </div>

            <div className="hero-trust">
              <div className="hero-trust-label">Deployed by 240+ teams</div>
              <div className="hero-trust-logos">
                <span className="logo">Northbeam</span>
                <span className="logo">Verdant Co.</span>
                <span className="logo">Atlas Freight</span>
                <span className="logo">Mira Health</span>
              </div>
            </div>
          </div>

          <div className="hero-mock-wrap">
            <div className="hero-callout">
              "Closed our books 9 days faster."
            </div>
            <div className="hero-mock">
              <div className="hero-mock-inner">
                <div className="agent-head">
                  <div className="agent-title-row">
                    <span className="agent-dot" />
                    <span>Accounts Payable agent</span>
                  </div>
                  <div className="agent-status">running · 14:02</div>
                </div>
                <div className="agent-steps">
                  {steps.map((s, i) => {
                    const state = i < activeStep ? "done" : i === activeStep ? "active" : "";
                    return (
                      <div key={i} className={`agent-step ${state}`}>
                        <span className="agent-step-icon">
                          {state === "done" ? <IconCheck size={10} strokeWidth={3} /> : s.icon}
                        </span>
                        <span>{s.label}</span>
                        <span className="agent-step-meta">{s.meta}</span>
                      </div>
                    );
                  })}
                </div>
                <div style={{
                  marginTop: 14, padding: 12, borderRadius: 12,
                  background: "var(--brand-50)", border: "1px solid var(--brand-100)",
                  display: "flex", alignItems: "center", gap: 10,
                  fontSize: 12.5, color: "var(--brand-700)"
                }}>
                  <IconBolt size={14} />
                  <span><strong>312 invoices</strong> processed today · est. 14 hrs saved</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ---------- How it works — two paired cards ---------- */
function HowItWorks() {
  return (
    <section id="how">
      <div className="container">
        <div className="section-header">
          <span className="eyebrow"><IconSpark size={11} /> How Stratos works</span>
          <h2 style={{ maxWidth: 760 }}>
            Agents that <em style={{ fontStyle: "normal", color: "var(--brand-600)" }}>actually finish the job</em>, with a human in the loop.
          </h2>
        </div>

        <div className="two-card">
          {/* Blue feature */}
          <div className="card-feature card-feature-blue">
            <h3 style={{ color: "white", marginBottom: 16 }}>
              Stratos <span className="pill-inline"><IconSpark size={10} /> works</span> on every workflow
            </h3>
            <p style={{ color: "rgba(255,255,255,0.85)", maxWidth: 360, fontSize: 15.5, lineHeight: 1.55 }}>
              Plug into the tools you already use. Each agent runs a real, multi-step process end-to-end — not a chatbot, not a copilot.
            </p>

            <div className="mock-surface">
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, fontWeight: 600 }}>
                  <span className="agent-dot" />
                  <span>Refund agent · live</span>
                </div>
                <span className="text-mono" style={{ fontSize: 11, color: "var(--ink-500)" }}>00:42</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {[
                  ["Ticket #8821 received", "done"],
                  ["Verified order in Shopify", "done"],
                  ["Refund issued via Stripe", "active"],
                  ["Sent confirmation to customer", ""],
                ].map(([t, st], i) => (
                  <div key={i} style={{
                    display: "flex", gap: 8, alignItems: "center",
                    fontSize: 12.5, padding: "8px 10px", borderRadius: 8,
                    background: st === "active" ? "var(--brand-50)" : st === "done" ? "transparent" : "transparent",
                    color: st === "active" ? "var(--brand-700)" : st === "done" ? "var(--ink-500)" : "var(--ink-400)",
                  }}>
                    <span style={{
                      width: 14, height: 14, borderRadius: "50%",
                      background: st === "done" ? "#10B981" : st === "active" ? "var(--brand-500)" : "var(--ink-200)",
                      display: "grid", placeItems: "center", color: "white"
                    }}>
                      {st === "done" && <IconCheck size={9} strokeWidth={3} />}
                    </span>
                    <span>{t}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Light feature */}
          <div className="card-feature card-feature-light">
            <h3 style={{ marginBottom: 16 }}>
              You stay in <span className="pill-inline light"><IconShield size={10} /> control</span> at every step
            </h3>
            <p style={{ color: "var(--ink-600)", maxWidth: 380, fontSize: 15.5, lineHeight: 1.55 }}>
              Set guardrails, approval thresholds, and audit policies. Watch every action in a live timeline. Pause or roll back anything.
            </p>

            <div className="approval-list">
              <div className="approval-row">
                <div className="ic"><IconCard size={14} /></div>
                <div>
                  <div style={{ fontWeight: 500 }}>Refund · Order #2087</div>
                  <div className="meta">$489.00 · over $250 threshold</div>
                </div>
                <span className="tag tag-pending">Awaiting</span>
              </div>
              <div className="approval-row">
                <div className="ic"><IconMail size={14} /></div>
                <div>
                  <div style={{ fontWeight: 500 }}>Reply · Lead from Hubspot</div>
                  <div className="meta">High-intent · sent automatically</div>
                </div>
                <span className="tag tag-approved">Sent</span>
              </div>
              <div className="approval-row">
                <div className="ic"><IconDoc size={14} /></div>
                <div>
                  <div style={{ fontWeight: 500 }}>Invoice · Vendor mismatch</div>
                  <div className="meta">Flagged for review</div>
                </div>
                <span className="tag tag-flagged">Flagged</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ---------- Metrics big numbers ---------- */
function Metrics() {
  const items = [
    { value: "42", unit: "%", label: "Average reduction in operating costs across 90 days" },
    { value: "8.4", unit: "×", label: "Throughput on repetitive tasks vs. manual handling" },
    { value: "$1.2", unit: "M", label: "Median annualized savings per deployed agent" },
    { value: "3", unit: "wks", label: "From kickoff to production-grade agent in your stack" },
  ];
  return (
    <section id="results">
      <div className="container">
        <div className="section-header">
          <span className="eyebrow"><IconDollar size={11} /> The numbers</span>
          <h2 style={{ maxWidth: 720 }}>What growing teams gain in the first quarter.</h2>
          <p className="lede">Aggregated across 60+ Stratos deployments at companies between 20 and 500 employees.</p>
        </div>

        <div className="metrics-row">
          {items.map((m, i) => (
            <div key={i} className="metric">
              <div className="metric-value">{m.value}<span className="unit">{m.unit}</span></div>
              <div className="metric-label">{m.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ---------- Case studies — 3 cards ---------- */
function CaseStudies() {
  const cases = [
    {
      tag: "Case study", art: "case-art-1",
      metric: "−63% cost",
      title: "Northbeam cut AP processing costs by 63%",
      desc: "Northbeam's finance team used Stratos to fully automate accounts payable for 4,200 invoices/month — closing books 9 days faster.",
      industry: "Logistics · 180 employees",
    },
    {
      tag: "Case study", art: "case-art-2",
      metric: "3.8× output",
      title: "Verdant tripled support throughput with one agent",
      desc: "A single support agent now handles 71% of inbound tickets end-to-end — refunds, returns, and product questions — without escalation.",
      industry: "Consumer goods · 45 employees",
    },
    {
      tag: "Case study", art: "case-art-3",
      metric: "$840K saved",
      title: "Atlas Freight saved $840K in dispatch operations",
      desc: "A scheduling agent reconciles driver hours, fuel, and routes nightly — replacing 11 hours of daily spreadsheet work.",
      industry: "Transportation · 90 employees",
    },
  ];

  return (
    <section id="customers">
      <div className="container">
        <div className="section-header">
          <span className="eyebrow"><IconNet size={11} /> Case studies</span>
          <h2 style={{ maxWidth: 760 }}>SMEs that shipped agents in weeks, not quarters.</h2>
        </div>

        <div className="case-grid">
          {cases.map((c, i) => (
            <article key={i} className="case-card">
              <div className={`case-art ${c.art}`}>
                <span className="case-tag pill" style={{ background: "rgba(255,255,255,0.92)", borderColor: "transparent" }}>
                  {c.tag}
                </span>
              </div>
              <div className="case-body">
                <div className="case-metric">{c.metric}</div>
                <div className="case-title">{c.title}</div>
                <div className="case-desc">{c.desc}</div>
                <div style={{ fontSize: 12.5, color: "var(--ink-500)", marginTop: 4 }}>{c.industry}</div>
                <div className="case-cta">Read case study <IconArrow size={14} /></div>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ---------- Capabilities — 3-up ---------- */
function Capabilities() {
  return (
    <section id="capabilities">
      <div className="container">
        <div className="section-header centered">
          <span className="eyebrow"><IconShield size={11} /> Built for enterprise teams</span>
          <h2>Trusted with the work that matters.</h2>
          <p className="lede">Production-grade infrastructure with the controls your security and finance teams will sign off on.</p>
        </div>

        <div className="cap-grid">
          <CapCard
            title="Works with your stack"
            desc="Native connectors to NetSuite, QuickBooks, Salesforce, HubSpot, Zendesk, Shopify, Slack, and 60+ more. No middleware to wrangle."
            visual={<IntegrationsVisual />}
          />
          <CapCard
            title="Reasoning you can audit"
            desc="Every decision shows its work — source documents, policy applied, and confidence score. Replay any run, step by step."
            visual={<ReasoningVisual />}
          />
          <CapCard
            title="SOC 2 & SSO ready"
            desc="Your data stays in your tenant. Encrypted at rest and in transit. Role-based access, scoped credentials, full activity log."
            visual={<SecurityVisual />}
          />
        </div>
      </div>
    </section>
  );
}

function CapCard({ title, desc, visual }) {
  return (
    <div className="cap-card">
      <div className="cap-visual">{visual}</div>
      <div className="cap-body">
        <div className="cap-title">{title}</div>
        <div className="cap-desc">{desc}</div>
      </div>
    </div>
  );
}

function IntegrationsVisual() {
  const tools = [
    { name: "NetSuite", color: "#1B5E20", letter: "N" },
    { name: "Salesforce", color: "#0176D3", letter: "S" },
    { name: "Slack", color: "#4A154B", letter: "#" },
    { name: "Shopify", color: "#5D8C40", letter: "S" },
    { name: "Zendesk", color: "#03363D", letter: "Z" },
    { name: "HubSpot", color: "#FF7A59", letter: "H" },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, height: "100%", alignContent: "center" }}>
      {tools.map((t) => (
        <div key={t.name} style={{
          background: "var(--ink-50)",
          border: "1px solid var(--ink-200)",
          borderRadius: 10,
          padding: "10px 8px",
          display: "flex", alignItems: "center", gap: 8,
          fontSize: 12, fontWeight: 500
        }}>
          <span style={{
            width: 22, height: 22, borderRadius: 6,
            background: t.color, color: "white",
            display: "grid", placeItems: "center",
            fontSize: 11, fontWeight: 600
          }}>{t.letter}</span>
          {t.name}
        </div>
      ))}
    </div>
  );
}

function ReasoningVisual() {
  return (
    <div style={{ fontFamily: "var(--font-mono)", fontSize: 11.5, lineHeight: 1.7, color: "var(--ink-700)" }}>
      <div style={{ color: "var(--ink-400)" }}>// trace.run.42a8</div>
      <div><span style={{ color: "var(--brand-600)" }}>match</span>(invoice, po) → <span style={{ color: "#10B981" }}>0.97</span></div>
      <div><span style={{ color: "var(--brand-600)" }}>policy</span>("3-way match") → <span style={{ color: "#10B981" }}>pass</span></div>
      <div><span style={{ color: "var(--brand-600)" }}>policy</span>("under $5k") → <span style={{ color: "#10B981" }}>pass</span></div>
      <div><span style={{ color: "var(--brand-600)" }}>route</span> → auto-approve</div>
      <div style={{ marginTop: 8, padding: 8, background: "var(--brand-50)", borderRadius: 6, color: "var(--brand-700)" }}>
        ✓ Action committed · audit_id: 8f3e
      </div>
    </div>
  );
}

function SecurityVisual() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {[
        ["SOC 2 Type II", "Audited Apr 2026"],
        ["HIPAA-ready", "BAA available"],
        ["SSO / SAML", "Okta, Google, Azure AD"],
        ["Data residency", "US, EU, AU regions"],
      ].map(([k, v], i) => (
        <div key={i} style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "8px 10px", borderRadius: 8,
          background: "var(--ink-50)", border: "1px solid var(--ink-200)"
        }}>
          <span style={{
            width: 18, height: 18, borderRadius: 4,
            background: "var(--brand-500)", color: "white",
            display: "grid", placeItems: "center"
          }}>
            <IconCheck size={11} strokeWidth={3} />
          </span>
          <div style={{ fontSize: 12.5, fontWeight: 500 }}>{k}</div>
          <div style={{ fontSize: 11.5, color: "var(--ink-500)", marginLeft: "auto" }}>{v}</div>
        </div>
      ))}
    </div>
  );
}

/* ---------- FAQ ---------- */
function FAQ() {
  const faqs = [
    {
      q: "How are Stratos agents different from a chatbot or copilot?",
      a: "Agents are autonomous workflows. Instead of suggesting an answer, they take action across your tools — reading data, making decisions against your policies, and committing changes — with you supervising rather than driving."
    },
    {
      q: "How long does a deployment actually take?",
      a: "Most teams ship their first production agent in 3 weeks. We run a discovery week, build a sandboxed agent against your real data in week two, and graduate to live traffic with guardrails in week three."
    },
    {
      q: "What does pricing look like for an SME?",
      a: "We price per agent, per month — typically between $1,500 and $4,500 depending on workflow complexity and volume. There's no per-seat charge and no usage cliff. Most customers pay back the contract within the first quarter."
    },
    {
      q: "How do you handle our data and security?",
      a: "We're SOC 2 Type II certified and offer SSO/SAML, scoped credentials, encryption at rest and in transit, and US/EU/AU data residency. Customer data is isolated per tenant and never used to train shared models."
    },
    {
      q: "What if an agent makes a mistake?",
      a: "Every action is logged with reasoning, can be approved or paused before execution, and can be rolled back. You set thresholds — anything above them queues for a human. You're always in control."
    },
    {
      q: "Do we need engineers to maintain it?",
      a: "No. Agents are configured in plain English, monitored from a dashboard, and adjusted by your ops team. Our solutions engineers stay engaged for the life of the contract for tuning and new workflows."
    },
  ];

  const [open, setOpen] = useStateS(0);

  return (
    <section id="faq">
      <div className="container-narrow">
        <div className="section-header">
          <span className="eyebrow"><IconNet size={11} /> FAQ</span>
          <h2>Questions, answered.</h2>
        </div>

        <div className="faq">
          {faqs.map((f, i) => (
            <div key={i} className={`faq-row ${open === i ? "open" : ""}`}>
              <button className="faq-q" onClick={() => setOpen(open === i ? -1 : i)}>
                <span>{f.q}</span>
                <span className="chevron"><IconChevron size={18} /></span>
              </button>
              <div className="faq-a">
                <div className="faq-a-inner">{f.a}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ---------- Final CTA — book a call card ---------- */
function BookSection({ onBook, rep }) {
  return (
    <section className="book-section">
      <div className="container">
        <div className="book-card">
          <div className="book-grid">
            <div>
              <span className="eyebrow" style={{
                background: "rgba(255,255,255,0.15)",
                color: "white",
                borderColor: "rgba(255,255,255,0.2)"
              }}>
                <span style={{ background: "white", boxShadow: "0 0 0 4px rgba(255,255,255,0.2)" }} />
                Free · No deck, no sales pitch
              </span>
              <h2 style={{ marginTop: 20 }}>
                30 minutes. One agent identified. Real numbers.
              </h2>
              <p className="lede" style={{ marginTop: 20, marginBottom: 28 }}>
                Bring an open workflow that's costing you time. We'll map what an agent would own, what it would save, and how fast it ships. If there's no clear win, we'll say so.
              </p>
              <button className="book-btn" onClick={onBook}>
                <IconCalendar size={16} />
                <span>Book with {rep.name.split(" ")[0]}</span>
                <span className="arrow"><IconArrow size={14} /></span>
              </button>
            </div>

            <div className="book-rep">
              <div className="rep-head">
                <div className="rep-avatar">{rep.initials}</div>
                <div>
                  <div className="rep-name">{rep.name}</div>
                  <div className="rep-role">{rep.role}</div>
                </div>
              </div>
              <div className="rep-availability">
                <div className="rep-availability-row">
                  <span className="check"><IconCheck size={10} strokeWidth={3} /></span>
                  <span>Replies within 2 business hours</span>
                </div>
                <div className="rep-availability-row">
                  <span className="check"><IconClock size={10} /></span>
                  <span>Next opening: <strong style={{ color: "white" }}>tomorrow, 10:30 AM</strong></span>
                </div>
                <div className="rep-availability-row">
                  <span className="check"><IconVideo size={10} /></span>
                  <span>Google Meet · automatic invite</span>
                </div>
              </div>
              <div style={{
                paddingTop: 14,
                borderTop: "1px solid rgba(255,255,255,0.15)",
                fontSize: 13,
                color: "rgba(255,255,255,0.8)",
                fontStyle: "italic"
              }}>
                "Show me one workflow that drains a person's day. I'll show you what an agent does with it."
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ---------- Footer ---------- */
function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-grid">
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
              <span style={{
                width: 28, height: 28, borderRadius: 8,
                background: "var(--ink-900)", color: "white",
                display: "grid", placeItems: "center"
              }}>
                <IconLogo size={16} />
              </span>
              <span style={{ fontWeight: 600, letterSpacing: "-0.02em" }}>Stratos</span>
            </div>
            <p style={{ fontSize: 13.5, color: "var(--ink-600)", maxWidth: 280, lineHeight: 1.6 }}>
              Production-grade agentic AI for finance, operations, and customer teams at growing companies.
            </p>
          </div>
          <div className="footer-col">
            <h4>Product</h4>
            <ul>
              <li><a href="#how">How it works</a></li>
              <li><a href="#capabilities">Capabilities</a></li>
              <li><a href="#customers">Case studies</a></li>
              <li><a href="#">Integrations</a></li>
            </ul>
          </div>
          <div className="footer-col">
            <h4>Company</h4>
            <ul>
              <li><a href="#">About</a></li>
              <li><a href="#">Careers</a></li>
              <li><a href="#">Blog</a></li>
              <li><a href="#">Press</a></li>
            </ul>
          </div>
          <div className="footer-col">
            <h4>Resources</h4>
            <ul>
              <li><a href="#">Docs</a></li>
              <li><a href="#">Security</a></li>
              <li><a href="#">SOC 2 report</a></li>
              <li><a href="#">Contact</a></li>
            </ul>
          </div>
        </div>
        <div className="footer-bottom">
          <span>© 2026 Stratos AI, Inc.</span>
          <span>San Francisco · Remote-first</span>
        </div>
      </div>
    </footer>
  );
}

Object.assign(window, { Hero, HowItWorks, Metrics, CaseStudies, Capabilities, FAQ, BookSection, Footer });
