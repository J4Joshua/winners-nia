// Main App — Nav, composition, BookingModal state, Tweaks
const { useState: useStateA, useEffect: useEffectA } = React;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent": "#4F6BFF",
  "headline": "The teammate that runs your back office while you sleep.",
  "tagline": "Stratos deploys task-specific AI agents that own end-to-end workflows in finance, operations, and customer support — proven to save growing teams 30%+ on operating costs in 90 days.",
  "rep": "Maya"
}/*EDITMODE-END*/;

const REPS = {
  "Maya":   { name: "Maya Okafor",  role: "Solutions Engineer · Stratos", initials: "MO" },
  "David":  { name: "David Chen",   role: "Head of Customer Success",     initials: "DC" },
  "Priya":  { name: "Priya Vasan",  role: "Founding Solutions Engineer",  initials: "PV" },
};

function Nav({ onBook }) {
  const [scrolled, setScrolled] = useStateA(false);
  useEffectA(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <nav className={`nav ${scrolled ? "scrolled" : ""}`}>
      <div className="nav-inner">
        <div className="nav-logo">
          <span className="glyph"><IconLogo size={16} /></span>
          Stratos
        </div>
        <div className="nav-links">
          <a href="#how">Product</a>
          <a href="#customers">Customers</a>
          <a href="#capabilities">Security</a>
          <a href="#faq">FAQ</a>
        </div>
        <button className="btn btn-primary" onClick={onBook} style={{ padding: "10px 18px", fontSize: 14 }}>
          Book a call <span className="arrow"><IconArrow size={13} /></span>
        </button>
      </div>
    </nav>
  );
}

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [modalOpen, setModalOpen] = useStateA(false);

  const rep = REPS[t.rep] || REPS.Maya;

  // Apply accent live via CSS var
  useEffectA(() => {
    document.documentElement.style.setProperty("--brand-500", t.accent);
  }, [t.accent]);

  const onBook = () => setModalOpen(true);

  return (
    <>
      <Nav onBook={onBook} />
      <Hero onBook={onBook} headline={t.headline} tagline={t.tagline} />
      <HowItWorks />
      <Metrics />
      <CaseStudies />
      <Capabilities />
      <FAQ />
      <BookSection onBook={onBook} rep={rep} />
      <Footer />

      <BookingModal open={modalOpen} onClose={() => setModalOpen(false)} rep={rep} />

      <TweaksPanel title="Tweaks">
        <TweakSection label="Brand" />
        <TweakColor
          label="Accent"
          value={t.accent}
          onChange={(v) => setTweak("accent", v)}
          options={["#4F6BFF", "#2A6FDB", "#0EA5E9", "#6E40C9", "#0F172A"]}
        />

        <TweakSection label="Hero copy" />
        <TweakText
          label="Headline"
          value={t.headline}
          onChange={(v) => setTweak("headline", v)}
        />
        <TweakText
          label="Tagline"
          value={t.tagline}
          onChange={(v) => setTweak("tagline", v)}
        />

        <TweakSection label="Sales rep" />
        <TweakRadio
          label="Who handles the call"
          value={t.rep}
          onChange={(v) => setTweak("rep", v)}
          options={["Maya", "David", "Priya"]}
        />

        <TweakSection label="Booking flow" />
        <TweakButton label="Open booking modal" onClick={onBook} />
      </TweaksPanel>
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
