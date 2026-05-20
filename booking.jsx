// Booking modal — 3 step flow: date → time → details → success
const { useState, useMemo } = React;

function BookingModal({ open, onClose, rep }) {
  const [step, setStep] = useState(0);
  const [date, setDate] = useState(null);
  const [time, setTime] = useState(null);
  const [form, setForm] = useState({
    name: "", email: "", company: "", teamSize: "11–50", topic: "Cost reduction audit"
  });
  const [done, setDone] = useState(false);

  // Generate next 14 weekdays
  const dates = useMemo(() => {
    const out = [];
    const today = new Date();
    let d = new Date(today);
    while (out.length < 12) {
      d.setDate(d.getDate() + 1);
      const dow = d.getDay();
      if (dow === 0 || dow === 6) continue;
      out.push(new Date(d));
    }
    return out;
  }, []);

  const slots = ["09:00", "09:30", "10:30", "11:00", "13:00", "14:00", "15:30", "16:00", "17:00"];

  const reset = () => {
    setStep(0); setDate(null); setTime(null); setDone(false);
    setForm({ name: "", email: "", company: "", teamSize: "11–50", topic: "Cost reduction audit" });
  };

  const close = () => { onClose(); setTimeout(reset, 220); };

  const next = () => setStep((s) => Math.min(s + 1, 2));
  const back = () => setStep((s) => Math.max(s - 1, 0));
  const submit = (e) => { e.preventDefault(); setDone(true); };

  const canNext = (step === 0 && date) || (step === 1 && time);

  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={close}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        {!done && (
          <>
            <div className="modal-head">
              <div>
                <h3>Book a strategy call</h3>
                <div className="modal-sub">30 min with {rep.name} · {rep.role}</div>
              </div>
              <button className="modal-close" onClick={close} aria-label="Close">
                <IconClose size={16} />
              </button>
            </div>

            <div className="modal-body">
              <div className="modal-steps">
                {[0,1,2].map((i) => (
                  <div key={i} className={`modal-step-pip ${i <= step ? "active" : ""}`} />
                ))}
              </div>

              {step === 0 && (
                <>
                  <div style={{ marginBottom: 14, fontSize: 14, color: "var(--ink-700)", fontWeight: 500 }}>
                    Pick a date — all times shown in your local timezone.
                  </div>
                  <div className="date-grid">
                    {dates.map((d, i) => {
                      const sel = date && d.toDateString() === date.toDateString();
                      return (
                        <button
                          key={i}
                          className={`date-cell ${sel ? "selected" : ""}`}
                          onClick={() => setDate(d)}
                        >
                          <div className="date-dow">{d.toLocaleDateString("en-US", { weekday: "short" })}</div>
                          <div className="date-day">{d.getDate()}</div>
                          <div className="date-dow" style={{ marginTop: 4 }}>
                            {d.toLocaleDateString("en-US", { month: "short" })}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </>
              )}

              {step === 1 && (
                <>
                  <div style={{ marginBottom: 14, fontSize: 14, color: "var(--ink-700)", fontWeight: 500 }}>
                    {date && date.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" })} — pick a time slot.
                  </div>
                  <div className="time-grid">
                    {slots.map((s) => (
                      <button
                        key={s}
                        className={`time-cell ${time === s ? "selected" : ""}`}
                        onClick={() => setTime(s)}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </>
              )}

              {step === 2 && (
                <form className="form-grid" onSubmit={submit}>
                  <div className="form-row-2">
                    <div className="form-field">
                      <label>Full name</label>
                      <input required value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="Alex Rivera" />
                    </div>
                    <div className="form-field">
                      <label>Work email</label>
                      <input required type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} placeholder="alex@company.com" />
                    </div>
                  </div>
                  <div className="form-field">
                    <label>Company</label>
                    <input required value={form.company} onChange={(e) => setForm({ ...form, company: e.target.value })} placeholder="Northbeam Logistics" />
                  </div>
                  <div className="form-row-2">
                    <div className="form-field">
                      <label>Team size</label>
                      <select value={form.teamSize} onChange={(e) => setForm({ ...form, teamSize: e.target.value })}>
                        <option>1–10</option>
                        <option>11–50</option>
                        <option>51–200</option>
                        <option>200–500</option>
                      </select>
                    </div>
                    <div className="form-field">
                      <label>I want to talk about</label>
                      <select value={form.topic} onChange={(e) => setForm({ ...form, topic: e.target.value })}>
                        <option>Cost reduction audit</option>
                        <option>Customer support agent</option>
                        <option>Finance & ops automation</option>
                        <option>Sales pipeline agent</option>
                        <option>Something else</option>
                      </select>
                    </div>
                  </div>

                  {/* hidden submit so Enter works */}
                  <button type="submit" style={{ display: "none" }} />
                </form>
              )}

              <div className="modal-actions">
                {step > 0 ? (
                  <button className="btn btn-ghost" onClick={back}>Back</button>
                ) : (
                  <button className="btn btn-ghost" onClick={close}>Cancel</button>
                )}
                {step < 2 ? (
                  <button
                    className="btn btn-primary"
                    onClick={next}
                    disabled={!canNext}
                    style={{ opacity: canNext ? 1 : 0.5 }}
                  >
                    Continue <span className="arrow"><IconArrow size={14} /></span>
                  </button>
                ) : (
                  <button
                    className="btn btn-primary"
                    onClick={submit}
                    disabled={!form.name || !form.email || !form.company}
                    style={{ opacity: (!form.name || !form.email || !form.company) ? 0.5 : 1 }}
                  >
                    Confirm booking <span className="arrow"><IconArrow size={14} /></span>
                  </button>
                )}
              </div>
            </div>
          </>
        )}

        {done && (
          <div className="modal-body" style={{ paddingTop: 28 }}>
            <div className="success-state">
              <div className="success-icon"><IconCheck size={28} strokeWidth={2.4} /></div>
              <h3>You're on the calendar</h3>
              <div className="modal-sub">A confirmation is heading to <strong style={{ color: "var(--ink-800)" }}>{form.email}</strong>.</div>

              <div className="success-summary">
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <IconCalendar size={14} />
                  <span>{date.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" })} at {time}</span>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <IconUser size={14} />
                  <span>{rep.name} · {rep.role}</span>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <IconVideo size={14} />
                  <span>Google Meet link in invite</span>
                </div>
              </div>

              <div className="modal-actions" style={{ justifyContent: "center" }}>
                <button className="btn btn-primary" onClick={close}>
                  Done
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

Object.assign(window, { BookingModal });
