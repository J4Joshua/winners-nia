// Icon components — minimal stroke icons, no decorative SVG slop
const { createElement: h } = React;

function IconBase({ children, size = 16, className = "", strokeWidth = 1.6, ...rest }) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24"
      fill="none" stroke="currentColor"
      strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round"
      className={className} {...rest}
    >
      {children}
    </svg>
  );
}

const IconArrow = (p) => <IconBase {...p}><path d="M5 12h14"/><path d="M13 6l6 6-6 6"/></IconBase>;
const IconCheck = (p) => <IconBase {...p}><path d="M5 12l5 5 9-11"/></IconBase>;
const IconChevron = (p) => <IconBase {...p}><path d="M6 9l6 6 6-6"/></IconBase>;
const IconClose = (p) => <IconBase {...p}><path d="M6 6l12 12M18 6L6 18"/></IconBase>;
const IconSpark = (p) => <IconBase {...p}><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/></IconBase>;
const IconBolt = (p) => <IconBase {...p}><path d="M13 3L4 14h7l-1 7 9-11h-7l1-7z"/></IconBase>;
const IconShield = (p) => <IconBase {...p}><path d="M12 3l8 3v6c0 5-3.5 8.5-8 9-4.5-.5-8-4-8-9V6l8-3z"/><path d="M9 12l2 2 4-4"/></IconBase>;
const IconNet = (p) => <IconBase {...p}><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/></IconBase>;
const IconCalendar = (p) => <IconBase {...p}><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 9h18M8 3v4M16 3v4"/></IconBase>;
const IconClock = (p) => <IconBase {...p}><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></IconBase>;
const IconVideo = (p) => <IconBase {...p}><rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3z"/></IconBase>;
const IconLogo = (p) => <IconBase {...p} strokeWidth={2}><path d="M4 16l8-12 8 12"/><path d="M4 16h16"/><path d="M8 16l4-6 4 6"/></IconBase>;
const IconPlay = (p) => <IconBase {...p}><path d="M7 4l13 8-13 8V4z" fill="currentColor"/></IconBase>;

// Mini icons for inline mock UIs
const IconDoc = (p) => <IconBase {...p}><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h5"/></IconBase>;
const IconMail = (p) => <IconBase {...p}><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 7l9 6 9-6"/></IconBase>;
const IconUser = (p) => <IconBase {...p}><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></IconBase>;
const IconCard = (p) => <IconBase {...p}><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 10h18M7 15h3"/></IconBase>;
const IconLink = (p) => <IconBase {...p}><path d="M10 14a4 4 0 0 0 5.66 0l3-3a4 4 0 0 0-5.66-5.66l-1 1"/><path d="M14 10a4 4 0 0 0-5.66 0l-3 3a4 4 0 1 0 5.66 5.66l1-1"/></IconBase>;
const IconDollar = (p) => <IconBase {...p}><path d="M12 3v18"/><path d="M16 7H10a3 3 0 0 0 0 6h4a3 3 0 0 1 0 6H7"/></IconBase>;
const IconTrend = (p) => <IconBase {...p}><path d="M3 17l6-6 4 4 8-9"/><path d="M14 6h7v7"/></IconBase>;

Object.assign(window, {
  IconArrow, IconCheck, IconChevron, IconClose, IconSpark, IconBolt,
  IconShield, IconNet, IconCalendar, IconClock, IconVideo, IconLogo,
  IconPlay, IconDoc, IconMail, IconUser, IconCard, IconLink,
  IconDollar, IconTrend,
});
