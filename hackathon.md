Tracks
🛰️ Always-On Agents 🛰️
Sponsor: Nia + Tensorlake

Build an agent that runs continuously in the background, remembers what it has learned across sessions, and acts without a human prompting it. The bar is high: removing either the background execution or the stateful memory should break your demo.

The dream shape is an agent that wakes up on its own and knows everything it has done before - a research monitor that learns your reading history over weeks, a support agent with durable per-customer memory across months, or a dev assistant that reviews PRs overnight and has internalised your codebase's conventions.

Examples of what to build
A research agent that monitors a topic continuously, tracks what you've read and dismissed, and sends weekly briefs that get sharper every week
A support agent that handles tickets 24/7 with durable per-customer memory
An overnight PR reviewer that clears your team's queue before standup
A personal agent that runs on your calendar and email, remembers what you asked last Tuesday, and learns your preferences over time
Sponsors
1 Nia is a context augmentation layer and MCP server that indexes repos, documentation sites, PDFs, Slack, Google Drive, local files - anything - so agents always have up-to-date, accurate context rather than hallucinating from stale training data. In this track, Nia keeps the agent's knowledge fresh and current across every run.

What they provide:

MCP server with 15+ specialised tools for semantic search across indexed sources
Indexes entire codebases, docs sites, research papers, datasets, and local folders
Nia Sync daemon for continuously syncing local data (files, iMessage, databases)
Nia Oracle - autonomous research agent for multi-source research workflows
Cross-session context: save findings in one agent, retrieve in another
Proven to reduce hallucination rates by 11.3% vs leading alternatives

Tensorlake provides the infrastructure for background and stateful agent execution. Agents built with Tensorlake run on their own - triggered by schedules, webhooks, or other agents - and remember what they have done across runs. Durable memory is a first-class primitive, not bolted on as a RAG afterthought.
What they provide:
Background execution: agents triggered by schedules, webhooks, events, or other agents
Stateful execution: durable memory that survives invocations, sessions, and restarts
Sandbox environment for real work - code, APIs, data transforms
Full agentic infrastructure: the agent runs while no one is watching
Judging criteria

🚀 Ship It - Full-Stack Agents 🚀
Sponsors: Nia + InsForge
Build a complete, production-deployed agentic application with real authentication, a live backend, and AI logic that does not break when users go off-script. The challenge: get it actually shipped and running, not just demo-ready.

Not localhost. Not a polished UI with no backend. Something a stranger can open, sign up for, and use today. Hyperspell's context graphs handle agent intelligence, InsForge gets it shipped end-to-end, and Nia ensures the agent never hallucinates on your own codebase or docs.

Examples of what to build
A deployable coding assistant that understands your private codebase via Nia and ships features autonomously
A fully live SaaS tool with auth, database, and real user flows - powered by an agent that handles the messy edge cases
A multi-agent app where different agents (planning, execution, review) share context via Nia and Hyperspell
Anything production-grade: if someone outside your team can use it end-to-end, you're on the right track
Sponsors
Nia is a context augmentation layer and MCP server that indexes repos, documentation sites, PDFs, Slack, Google Drive, local files - anything - so agents always have up-to-date, accurate context rather than hallucinating from stale training data. In this track, Nia keeps the agent's knowledge fresh and current across every run.

What they provide:

MCP server with 15+ specialised tools for semantic search across indexed sources
Indexes entire codebases, docs sites, research papers, datasets, and local folders
Nia Sync daemon for continuously syncing local data (files, iMessage, databases)
Nia Oracle - autonomous research agent for multi-source research workflows
Cross-session context: save findings in one agent, retrieve in another
Proven to reduce hallucination rates by 11.3% vs leading alternatives

InsForge is about production, not demos. Set up authentication, database, and deployment - not just UI, not just localhost. Your app should be usable end-to-end by someone outside your team by the end of the day.

What they provide:
Production-ready backend scaffolding with auth and database out of the box
Live deployment so your app is accessible to real users on the day
Focus on reliability and real user flows, not just a polished front end
Judging criteria

🎯 AI-Native Growth Tools 🎯
Sponsors: Nia + Reacher

Build tools that do revenue work autonomously - creator scouting, outreach automation, campaign building, or attribution analysis. Think beyond dashboards: the best submissions will have the agent doing the work, not just reporting on it.

Reacher provides per-team TikTok Shop demo data, market-wide Social Intelligence, and sandboxed write endpoints so agents can create real campaigns, sample requests, and outreach drafts without any external dispatch. Nia adds the context layer - index market reports, creator briefs, or campaign history so your agent reasons with the full picture, not just the latest query.

Examples of what to build
A TikTok creator scouting agent: brief → ranked list with evidence, searching beyond a single network
An always-on market monitor: continuous watch on category and trending-video shifts, with alerts on competitor creator spikes
An end-to-end TikTok Shops campaign builder: discovery → vetting → outreach creation → follow-up, fully sandboxed
Attribution forensics: explain week-over-week GMV changes using creator, video, and category signals
Sponsors
Nia is a context augmentation layer and MCP server that indexes repos, documentation sites, PDFs, Slack, Google Drive, local files - anything - so agents always have up-to-date, accurate context rather than hallucinating from stale training data. In this track, Nia keeps the agent's knowledge fresh and current across every run.

What they provide:

MCP server with 15+ specialised tools for semantic search across indexed sources
Indexes entire codebases, docs sites, research papers, datasets, and local folders
Nia Sync daemon for continuously syncing local data (files, iMessage, databases)
Nia Oracle - autonomous research agent for multi-source research workflows
Cross-session context: save findings in one agent, retrieve in another
Proven to reduce hallucination rates by 11.3% vs leading alternatives

Reacher provides the TikTok Shop data infrastructure for this track. Every team gets a sandboxed demo environment with realistic data - no payment card, no onboarding friction.

What they provide:
MCP server at api.reacherapp.com/mcp with 33 tools covering creators, products, videos, samples, GMV metrics, and the full Social Intelligence market catalogue
Social Intelligence: market-wide creator, seller, and trending-video data across the entire TikTok Shop ecosystem
Per-team demo dataset - a scoped clone of a dummy account with creators, videos, GMV history, and samples. Realistic enough for multi-step agent reasoning, isolated so teams don't interfere
Sandboxed write endpoints - POST /automations, /samples/request, /outreach/draft persist records for demo purposes but never dispatch to TikTok or trigger real emails
One-click MCP connector in Claude.ai, Cursor, and any MCP client
Judging criteria

🧠 The Company Brain 🧠
Sponsors: Nia + Hyperspell
Build an agent that synthesizes context across a company's data and does something genuinely valuable with it. The best submissions won't just answer questions, they'll do work. Sales agents that know every deal in the pipeline. Support agents that know every past ticket. Marketing agents that have internalized the brand voice. Coding agents that know the codebase, the team's conventions, and last week's design review.
Hyperspell gives you ingestion (Slack, Gmail, Drive, GitHub, Notion, and more) and search across all of it. You decide what the brain looks like and what agents do with the context.
Examples of what to build

A sales agent that drafts deal-specific outreach by synthesizing CRM, email threads, and past won deals
A marketing agent that ships campaigns matching brand voice learned from every doc and Slack the team has ever written
An onboarding agent that knows the company's tech stack, Slack history, and processes better than any human
A support agent that resolves tickets with full context across product decisions, past tickets, and customer history
Sponsors
Nia is a context augmentation layer and MCP server that indexes repos, documentation sites, PDFs, Slack, Google Drive, local files - anything - so agents always have up-to-date, accurate context rather than hallucinating from stale training data. In this track, Nia keeps the agent's knowledge fresh and current across every run.

What they provide:

MCP server with 15+ specialised tools for semantic search across indexed sources
Indexes entire codebases, docs sites, research papers, datasets, and local folders
Nia Sync daemon for continuously syncing local data (files, iMessage, databases)
Nia Oracle - autonomous research agent for multi-source research workflows
Cross-session context: save findings in one agent, retrieve in another
Proven to reduce hallucination rates by 11.3% vs leading alternative

Hyperspell is the company brain for AI agents. They ingest all of an organization's data and synthesizes it into context any agent can read.
What they provide:
Ingestion across Slack, Gmail, Google Drive, GitHub, Notion, and more
Semantic search across all sources, synthesis, and presentation as a filesystem for all agents
Built for agents like Claude Code, Hermes, and OpenClaw
Judging criteria

Submission
Please read before submitting:
One submission per team. Only one team member should fill out this form. Duplicate submissions will be disqualified.
Submissions close at 6:00pm sharp. Late submissions may not be accepted.
You must be present for judging. At least one team member must be available for the in-person judging session starting at 6:10pm.
Have these ready before you start: your deployed demo link, GitHub repo URL, and the names and emails of all team members.
Incomplete submissions may be disqualified. Make sure every field is filled out - especially the demo link. A localhost link is not a valid demo.
Judging Format
Step 1: Submit your project here https://forms.gle/fkoFXRo3L2MVkkz87
Step 2: Judging will be conducted in person first. Each team will receive 3 mins to present their project directly to the judges. Judges will individually score and the average will be marked as the final grade.
Step 3: The top 6 across all tracks will present live in front of the entire room and in front of the final judges.
Judging Method
There are 4 tracks. Each track is judged separately by the judges. However, the final rankings are track-agnostic. The top 6 finalists and top 3 winners are determined purely by score, regardless of which track you submitted to. This means you're ultimately competing against everyone, not just teams in your track.

We understand this may seem unusual but we've designed it this way to remove bias and ensure the best projects win, period.

Judges score individually and the average is your final grade.

Rules
Max of 3 participants per team, solo is also welcomed
No re-using personal projects
Project must be submitted before 6pm
Any form of cheating or rule-breaking disqualifies the team immediately
Organisers reserve the right to disqualify any project that violates the spirit of the event
Winners must be present at the awards to claim prizes
Prize fulfilment handled by sponsors directly - timelines may vary
Teams can only join and present for 1 track
