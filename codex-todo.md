# Codex TODO For speech-filters

## Why this exists

This note is a handoff from work done in:

- `C:\Users\rkafl\Documents\Projects\speech-intellegence`

When I switch to this folder, I should read this first.

## What has already been built in speech-intellegence

The main app repo already has:

- an agentic backend foundation
- session conductor structure
- output filter expert structure
- child and parent communication profiles
- environment profile and room-check logic
- month-one curriculum scaffolding for 20 targets
- multimodal vector/reference scaffolding
- repository layer for curriculum, vectors, and environment access
- workflow queue scaffolding
- Supabase schema draft
- notes that define product direction

## Important product direction already decided

The app must support:

- tablet
- TV
- desktop

The product purpose is:

- encourage the child to speak
- improve imitation and speaking confidence

The product is not mainly for:

- familiarizing the child with words
- familiarizing the child with text
- passive vocabulary exposure
- text-heavy learning

## What speech-filters should focus on

This folder should focus on the special filter system.

That filter should sit before any child-facing or parent-facing output.

It should make output:

- constructive
- user friendly
- calming
- peaceful
- lower-arousal
- non-chatty
- non-irritating

It should adapt to:

- child state
- parent state
- session context
- environment context
- speaking-first purpose

## What should be watched or modified here

Work to do in `speech-filters`:

1. Define the filter architecture clearly.
2. Decide whether the filter is:
- rules only
- model only
- hybrid rules + model
- third-party API assisted

3. Define separate output handling for:
- child output
- parent output
- caregiver alerts
- retry prompts
- praise and reinforcement
- escalation messages
- environment-adjustment requests

4. Make sure the filter reduces:
- verbosity
- overstimulation
- emotional intensity
- repetitive chatter
- distracting phrasing

5. Make sure the filter preserves:
- encouragement to speak
- calm guidance
- short prompts
- low-pressure repetition
- parent clarity

6. Think about profile-aware filtering:
- child communication profiles
- parent communication profiles
- sensory preferences
- banned styles
- preferred phrasing
- verbosity limits

7. Think about environment-aware filtering:
- if room is distracting, parent guidance should stay calm and actionable
- environment corrections should not sound critical or noisy

8. Define what should be deleted or blocked by the filter:
- text-heavy output for child mode
- noisy praise
- too many words per prompt
- emotionally intense correction
- chatter-box behavior

## What to compare against the main app

When working here, compare every filter idea against the current app in `speech-intellegence`:

- output_filter_expert
- communication profiles
- environment checks
- session-start parent guidance
- notes in `notes\working-process.md`
- notes in `notes\vector-db-design.md`
- notes in `notes\codex-todo.md`

## Best design standard

Every filter decision should answer:

- Does this help the child speak more?
- Does this keep the parent calm and clear?
- Does this reduce irritation and noise?
- Does this avoid turning the app into a chatter box?

If not, it likely needs to be changed or removed.

## First thing to do after switching here

1. Read this file.
2. Inspect the files in this folder.
3. Build the filter architecture so it can later plug back into `speech-intellegence` cleanly.
