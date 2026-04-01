create extension if not exists vector;

create table if not exists communication_profiles (
    profile_id text primary key,
    audience text not null check (audience in ('child', 'parent')),
    owner_id text not null,
    preferred_tone text not null,
    preferred_pacing text not null,
    sensory_notes jsonb not null default '[]'::jsonb,
    banned_styles jsonb not null default '[]'::jsonb,
    preferred_phrases jsonb not null default '[]'::jsonb,
    policy jsonb not null,
    created_at timestamptz not null default now()
);

create unique index if not exists idx_communication_profiles_owner_id
    on communication_profiles(owner_id, audience);

create table if not exists target_profiles (
    target_id text primary key,
    target_type text not null check (target_type in ('letter', 'number', 'word')),
    display_text text not null,
    phoneme_group text not null default '',
    difficulty_level int not null default 1 check (difficulty_level between 1 and 10),
    active boolean not null default true,
    created_at timestamptz not null default now()
);

create table if not exists reference_vectors (
    reference_id text primary key,
    target_id text not null references target_profiles(target_id) on delete cascade,
    modality text not null check (modality in ('audio', 'noise', 'lip', 'emotion')),
    embedding vector(8),
    source_label text not null default '',
    quality_score double precision not null default 0 check (quality_score between 0 and 1),
    age_band text not null default '',
    notes jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists child_attempt_vectors (
    attempt_id text primary key,
    child_id text not null,
    target_id text not null references target_profiles(target_id) on delete cascade,
    session_id text not null,
    audio_embedding vector(8),
    lip_embedding vector(8),
    emotion_embedding vector(8),
    noise_embedding vector(8),
    top_match_reference_id text references reference_vectors(reference_id) on delete set null,
    cosine_similarity double precision check (cosine_similarity between -1 and 1),
    success_flag boolean,
    created_at timestamptz not null default now()
);

create table if not exists output_filter_profiles (
    profile_id text primary key,
    child_id text,
    caregiver_id text,
    preferred_tone_embedding vector(8),
    safe_expression_embedding vector(8),
    best_reengagement_style vector(8),
    parent_guidance_style vector(8),
    overstimulation_flags jsonb not null default '[]'::jsonb,
    verbosity_limit int not null default 100 check (verbosity_limit between 20 and 240),
    calming_style_vector vector(8),
    updated_at timestamptz not null default now(),
    check (child_id is not null or caregiver_id is not null)
);

create table if not exists environment_standard_profiles (
    environment_profile_id text primary key,
    child_id text not null,
    baseline_room_embedding vector(8),
    baseline_visual_clutter_score double precision not null default 0 check (baseline_visual_clutter_score between 0 and 1),
    baseline_noise_score double precision not null default 0 check (baseline_noise_score between 0 and 1),
    baseline_lighting_score double precision not null default 0 check (baseline_lighting_score between 0 and 1),
    baseline_distraction_notes jsonb not null default '[]'::jsonb,
    recommended_adjustments jsonb not null default '[]'::jsonb,
    updated_at timestamptz not null default now()
);

create unique index if not exists idx_environment_standard_profiles_child_id
    on environment_standard_profiles(child_id);

create index if not exists idx_reference_vectors_target_id on reference_vectors(target_id);
create index if not exists idx_reference_vectors_modality on reference_vectors(modality);
create index if not exists idx_child_attempt_vectors_child_id on child_attempt_vectors(child_id);
create index if not exists idx_child_attempt_vectors_session_id on child_attempt_vectors(session_id);
create index if not exists idx_output_filter_profiles_child_id on output_filter_profiles(child_id);
create index if not exists idx_output_filter_profiles_caregiver_id on output_filter_profiles(caregiver_id);
