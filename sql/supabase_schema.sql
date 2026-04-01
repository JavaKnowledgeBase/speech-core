create extension if not exists vector;

create table if not exists communication_profiles (
    profile_id text primary key,
    audience text not null,
    owner_id text not null,
    preferred_tone text not null,
    preferred_pacing text not null,
    sensory_notes jsonb not null default '[]'::jsonb,
    banned_styles jsonb not null default '[]'::jsonb,
    preferred_phrases jsonb not null default '[]'::jsonb,
    policy jsonb not null
);

create table if not exists target_profiles (
    target_id text primary key,
    target_type text not null,
    display_text text not null,
    phoneme_group text not null default '',
    difficulty_level int not null default 1,
    active boolean not null default true,
    created_at timestamptz not null default now()
);

create table if not exists reference_vectors (
    reference_id text primary key,
    target_id text not null references target_profiles(target_id) on delete cascade,
    modality text not null,
    embedding vector(8),
    source_label text not null default '',
    quality_score double precision not null default 0,
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
    top_match_reference_id text,
    cosine_similarity double precision,
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
    verbosity_limit int not null default 100,
    calming_style_vector vector(8),
    updated_at timestamptz not null default now()
);

create table if not exists environment_standard_profiles (
    environment_profile_id text primary key,
    child_id text not null,
    baseline_room_embedding vector(8),
    baseline_visual_clutter_score double precision not null default 0,
    baseline_noise_score double precision not null default 0,
    baseline_lighting_score double precision not null default 0,
    baseline_distraction_notes jsonb not null default '[]'::jsonb,
    recommended_adjustments jsonb not null default '[]'::jsonb,
    updated_at timestamptz not null default now()
);

create index if not exists idx_reference_vectors_target_id on reference_vectors(target_id);
create index if not exists idx_child_attempt_vectors_child_id on child_attempt_vectors(child_id);
create index if not exists idx_output_filter_profiles_child_id on output_filter_profiles(child_id);
create index if not exists idx_environment_standard_profiles_child_id on environment_standard_profiles(child_id);
