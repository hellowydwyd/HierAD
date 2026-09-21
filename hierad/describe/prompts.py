"""
描述模块 Prompt 模板 — v2 架构

Stage 1: 全片场景 clip VLM + LLM 全局增强
Stage 2: AD gap VLM（混合复用 Stage1）
Stage 3: LLM AD 精炼
"""

# ==================== Stage 1: 场景 clip VLM ====================

STAGE1_SCENE_SYSTEM = """You are a video description assistant for blind audiences.
The video clip has CHARACTER NAME LABELS drawn on faces (green ellipses) — treat those labels as ground truth for identity.

Jointly use VISUALS + DIALOGUE to infer what is happening and each character's state
(who they are with, what they hold, what they are reacting to, emotional/attentional state).

DIALOGUE USAGE:
- Dialogue explains intent and names props/topics — combine it with what you see.
- If dialogue mentions an object/topic (e.g. chocolate, shoes), prefer that reading when the object is on screen.
- Use speaker / addressee cues to resolve roles when faces are unclear.
- Describe the resulting state and action; do not treat the clip as a transcript dump.

OUTPUT FORMAT (exactly 3 lines):
SETTING: [location/environment, or "Same" if unchanged from previous clip]
CHARACTERS: [names from on-screen labels in CAPS; if no label use appearance+role]
ACTION: [what happens and character state — 20-35 words, present tense]

RULES:
1. Use PREVIOUS CLIPS context to avoid repeating unchanged setting/actions.
2. NO camera language ("camera pans", "we see", "the shot").
3. NEVER use "continues", "still", "keeps" — describe the current state directly.
4. Prefer labeled character names over generic descriptions.
5. When dialogue and a naive visual guess conflict, prefer dialogue + visible evidence.
"""

STAGE1_SCENE_USER_FIRST = """This is the OPENING scene clip of the movie.

You MUST describe SETTING fully (never "Same" for the first clip).
Describe all visible CHARACTERS and the ACTION / character state in detail.

DIALOGUE IN THIS CLIP (use with visuals to infer state and props):
{dialogue_overlap}

Describe this scene clip:
SETTING:
CHARACTERS:
ACTION:"""

STAGE1_SCENE_USER = """=== PREVIOUS SCENES (do NOT repeat unchanged content) ===
{recent_context}

DIALOGUE IN THIS CLIP (use with visuals to infer state and props):
{dialogue_overlap}

=== DESCRIBE THIS SCENE CLIP (what is NEW or CHANGED) ===
SETTING:
CHARACTERS:
ACTION:"""


# ==================== Stage 1: LLM 全局增强（沿用） ====================

SCENE_SUMMARY_SYSTEM = """You summarize video scene descriptions into concise story beats.

OUTPUT FORMAT (exactly 3 lines):
EVENT: [Main event - past tense, one sentence]
PROGRESSION: [key actions in order: a → b → c]
CHARACTERS: [names in CAPS]
"""

SCENE_SUMMARY_USER = """Scene clip descriptions:
{segments}

Summarize:
EVENT:
PROGRESSION:
CHARACTERS:"""

CHARACTER_GRAPH_SYSTEM = """You analyze video descriptions to identify characters and relationships.
OUTPUT JSON only:
{"characters":[{"name":"NAME","description":"...","role":"protagonist|supporting|minor"}],"relations":[{"from":"A","to":"B","relation":"..."}]}
Names in UPPERCASE. Infer relationships from context."""

CHARACTER_GRAPH_USER = """Analyze these clip descriptions:
{descriptions}
Output character graph JSON:"""

EVENT_CHAIN_SYSTEM = """Extract KEY EVENTS with causal links from scene summaries.
OUTPUT JSON only:
{"events":[{"id":0,"clip_id":0,"description":"...","characters":["A"],"importance":3,"is_turning_point":false,"caused_by":[]}]}
Maximum 12 events."""

EVENT_CHAIN_USER = """Scene summaries:
{scene_summaries}
Output events JSON:"""

STORY_STRUCTURE_SYSTEM = """Analyze narrative structure.
OUTPUT JSON only:
{"setting":"...","core_conflict":"...","turning_points":[1],"phases":[{"name":"SETUP","clip_ids":[0,1],"summary":"..."}]}
"""

STORY_STRUCTURE_USER = """CHARACTERS:
{character_info}

EVENTS:
{event_chain}

Output story structure JSON:"""


# ==================== Stage 2: AD gap VLM ====================

STAGE2_AD_SYSTEM = """You write audio description for ONE SHORT SILENT WINDOW (no speech inside the window).
Character name labels on faces (green ellipses) are ground truth for identity.

CRITICAL: Focus on what is SPECIFICALLY VISIBLE in THIS window right now.
- Describe the concrete action, movement, gesture, expression, or object interaction happening in this exact moment.
- Do NOT repeat the general scene description — the viewer already knows the setting.
- Each window is different: focus on what CHANGES — who moves, what they hold, where they look, what they dodge or approach.
- If multiple windows share the same setting, differentiate by the SPECIFIC action in each.

Use VISUALS + surrounding dialogue together to infer character state in this beat
(what they hold, who they face, what they are about to do or react to).

DIALOGUE USAGE:
- BEFORE = just said; AFTER = about to be said — both explain the silent beat.
- Example: AFTER mentions chocolate → the held object is a chocolate box, not a book/paper.
- Example: BEFORE/AFTER discuss shoes → prioritize footwear and related gestures.
- Focus on the character's present state and visible action grounded in that context.

OUTPUT FORMAT (exactly 3 lines):
SETTING: [brief location, or "Same"]
CHARACTERS: [names in CAPS from labels if visible]
ACTION: [specific visible action in THIS window — 12-22 words, present tense, focus on movement/detail]

NO camera language."""

STAGE2_AD_USER = """=== GLOBAL STORY CONTEXT (background only, do NOT repeat) ===
{global_summary}

=== SCENE CONTEXT (background only, do NOT repeat) ===
{scene_context}

DIALOGUE BEFORE (just said — use with visuals to infer state):
{dialogue_before}

UPCOMING DIALOGUE AFTER (about to be said — use with visuals to infer state/props):
{dialogue_after}

Describe ONLY what is specifically visible in THIS silent window right now.
Do NOT repeat the scene context above. Focus on the unique action/movement/detail of this moment.
SETTING:
CHARACTERS:
ACTION:"""


# ==================== Stage 3: AD 精炼（单段、无跨段记忆）====================

STAGE3_SYSTEM = """You REFINE one visual description into professional audio description (AD).

OUTPUT: ONE natural English sentence. Speakable within the MAX WORDS budget.

GROUNDING (critical):
- Use ONLY the VISUAL DESCRIPTION of THIS silent window.
- Do NOT use prior AD lines, story memory, or off-screen plot.
- If Characters is NONE: describe setting/objects/vehicles/environment only — NEVER invent person names.
- If Characters lists names: you may use those names; do not add others.
- Do not invent actions unsupported by the visual text.
- NEVER use camera language (cut, pan, shot, camera).
- NEVER use "continues to", "still", "begins to".
- English only, one sentence, no prefix or quotes.
- Prefer short concrete verbs; cut fillers when the budget is tight."""

STAGE3_USER = """=== VISUAL DESCRIPTION (this window only) ===
{visual_description}

Characters field: {character_info}

Silent gap: {gap_sec:.1f}s → MAX WORDS: {max_words}

Write ONE AD sentence (≤{max_words} words) grounded ONLY in the visual above:"""

# 兼容旧 import
STAGE1_SYSTEM = STAGE1_SCENE_SYSTEM
STAGE1_USER_FIRST = STAGE1_SCENE_USER_FIRST
STAGE1_USER = STAGE1_SCENE_USER
STAGE1_GAP_NOTE = ""
STAGE1_FACE_HINT = ""
