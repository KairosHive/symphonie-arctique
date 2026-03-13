# Symphonie Arctique

**An AI-Assisted Audiovisual Creation Project**  
*Kairos Création for Laboratoire AgeTeQ — CIUSSS Centre-Sud-de-l'Île-de-Montréal*  
*March 2026*

## Overview

Symphonie Arctique is an experimental audiovisual work exploring Arctic themes through a unique collaboration between artificial intelligence and human artistic curation. The project transforms audio compositions—blending Inuit and non-Indigenous musical elements—into cohesive visual narratives.

Rather than fully automating the creative process, this pipeline uses AI as a generator of possibilities that artists shape and refine. The result is a carefully curated synthesis where algorithmic analysis meets human aesthetic judgment at every critical decision point.

## Project Philosophy

This work demonstrates a **hybrid creative methodology**:

- 🤖 **AI Agents** handle computational tasks: audio analysis, prompt generation, image synthesis, and animation
- 👤 **Human Artists** provide creative direction: designing vocabularies, curating outputs, directing motion, and assembling the final narrative

The interplay between these roles is intentional—AI expands the space of possibilities while human judgment ensures artistic coherence and emotional resonance.

---

## Visual Content Pipeline

### 1. Audio Analysis & Descriptor Extraction

**Script:** `scripts/CLAP_extractor.py`

**Purpose:** Translates sound into semantic concepts by analyzing audio segments and identifying descriptive words that match the sonic character of each moment.

**How it works:**
- Segments audio files based on predefined timelines
- Uses the CLAP (Contrastive Language-Audio Pretraining) model to compute similarity between audio chunks and a curated vocabulary
- Outputs ranked lists of descriptors for each second of audio

**Human role:** Artists design the descriptor dictionaries (`assets/descriptors-*.json`) that define the symbolic and lexical universe of the work—organized into themes like Auditory, Temporal, Aqueous, and Emotional qualities.

### 2. Prompt Synthesis

**Script:** `scripts/txt2img_prompt.py`

**Purpose:** Transforms lists of audio-matched keywords into poetic, visually-oriented text prompts suitable for image generation.

**How it works:**
- Aggregates descriptors from the audio analysis phase
- Sends them to a large language model (via Ollama) with specific stylistic instructions
- Produces refined prompt texts optimized for visual generation

**Human role:** Artists write the system prompt (`assets/system_prompt_txt2img.md`) that establishes the narrative voice, stylistic constraints, and thematic guidelines the AI must follow.

### 3. Text-to-Image Generation

**Script:** `scripts/comfyui/comfyui_txt2img.py`

**Purpose:** Generates visual imagery from text prompts using state-of-the-art diffusion models.

**How it works:**
- Submits prompts to a ComfyUI server running the FLUX-dev model with LoRA style adaptations
- Generates multiple variations of each concept to provide artistic choice
- Outputs static images at 1280x720 resolution with associated metadata

**Human role:** Artists manually curate hundreds of generated images, selecting only those that meet aesthetic standards and narrative coherence (✅ checkmark process shown in methodology diagram).

### 4. Image-to-Video Animation

**Script:** `scripts/comfyui/comfyui_i2v.py`

**Purpose:** Brings selected still images to life with motion, synchronized to the audio timeline.

**How it works:**
- Animates curated images using image-to-video models
- Synchronizes video segments with corresponding audio sections
- Produces raw video clips ready for final editing

**Human role:** Artists define motion characteristics (camera movements, pacing) and perform final video editing—assembling clips, creating transitions, color grading, and ensuring visual events align with musical moments.

---

## Audio Content Pipeline

### Hybrid Sound Creation

**Script:** `audio-blending/submit_comfy.py`  
**Workflow:** `audio-blending/workflow_api.json`

**Purpose:** Creates new hybrid musical material by fusing different instrumental tracks in latent space, blending Inuit and non-Indigenous musical sources while preserving expressive qualities.

**How it works:**
- Uses the ACE-Step generative audio model within ComfyUI
- Encodes two audio sources into latent representations using a Variational Autoencoder (VAE)
- Blends these representations through interpolation (typically 50/50 ratio)
- Applies a diffusion process to regularize and refine the fusion
- Decodes the result back to audio, creating novel sounds that combine characteristics of both sources

**Human role:** Artists apply this process combinatorially to multiple track pairs, then manually select and arrange the most successful fusions into a final composition, ensuring musical coherence and emotional narrative.

---

## Repository Structure

```
symphonie-arctique/
│
├── assets/                          # Core creative resources
│   ├── descriptors-*.json          # Curated vocabulary for audio-visual matching
│   ├── timeline_*.json             # Temporal segmentation of compositions
│   └── system_prompt_txt2img.md    # Stylistic instructions for prompt generation
│
├── scripts/                         # Processing pipeline
│   ├── CLAP_extractor.py           # Audio → Semantic descriptors
│   ├── txt2img_prompt.py           # Descriptors → Visual prompts
│   └── comfyui/                    # ComfyUI integration scripts
│       ├── comfyui_txt2img.py      # Prompts → Images
│       └── comfyui_i2v.py          # Images → Video clips
│
├── audio-blending/                  # Hybrid audio generation
│   ├── submit_comfy.py             # ACE-Step fusion pipeline
│   └── workflow_api.json           # ComfyUI workflow definition
│
├── workflows/                       # ComfyUI workflow configurations
│
├── results/                         # Generated prompts and metadata
│
├── footage/                         # Generated and curated visual content
│   ├── i2v_output/                 # Raw animated clips
│   ├── refined_curated/            # Human-selected content
│   └── scene-*/                    # Organized by narrative segments
│
└── audio_output/                    # Generated hybrid audio compositions
```

---

## Technical Stack

- **Audio Analysis:** CLAP (Contrastive Language-Audio Pretraining)
- **Prompt Generation:** Ollama (Gemma, Hermes LLMs)
- **Image Generation:** ComfyUI + FLUX-dev + LoRA style adaptations
- **Video Generation:** ComfyUI image-to-video models
- **Audio Fusion:** ACE-Step (latent diffusion for music generation)
- **Orchestration:** Python scripts + ComfyUI API

---

## Artistic Statement

Symphonie Arctique demonstrates that effective AI-assisted art relies not on automation, but on **structured collaboration**. The machine operates within creative constraints defined by human artists—vocabularies, themes, aesthetic criteria—while artists leverage the machine's ability to explore vast possibility spaces.

The Arctic serves as both subject and metaphor: a vast, transformed landscape where different cultural expressions meet, merge, and create something new while respecting their distinct identities.

---

## Project Team

**Production:** Kairos Création  
**Commission:** Laboratoire AgeTeQ — CIUSSS Centre-Sud-de-l'Île-de-Montréal  
**Completion:** March 2026


---

**For questions or collaboration inquiries, please contact the project team.**