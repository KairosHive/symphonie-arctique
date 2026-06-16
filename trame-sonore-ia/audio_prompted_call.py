"""
Custom Stable Audio Open call with proper init-audio "strength" control.

The stock diffusers `StableAudioPipeline` adds full-magnitude noise on top of
encoded init audio (`latents = encoded_audio + noise * init_noise_sigma`,
where `init_noise_sigma ~ 500`). The encoded audio's typical magnitude is on
the order of 1-3, so the init is essentially a tiny perturbation buried in
noise — the model effectively ignores it.

This helper mirrors what an img2img pipeline does:

  1. Encode init audio to VAE latents.
  2. Pick a start step based on `strength` (1.0 = stock behavior = full noise,
     init ignored; 0.3 = very strong init influence).
  3. Add noise scaled by sigmas[start_step] (much smaller than init_noise_sigma).
  4. Run denoising from start_step onward, skipping the high-noise early
     iterations the model would use to "explore away" from the init.

Result: the EMD chord audibly anchors the generated audio's spectrum.
"""
from __future__ import annotations

import torch
from diffusers.models.embeddings import get_1d_rotary_pos_embed


@torch.no_grad()
def audio_prompted_generate(
    pipe,
    prompt: str,
    *,
    init_audio: torch.Tensor,
    init_sample_rate: int,
    strength: float = 0.7,
    num_inference_steps: int = 200,
    audio_end_in_s: float = 47.0,
    guidance_scale: float = 7.0,
    negative_prompt: str = "",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Returns generated audio as torch.Tensor of shape (1, channels, samples), fp32.

    strength: float in (0, 1].
        1.0 -> identical to stock pipeline (full noise; init audio ignored).
        0.7 -> moderate (a good first try).
        0.4 -> strong init influence; output noticeably anchored on the chord.
        0.2 -> very strong; output may sound near-sine-wavey.
    """
    device = pipe._execution_device
    do_cfg = guidance_scale > 1.0

    # ------------------------------------------------------------------
    # 1. Encode text prompt (returns already-CFG-concatenated embeds if
    #    do_cfg=True and negative_prompt is provided)
    # ------------------------------------------------------------------
    neg_for_encode = negative_prompt if negative_prompt else None
    prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=neg_for_encode,
    )

    # ------------------------------------------------------------------
    # 2. Encode duration (seconds_start=0, seconds_end=audio_end_in_s).
    #    encode_duration handles CFG-doubling internally.
    # ------------------------------------------------------------------
    seconds_start_hidden_states, seconds_end_hidden_states = pipe.encode_duration(
        audio_start_in_s=0.0,
        audio_end_in_s=audio_end_in_s,
        device=device,
        do_classifier_free_guidance=do_cfg,
        batch_size=1,
    )

    # ------------------------------------------------------------------
    # 3. Build text_audio_duration_embeds and audio_duration_embeds
    # ------------------------------------------------------------------
    text_audio_duration_embeds = torch.cat(
        [prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
    )
    audio_duration_embeds = torch.cat(
        [seconds_start_hidden_states, seconds_end_hidden_states], dim=2
    )

    # CFG without an explicit negative_prompt: zero out the negative side
    # (matches pipeline behavior at lines ~170-180 of pipeline_stable_audio.py)
    if do_cfg and neg_for_encode is None:
        neg_text = torch.zeros_like(text_audio_duration_embeds)
        text_audio_duration_embeds = torch.cat([neg_text, text_audio_duration_embeds], dim=0)
        audio_duration_embeds = torch.cat([audio_duration_embeds, audio_duration_embeds], dim=0)

    # ------------------------------------------------------------------
    # 4. Scheduler: set timesteps, choose start step from `strength`
    # ------------------------------------------------------------------
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    all_timesteps = pipe.scheduler.timesteps
    all_sigmas = pipe.scheduler.sigmas

    strength_clipped = float(max(0.05, min(1.0, strength)))
    start_idx = int(round(num_inference_steps * (1.0 - strength_clipped)))
    start_idx = max(0, min(start_idx, num_inference_steps - 1))
    sigma_start = float(all_sigmas[start_idx].item())
    timesteps = all_timesteps[start_idx:]

    # ------------------------------------------------------------------
    # 5. Encode init audio via VAE
    # ------------------------------------------------------------------
    waveform_length = int(pipe.transformer.config.sample_size)
    downsample_ratio = pipe.vae.hop_length
    audio_vae_length = waveform_length * downsample_ratio

    init_audio = init_audio.to(device=device, dtype=pipe.transformer.dtype)
    if init_audio.ndim == 2:
        init_audio = init_audio.unsqueeze(0)
    if init_audio.shape[-1] < audio_vae_length:
        pad = audio_vae_length - init_audio.shape[-1]
        init_audio = torch.nn.functional.pad(init_audio, (0, pad))
    elif init_audio.shape[-1] > audio_vae_length:
        init_audio = init_audio[..., :audio_vae_length]

    init_latents = pipe.vae.encode(init_audio).latent_dist.sample(generator=generator)

    # ------------------------------------------------------------------
    # 6. Add noise scaled by sigma_start (the actual "strength" effect)
    # ------------------------------------------------------------------
    noise = torch.randn(init_latents.shape, device=device, dtype=init_latents.dtype, generator=generator)
    latents = init_latents + noise * sigma_start

    # ------------------------------------------------------------------
    # 7. Rotary positional embedding
    # ------------------------------------------------------------------
    rotary_embedding = get_1d_rotary_pos_embed(
        pipe.rotary_embed_dim,
        latents.shape[2] + audio_duration_embeds.shape[1],
        use_real=True,
        repeat_interleave_real=False,
    )

    # ------------------------------------------------------------------
    # 8. Denoising loop (replicate the pipeline's exact step structure)
    # ------------------------------------------------------------------
    for t in timesteps:
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = pipe.transformer(
            latent_model_input,
            t.unsqueeze(0) if t.ndim == 0 else t,
            encoder_hidden_states=text_audio_duration_embeds,
            global_hidden_states=audio_duration_embeds,
            rotary_embedding=rotary_embedding,
            return_dict=False,
        )[0]

        if do_cfg:
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # ------------------------------------------------------------------
    # 9. Decode and crop to requested duration
    # ------------------------------------------------------------------
    audio = pipe.vae.decode(latents).sample
    waveform_end = int(audio_end_in_s * pipe.vae.config.sampling_rate)
    audio = audio[:, :, :waveform_end]

    return audio
