# Local image description feature

Research date: September 11, 2026. This document records the original implementation plan.

Implementation update: the first working version is now delivered. See `DESCRIPTION_FEATURE_HANDOFF.md` for verified runtime results, implementation choices, and optional follow-up work. The original research below is retained for context.

## Recommendation and verified prerequisites

Use LM Studio as the local inference host, with the existing Python `requests` dependency calling its native REST API. Add a **Write description** button and an editable `QPlainTextEdit` below the editor preview. Keep inference in a background worker, following the application's existing Qt worker approach.

The selected model is suitable for an initial trial: its publisher documents working image support with a separate projector in the same folder. This establishes advertised capability, not measured description quality. “Uncensored” is the publisher's characterization, not a guarantee of output behavior or accuracy. [Model card](https://huggingface.co/DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF)

Read-only inspection found both files locally:

```text
C:\Users\cbein\.lmstudio\models\DavidAU\Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF\
  Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-Q4_K_S.gguf
  mmproj-F32.gguf
```

The path in the request had extra separators around the quantization suffix; the actual filename contains `Q4_K_S`. File sizes are 6,551,514,400 and 1,824,062,016 bytes, approximately 7.80 GiB combined. This particular model filename is the regular variant, without an MTP suffix; speculative decoding is unnecessary for the feature.

The machine reports an NVIDIA RTX 3080 Ti with 12,288 MiB VRAM. Trying this quantization with a modest context is reasonable, but file sizes do not establish runtime memory consumption: vision processing, cache, buffers, the desktop, and ComfyUI also consume memory. Start at 8,192 context tokens and measure peak VRAM and latency. Reduce GPU offload through LM Studio if necessary; avoid loading a second instance.

`lms.exe` exists. `lms ls --json` could not connect to its daemon and reported an installation-discovery error; `lms server status` reported no running server. These checks do not prove the installation is missing or faulty. The installed application/runtime version, API model identifier, successful projector loading, and actual inference remain unverified. If restricted access prevents checking them during implementation, request the necessary access before diagnosing a fault, per AGENTS.md.

## Integration design

1. **Verify the runtime first.** Open the installed LM Studio, check its version and llama.cpp runtime support for Qwen3.5, select this GGUF, and verify image attachment works with the existing projector. Run one ordinary image description before integrating the UI. The native v1 API requires LM Studio 0.4.0 or later; an older version should produce an actionable update message. No automatic model downloads or runtime installation. [REST API version and endpoints](https://lmstudio.ai/docs/developer/rest)

2. **Connect and discover the model.** Default to `http://127.0.0.1:1234`. Provide a compact settings area with server address, Refresh models, model selection, Load model, and Unload model. Persist non-secret settings using the existing `QSettings`. Discover the real model key through `GET /api/v1/models`, match the downloaded quantization, and inspect vision capability and loaded instances. Do not submit the Windows filesystem path as an assumed API model ID. [Model discovery](https://lmstudio.ai/docs/developer/rest/list)

3. **Load on demand.** Write description should reuse the selected loaded instance or load the selected model through `POST /api/v1/models/load`; show Loading model status. Track the returned instance ID and which instances this app loaded. Do not unload unrelated sessions. [Model loading](https://lmstudio.ai/docs/developer/rest/load)

4. **Server lifecycle.** For the first version, have LM Studio running with its local server enabled. Document `lms server start --port 1234 --bind 127.0.0.1`; do not require CORS for the desktop client. A hidden, explicit Start local server action can follow once installation discovery is verified. Loading a model and starting the server are separate operations. [Server CLI](https://lmstudio.ai/docs/cli/serve/server-start)

5. **Generate text locally.** Use `POST /api/v1/chat` with text plus a base64 image data URL, `store: false`, a fresh request for each image, and no integrations. Start with a 512-token output budget. Request reasoning off only when the model advertises that option. Display only message output; keep reasoning separate. Streaming can populate the widget incrementally using SSE. [Image inputs, output types, and request settings](https://lmstudio.ai/docs/developer/rest/chat)

The official Python SDK is also viable and directly supports image bytes, but introduces another dependency. Native REST fits the current codebase and supplies model management and explicit image requests. Embedding llama.cpp directly would add runtime packaging, GPU configuration, and projector management that LM Studio already handles. [SDK image support](https://lmstudio.ai/docs/python/llm-prediction/image-input)

## UI and image behavior

- Put a collapsible Description panel below the image, keeping the artwork visible. Include **Write description**, **Cancel**, **Copy**, status text, and an editable multiline output widget. Disable generation until an image is loaded and while another description runs; retain the prior successful text if regeneration fails.
- Default the source selector to **Edited artwork (without censor blur)**, matching the existing upscale source and intended artwork-description use. Offer **Censored preview** explicitly. Label the selected source clearly. Neither mode includes split comparison, zoom cropping, circle outlines, or other UI overlays.
- Capture an immutable snapshot of image pixels and current filter settings when clicked. Reuse the color-processing pipeline underlying `_render_color_corrected_rgb8()`; use the censor-rendering behavior for the alternate source. Perform expensive rendering and encoding in the worker rather than calling the full-resolution renderer on the UI thread.
- Resize the rendered result proportionally to a proposed 1,280-pixel longest edge, without upscaling small inputs; encode an in-memory PNG without metadata. Validate this resolution against detail quality and memory usage in the smoke test.
- Tag requests with an image/edit revision. If the image or relevant edits change, mark existing text as out of date and prevent an old worker result from becoming the new image's description. Cancel or discard pending output on image replacement. Each request is independent, without previous-image chat history.
- Use an editable prompt, initially: “Write a concise artwork description for a gallery listing. Describe visible subjects, composition, colors, lighting, and mood. Stay grounded in the image; do not invent backstory. Return only the description.” Target roughly 80–150 words. Prompt controls can adjust style and length. Review actual model behavior on the user's intended artwork during acceptance testing.
- Keep image data on the configured loopback service, bypass HTTP proxy environment settings for that client, reject remote endpoints/redirects in this local-only feature, and avoid logging image payloads or generated text. Support an optional authentication token if the server requires it, without writing it to logs or ordinary settings. Copy is explicit; saving descriptions beside artwork can be a later addition.

## Files and worker lifecycle

| File | Planned changes |
| --- | --- |
| `cb_color_correct/description.py` | Settings/request dataclasses, local API client, image encoding, response handling, and a Qt description worker with status/text/error/completion signals. |
| `main.py` | Panel construction, settings persistence, image/edit snapshotting, request revisions, Copy/Cancel handlers, and shutdown integration. |
| `tests/test_description.py` | Mock API tests for discovery/loading, image payloads, streaming boundaries, errors, and cancellation. |
| `tests/test_main_description.py` | Offscreen Qt tests for correct image source, stale responses, enabled states, and shutdown. |
| `README.md` | Setup instructions, model/projector requirements, source selection, and troubleshooting. |

`upscale.py` can remain focused on ComfyUI. Coordinate the two operations in `MainWindow`: prevent simultaneous description/upscale jobs initially. Serialization alone does not free resident GPU models, so provide explicit unload controls and explain out-of-memory recovery. Track model ownership before considering automatic unloading; do not stop an external ComfyUI or LM Studio process.

Use bounded connection/read timeouts and a total request deadline. Keep all widget mutations on the GUI thread. Cancel should close the stream and discard subsequent output; test whether that also stops backend computation, without assuming it does. Handle cancellation during model loading as well as generation. Window close must request cancellation and wait asynchronously for worker cleanup, retaining the worker until it ends; do not copy an unbounded GUI-thread `wait()` pattern.

## Acceptance and implementation order

1. Runtime smoke test: exact GGUF/projector loads, API reports vision capability, a known image receives an accurate description; record application/runtime versions, model key, load time, generation time, and peak VRAM.
2. Implement the client and worker, with mocked checks for server offline, authentication failure, missing/non-vision model, unsupported API, load failure/OOM, malformed or empty output, timeout, and cancellation. A failed vision request must never silently retry as text-only.
3. Add the panel and revision tracking. Verify descriptions reflect color edits and source selection, the UI stays responsive, old results cannot overwrite a newly loaded image, and copying returns plain text.
4. Test generation followed by ComfyUI upscaling on the 12 GB GPU, including resident-model unload recovery. Check window close during both model loading and generation.
5. Run the existing tests plus the new focused tests, then manually verify normal, portrait, landscape, transparent, and large images. Confirm request traffic targets only the local service. Document any measured quality or memory limitations.

At the time of the original research, only this plan was delivered. See the implementation update above for current status.
