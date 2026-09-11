# Description feature implementation and validation

September 11, 2026. The first working version is implemented; this supersedes the original plan's unverified-runtime notes.

## Delivered

- `cb_color_correct/description.py`: local-only LM Studio native v1 client, full-resolution edit rendering and bounded PNG encoding, model discovery/load/unload, cancellable Qt worker.
- `cb_color_correct/description_panel.py`: editor output, Write description/Copy/Cancel, source selection, model settings, ownership tracking, and result invalidation.
- `main.py`: snapshots, preview-debounce handling, GPU-job exclusion, asynchronous description shutdown, and editable-field keyboard shortcuts.

UI update: the description panel now lives in the collapsible right Tools sidebar, in a Description tab beside Modifications. It no longer occupies space below the canvas. Its controls stack vertically for the sidebar, and settings remain scrollable. The M shortcut expands/collapses the shared sidebar. All 37 tests still pass after the relocation.
- `tests/test_description.py` and `tests/test_main_description.py`: network error/cancel tests against a local test server, payload/source tests, and UI behavior checks.
- `tests/manual_description_smoke.py`: opt-in synthetic-image test against the real local model, including UI screenshots and sampled GPU usage. Never uses the user's current artwork.

## Confirmed locally

Restricted installation/runtime access was resolved through execution approval. LM Studio 0.4.24 is installed; its server was started on 127.0.0.1:1234. Native model discovery reports the downloaded Q4_K_S as vision-capable with reasoning off supported. Its API key is `qwen3.5-9b-the-defiant-fable-uncensored-heretic-neo-imatrix-max-mtp` (the repository/API name contains MTP; the actual downloaded file is the regular variant).

The real GUI worker sent a synthetic red circle, blue square, green triangle, and printed title. The model correctly described these elements. One run took 12.95 seconds including model loading, with 1,870 MiB baseline and 10,347 MiB peak sampled total GPU usage. The test unloaded the instance it loaded. The local server remains available.

All 37 automated tests passed, including the pre-existing censor/upscale tests. Offscreen UI screenshots were inspected at the default 1200×800 size. The smoke-test script explicitly registers a Windows font because Qt's offscreen platform otherwise rendered missing-glyph boxes; this is confined to the test harness.

## Intentional first-version choices

Uses QtNetwork, already bundled with PySide6-Essentials, instead of requests. A short worker-thread timer aborts network requests on cancellation or deadline, including while waiting for HTTP headers. No additional dependency is required. The UI displays completed text rather than incremental tokens; the previous draft survives failure. Local request deadlines are 15 seconds for discovery, 180 for loading, 300 for description, and 60 for unloading. CPU rendering is checked for cancellation before/after and must finish before its worker exits.

Only instances loaded successfully by this app session are eligible for its unload button. Cancellation during loading can leave a backend instance whose ID the client never received; manage it in LM Studio. Aborting the HTTP request is verified locally, but backend inference cancellation is not guaranteed. Sequential jobs do not automatically release other applications' GPU allocations.

## Optional follow-up work

No additional agent work is required for the implemented copy/paste workflow. Useful future improvements, if requested:

- Tune the editable prompt against the user's actual artwork and preferred writing style. Only a synthetic image has been used for live inference so far.
- Test real ComfyUI upscaling after description/unload under the user's normal GPU workload. Simultaneous job exclusion is covered by UI tests; full sequential ComfyUI inference was not run for this change.
- Stream native message events while keeping reasoning separate, and verify backend cancellation semantics for this exact LM Studio runtime.
- Add an explicit server-start action if desired. Currently the user starts LM Studio's server; no runtime installation or automatic downloads occur.
- Optionally add text export or per-image draft persistence. Current output is intentionally a session-only editable field for manual copying.

See README.md for the user workflow. New agents should read AGENTS.md before accessing restricted installations; request access before diagnosing missing or faulty runtimes.
