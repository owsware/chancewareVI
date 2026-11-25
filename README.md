<h1 align="center">
  <img src="https://raw.githubusercontent.com/owsware/chancewareVI/readme-artwork/chancewarevi.png" alt="chancewareVI" width="880">
</h1>

[![GitHub stars](https://img.shields.io/github/stars/owsware/chancewareVI?style=social)](https://github.com/owsware/chancewareVI/stargazers)
[![GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)
[![Windows x64](https://img.shields.io/badge/platform-Windows%20x64-0078D4)](#start-here)

**Run Xbox Series software on Windows. Translate console GPU work to Direct3D 12.** chancewareVI is an experimental source fork of [VIctory](https://github.com/Marvisak/VIctory). It builds on the Radeon renderer and adds NVIDIA RTX compatibility work, with the goal of one renderer for both vendors.

> [!IMPORTANT]
> Bring an authorized, unpacked Xbox Series x64 executable and its accompanying files. A source build also needs the [separate BCPack table input](#bcpack-build-input). No title files, keys, or SDK binaries are bundled. There is no general compatibility guarantee.

🚀 [Get started](#start-here) · 🧩 [How it works](#how-it-works) · 🎮 [Compatibility roadmap](#compatibility-roadmap) · 🧪 [Testing](#testing) · 🤝 [Contributing](#contributing) · 🛡️ [Asset policy](#asset-policy)

## Start here

1. Use **Windows x64** and a current GPU driver. The runtime requests **Direct3D feature level 12.0**; the tested workload also uses **Shader Model 6.6 and mesh shaders**.
2. Install [Rust for Windows](https://rustup.rs/) and [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Select **Desktop development with C++**, including the x64 MSVC tools and a Windows SDK.
3. Prepare the [BCPack build input](#bcpack-build-input), then run **`Setup.cmd`** from the source folder. Select your local Xbox Series executable, or pass `-GamePath` from PowerShell. Setup compiles the runtime and adds the `Chanceware` Start menu shortcut.
4. Start the normal shortcut first. The `Chanceware (Experimental RT)` shortcut also enables the acceleration-structure builder, traversal and lighting resolve; it can be significantly slower or render incorrectly.

Setup installs `chanceware.exe`, `chanceware-play.exe`, and `chanceware-warm.exe` under `%LOCALAPPDATA%\Chanceware`. It stores saves in `state`, shaders in `cache`, and logs in `logs`. `logs\latest.txt` points to the latest play log.

Pass the executable inside **your** local title folder; no fixed folder or filename is required. The caption is **`<title name> — Chanceware`**. It reads a literal name and local 150×150 PNG from `MicrosoftGame.config` if present, with `xbs_logo_150.png` as an optional local image. Otherwise it uses the executable's filename and embedded icon when available. No title images are bundled.

<details>
<summary>💾 Update, keep existing saves, or select another executable</summary>

Run `Setup.cmd` again to update. It reuses the stored executable path and state in the selected install directory. If you are moving from an older VIctory-based launcher, close it, back up `state`, then pass the existing install directory explicitly. A new default install will not import old saves automatically.

```powershell
.\Setup.cmd -InstallDir "D:\path\to\existing\install"
.\Setup.cmd -GamePath "D:\XboxTitles\Example\title.exe"
```

</details>

## Compatibility roadmap

**Goal: one renderer that works on both AMD Radeon and NVIDIA RTX.** The inherited [Radeon baseline](#credits-and-license) is the starting point, not a guarantee that newer changes preserve AMD behavior. The upstream evidence below is from an RX 7900 XTX; the newer development evidence is from an RTX 5080. These are not matched runs, and the current revision has not been retested on Radeon.

**What this fork adds:** fixes to preserve full-wave builder counts on Wave32 GPUs, release completed descriptor tables earlier, and order a shader-written depth update before a later clear. These are concrete changes beyond the pinned Radeon baseline. They are not proof that this fork is faster or more compatible overall than the latest VIctory or DexNom builds.

| Area | AMD Radeon — inherited evidence | NVIDIA RTX — newer evidence | What needs work |
| --- | --- | --- | --- |
| **Launch** | Upstream ran indoor and outdoor scenes. | Fixed a first-present stall caused by synchronous window icon updates. A fresh export presented with empty runtime state and reached an in-engine sequence in a separate replay. | Repeat the same launch and scene checks on Radeon and on more RTX configurations. |
| **Shaders and ray tracing** | The 52 standalone builder shaders created pipelines; selected traversal recorded valid intersections. | The 52 standalone builder shaders created pipelines; three translated prefix scans matched a CPU reference. The runtime reported all 53 registered builder pipelines ready. | Some meshlet and heap-relocation compute pipelines still fail on RTX. Fix full-wave counts and shared-memory ordering, then compare GPU outputs. Builder availability is not lighting correctness. |
| **Lighting, shadows, and materials** | Interior shading still differed in upstream captures, even after the texture queue drained. | Fixed one depth/stencil ordering bug; an RTX 5080 regression passed with the D3D12 debug layer on driver 576.80. Interior exposure and material or shadow artifacts remain. | Find the first differing pass by comparing HDR color, material/depth masks, texture views, bloom, and exposure before and after composition. |
| **Streaming and draw submission** | Texture streaming improved in measured scenes; dark surfaces remained. | Completed GPU work releases temporary descriptor tables before CPU readback. An RTX smoke interval reported no indirect drops, but a refused direct compute dispatch remained. | Retest sustained descriptor pressure. Check residency, view formats, and null bindings against missing surfaces instead of treating queue counters as proof. |
| **Performance** | A 50-frame upstream sample averaged **1.03 FPS** with experimental RT enabled. | Builder-enabled diagnostics sampled **0.3–1.0 FPS**; this was not the full RT preset. | Measure identical workloads on both GPUs. Target CPU memory tracking and readback waits; do not trade correct output for FPS. |

**Clean source verification:** With separately supplied BCPack tables, a fresh Windows export built and passed the asset-policy check. Synthetic loader and D3D12 regressions passed during development. On an RTX 5080 (driver 576.80), the fixed export presented with fresh runtime state. A second run used a local save and a warmed cache, rendered into the opening sequence, and reported all 53 builder pipelines ready. It still showed rendering artifacts and one heavy sample ran at 0.6 FPS. This is a launch and rendering milestone, not visual or performance parity.

### Next milestones

1. **Establish a shared baseline.** Run the same revision, local workload, camera, resolution, and cache conditions on Radeon and RTX. Record normal and experimental RT modes separately.
2. **Fix the first visual and compute failures.** Match intermediate render targets and verify cross-wave GPU output. Use synthetic tests for regressions where practical.
3. **Prove sustained rendering before optimizing.** Recheck streaming and dropped work; then compare median and 95th-percentile frame times with matching output. A matched console comparison is still required before claiming console parity.

## Build and diagnose from PowerShell

Start from a fresh source snapshot or checkout. In PowerShell at its root, use a permitted local copy of the required tables and replace the example title path:

```powershell
$title = 'D:\XboxTitles\Example\title.exe'
.\Initialize-FormatTables.ps1 -FromCheckout 'D:\path\to\authorized\checkout'
cargo build --locked --profile diag -p victory --features diagnostics --bins
.\target\diag\chanceware.exe inspect $title --imports
.\launch_with_metrics.ps1 -GamePath $title -RuntimeBinary '.\target\diag\chanceware.exe'
```

`inspect` does not run the executable. The diagnostics launcher executes it and prints the log path. A successful inspection or a visible window is only a smoke check, not proof that a title renders correctly.

`victory` is still the internal Cargo package name. Existing `VICTORY_*` settings and [`victory.toml`](victory.example.toml) remain compatible. For a release build, replace `--profile diag` with `--release`. To warm the installed shader cache, run `chanceware-warm.exe`; graphics pipeline creation still depends on the draw state seen during play.

<details>
<summary>🛠️ Start development without title files</summary>

From the source root, run the policy check first, then prepare the build input and run the tests. The policy checker uses Git's ignore rules in a checkout, but checks every source file in an extracted ZIP.

```powershell
cargo xtask policy
.\Initialize-FormatTables.ps1 -FromCheckout 'D:\path\to\authorized\checkout'
cargo test --locked -p victory-loader --test load_probe
cargo test --locked -p victory-gpu --features diagnostics --lib depth_clear_after_shader_write_preserves_both_the_clear_and_other_texels
```

The loader probe builds and runs an original synthetic fixture. The last test exercises D3D12 clear ordering on a compatible host adapter. Neither needs a title executable, save, shader cache, or developer machine layout. Follow [AGENTS.md](AGENTS.md#work-from-a-current-base) to start a topic branch and record results.

</details>

## Testing

<details>
<summary>🔧 Setup fails, the window closes, or warmup finds no shaders</summary>

- For setup failures, check the x64 C++ workload, Windows SDK, and the [BCPack input](#bcpack-build-input). For a missing `VCRUNTIME140.dll`, install the [Microsoft x64 C++ redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
- For a failed launch, start with the log named in `%LOCALAPPDATA%\Chanceware\logs\latest.txt`. `chanceware.exe inspect <file>` reads PE headers without running the executable. `chanceware.exe census --missing <directory>` lists uncovered console imports.
- Shader warmup reads only local files. It can populate the shader cache and build some compute pipelines, but an empty scan does not prepare the cache for play.

</details>

<details>
<summary>🧪 Compare Radeon and RTX, or report a regression</summary>

Use the same revision, resolution, save, and camera. Record GPU and driver versions and keep normal and Experimental RT runs separate. Leave pipeline-skipping overrides off. Measure steady-state frames separately from cold streaming and shader compilation.

For a builder-only smoke test, set `$env:VICTORY_BVH_BUILDER = '1'` in a fresh PowerShell session and run the normal diagnostics command above. This is not the Experimental RT shortcut, which also changes traversal and resolve settings. Record the run as builder-only.

Compare a static interior for exposure and materials, then a repeatable streaming route for geometry, textures, and shadows. If the final frame differs, trace back to the first different pass or resource. Compare median and 95th-percentile frame time, refused pipelines, dropped operations, CPU memory tracking, and readback waits. Radeon is a comparison point, not an Xbox reference.

```powershell
.\launch_with_metrics.ps1 -GamePath "D:\XboxTitles\Example\title.exe" -D3d12Debug -Dred
.\target\diag\chanceware.exe translate "D:\local-debug\test-shader.dxbc" --validate --pipelines
```

Do not use debug-layer timings for performance comparisons. DXIL validation and pipeline creation do not prove correct pixels or buffer contents. Share the revision, hardware, settings, and sanitized error codes or counters; do not attach raw logs or captures.

</details>

## How it works

chancewareVI is a compatibility layer, not a virtual Xbox. The title's x86-64 CPU code runs natively. The runtime supplies the console APIs the Windows host does not provide and translates GPU work to desktop Direct3D 12.

1. **Load the executable.** `victory-pe` reads the PE image. `victory-loader` maps it, applies relocations, sets up thread-local storage, and binds imports. The loader tries the layer's console implementation first, then an allowed Windows API, then a DLL from the local title folder. Unsupported imports can stop execution if called.
2. **Provide console services.** `victory-hle` implements the supported portions of console memory, storage, input, audio, and runtime APIs. It maps console file paths to local folders and keeps persistent state separate from shader caches. Online services are not promised.
3. **Translate graphics.** `victory-gpu` maps D3D12.X resources, descriptors, barriers, and command lists onto desktop D3D12. `victory-shader` recompiles console RDNA2 programs to DXIL as pipelines are created. `victory-formats` handles console texture layouts and decoding. The experimental ray path runs translated compute traversal; it does not replace the title's acceleration structures with host DXR objects.

<details>
<summary>📚 Which files and libraries do I need?</summary>

| Source | What it provides |
| --- | --- |
| **Your local title folder** | The compatible Xbox Series executable and the original accompanying data and DLLs it imports. Keep the supplied folder layout intact. There is no universal set of title DLLs. `MicrosoftGame.config` and local artwork are optional for the window caption and icon. |
| **Windows and GPU driver** | Desktop `d3d12.dll` and `dxgi.dll`, Win32 and the Universal CRT. Supported paths use host XInput for controllers, XAudio2 (`xaudio2_9.dll`) for audio, and Media Foundation for media. Use Windows components from Windows, not DLL downloads. |
| **Microsoft x64 C++ Redistributable** | Install it from Microsoft if a required `vcruntime140*` or `msvcp140*` library is missing. |
| **Source build** | The Rust toolchain, x64 MSVC tools, Windows SDK, Cargo dependencies, and the [separate BCPack table input](#bcpack-build-input). Building the layer does not require the Xbox GDK. |
| **Optional diagnostics** | `dxil.dll` from the Windows SDK for `translate --validate`. The Windows D3D12 debug layer is for diagnosing graphics errors, not normal play. |

Names such as `kernelx.dll`, `xmem.dll`, `xgameruntime.dll`, `d3d12_xs.dll`, `gameinput.dll`, and `xcrdapi.dll` are console import names. The layer implements supported portions itself; do not download replacement console SDK DLLs. Its storage implementation uses local file I/O, zlib, and BCPack for supported requests.

To check your own files without executing the title:

```powershell
.\target\diag\chanceware.exe inspect "D:\XboxTitles\Example\title.exe" --imports
.\target\diag\chanceware.exe census --missing "D:\XboxTitles\Example"
```

A missing import is not automatically a missing file; the layer may not implement it. A clean static report does not guarantee coverage of DLLs loaded later or correct runtime behavior.

</details>

## Contributing

Start with [AGENTS.md](AGENTS.md) for local branches, review, and [issue triage and crash debugging](AGENTS.md#triage-issues-and-crashes).

**Before pushing:** Confirm the destination and review every outgoing commit, including its message. Keep title names, build fingerprints, executable offsets, symbols, research notes, and personal paths out of public commits, issues, logs, and PRs. Use your own authorized local files for testing. Share synthetic fixtures and sanitized platform-level findings, not title data or private prompts.

**In your pull request:** Explain the change, list the checks you ran and their results, and note known limitations. For runtime tests, include [device specifications and the exact graphics driver](AGENTS.md#device-specifications-for-runtime-reports) and describe what you observed. Run `cargo xtask policy`, preserve licensing and attribution, and get human review before merge. Do not push changes directly to the public default branch.

**AI assistance:** Say `AI assistance: none`, or list the tool or client, exact model ID or version (`not shown` if unavailable), and what each materially contributing model did. Explain what a human reviewed and how the change was verified. Do not attach prompts, transcripts, or local inputs.

Optional tools: **Ponytail** for small changes, **Caveman** for concise notes, **Graphify** for source navigation, and **REA** or **IDA Pro MCP** for authorized local analysis. Keep private analysis files out of hosted tools and shared memory.

## Asset policy

Publish only redistributable compatibility-layer source and original synthetic fixtures. Do not upload title executables, assets, saves, symbols, screenshots, disassembly, shader dumps, logs with local paths, or data derived from a title. Keep all user-supplied inputs and generated data outside the checkout. `cargo xtask policy` is a source-tree check, not a license or history review.

### BCPack build input

The BCPack decoder embeds separately supplied tables that the upstream GPL notice does not cover. Use an authorized local copy with `Initialize-FormatTables.ps1 -FromCheckout`, or put that copy at `crates/victory-formats/src/bcpack-tables.bin`. With neither input present the script tries the pinned upstream URL; availability and permission to use that copy are not guaranteed. The SHA-256 check verifies bytes, not rights. Without a verified copy, the current build cannot be reproduced. Do not distribute executables embedding the tables without separate clearance.

<details>
<summary>📦 Maintainers: review a source-only release</summary>

Do not push or mirror private development history. A [GitHub-native fork](https://docs.github.com/en/pull-requests/reference/forks) retains public upstream history, including older copies of the separately supplied BCPack tables. The current tree and source archive omit those tables; this does not remove them from history or grant redistribution rights. `.gitignore` does not remove tracked files. The source archive includes only this README and `AGENTS.md` as Markdown.

```powershell
cargo xtask policy
$archive = Join-Path $env:TEMP 'chancewareVI-source.zip'
git archive --format=zip --prefix=chancewareVI/ --output="$archive" 'HEAD^{tree}'
```

Review the extracted tree, embedded data, fixtures, workflows, and notices. Build and smoke-test it with a permitted local dependency. Preserve `LICENSE`, the dated modification notice, and in-source third-party notices. If object code is ever distributed, provide its exact Corresponding Source under GPL §6 and separately confirm the rights to every embedded dependency. Rebranding and exclusion rules do not guarantee protection from copyright claims.

</details>

## Credits and license

chancewareVI is independent and is not affiliated with or endorsed by any console manufacturer or title publisher. Source is [GPL-3.0-or-later](LICENSE).

**Modification notice — 25 November 2026.** The chancewareVI contributors modified VIctory, including RTX shader translation, D3D12 resource and descriptor handling, and the Windows launcher. The modified source remains GPL-3.0-or-later and comes without warranty. Applicable third-party notices remain in source.

- [Marvisak/VIctory](https://github.com/Marvisak/VIctory) and its contributors created the original runtime.
- [DexNom/VIctory](https://github.com/DexNom/VIctory) at [`2e01ab4`](https://github.com/DexNom/VIctory/commit/2e01ab4760db91820ef5f9a9f85761f83cd6b5d0) is the Radeon development baseline. [VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD) provided selected changes. [victory-world-runtime](https://github.com/ZSlayerHQ/victory-world-runtime), including work from triippywhale and asdaasdsaddf, provided the earlier rendering baseline.
- The [XCL contributors](https://github.com/xcl-dev/xcl) built an independent compatibility layer. Its published work served as a technical reference, and its README informed this one.
- OpenAI Codex (`openai-codex/gpt-6.2-sol`) assisted with Windows launcher changes, verification, and contributor documentation. Local binary analysis used IDA Pro MCP.
- [AMD's addrlib in Mesa](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/amd/addrlib) informed [texture layout support](crates/victory-formats/src/texture.rs). The [Mesa ACO opcode definitions](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-22.3.0/src/amd/compiler/aco_opcodes.py), credited to Valve, are the source for the generated [GFX10 opcode tables](crates/victory-shader/src/gfx10/opcodes.rs). Both notices remain in source.
