# MLB Studio V1.0

Visual model studio for building, training, generating, serving, and managing MLBricks models.

> **MLB Studio V1.0 package:** use the `mlb_studio` Python module and install from the `MLBricks-Studio` GitHub repository.

## Install on Kaggle

```python
%pip install -U "git+https://github.com/MlBricks/MLBricks-Studio.git"
```

Then:

```python
from mlb_studio import Builder

builder = Builder()
builder
```

MLB Studio V1.0 uses Jupyter's standard HTML representation protocol instead of `anywidget`, so Kaggle does not need a custom frontend widget module.


## V1.0 production hardening

- Legacy PyTorch checkpoints use restricted `torch.load(..., weights_only=True)` loading by default. Pickle-based legacy files are blocked unless unsafe legacy loading is explicitly enabled for a trusted file.
- Projects loaded from local files, cloud bundles, or Hub are **untrusted for the current session**. Embedded Python/API imports are blocked until you review the project and call `builder.trust_project()`. Trust is never serialized into a project file.
- Builder artifacts are stored under `mlbricks_workspace/` rather than a top-level `mlbricks/` directory, preventing collisions with the installed Python package namespace.
- Cloud ZIP extraction rejects path traversal and symlink entries.
- The built-in model server binds to `127.0.0.1` by default, requires an API key by default, uses same-origin CORS by default, escapes model names, adds browser security headers, limits request size/prompt length/token count, enforces concurrency and rate limits, and stops overlong generation requests.
- For permanent internet-facing deployment, place the server behind a production TLS/reverse proxy. The optional ngrok tunnel is intended for temporary notebook sharing and requires API-key protection.

## MLB Studio V1.0 API editor layout

- **Add Function** lives in the API Component top toolbar beside **Add Module**.
- Adjacent graph nodes connect directly side-to-side; only long/skip routes use outside rails.
- Persistent canvas instruction/helper bars are removed across all Studio workspaces; only temporary action-specific port guidance is shown while wiring.
- API nodes now distinguish **Function**, **Static Method**, **Class Method**, **Instance Method**, and **Create Object** calls.
- Instance-method nodes can **Create New Object** or **Use Existing Object** from the same API Component.
- Objects use stable internal IDs tied to their source node, so renaming the visible node does not break references.
- A created instance is constructed once per compiled API graph and reused by later nodes, preserving object state without recreating it.
- API results can optionally be registered as reusable objects for later nodes.

MLBricks Kit remains a separate dependency while Studio integration is being validated. The current compatible distribution is `mlbricks-kit==1.0.0b1`; its Python import namespace remains `mlbricks`.

## v0.1.3 UI

Compact narrow layer cards, full left component palette, right-side API inspector, layer/model tabs, and a layout matching the MLB Studio mockup more closely.


## v0.2.0 — Layer-by-layer workflow

The Builder now includes:

- Compact left-to-right layer cards matching the MLB Studio design direction.
- Clickable input/output ports for manual ComfyUI-style connections.
- Shift + input-port connection for a residual/skip edge.
- Beginner-friendly Auto Connect toggle.
- Component-specific MLBricks API inspector on the right.
- Nested reusable layer architecture with double-click / Open Architecture.
- Override and Save As New workflow for custom components.
- TinyStories 30M starter preset:
  - 6 nested model layers
  - target ~30M parameters
  - 512 context
  - batch size 16
  - TinyStories dataset
  - each model layer contains ESA → RMSNorm → FFN → Residual
- `Builder(preset="tinystories")` to open the starter directly.

Example:

```python
from mlb_studio import Builder

Builder(preset="tinystories")
```

The preset's ~30M value is an architecture target/estimate. Exact trainable parameters should be calculated by the installed MLBricks runtime because implementation details, vocabulary size and weight tying can change between MLBricks versions.


## v0.2.1 — Real MLBricks API inspector

The right inspector is built from the currently installed MLBricks package with `inspect.signature`. No MLBricks algorithms are copied into Builder. Updating/reinstalling MLBricks updates the available constructor parameters shown by Builder.

Examples found in MLBricks Kit 1.0.0b1 include `ESA(embd, head=4, ..., backend="auto", precision="fp16", compass="auto", ..., device="auto")`, `FFN(hidden_size, intermediate_size=None, activation="gelu", ...)`, `StateAwareFFN(d_model, state_dim=256, ...)`, and `Bolt(d_model, num_heads, latent_dim=32, ...)`.

Use `builder.component_api("esa")` to inspect the metadata in Python.


## v0.3.0

Full dark ComfyUI-style MLB Studio frontend with layer-by-layer layout, curved manual connections, residual edges, minimap, dark inspector, real installed MLBricks API forms, nested custom components, and TinyStories 30M preset.


## v0.3.1 — Kaggle stale-renderer + real API fix

This release fixes a notebook-specific bug in v0.3.0. Kaggle keeps JavaScript
globals alive in the browser page. Older Builder outputs had registered
`window.MLBricksBuilder`, and v0.3.0 incorrectly returned early when it found
that global. The result was new CSS applied to an old renderer.

v0.3.1 always replaces the old renderer before mounting, and shows `v0.3.1`
visibly in the Builder header.

It also aligns the TinyStories starter with the real MLBricks Kit 1.0.0b1 constructor
arguments from the uploaded library:

- `ESA(embd=384, head=6, batch=16, block=512, ...)`
- `Embedding(vocab_size=32000, embedding_dim=384)`
- `RMSNorm(normalized_shape=384, ...)`
- `FFN(hidden_size=384, intermediate_size=1536, ...)`
- `Residual(dropout=0.0)`
- `LMHead(hidden_size=384, vocab_size=32000, ...)`

Run:

```python
builder = Builder(preset="tinystories")
builder.diagnostics()
```

to verify which real MLBricks APIs were discovered.


## v0.3.2 — Source-backed API schema

The API inspector no longer depends on importing every MLBricks component
successfully at notebook startup.

This release's API schema was generated directly from the supplied
`MLBricks Kit 1.0.0b1` source API. Runtime introspection is
still attempted; when it works, it takes precedence. If it does not work,
the exact source-derived constructor/config schema remains available.

Examples:

- ESA: `embd`, `head`, `batch`, `block`, `backend`, `precision`, `compass`,
  `dropout`, `gate_min`, `gate_max`, `eps`, `device`, `auto_compile`,
  `compile_mode`, `auto_move_input`, `strict_checks`
- Bolt: `d_model`, `num_heads`, `latent_dim`, `bias`, `dropout`, `causal`,
  `backend`, `autotune_kernels`, `eps`, `use_sdpa`, `position`,
  `native_full_sequence`
- FFN: `hidden_size`, `intermediate_size`, `activation`, `dropout`, `bias`,
  `gated`, `device`, `dtype`
- StateAwareFFN, MicroVirtualFFN, VirtualStateAwareFFN
- RMSNorm, LayerNorm, Residual, ResController
- Previous Value Buffer (zero-init/hold utility for physical-depth state wiring)
- Vesa/VesaConfig and VisionBolt/VisionBoltConfig
- ElasticBit, ElasticLinear, ElasticEmbedding
- RoPE, LearnedPosition, SinusoidalPosition
- Brick and Bricks

Builder-owned input/output nodes also have their own valid interface and never
display “API unavailable”.


## v0.3.3 — residual pipeline + custom ports

This release adds the workflow behavior requested for kids and custom builders:

- residual connections are drawn like a top pipeline / bus
- auto-connect still builds the normal left-to-right layer flow
- users can also create manual custom connections between layers
- custom layers can expose a chosen number of input and output ports
- `Residual Add` defaults to 2 inputs and 1 output
- nested custom layers save and reload their chosen public interface
- hold **Shift** while connecting to create a residual pipeline


## v0.3.4 — per-node residual ports and removable connections

This release removes the residual top bus and replaces it with clearer
per-node residual ports:

- every node has a **top residual input port**
- every node has a **bottom residual output port**
- side ports remain for the normal left-to-right data flow
- residual connections are created using **bottom residual port → top residual port**
- users can create multiple residual/skip connections
- the inspector now lists all connections for the selected node and gives a
  **Remove** button for each connection
- a **Remove All Links** action is also provided


## v0.3.5 — fixed 3-lane left-to-right ports

Every node now has exactly three inputs on the left and three outputs on the right:

- top: **Skip In / Skip Out** — residual and skip connections, routed above intervening nodes
- middle: **Main In / Main Out** — normal model flow; Auto Connect uses this lane
- bottom: **Extra In / Extra Out** — auxiliary/custom signals, routed below intervening nodes

Multiple skip connections are supported. Skip routes receive separate vertical offsets so multiple residual paths stay readable. Existing per-connection **Remove** buttons and **Remove All Links** remain available in the Inspector. Custom/nested layers use the same fixed three-lane public interface.


## v0.3.6 — compact Kaggle workspace

The Builder is now constrained to an app-like notebook height instead of
growing with the Brick Library or Inspector.

- left Brick Library scrolls independently
- right API Inspector scrolls independently
- Input/Core/Advanced/Position/Heads/Outputs sections are clickable
  collapse/expand controls
- Advanced, Position, Heads, and Outputs start collapsed to save space
- search temporarily expands matching categories
- all categories can be expanded and the library remains scrollable
- the center graph canvas keeps the remaining vertical space
- large blank space above/below the node row was reduced
- Presets/Graph Info/Compute/Shortcuts moved into a compact **Model Details**
  drawer that is collapsed by default


## v0.3.7 — sketch-accurate 3-in / 3-out node terminals

The six terminals now match the hand-drawn design instead of putting all three
inputs on the left and all three outputs on the right.

Each node has:

- **Top edge**
  - input near the top-left
  - output near the top-right
- **Middle**
  - input on the left side
  - output on the right side
- **Bottom edge**
  - input near the bottom-left
  - output near the bottom-right

This keeps all signals left-to-right while allowing:

- top routes to travel above intervening nodes
- normal main routes through the center
- bottom routes below intervening nodes

The Builder notebook workspace is also about 20% taller than v0.3.6.


## v0.3.8 — persistent canvas HUD spacing

The canvas overlays no longer move when the graph is horizontally scrolled:

- **Blueprint** stays pinned above the node row
- the port-layout instruction banner stays pinned near the bottom
- extra vertical runway keeps a visible gap between Blueprint/top routes and nodes
- **Model Details** is now an overlay drawer
- expanding Model Details does **not** resize the canvas or move the node row/banner
- when Model Details opens, it covers the lower canvas/banner area as requested
  instead of pushing the banner upward


## v0.3.9 — horizontal-only HUD persistence

Blueprint and the node-layout instruction banner now remain horizontally
persistent during left/right graph scrolling, but they are **not vertically
pinned**.

The vertical layout behaves as one continuous stack:

`Blueprint → fixed gap → node row → fixed gap → instruction banner`

So when the user scrolls the canvas vertically, all three move together and
their relative vertical distances stay unchanged.

Model Details remains an overlay drawer and does not reflow the graph.


## v0.3.10 — zoom-safe graph + Undo / Redo

### Zoom fix
The graph wrapper now grows to the scaled visual dimensions. At 110–150% zoom,
right-side nodes, bottom ports, routed edges and the instruction banner remain
inside the real scrollable canvas instead of being clipped or overlapping due
to CSS transform layout mismatch.

### Undo / Redo
The toolbar now has **Undo** and **Redo** buttons backed by a 60-step model
history. History includes node add/delete/duplicate, connections, connection
removal, API parameter edits, custom bricks, TinyStories preset loading, Clear,
and Auto Connect changes. Zoom and sidebar UI state are intentionally not model
history operations.


## v0.3.11 — insertion at selection + layer reordering

- Brick Library labels/descriptions are consistently left-aligned and no longer overlap.
- Adding a built-in or custom brick inserts it **immediately after the selected node**.
- If no node is selected, the new brick is appended to the end.
- With Auto Connect enabled, the middle Main lane is rebuilt automatically after insertion,
  deletion, duplication, or movement while Skip and Extra connections are preserved.
- The Inspector now provides **Move Left** and **Move Right** controls for the selected layer.
- Move/add/delete/reorder operations are included in Undo/Redo history.


## v0.3.12 — empty custom-brick shells + unique names

Custom-brick creation is now isolated from the current model canvas.

- **Create Custom Brick** always creates an empty nested component:
  - `nodes = []`
  - `edges = []`
- Existing model nodes/siblings are never copied into a newly created custom brick.
- The new empty custom brick opens immediately for internal editing.
- Custom brick names must be unique after trimming spaces and ignoring case.
  For example, `SAM`, `sam`, and ` Sam ` are treated as the same name.
- The same unique-name rule applies to **Save As New**.
- Empty shells are visibly marked as `Empty` in **My Bricks**.


## v0.4.0 — Data + Text Processing pipeline

MLB Studio can now design the dataset path together with the model.

### Data sources

- **Hugging Face Dataset**
- **Kaggle Dataset**
- **URL Dataset**
- **Local Dataset**

### Text processing

- **Text Processing** — Unicode normalization, whitespace cleanup, lowercase,
  empty filtering, minimum/maximum text length
- **Train / Test Split** — configurable train/test ratio, seed and shuffle
- **Tokenize Text** — Hugging Face tokenizer, context length, truncation,
  padding and special-token controls

Example visual pipeline:

`Hugging Face Dataset → Text Processing → Train/Test Split → Tokenize Text → Embedding → Model`

The Inspector shows runnable **DATA PYTHON** for data/text nodes using
`mlb_studio.data`. These are Builder APIs, not fake `mlbricks` imports.

### Save / Load Design

The top **Save** button now downloads the complete design as
`<project>.mlbricks.json`, including:

- data pipeline
- preprocessing settings
- train/test split configuration
- model nodes and connections
- custom bricks
- project settings

The **Load** button restores the same design from JSON.

### Optional data dependencies

Keep the normal Builder installation light. Install data features with:

```bash
pip install -U mlb-studio
```

or, when installing Builder from GitHub in Kaggle, install:

```bash
# Data Processing dependencies are installed automatically with mlb-studio
```

Authentication tokens/credentials are deliberately **not** stored in design
files. Hugging Face and Kaggle use their normal notebook/environment login.


## v0.4.1 — Text Input as the kid-friendly text workspace

For beginners, all common text preparation now lives inside **Text Input**.

Click Text Input to get four simple collapsible sections:

1. **Text Source** — Manual, Hugging Face, Kaggle, URL, or Local File
2. **Clean Text** — cleanup/normalization/filtering
3. **Train / Test** — split percentages, seed, shuffle
4. **Tokenization** — tokenizer, context, truncation, padding

The UI is conditional: selecting Hugging Face shows Hugging Face fields;
selecting Kaggle shows Kaggle fields; turning a processing step off hides its
advanced controls.

Standalone Data/Text Processing bricks remain available for advanced workflows,
but those categories start collapsed. The beginner path is one Text Input node.

The Text Input configuration is saved in the normal `.mlbricks.json` design.
Its runnable code uses the real Builder helper
`mlb_studio.data.prepare_text_input(...)`; it never generates a fake
`from mlbricks import Text Input`.


## v0.5.0 — separate Model Builder and Data Processing workspaces

The category-chip row has been replaced by a simple **Build Workspace** selector:

- **Model Builder**
- **Data Processing**

Each workspace has its own independent graph and both graphs are saved in the
same `.mlbricks.json` project.

### Model Builder

Shows model-building components only:

- Inputs
- Core Blocks
- normalization
- advanced blocks
- position
- heads
- outputs
- My Bricks / custom components

`Text Input` is simple again: it represents prompt/text entering the model.
Dataset downloading, train/test splitting and tokenization are no longer hidden
inside Text Input.

### Data Processing

Shows data operations only:

- **Data Source**
  - Manual Text Data
  - Hugging Face Dataset
  - Kaggle Dataset
  - URL Dataset
  - Local Dataset
- **Splitting**
  - Train / Validation / Test Split
- **Text**
  - Text Processing
  - Tokenize Text
- **Image**
  - Image Processing
- **Audio**
  - Audio Processing
- **Dataset**
  - Batch / DataLoader
- **Output**
  - Prepared Dataset

The right Inspector shows **Builder Data API** code for these processing nodes.
The helper functions are implemented in `mlb_studio.data`; they are not
fake `mlbricks` imports.

### Saved project structure

Conceptually:

```text
MLBricks Project
├── Model Builder graph
├── Data Processing graph
├── Custom Bricks
└── Project Settings
```

Switching workspaces preserves each canvas, nested model view, and scroll
position. Legacy pre-v0.5 designs are migrated in the browser by creating a new
empty Data Processing workspace while preserving the existing model graph.

### Beginner data starter

In Data Processing mode, **Text Data Starter** builds:

`Hugging Face → Clean Text → Train/Val/Test → Tokenize → Prepared Dataset`


## v0.5.1 — proper train/validation/test UI + beginner data pipeline

### Split interface

The Hugging Face source field previously labelled `Split` is now labelled
**Hub Source Split**. It only chooses which existing Hub split to download.
It is deliberately separated from dataset percentages.

The real **Train / Validation / Test Split** processing step now has:

- Training percentage slider + number input
- Validation percentage slider + number input
- Testing percentage slider + number input
- live split preview
- live total validation (`100%` required)
- beginner presets: `90/5/5`, `80/10/10`, `90/10/0`
- random seed and shuffle controls

Its executable Builder Data API calls
`train_validation_test_split(dataset, train_size=..., validation_size=..., test_size=...)`.
The backend validates that the three proportions sum to exactly 1.0.

### Default Data Processing pipeline

Every brand-new project now already contains:

`Hugging Face → Text Processing → Train/Validation/Test → Tokenize → Batch/DataLoader → Prepared Dataset`

The **Text Data Starter** button rebuilds the same beginner-ready pipeline.
Legacy projects that do not yet contain a Data Processing workspace are migrated
with this starter pipeline instead of a blank canvas.

### Binary project files

The top toolbar now includes **BIN** next to Save.

- **Save** → `<project>.mlbricks.json`
- **BIN** → `<project>.mlbricks.bin`
- **Load** automatically accepts either format

The binary file uses an `MLBRICKS-BIN-1` header followed by the project payload.
Python helpers are also available in `mlb_studio.design_io`.


## v0.5.2 — executable Data Run + live node progress

### Correct default beginner pipeline

Every new Data Processing workspace now starts with:

`Hugging Face → Text Processing → Train/Validation/Test → Tokenize → Prepared Dataset`

The default Hugging Face source is capped at 10,000 rows so a beginner does not
accidentally start by processing an entire large dataset. Set Max Rows to 0 to
use all rows.

**Batch / DataLoader** remains available as an optional advanced step.

### Run now executes the Python data pipeline

Builder uses a bridge made only from **standard ipywidgets** (no AnyWidget or
custom frontend extension). In Jupyter/Kaggle, the visual Run button sends the
current graph to Python and executes `mlb_studio.runner`.

The active node is visibly highlighted:

- `QUEUED`
- `RUNNING`
- `DONE`
- `ERROR`
- `STOPPED`

The toolbar shows overall step progress and the Inspector shows the selected
node's live execution state. Long operations use an indeterminate activity bar
rather than inventing a fake percentage.

**Stop** requests cancellation after the currently active processing step.

If a notebook frontend does not expose standard ipywidgets comms, the Builder
continues to render normally and the same real runner is available through:

```python
builder.run_data_pipeline()
```

### Beginner validation

Run refuses obviously invalid data pipelines before downloading or processing
anything. It checks for:

- exactly one Data Source
- exactly one Prepared Dataset output
- Prepared Dataset being the final step
- Train + Validation + Test totaling 100%
- disconnected Main-lane steps
- cycles / unsupported branching in the beginner runner

Invalid nodes are highlighted in red with a readable explanation.

Use **Default Data Pipeline** to instantly restore the known-good beginner
pipeline.

JSON and `.mlbricks.bin` project Save/Load support remains available.


## v0.5.3 — Kaggle Run bridge hardening + contained node text

### Node-card overflow fix

Long dataset IDs, URLs and filesystem paths can no longer escape the node card.
Mini fields use bounded two-column layouts with ellipsis. Hovering a clipped
label/value shows the full text via a tooltip.

This fixes examples such as:

- `roneneldan/TinyStories`
- `/kaggle/working/prepared_dataset`
- long URL dataset links

### Run bridge fix for Kaggle/Jupyter

The Run bridge no longer searches only the HTML output document. It now searches:

- the Builder output document
- the parent notebook document
- the top notebook document
- accessible sibling/child frame documents
- open shadow roots

This matters in notebook frontends such as Kaggle where raw HTML output and
standard ipywidgets may be mounted in different document contexts.

Writing the current graph into the hidden standard Textarea now uses the native
DOM value setter plus `input`/`change` events from the target document. Run and
Stop activate standard ipywidgets buttons in the document where they are
actually mounted.

The Data toolbar now shows:

- **Kernel Connected** — visual Run can execute Python
- **Kernel Offline** — re-run the Builder cell before trying Run

After Run is clicked, Builder waits for a Python acknowledgement. If the kernel
does not respond within three seconds, it displays a clear error instead of
silently appearing to do nothing.

The direct Python fallback remains:

```python
builder.run_data_pipeline()
```


## v0.5.4 — Dataset Registry + automatic Model Text Input binding

A successful Data Processing run now creates a reusable **Prepared Dataset**
entry in the project.

### Completion result

After the pipeline reaches `DONE`, Builder reports real split row counts, e.g.:

- Train: 9,000
- Validation: 500
- Test: 500

The Prepared Dataset Inspector shows these counts and whether the data is in
memory or also saved to disk.

### Multiple datasets

Prepared Dataset now has a **Dataset Name** setting.

Runs with different names are kept as separate datasets in the project-level
Dataset Registry. Re-running the same name refreshes that registry entry rather
than creating duplicate names.

### Model Builder integration

Text Input now supports:

- **Prompt**
- **Prepared Dataset**

When Prepared Dataset is selected it shows:

- **Available Dataset** — dropdown of every completed dataset
- **Use Split** — dropdown populated from that dataset's actual splits

After a Data Processing run completes, the newest dataset is automatically
selected by existing Text Input nodes in the Model Builder. A Text Input added
later also defaults to the newest prepared dataset.

The Text Input Inspector shows the selected dataset's Train / Validation / Test
counts.

### Python access

Actual Dataset/DatasetDict objects remain in the Builder's Python registry:

```python
builder.available_datasets()
dataset = builder.get_prepared_dataset("TinyStories Prepared")
train = builder.get_prepared_dataset("TinyStories Prepared", split="train")
```

Dataset metadata is saved in the project design. Actual data stays in memory
unless **Save To Disk** is enabled on Prepared Dataset. If a design is loaded
in a new Python session, re-run the pipeline or load the disk-backed dataset.


## v0.5.5 — Output Directory + unified Files view

The bottom project drawer now has three views:

- **Pipeline Details / Model Details**
- **Output Directory**
- **Files**

### Output Directory

The content follows the active workspace.

**Data Processing**
shows all completed prepared datasets with:

- Train / Validation / Test row counts
- total rows
- memory/disk status
- saved path
- **Use in Model**

**Model Builder**
shows:

- current editable model design
- layer/link/context/batch information
- selected prepared dataset
- registered trained/exported model artifacts as those runtimes are added

### Files

Files is workspace-independent and collects known project files in one place.

It includes:

- Builder JSON design (`.mlbricks.json`)
- Builder binary design (`.mlbricks.bin`)
- generated model config (`.model-config.json`)
- disk-backed prepared datasets
- in-memory prepared dataset entries
- registered model artifacts

Filters:

- **All**
- **Data**
- **Models**
- **Config**
- **Design**

Known paths such as `/kaggle/working/prepared_dataset` are displayed directly.
In-memory datasets are clearly marked as `Python memory`.

Files also provides direct actions:

- Save JSON
- Save BIN
- Download model config
- Use a prepared dataset in Model Builder

The schema now reserves `model_outputs` and `project_files` registries so future
training, checkpoints, weights, tokenizer files, exports and other artifacts can
appear in the same Files browser without redesigning the UI.


## v0.5.6 — compact Output Directory + full dataset Inspector

Prepared dataset cards in Output Directory are now compact, close to the Starter card footprint. Click a card to inspect source, split, cleaning, tokenizer, context and storage settings in the right Inspector.

Also fixes DatasetDict split counting: a three-split DatasetDict no longer appears as `Train = 3`. Re-run the data pipeline once after upgrading to refresh old registry metadata.


## v0.6.0 — Model Build lifecycle + data compatibility gate

Model Builder no longer presents the data-processing **Run** action.

### Model workflow

`Design → Build → Select Built Model → Check Data → Train → Generate`

In **Model Builder**, the top action is now **Build**.

Build:

- validates that the architecture has an input and output/head
- rejects disconnected components
- rejects Main-lane cycles
- visually walks through the model nodes
- snapshots the current architecture into **Model Outputs**
- preserves revisions when the same model is rebuilt

Data Processing still uses **Run Data**.

### Built Model Inspector

Click a built model in **Output Directory** to open the right-side model panel.

It shows:

- model status / build revision
- layers and connections
- input modality
- output type
- context length
- batch size
- parameter estimate
- build time
- prepared-dataset selector

### Compatibility gate

A user can choose any prepared dataset from the project. Builder checks:

- input/data modality
- Train split existence
- tokenizer availability for text language models
- data context length versus model context length
- `input_ids` availability when split-column metadata is available

When compatible, the **Train** action appears.

When incompatible, Train is hidden and the Inspector shows the exact failed
checks.

The model's dataset selection also updates its editable Text Input binding.

### Generation

**Generate Tokens** is shown for text models but remains disabled until the model
has trained or loaded weights.

Build is currently an architecture validation/snapshot step. This version does
not pretend to execute model training or token generation without a model
runtime/compiler. Train records readiness after compatibility passes; the
training executor is the next runtime layer to connect.


## v0.6.1 — Training / Generation setup workspace + available devices

Clicking **Train** or **Generate / Configure Generation** on a built model now
replaces the center graph area with a guided runtime configuration workspace.
The model graph is preserved and **Back to Model Graph** returns to it.

### Training setup

- budget by **steps**, **tokens**, or **epochs**
- training steps / token budget / epochs
- batch size and gradient accumulation
- optimizer, learning rate, weight decay, warmup
- validation split
- validate every N steps
- validation steps
- **generate a sample during validation**
- validation prompt + generated-token count
- checkpoint cadence
- seed
- output directory
- device / backend / execution / compile mode / precision

### Generation setup

- prompt
- new-token count
- temperature
- top-k / top-p
- seed
- device / backend / execution / compile mode / precision

### Available devices

Builder now detects the devices visible to the Python kernel and shows them as
selectable cards and in the Device dropdown. CPU is always shown. CUDA GPUs are
listed individually with GPU name, VRAM and compute capability when available;
MPS/XPU are also detected when PyTorch exposes them.

Runtime choices include:

- Device: Auto / CPU / each available GPU
- Backend: Auto / Native / PyTorch
- Execution: Eager / Compiled
- Compile mode: Default / Reduce Overhead / Max Autotune
- Precision: Auto / FP32 / FP16 / BF16

The runtime configuration is saved on the built-model entry. This version makes
the Train/Generate buttons functional as configuration workflows, but does **not**
fake actual model training or generation: the MLBricks graph compiler/model
executor still needs to be connected before Start Training or Generate can run
real model computation.


## v0.6.2 — real training + generation executor

`Start Training` is now connected to the Python kernel. It no longer stops at
"Training configuration saved".

For supported text language-model graphs, Builder now:

- compiles the visual graph into a real `torch.nn.Module` using MLBricks layers
- consumes the selected prepared `train` / validation splits
- runs AdamW / Adam / SGD optimization
- supports step, token and epoch budgets
- gradient accumulation
- warmup
- selected CPU / CUDA device
- Auto / Native / PyTorch ESA backend policy
- eager or `torch.compile` execution
- fp32 / fp16 / bf16 autocast
- validation cadence and validation-step limits
- validation sample generation
- checkpointing and final checkpoint output
- live step/loss/validation/token progress in the runtime panel
- Stop Training

After training, the built model is marked `weights_ready`, the final **MLBricks model artifact**
(directory containing `model.pt` + `metadata.json`) is registered on the model output, and **Generate Tokens** becomes executable.
Generation uses the configured prompt, token count, sampling settings, device,
execution mode and precision and streams generated text back into the runtime
panel.

### Current executable graph coverage

The first real compiler deliberately supports the components needed by the
TinyStories ESA starter and similar language models:

- Text Input / Text Output
- Embedding
- ESA
- RMSNorm / LayerNorm
- FFN
- Residual
- Dropout
- LM Head
- nested custom bricks composed from those parts

Unsupported advanced model bricks fail with a clear compiler error instead of
pretending to train.

The executor automatically expands the runtime vocabulary when the prepared
Hugging Face tokenizer is larger than the visual Embedding/LM Head vocabulary,
so token IDs cannot index outside the model embedding table.


## v0.6.3 — Training Status + Generation Status tabs

The runtime workspace no longer uses a `← Model Graph` button. Training and
generation are organized as two-tab workflows:

- **Training Setup** / **Training Status**
- **Generation Setup** / **Generation Status**

Clicking **Start Training** automatically switches to Training Status. Clicking
**Generate Tokens** automatically switches to Generation Status.

### Training Status

Shows live Python-kernel events:

- progress percentage
- step / max steps
- train loss
- latest validation loss
- best validation loss
- tokens seen
- elapsed time
- validation schedule
- validation generated sample
- chronological training log
- checkpoint events and latest checkpoint path
- output directory
- weights/training status
- Stop Training

Validation completion and checkpoint saves are now explicit runtime events, so
they appear in the status/log view rather than being hidden inside a generic
step message.

### Generation Status

Uses the same pattern and shows:

- generated tokens / requested tokens
- live percentage
- prompt
- live/final generated text
- temperature / top-k / top-p / seed
- runtime/device/backend/execution/compile/precision
- generation event log
- Stop Generation

Runtime progress events now carry the built model id so status/history is kept
on the correct model even when multiple built models exist.


## v0.6.4 — training/generation null-safety

Fixes the step-0 runtime failure:

`TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'`

Older saved runtime configurations could contain explicit JSON `null` values.
Those values previously overwrote safe defaults and later reached Python
`int(...)`/`float(...)` conversions.

v0.6.4:

- ignores null/blank legacy values while merging runtime defaults
- safely normalizes all numeric training fields
- safely normalizes generation numeric fields
- makes supported model-component numeric parameters null-safe
- reports field-specific errors such as `Batch Size must be a number`
- adds **Reset Runtime Defaults** to both Training Setup and Generation Setup
- keeps blank required fields blocked before Start Training

Opening Training Setup after upgrading automatically repairs old null settings.


## v0.6.5 — Model Settings + focused runtime mode

### Model Settings

A built model now exposes editable **MODEL SETTINGS** in the right Inspector:

- Embedding Size
- Heads
- Block / Context
- Default Batch
- Vocabulary
- Precision

Changes synchronize compatible model-wide fields across the editable graph and
nested blocks (Embedding, ESA, norms, FFN, LM Head and supported related
components).

`Block / Context` updates the project context and ESA block size.
`Default Batch` updates the model default and seeds Training Setup's batch size.
Training Setup can still override that batch for an individual run.

Architecture-affecting changes mark the built model **Rebuild Required**.
Compatibility then blocks Train until **Build** is clicked again, preventing a
stale build from being trained accidentally.

### Focused training/generation workspace

When Train or Generate opens a runtime workspace, the bottom **MODEL WORKSPACE**
drawer is hidden completely. The center is reserved for Training
Setup/Status or Generation Setup/Status.

### Stable animated Build button

The top Build button has a fixed 82 px width, so its label no longer expands and
shrinks the toolbar.

During training it becomes a fixed-width animated `◆ Training` indicator.
During generation it becomes `◆ Generating`. A subtle pulse and moving highlight
show activity without changing the button's dimensions. When runtime activity
finishes, it returns to `◆ Build`.


## v0.6.6 — Hugging Face Hub push/load

The bottom project drawer now includes **Hugging Face**.

### Authentication

Builder uses the notebook's existing Hugging Face credentials:

```bash
hf auth login
```

or the `HF_TOKEN` environment variable.

The token is **never stored** in Builder state, JSON, BIN, dataset metadata,
model metadata, or project files.

### Push

The Hub panel can push:

- **Prepared Dataset**
  - uploads Dataset / DatasetDict splits using `datasets.push_to_hub`
  - writes `mlbricks_dataset.json` so Builder-specific processing and tokenizer
    metadata can be restored later
- **Built / Trained Model**
  - uploads the Builder model graph and model metadata
  - uploads the complete directory-based MLBricks artifact when trained weights exist
  - keeps legacy `weights/last.pt` support for older Builder repositories
  - includes a locally available tokenizer when possible
- **Builder Project**
  - uploads the complete Builder project state as `mlbricks_project.json`

Repositories can be private or public.

### Load

The same panel can load:

- a Hub dataset into the Prepared Dataset registry
- an MLB Studio model into Model Builder / Model Outputs
- a complete MLB Studio project

Public repositories can load without authentication. Private repositories use
the locally authenticated Hugging Face token.

Loaded trained model packages restore their MLBricks artifact directory (or a legacy checkpoint) from the Hugging
Face cache and can be opened for token generation. A newly selected local
Prepared Dataset can be used for compatibility checking and further training.


## v0.6.7 — collapsed runtime drawer + Cloud & Repositories

### MODEL WORKSPACE behavior

Entering Training or Generation now **collapses** MODEL WORKSPACE instead of
removing it.

The bar remains visible at the bottom and can be manually expanded while the
runtime screen is open. Returning to the graph is not required.

### Cloud & Repositories

The previous Hugging Face-only view is now **Cloud & Repositories**.

Providers:

- Hugging Face
- GitHub
- AWS S3
- Google Cloud Storage
- Azure Blob Storage

Content:

- Prepared Dataset
- Built / Trained Model
- Complete Builder Project

Hugging Face continues to use native Hub dataset/model repositories.

GitHub, S3, GCS and Azure store portable `.mlbricks.zip` bundles containing the
selected dataset/model/project so the same content can be restored into Builder.

### Credentials + local secure references

The Cloud panel includes masked credential fields for:

- Hugging Face API/access token
- GitHub personal access token
- AWS access key / secret key / optional session token
- Google Cloud service-account JSON
- Azure Storage connection string

Credentials can be given a local **Credential Name** and saved. MLBricks Studio
keeps only masked credential metadata in its SQLite Studio database. The real
secret is stored in the operating-system credential store through `keyring` when
a secure backend is available. In notebook/headless environments without an OS
keyring, the real secret remains session-only and the UI marks the saved reference
as requiring re-entry after the session ends.

Real credentials are never included in Builder state, autosaved drafts, Gallery
items, JSON/BIN exports, model designs, dataset metadata, or cloud bundles.
Environment/default credentials continue to work when supported.

### Optional cloud packages

```bash
pip install "mlb-studio[cloud]"
```

or install individual provider packages:

```bash
pip install boto3
pip install google-cloud-storage google-auth
pip install azure-storage-blob
```

GitHub support uses Python's standard HTTP library and needs no extra package.

## v0.6.8 — Local / Kaggle filesystem loading

Adds **Local / Kaggle** to the bottom workspace selector. Builder can now scan and directly load content from `/kaggle/working`, `/kaggle/input`, Colab `/content`, the current working directory, or any absolute path.

Supported local data: Hugging Face `Dataset.save_to_disk()` / `DatasetDict.save_to_disk()` folders and raw TXT/CSV/JSON/JSONL/Parquet files. Loaded data is registered as Prepared Dataset and becomes available to Model Builder.

Supported local models: MLBricks model artifact directories (`model.pt` + `metadata.json`), legacy `last.pt` / periodic `.pt` checkpoints, `.pth` / `.ckpt`, plus `.mlbricks.zip` bundles. v0.6.8 training checkpoints now embed the Builder model graph, nested custom-component definitions, project settings and dataset metadata so a new checkpoint can restore after a kernel restart.

Older checkpoints can still load when the matching Builder project/custom definitions are already open. If they lack embedded nested definitions, Builder now explains that the matching project must be loaded first.

Local projects: `.mlbricks.json`, `.mlbricks.bin`, and `.mlbricks.zip` project bundles.


## v0.7.0 — Serve trained models through generated links

A trained or locally loaded model now exposes **Serve Model / API**.

### Links

Starting the server generates:

- `http://127.0.0.1:<port>` for apps on the same machine
- a LAN URL for phones/devices on the same network
- optional **ngrok Public HTTPS** for Kaggle, Colab, or remote access

Kaggle/Colab localhost belongs to the remote kernel. To reach that model from
your phone or a web app on your own computer, enable the public HTTPS tunnel.

### HTTP API

- `GET /` responsive browser playground
- `GET /health`
- `GET /v1/model`
- `GET /v1/models`
- `POST /v1/generate`
- `POST /v1/completions` OpenAI-style text-completion response

The trained model is loaded once when the server starts and remains resident.

### Security

Bearer API-key protection is enabled by default. If the API-key field is empty,
Builder generates a random key. API keys and ngrok tokens are session-only and
are not written into Builder project files. V1.0 also defaults to loopback-only
binding and same-origin CORS, adds browser security headers, request/prompt/token
limits, concurrency/rate limiting, and a generation timeout. Public ngrok tunnels
refuse to start when API-key authentication is disabled.

For public tunnels install:

```bash
pip install pyngrok
```

or:

```bash
pip install "mlb-studio[serve]"
```


## v0.7.1 — path-only recursive model import

**Local / Kaggle Models** now needs only one base path.

Examples:

```text
/kaggle/working
/kaggle/input
/kaggle/input/my-training-run
/content/models
```

Click **Scan & Import Models**. Builder recursively scans all subdirectories,
detects MLBricks model checkpoints/bundles, restores every compatible model,
skips duplicate paths and continues past incompatible/older files.

After import, Builder switches automatically to **Model Repository** so the
imported models are immediately available for Generate Tokens, Serve Model/API,
or further training when compatible data is selected.


## v0.7.2 — legacy checkpoint recovery + true model repository compilation

### Legacy custom-brick recovery

Older checkpoints (before v0.6.8) may contain the root model architecture but
not the reusable/nested custom-brick definition table.

When importing such a checkpoint, Builder now:

1. detects the stale/missing custom definition IDs
2. compares the checkpoint top-level node names/types and edge flow with the
   model currently open in Builder
3. only on an exact architecture-shape match, remaps the stale custom IDs to
   the matching current reusable brick definitions
4. imports the checkpoint as **Recovered Legacy Checkpoint**

This lets old TinyStories checkpoints restore when the matching TinyStories
model is already open, even if IDs such as `custom_xxx` changed between sessions.

If the current graph is not an exact structural match, Builder still rejects the
checkpoint rather than guessing.

### Model Repository correctness

`compile_builder_model()` now compiles the selected repository model's own
captured `architecture` and `custom_components_snapshot`.

Previously, an imported model could accidentally compile whichever graph was
currently open in Model Builder. This matters once multiple different models
exist in Model Repository.

Newly built models now snapshot their custom component definitions as well.


## v0.7.3 — path-only recursive data import

The same one-path workflow used by Local / Kaggle Models is now available for
**Local / Kaggle Data**.

In Data Processing, provide only a base path:

```text
/kaggle/working
/kaggle/input
/kaggle/input/my-dataset
/content/data
```

Click **Scan & Import Data**.

Builder recursively scans all subdirectories and detects:

- Hugging Face `Dataset.save_to_disk()` folders
- Hugging Face `DatasetDict.save_to_disk()` folders
- TXT
- CSV
- JSON
- JSONL
- Parquet
- Arrow
- MLBricks `.mlbricks.zip` dataset bundles

Compatible datasets are registered automatically in **Data Repository**.
Duplicate local paths are skipped. One incompatible file does not stop the rest
of the directory import; it is reported in the import summary.

After the scan, Builder automatically switches to the Data Processing workspace
and opens **Data Repository**, selecting the most recently imported dataset.

Raw tabular files do not require a column named `text` during import. They can
be loaded first and then configured/processed inside Data Processing.


## v0.7.4 — automatic API port fallback

If the requested API port is already in use (common in Kaggle after a previous
server attempt), Serve Model/API no longer fails with `Errno 98`.

For a requested port such as `8000`, Builder now tries:

```text
8000, 8001, 8002, ... 8020
```

and finally asks the OS for any free port if necessary.

The API Server Status panel shows the actual selected port and all Localhost,
LAN, Public HTTPS and generated web-app examples use that real port.

Startup errors are now displayed directly as **ERROR** instead of appearing as
a generic **STOPPED** state.

The Web App Example code block also renders real line breaks instead of literal
`\n` characters.


## v0.7.5 — do not lose the HTTP server when ngrok fails

Previously the serving flow was:

```text
HTTP server starts
→ ngrok fails
→ whole start command reports failure
→ HTTP server thread may remain alive but Builder loses its handle
→ next retry finds another occupied port
```

v0.7.5 registers the HTTP server immediately after it binds.

If ngrok fails:

- API Server Status remains **RUNNING · LOCAL**
- Localhost/LAN links are shown
- generated API key is shown
- the exact ngrok error is displayed separately
- Stop API Server can clean up the live server
- Restart API Server closes the previous server before starting again

This prevents orphaned model-server threads from accumulating occupied ports.

On Kaggle the local link still is not externally reachable; fix the displayed
ngrok error and restart to obtain the Public HTTPS URL.


## v0.7.6 — Kaggle bridge reliability, copy fallback, workspace default

- MODEL/DATA WORKSPACE opens by default.
- Entering Train or Generate collapses it automatically. Serve Model/API does not.
- Copy API Key now falls back from Clipboard API to `execCommand("copy")`, then a manual copy prompt for restrictive Kaggle iframes.
- The API key is no longer copied into serialized Builder state.
- Local model/data, cloud and runtime actions use a dedicated standard-ipywidgets command channel instead of embedding commands in the complete project JSON.
- Local data import reports a single actionable dependency error if `datasets` is not installed.
- The visible version badge now follows the actual frontend version instead of the stale hard-coded `v0.6.7`.


## v0.7.7 — uniform Brick Library cards

The left Brick Library now uses one universal card size for every component.

- fixed 58px component-card height
- fixed icon geometry
- component title limited to one line with ellipsis
- component description limited to a maximum of two lines
- long words/technical API descriptions wrap safely
- overflowing text is clipped inside the card
- built-in and custom component cards use the same dimensions

This prevents descriptions such as Embedding/ESA/VESA/FFN/Residual text from
leaking below the card boundary and keeps the complete library visually aligned.


## v0.7.8 — vertical alignment cleanup

This release refines the Brick Library card layout introduced in v0.7.7.

- icon and text block are vertically centered inside each component card
- title and description now align visually in the middle of the card
- fixed-height card system remains unchanged
- title still uses one-line clamp
- description still uses two-line clamp
- built-in and custom brick cards stay visually consistent


## v0.7.9 — toolbar + info + library polish

This release improves the Builder UI and Inspector.

### Updated
- fixed vertical alignment for library card icons and text
- made library cards slightly roomier with a more stable fixed height
- first item under each collapsible section now aligns correctly
- added bottom spacing for **Create Custom Brick**
- changed **Batch / DataLoader** icon from `BATCH` to `BTC`
- renamed **Node Info** tab to **Info**
- added **Config** guidance in the inspector: what the element does and why it is used

### Toolbar
- removed the separate **BIN** button
- **Save** now asks whether to save **BIN** or **JSON**
- wired **Export**, **Share**, **Help**, and **Settings** actions
- improved top-toolbar vertical alignment


## v0.7.10 — blank Builder screen fix

v0.7.9 introduced a JavaScript syntax error in the new Share action because
multiline text was emitted as literal newlines inside a quoted JavaScript string.
That prevented the frontend from mounting and produced a blank dark Builder panel.

v0.7.10 rebuilds the Share summary using an array of lines joined with `\n` and
passes JavaScript syntax validation before packaging.


## v0.7.11 — simplify visual Brick Library

`Brick` and `Bricks Model` are no longer shown in the visual **Brick Library**.
They are MLBricks code-level composition/container APIs, while Builder already
provides visual composition through **Custom Brick** and the model canvas.

Their underlying catalog/API entries remain available internally so older saved
Builder projects that already contain these node types can still load and render.


## v0.7.12 — editable/locked layouts and personal Gallery

### Toolbar cleanup
- removed the top **Save** button because **Export** is the primary artifact action
- removed **Auto Layout** and **Add Layer / Add Step**
- added one layout-mode toggle: **Lock Layout** / **Edit Layout**
- added **Rename Layout**
- added **Gallery**

### Layout lock
Lock Layout protects structural edits: adding, deleting, duplicating, moving or
reconnecting components, clearing the graph and changing Auto Connect. Node
configuration remains inspectable/editable. Unlock with **Edit Layout**.

### Unique names
Components in the same layout are automatically given unique names. Repeated
components become `ESA`, `ESA 2`, `ESA 3`, etc. Component and layout rename
operations reject duplicate names.

### Gallery
The new Gallery stores reusable user-created assets:
- **My Components** — custom bricks can be saved and re-added to My Bricks
- **My Models** — model layouts can be saved and loaded back to the canvas

Gallery data is part of Builder state and is also mirrored to browser local
storage when the notebook/browser environment permits it.


## v0.7.13 — clean component labels + stable library scroll

- repeated components keep unique internal names/IDs but display clean labels such as
  `ESA` and `Text Input` instead of `ESA 2`, `ESA 3`, `Text Input 2`
- manual component renames remain unique within a layout
- the Brick/Data Library preserves both vertical and horizontal scroll position
  when adding a component or expanding/collapsing a category
- switching workspaces preserves each workspace's sidebar position independently
- removed the **What it does / Why use it** information card from Inspector/Config


## v0.7.14 — environment-aware local import

The bottom drawer no longer says **Local / Kaggle Models/Data**. It now uses
**Local Environment** and detects the Python notebook environment on the backend.

Examples:

- Kaggle → `/kaggle/working`, `/kaggle/input`
- Google Colab → `/content` and mounted Drive when available
- Lightning AI → current workspace / Teamspace roots
- GitHub Codespaces → current workspace / `/workspaces`
- Amazon SageMaker → current notebook/SageMaker roots
- other Python/Jupyter environments → current working directory and user home

**Scan Environment Models/Data** scans all detected roots recursively. A
secondary **Scan This Path** action remains available for a specific directory.
No browser-side filesystem guessing is used; detection comes from the Python
runtime that actually owns the files.


## v0.7.15 — Full Window Builder

The top bar includes **↗ Full Window**. It opens the Builder in a separate tab
that occupies the whole browser viewport. The notebook tab stays open as the
Python execution host, while a same-browser BroadcastChannel proxies Run Data,
Train, Generate, Serve Model/API, Local Environment import, Cloud actions and
Stop between the Full Window tab and the notebook's standard-ipywidgets bridge.

Visual edits in Full Window synchronize back to the notebook Builder. Keep the
notebook tab/session open for Python-backed actions. No ngrok or public server is
required simply to use Full Window mode.

## v0.7.16 — Full Window reliability fix

v0.7.15 could open a new tab but leave the generated Builder page unmounted. The
popout HTML used an escaped `<\/script>` sequence as the generated script closing
tag, so the browser treated the rest of the page as part of the first script.

v0.7.16 fixes the generated HTML and hardens Full Window for notebook sandboxes:

- **Full Window** uses a self-contained Blob page
- the generated page receives real script closing tags and mounts normally
- notebook ↔ Full Window communication uses `window.postMessage` with
  `BroadcastChannel` fallback
- Run and Stop have distinct proxy bridge identifiers in the popout
- repeated hello/handshake attempts help when the new tab loads slowly
- Full Window remains 100vw × 100vh and keeps the notebook tab as the Python host
- local scan status text is environment-neutral instead of Kaggle-specific

## v0.7.17 — dedicated Full Window kernel channel

v0.7.16 could render the Full Window correctly but some sandboxed notebook
environments could strip or isolate `window.opener` / `BroadcastChannel`, leaving
the new tab showing **Kernel bridge is offline**.

v0.7.17 makes the notebook tab explicitly own the popout and transfers a dedicated
`MessageChannel` to it:

- Full Window is opened directly from the user click so the notebook retains its
  `WindowProxy` even when the popup cannot use `window.opener`
- the notebook transfers a two-way `MessagePort` to the Full Window
- fresh channel offers retry during slow popup startup
- opener, `postMessage`, and `BroadcastChannel` remain fallbacks
- duplicate transport messages are tolerated rather than letting one stale route
  hide another working route
- state sync, Run, Stop, Train, Generate, Serve, local import, and cloud commands
  all use the same dedicated transport back to the notebook Python bridge


## v0.7.20 — clean about:blank new-tab URL

The Builder popout keeps the existing notebook-owned execution bridge, but the
new tab no longer starts from a `blob:https://...kaggle.net/...` URL. **Full
Window** now opens `about:blank` and writes the same self-contained Builder page
directly into that tab.

- no pyngrok is used for opening Builder in a new tab
- no standalone web-app server is required
- the address bar stays `about:blank` instead of exposing the Kaggle/Colab host
  inside a Blob URL
- the original notebook tab remains the Python execution host
- the existing MessageChannel / postMessage / BroadcastChannel bridge is kept
  unchanged

## v0.7.22 — readable AIBuilder workspace

- Full-window browser tab title is now **MLB Studio V1.0**. The browser address remains `about:blank` by design because a script-injected tab cannot claim a custom URL without a real hosted origin.
- Raised tiny 5–9 px UI text to readable sizes across the Builder, Inspector, runtime views, repositories, Gallery and Local Environment panels.
- Enlarged sidebars, nodes and key controls to match the new typography.
- Build / Training / Generating and Stop are centered in the top bar.
- Removed the `?` Help and gear Settings buttons from the top-right toolbar.
- Local Environment Base Path now defaults to the current writable workspace root.
- Builder creates `<workspace>/mlbricks/` with `models/`, `data/`, `training/`, `projects/` and `exports/` directories.
- Default model training output is under `mlbricks/models/`; prepared datasets default under `mlbricks/data/`.



## v0.7.22 — classic graph typography + running-port fix

- Restores Brick/Data Library and graph-node typography to the compact v0.7.19 scale while keeping the larger v0.7.21 toolbar and inspector.
- Moves the live running/progress track above the bottom edge and raises connector z-order so the Extra-lane dots remain fully visible.
- The detached tab title is `MLB Studio V1.0`. The browser address remains `about:blank`; browsers do not permit a script to replace the address bar with arbitrary non-URL text.


## v0.7.24 — stable editing and clearer data controls

- Removed the top-toolbar **Rename Layout** and **Gallery** buttons. Gallery remains available from the workspace drawer.
- The model/project title is now editable inline from the top bar, similar to notebook title editing.
- Search now keeps keyboard focus and caret position while results update, with a clear **Search...** placeholder.
- Inspector scroll position is preserved per selected node/output so edits and actions no longer jump the right panel to the top.
- Training dataset selector uses compact dataset names so the native option menu stays contained.
- Split preview is slightly larger and the old numeric-only preset buttons are replaced with themed, explicit Train / Validation / Test labels.


## v0.7.24 — sample models and data moved to Gallery

- Removed TinyStories 30M and Default Data Pipeline sample shortcuts from the canvas toolbar.
- Added built-in **Sample Models** and **Sample Data** sections to Gallery.
- TinyStories 30M now loads from Gallery.
- The TinyStories text-processing pipeline now loads from Gallery.
- Data Processing pipelines can now be saved to and loaded from **My Data Pipelines** in Gallery.
- Bottom workspace sample cards now open Gallery instead of loading a preset directly.


## v0.7.26 — center Gallery workspace

Gallery is now a first-class center workspace, opened from the top toolbar before Undo. It has Models, Components, and Data tabs, built-in sample areas, user-saved items, contextual save actions, and a Close Gallery button. The old Gallery entry was removed from the bottom drawer.


### Hosted AIBuilder launcher

The separate-tab launcher is configured for `https://builder.mlbricks.io/`. Deploy `web/builder.mlbricks.io/index.html` at that origin. The launcher contains no model runtime; it receives the current Builder UI from the already-open notebook and keeps Python execution bridged through that notebook tab. If the hosted launcher is unavailable, Builder falls back to the working `about:blank` injected tab rather than failing.


## v0.7.28 — Gallery blank-screen fix + hosted AIBuilder URL

- Fixed the v0.7.26 startup regression where the generic button helper referenced an undefined Gallery item, causing a blank/black Builder screen.
- Gallery remains a center workspace with Models / Components / Data tabs.
- Full-window launch is configured for `https://builder.mlbricks.io/` when its launcher page is deployed.
- Added `web/builder.mlbricks.io/index.html`, the static launcher to deploy at that origin.
- If the hosted launcher is unavailable, Builder safely falls back to the working injected `about:blank` tab rather than breaking.


## v0.7.28 — Center Cloud workspace + launcher stability

- Removed the large Gallery title/subtitle banner; Gallery now opens directly into Models / Components / Data tabs.
- Added a top-level **Cloud & Repositories** button beside Gallery and moved the cloud provider controls into the center workspace.
- Removed Cloud & Repositories from the bottom drawer selector.
- Full Window now opens the working Builder immediately and only upgrades to `https://builder.mlbricks.io/` after the hosted launcher proves it is deployed, preventing the visible URL-to-`about:blank` bounce.


## v0.7.29 — Gallery + Repositories center workspaces

- Gallery, Build, and Repositories are now a centered segmented top control with no overlap.
- Gallery is a full center workspace with Models, Components, and Data tabs plus visual cards.
- Cloud & Repositories is a full center workspace with provider/connection, session credentials, Push, and Load sections.
- Gallery/Repositories hide the secondary graph toolbar and bottom drawer while open so the center workspace gets the full available area.
- Close buttons return to the Builder canvas without changing the current graph.


## v0.7.30 — Scrollable Gallery and separated top actions

- Gallery now has its own reliable vertical scroll area for large prebuilt and saved collections.
- Top-center controls are separate `Build`, `Gallery`, and `Cloud & Repositories` buttons instead of a segmented switch.
- Restores the compact Gallery/Repositories banner treatment while keeping close actions.
- Cloud & Repositories remains a center workspace and scrolls independently when content exceeds the available height.


## v0.7.31 — Fixed Gallery sizing and Cloud action placement

- Gallery banner/tabs keep a fixed height; only Gallery contents scroll as collections grow.
- Restored compact v0.7.27-style Gallery model cards with parameters, batch and block metadata.
- Removed Share and placed Cloud & Repositories in its former top-right slot.



## v0.7.35 — Fetch lifecycle and two-column Components/Data Gallery

- Data action is now **Fetch Data**.
- **Stop** appears beside Fetch Data only while the data pipeline is actively running.
- Gallery **Components** and **Data** saved items use the same two-column layout as saved Models.
- Data progress is explicitly tagged as the data runtime so toolbar state stays accurate.

## v0.7.32 — Two-column saved models and Gallery file actions

- Saved models now flow across both Gallery columns instead of being confined to one side.
- Load and Export moved from the global toolbar into Gallery to free top-bar space.
- Prebuilt Models and My Models are full-width Gallery sections with responsive two-column card grids.


## v0.7.38 — SOUP and ElasticBit 4–32 components

- Added the MLBricks SOUP architecture to the model component library and training compiler.
- SOUP supports ESA/BOLT mixer selection, SAFFN/FFN selection, per-layer comma-separated routing, JSON mixer/FFN configs, observer memory and fusion settings.
- Added ElasticBit 4–32 as the primary adaptive native runtime component with compact/fast execution and 4–32 bit analysis controls.
- Removed the legacy ElasticBit 2–8 bit quantizer from the AI Builder component library; ElasticBit 4–32 is now the only ElasticBit component exposed in the Builder.

## v0.7.38 — Whole-model training parity + MLBricks lifecycle API

- Eager and compiled language-model training now use the same packed fixed-shape `[batch, context]` batches, eliminating padding-biased throughput comparisons.
- Compiled training captures one full causal-LM graph (model + LM head + cross entropy) through one explicit `torch.compile(..., mode="reduce-overhead", fullgraph=True, dynamic=False)` call and performs two untimed forward/backward warm-up passes. Builder uses the explicit PyTorch call so these benchmark flags are forwarded unchanged for visual `TensorGraph` models.
- Training reports **GPU Tok/s** separately from **end-to-end Tok/s** so data preparation/H2D time is no longer confused with model compute throughput.
- Default AdamW settings now match the validated notebook (`lr=5e-4`, betas `0.9/0.95`, weight decay `0.1`, no LR warmup); Beta 1/Beta 2 are editable in Training Setup.
- The TinyStories ~30M starter now mirrors the validated 10-layer/330-width/6-head ESA benchmark, including learned positions, two pre-norm residuals per layer, final LayerNorm, 4× GELU FFN, and tied token-embedding/LM-head weights.
- New training outputs/checkpoints use the package-level MLBricks lifecycle API: `mlbricks.save`, `mlbricks.load`, and `mlbricks.inspect`. Directory model artifacts (`model.pt` + `metadata.json`) are supported by local loading, cloud bundles, generation restore, and Hugging Face push/load. Legacy Builder `.pt/.pth/.ckpt` checkpoints remain loadable.
- Builder resolves the current `LMHead(..., tie_to=...)` module-reference API visually through a `Tie Embeddings` setting.
- Learned and sinusoidal position modules are executable in the Builder runtime.
- SOUP and ElasticBit 4–32 remain aligned with the supplied MLBricks Kit 1.0.0b1 source API.
- Generic `Brick` / `Bricks` composition-container APIs are not exposed as Builder palette components; reusable visual compositions live under **My Components** instead.


## v0.7.38 — 200M ESA/SOUP and 30M one-layer SOUP presets

- **StateAware ESA 200M** — 8 layers, d_model 384, 6 ESA heads, state_dim 2749, context 256, **199,982,344 parameters**.
- **SOUP 200M** — 3 layers, d_model 1152, state width 2864, 18-head ESA mixers, observer memory 256, fusion hidden 1728, **199,916,160 parameters**.
- **SOUP 30M 1L** — one SOUP layer, d_model 384, state width 1408, 6-head ESA mixer, context 512, **30,003,528 parameters**.

Programmatic presets: `stateaware-esa-200m`, `soup-200m`, `soup-30m-1l`.

## v0.7.40 — Lazy MLBricks import pool

- Added a central lazy import pool for all MLBricks-backed Builder components.
- Canonical submodule imports are preferred; compact top-level exports are fallback-only.
- Adding a component asks the Python bridge to warm that component import and caches it for reuse.
- Model compilation preflights only APIs actually referenced by the graph, including nested custom components.
- Constructor previews now emit canonical import paths.
- Lifecycle save/load/inspect and Adam/AdamW also resolve through the same pool.

## v0.7.41 — Runtime context repacking compatibility

- Fixed dataset/model context compatibility at the shared compatibility-engine level rather than patching individual presets or architectures.
- Token IDs no longer inherit a hard context requirement from the tokenizer preparation step. `input_ids` datasets are treated as repackable token streams for every text model.
- Model context now controls runtime LM packing; prepared tokenizer max length is informational metadata only.
- A 512-max-length prepared dataset can train a 256-context, 512-context, 1024-context, or other text model without a false incompatibility gate.
- `_PackedLMBatcher` strips tokenizer padding via `attention_mask` before runtime repacking, so padded prepared datasets do not inject padding into the continuous LM token stream.
- Prepared dataset metadata now declares `runtime_context_repack` centrally when `input_ids` are present; older datasets remain supported through column-based capability inference.
- Renamed the data-preparation field display to **Tokenizer Max Length** and palette labels to **Component Library / Core Components** to avoid implying that preprocessing blocks define model context.




## v0.7.44 — Custom component Gallery lifecycle

- **Save Component** now saves/upserts the definition in **Gallery → Custom Components** instead of automatically installing it in the left **My Components** palette.
- Gallery custom components remain reusable templates; **Add to My Components** explicitly installs a saved component into the left palette.
- Every custom component installed in the left palette now has a compact **✎ edit icon** with **Edit**, **Rename**, and **Remove** actions.
- Removing a custom component asks for confirmation once. Removing from **My Components** hides the palette template while preserving already-placed model instances.
- Gallery removals also require one confirmation.

## v0.7.42 — Custom API components

Gallery → Components can now create reusable **API-bound custom components** in addition to visual nested components. A custom API component stores a dotted Python import path (for example `torch.nn.Linear` or `mamba_ssm.Mamba`), a target kind (`module` or `function`), and typed constructor/call parameters. Parameters can be exposed to the component Inspector or bound to the Main, Skip, or Extra tensor lanes and Builder model settings such as model dimension, head count, context, batch, device, and dtype.

External APIs use the same lazy import pool design as MLBricks components: the target is imported only when tested or used and then cached. The runtime supports arbitrary installed PyTorch `nn.Module` classes and callable functions, including multi-input functions through the three Builder tensor lanes. Tuple/list/dict outputs can be reduced with an output selector.


### v0.7.44 — simplified custom component flow

- Custom components are created from **Gallery → Components** only; the duplicate sidebar **+ Create Component** button is removed.
- Gallery creation is reduced to two compact actions: **API Component** and **Component**.
- Custom component editing uses **Save** and **Save As New** in the right Inspector.
- The Gallery toolbar no longer shows a second **Save Current Component** action while editing a custom component.


## v0.7.45 — API Component function graph composer

API Components are now explicit Python/PyTorch function DAGs rather than a single binding. Creating an API Component starts with one Function block. Each block can bind an import module + function/class, choose Module or Function execution, define init/call parameters, and bind tensor arguments to Main / Skip / Extra lanes. `+ Add Function` adds another operation block; visual links can be added or removed to build serial chains, parallel fan-out, and three-input merges such as Q/K/V-style APIs. Nested Builder components are intentionally disabled inside API Components.

Saving an API Component now returns directly to Gallery → Components. Saved custom API components expose Add to My Components, Edit, and Remove actions. The runtime executes the saved API graph as a differentiable PyTorch DAG, lazily resolves every external import, registers module steps as submodules, and retains compatibility with older single-binding API Components.


## v0.7.46 — mixed API + MLBricks component graphs

API Component editing now supports a true mixed execution graph. API function/module blocks can be combined with supported built-in MLBricks components such as FFN, RMSNorm, Linear, Residual, ESA and SOUP directly from the left Component Library. New blocks connect after the selected block by default and all connections remain manually editable for serial, branch and merge execution. Saved custom components are still excluded from API Component internals to prevent circular nesting. The runtime compiler and lazy import preflight now execute/import both API steps and built-in nodes inside the same saved API Component. API composer node spacing was increased for clearer wiring.

## v0.7.47 — focused nested component composition

- Component editors are now focused workspaces: Gallery navigation is hidden/blocked until the top-level component is saved.
- Saved custom components are available directly in the left Component Library while composing another Component, so nested components do not require a Gallery detour.
- Small `+` insertion controls appear between layers and at the end of a Component graph; the Inspector also has `+ Add Component`.
- The component picker can insert an existing saved Component/API Component or create a new nested Component in-place.
- Saving a nested child returns to its parent editor and inserts/updates it there; saving the top-level component returns to Gallery → Components.
- Circular component nesting is rejected in the UI and guarded again by the Python runtime compiler.
- Gallery component snapshots now carry nested dependency definitions so reusable nested Components remain self-contained when restored later.

## v0.7.52 — in-canvas Module insertion

- **+ Add Module** in the API editor now opens the Module chooser over the canvas instead of replacing Inspector content.
- Selecting a saved Module inserts it directly into the API execution canvas at the selected position.
- **+ New Module** starts a nested Module editor and returns the completed Module to the parent API canvas.
- The Inspector remains dedicated to the selected API function/component while the Module chooser is open.
- Clicking the canvas backdrop or **Cancel** closes the chooser without changing the graph.

## v0.7.49 — focused Module editor and atomic nested save

Reusable visual custom graphs are now presented as **Modules**. Low-level items remain Components, while API Component continues to describe Python/PyTorch API graphs.

- Opening or creating a Module enters a focused Module Editor. The outer Module is the editor root instead of inheriting `Untitled Model` in the breadcrumb/title.
- Model **Build**, **Gallery**, **Cloud & Repositories**, and the Model Workspace drawer are hidden while a Module or API Component is being authored. Saving the outer item returns to Gallery.
- In the Module Editor, **+ New Module** replaces Auto Connect. Compact `+` insertion controls remain between layers and after the last layer, and the redundant empty-canvas `+ Add Component` action was removed.
- Nested Modules use **Done**, not Save. Done applies the child to its parent draft and returns one level. Only the outermost **Save / Save As New** writes a Gallery item.
- The outer Gallery item carries recursive dependency snapshots, so the saved Module is a self-contained graph with its nested Modules/API Components rather than a stack of separately saved inner layers.
- Module identity is ID-based, not name-based. Parent and child Modules can therefore have the same display name while circular nesting is still rejected using the dependency-ID graph.
- Gallery/model restoration remaps definition IDs instead of merging definitions by visible name, preserving same-name nested Modules correctly.


## v0.7.52 — direct nested Module creation

- `+ Add Module` now creates a blank nested Module immediately; no chooser/notification is shown.
- Cancelling that nested Module rolls the transaction back completely, so the new Module disappears from its parent.
- Saved Modules are available directly from the left library while editing an API Component; nested API Components remain blocked.
- Module/API names can be edited from the Inspector, so auto-created nested Modules do not need a naming popup.
- Module editor and API editor use the same direct Add Module behavior and keep editor undo/redo isolated.


## v0.7.52 — direct editor insertion cleanup

- Removed the ADD MODULE / API COMPONENT chooser overlay from Module and API editing workflows.
- Removed inline per-layer `+` insertion buttons and the trailing `+` from the canvas.
- Components are inserted directly from the left Component Library after the selected layer.
- `Add Module` remains the single direct nested-Module action and immediately opens an empty nested Module editor.
- Cancel discards that nested Module; Done returns it to the parent draft.



## v0.7.53 — Add Module click fix

- Restores the missing direct nested-Module creation handler removed during the v0.7.52 editor cleanup.
- `Add Module` now immediately opens a blank child Module in both Module Editor and API Component Editor.
- `Done` inserts the child at the selected position; `Cancel` discards the child and restores the parent editor.
- No picker/overlay or per-layer `+` buttons are used.

## User Defined Functions in API Components

API Components can include a **User Defined Function** node. Write the Python function directly in the Inspector, validate its syntax against the notebook kernel, and connect it to normal MLB Studio components.

A User Defined Function has one central **Visual ↔ Function Mapping** section. The fixed visual inputs map directly to Python arguments (`Top Input`, `Main Input`, `Bottom Input`), while the fixed outputs map function return indexes/keys back to (`Top Output`, `Main Output`, `Bottom Output`). Custom terminals appear in the same mapping table, so terminal geometry and function binding are no longer configured in separate places.

For example:

```python
def split_state(x, residual):
    main = x
    skip = residual
    extra = x + residual
    return main, skip, extra
```

The visual mapper can bind **Main Input → `x`**, **Top Input → `residual`**, **Main Output ← `0`**, **Top Output ← `1`**, and **Bottom Output ← `2`**. Tuple/list indexes, dictionary keys, and object attributes are accepted as output selectors. Non-visual constants/settings remain under **Parameters**. `torch` and `torch.nn` are available automatically; other installed packages can be imported inside the user function source.

## User-defined source components

MLB Studio API Components can embed **User Defined Function** and **User Defined Class** source code. When a component is saved, Studio stores the source, entry point, detected import dependencies, source revision, and source hash in the component cache and embeds that cache in Gallery/project exports. Reopening or importing the component restores the source automatically.

Third-party Python libraries are **not installed automatically**. The dependency list is validated against the active notebook/local Python environment; install missing packages explicitly before building or running the component.

A User Defined Class node constructs one reusable object instance. Later Instance Method nodes can select that object from the object registry, so stateful objects can be reused across intervening graph operations without recreating the class instance.

### Custom named ports

User Defined Function nodes can use either the standard Main/Skip/Extra interface or **Custom Named Ports**. Named mode allows an arbitrary number of visual inputs and outputs. Each input maps to a Python function parameter; each output maps to `auto`, a tuple/list index, or a dict/object key. Named User Function ports can connect to other named User Functions or directly to standard MLB Studio Main/Skip/Extra ports.

## MLBricks maintained Data Gallery presets

Gallery → Data includes six ready-to-open text pipelines backed by MLBricks-maintained
Hugging Face dataset repositories:

- `MlBricks/tinystories` — full TinyStories mirror.
- `MlBricks/wikipedia-en-1b` — curated English Wikipedia edition at about 1B GPT-2 tokens.
- `MlBricks/cosmopedia` — full Cosmopedia mirror; Studio opens the `openstax` config.
- `MlBricks/fineweb-edu-1b` — curated FineWeb-Edu edition at about 1B GPT-2 tokens.
- `MlBricks/openwebmath-1b` — curated OpenWebMath edition at about 1B GPT-2 tokens.
- `MlBricks/ultrachat-200k` — normalized UltraChat SFT mirror with a common `text` column.

Every prebuilt data pipeline defaults to a 10,000-row quickstart so opening a Gallery card
does not accidentally process the entire maintained edition. Set **Max Rows = 0** on the
Hugging Face Dataset source only when full-dataset processing is intended.

The publishing utility is `tools/publish_mlbricks_datasets.py` (with a PowerShell wrapper at
`tools/publish_mlbricks_datasets.ps1`). It records upstream repository/revision provenance and
keeps the original dataset license/attribution requirements visible in the destination dataset
card/manifest. Hugging Face credentials are read from `HF_TOKEN` or the normal `hf auth login`
configuration and are never stored in Studio project files.


## v1.0 local persistence — drafts + Local Repository

MLBricks Studio now keeps lightweight design work across restarts instead of treating
the browser session as disposable:

- every model/data/component edit is browser-autosaved as a recovery draft;
- the Python side mirrors drafts into a local SQLite Studio store when the kernel bridge is available;
- **Cloud & Repositories → Local Studio Storage** lists recent drafts and named Local Repository items;
- users can save model designs, data pipelines, complete projects, and reusable Module/API Component definitions locally;
- saved components are also mirrored into the Local Repository when they are saved to Gallery;
- Hugging Face data nodes can reference a saved credential profile (default: `Default`) for private/gated datasets.

The local database is deliberately **design-only**. It stores graphs, component source/config,
layout, metadata, hyperparameters, and references/paths to external artifacts. It does **not**
store model parameter tensors, optimizer state, checkpoint bodies, or dataset contents. Large
artifacts remain in their proper local/cloud repository and Studio stores only the reference.
