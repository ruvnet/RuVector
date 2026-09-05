# The 26-Second Microseism: Literature Review

> Scope note. This review separates **established facts** (well-replicated
> across independent groups) from **open questions** (actively debated). Where
> exact citation details are uncertain, older work is framed generically rather
> than fabricated. The single firm modern anchor is Bruland & Hadziioannou
> (2023). Treat everything else as best-effort summary subject to revision when
> the papers corpus (`data/papers/`) is populated.

## 1. What is the 26-second pulse?

There is a persistent, narrowband seismic signal at a period of roughly
**26 seconds** (a frequency near **0.038 Hz**) that has been recorded on
broadband and long-period seismometers across multiple continents for decades.
It is not a one-off event: it recurs, it is remarkably stable in frequency, and
it has been observed in array studies to back-azimuth toward a fixed region in
the **Gulf of Guinea**, in the **Bight of Bonny near the island of São Tomé**
(off the coast of West Africa).

The signal is weak — typically only resolvable by stacking, array beamforming,
or long-duration spectral analysis — but it is one of the most temporally
**persistent** narrowband microseisms known. This persistence and the fixed
source location are the two facts that make it scientifically interesting: most
microseism energy migrates seasonally with storm tracks and ocean-wave climate,
whereas this pulse stays put.

### Established facts (high confidence)
- A narrowband signal exists at ~26 s period; it is **reproducible** across
  independent stations and independent analysis groups.
- Source localization (multiple array studies) points to the **Gulf of
  Guinea / Bight of Bonny** region, near São Tomé.
- The signal has been **observed for decades**, not a transient artifact of one
  instrument generation.
- The amplitude is **time-varying** and appears to have seasonal structure.

### Open questions (low/medium confidence)
- The **physical mechanism** generating the carrier frequency is unresolved.
- Whether the 26 s carrier and the associated **gliding tremors** share a
  single source process or are two coupled processes.
- Why the frequency is so **stable** over decades despite a presumably
  changing ocean/atmosphere forcing.

## 2. Discovery history

The 26 s signal was, by the generally repeated account in the literature, first
noted by **Jack Oliver** at the Lamont Geological Observatory in the **early
1960s**, in the course of cataloguing long-period background seismic noise. At
the time the global broadband network was sparse and the tools for
source localization were limited, so the observation remained a curiosity
rather than a research program.

The signal was independently re-noticed and partially localized in the **2000s**
as digital broadband arrays and modern beamforming made it tractable to point a
back-azimuth at the Gulf of Guinea. A handful of array-seismology and
ambient-noise studies over the following two decades refined the source region
toward the Bight of Bonny and confirmed the long-term persistence.

> Citation honesty: the Oliver attribution and the 2000s re-localization are
> reported consistently in secondary sources, but this review does **not** fix
> exact DOIs or author lists for that older work. Those entries in
> `data/papers/` must be filled with primary sources before any claim depending
> on them is promoted past Level 2 of the discovery ladder (see
> `benchmark-design.md`).

## 3. Microseism theory background

To reason about candidate mechanisms it helps to recall how ordinary
microseisms are generated. "Microseisms" are continuous, low-amplitude ground
oscillations driven primarily by ocean waves coupling into the solid Earth.
They fall into two well-established classes.

### 3.1 Primary microseisms
- **Frequency band:** roughly the same as ocean swell, ~0.05–0.12 Hz
  (periods ~8–20 s).
- **Mechanism:** ocean swell propagating into shallow water interacts with
  **sloping bathymetry and the continental shelf**, transferring pressure
  fluctuations to the seabed at the **swell frequency itself** (a linear,
  single-frequency coupling).
- **Key dependence:** requires waves to reach shallow, sloping seafloor;
  strongly modulated by **coastline geometry and shelf width**.

### 3.2 Secondary microseisms
- **Frequency band:** roughly **double** the swell frequency, ~0.1–0.35 Hz
  (periods ~3–10 s); usually the dominant microseism peak.
- **Mechanism:** **nonlinear wave–wave interaction** between two wave trains of
  similar frequency travelling in nearly opposite directions
  (the Longuet-Higgins / Hasselmann mechanism). Two waves of frequency *f*
  produce a standing pressure fluctuation at **2f** that does not decay with
  depth, coupling efficiently to the seabed.
- **Key dependence:** requires **opposing wave trains** (e.g. coastal
  reflection, or two storm systems), so it is sensitive to wave directionality.

### 3.3 Why long-period mechanisms remain debated
The 26 s pulse sits at a period (**~26 s**) that is **longer** than typical
primary microseisms. That immediately raises a problem: ordinary swell does not
carry much energy at 26 s, and neither the standard primary nor the standard
secondary mechanism naturally produces a narrowband line at exactly this
period that stays fixed for decades. Candidate long-period explanations
therefore reach for less standard physics:

- **Resonance / mode selection** — a geometric resonator (shelf, bay,
  water-column mode) that picks out one frequency from broadband forcing.
- **Infragravity waves** — very-long-period ocean waves (periods of tens of
  seconds to minutes) generated by nonlinear interaction of swell near coasts;
  these *can* reach the 26 s band, but tying them to a single fixed line is
  hard.
- **Volcanic / hydrothermal tremor** — a sustained subsurface fluid or magmatic
  oscillator near São Tomé, which is a volcanically active region. This more
  naturally explains a *fixed source* and *narrowband stability* but must then
  explain the apparent **ocean / seasonal modulation**.

None of these is established. They are the hypothesis space, catalogued and
scored in `hypothesis-catalog.md`.

## 4. The modern anchor: Bruland & Hadziioannou (2023)

The most important recent contribution, and the reason this harness exists, is
the work of **Bruland & Hadziioannou (2023)**. Their key reported findings:

1. **Gliding tremors.** Alongside the steady ~26 s line, they identify
   associated **gliding tremors** — signals whose frequency drifts (glides)
   over time — that **start from the same frequency** as the 26 s microseism
   and emanate from the **same fixed source region** (Gulf of Guinea / Bight of
   Bonny). The gliding behavior is a qualitatively new observable that a
   correct mechanism must reproduce, not just the steady carrier.

2. **Shared origin.** The fact that the glides initiate at the carrier
   frequency and share the source location argues that the 26 s line and the
   glides are **physically linked**, not coincidental.

3. **Interpretation.** They argue that the combination of **stability**,
   **spatial fixity**, and **decades-long persistence** points to a **gap in
   our understanding of long-period oceanic and volcanic signals** — i.e. that
   neither the standard ocean-wave picture nor a naive volcanic picture, on its
   own, comfortably accounts for all the observables.

This is the framing the harness adopts: the pulse is not a mystery to be
narrated but a **prediction target** for which competing mechanisms make
**different, testable** forecasts (see `benchmark-design.md`).

### What Bruland & Hadziioannou (2023) establishes vs. leaves open
- **Establishes (taken as anchor fact in this harness):** existence of gliding
  tremors associated with the 26 s microseism, starting at the carrier
  frequency, from the fixed Gulf of Guinea source.
- **Leaves open:** the generating mechanism; whether one process or two coupled
  processes are responsible; the quantitative dependence of amplitude on ocean
  state.

## 5. Competing mechanism families (summary)

| Family | Natural strength | Natural difficulty |
|---|---|---|
| Ocean-wave / storm forcing (shelf resonance, infragravity) | Explains seasonal/ocean modulation; standard physics | Hard to produce one fixed narrowband line stable for decades |
| Volcanic / hydrothermal origin | Explains fixed location and narrowband stability | Must explain apparent ocean/seasonal modulation and gliding |
| Coupled ocean + geology | Can explain *both* the stable carrier and modulated/gliding behavior | More parameters; harder to falsify cleanly |

The detailed, scored version of this table — with killer contradictions and the
distinguishing ruVector test for each — lives in `hypothesis-catalog.md`.

## 6. Implications for the harness

The literature gives us three things to exploit:

1. **A fixed source** means source-region ocean and weather variables are
   well-defined inputs, not a moving target.
2. **A stable carrier plus gliding tremors** gives at least two distinct
   observables (steady amplitude; glide onset/rate) that mechanisms must
   jointly explain — a strong discriminator.
3. **Decades of records** mean there is, in principle, enough held-out data to
   run an honest train/test split over months and years.

The danger to avoid: **over-narrating**. Every claim that "Earth has a
heartbeat" must be reducible to "mechanism *M* predicts observable *O* with
skill *S* better than baseline *B*." That reduction is the job of
`benchmark-design.md`.

## References

> Conventions: the one firm modern citation is given with author/year. Older
> and contextual work is described generically; precise DOIs and author lists
> are intentionally **omitted rather than invented**. Populate
> `data/papers/` with verified primary sources and update this list; every
> promoted claim must map to a file there.

1. **Bruland, S. & Hadziioannou, C. (2023).** Study identifying gliding tremors
   associated with the 26-second microseism originating from a fixed source in
   the Gulf of Guinea (Bight of Bonny, near São Tomé), arguing that the signal's
   stability, spatial fixity, and decades-long persistence indicate a gap in
   understanding of long-period oceanic and volcanic signals. *(Primary anchor
   for this harness; full bibliographic record to be verified and stored in
   `data/papers/`.)*

2. **Oliver, J. (early 1960s, Lamont Geological Observatory).** First reported
   notice of the ~26 s long-period background signal during cataloguing of
   long-period seismic noise. *(Attribution per repeated secondary accounts;
   primary source to be located and verified.)*

3. **Array-seismology / ambient-noise localization studies (2000s–2010s).**
   Independent re-detection and source localization of the 26 s signal to the
   Gulf of Guinea / Bight of Bonny using modern broadband arrays and
   beamforming. *(Multiple groups; specific citations to be verified and
   stored.)*

4. **Longuet-Higgins (secondary microseism theory) and Hasselmann
   (statistical wave-forcing theory).** Foundational theory of microseism
   generation by nonlinear wave–wave interaction producing energy at twice the
   swell frequency. *(Classic foundational work; cited generically here.)*

5. **Primary-microseism / shelf-coupling literature.** Body of work on swell
   coupling to sloping bathymetry and continental shelves at the swell
   frequency. *(Cited generically; specific sources to be stored.)*

6. **Infragravity-wave literature.** Work on very-long-period ocean waves
   (tens of seconds to minutes) generated by nonlinear processes near coasts,
   relevant to long-period seismic coupling. *(Cited generically.)*
