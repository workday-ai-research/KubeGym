# Unverified literature, claims and attributions — KubeGym

_Compiled 2026-08-29. Indices consulted: OpenAlex REST API (authenticated), Crossref REST API,
arXiv Atom API. Semantic Scholar returned HTTP 429 for every anonymous request and could not be
used. Per-entry query logs are in `citation_verification.json`._

Totals: **74 citations verified**, of which **14 carry a venue-only human check**;
**3 items could not be verified at all**; **6 comparative or descriptive claims** are flagged below.

---

## A. Citations that could not be verified at all — do not present as confirmed

### A1. `qiu2023aware` — "AWARE: Automate Workload Autoscaling with Reinforcement Learning in Production Cloud Systems"
Believed USENIX ATC 2023. **Not found.** Queries issued: OpenAlex `title.search` (raw and
punctuation-stripped), OpenAlex `title_and_abstract.search`, OpenAlex
`raw_author_name.search:"Haoran Qiu"` (40 records scanned), Crossref `query.bibliographic`
(two phrasings), arXiv `ti:` exact-phrase and loose title query, Semantic Scholar (HTTP 429).
USENIX proceedings carry no DOIs and are thinly covered by OpenAlex, so absence here is weak
evidence of non-existence — but it is not confirmation either. Author list, exact title and venue
are all unverified. Commented-out placeholder in `refs.bib`.
**Verified substitutes by the same suspected author, if a citation is needed:** "Reinforcement
learning for resource management in multi-tenant serverless platforms" (doi:10.1145/3517207.3526971)
and "SIMPPO" (doi:10.1145/3542929.3563475).

### A2. Two further works (not cited)
Two related works have no public version that could be located. No identifier, venue or
year has been invented for either, and neither is cited.

## B. Verified title/authors/year, but VENUE inferred rather than asserted by an index

Title, author list and year for all 14 are index-confirmed. Only the venue string needs a human
glance. Grouped by why the venue was unavailable.

**No DOI; venue reconstructed from OpenAlex hosting-source name plus volume/page corroboration.**
Corroboration is strong for these; risk is low.

- `mao2019park` — Park. OpenAlex location source "Neural Information Processing Systems"
  [conference]; biblio vol 32, pp. 2490–2502. Rendered as *Advances in NeurIPS 32* (2019).
- `raffin2021sb3` — Stable-Baselines3. OpenAlex location source "Journal of Machine Learning
  Research" [journal]; biblio 22(268):1–8. Rendered as JMLR.
- `liang2018rllib` — RLlib. OpenAlex location source "International Conference on Machine
  Learning"; pp. 3053–3062. Rendered as ICML 2018.
- `jordan2020evaluating` — OpenAlex location source "International Conference on Machine
  Learning"; pp. 4962–4973. Rendered as ICML 2020.
- `herbst2013elasticity` — OpenAlex location source "International Conference on Autonomic
  Computing"; pp. 23–27. Rendered as ICAC 2013.
- `dutot2016batsim` — Crossref DOI 10.1007/978-3-319-61756-5_10, LNCS book series, pp. 178–197.
  Rendered as JSSPP / LNCS. Note the LNCS volume is dated 2017 while the workshop was 2016.
- `andrychowicz2021what` — OpenAlex primary-location source "International Conference on Learning
  Representations", no DOI. Rendered as ICLR 2021.
- `fan2021dras` — Crossref confirms *Software Impacts* vol 8, art. 100077; the arXiv version
  (2105.07526) is dated 2021.

**USENIX venues: no DOI and no venue string in any index. arXiv version cited instead where one
exists. Venue is an inference from page range alone — weaker.**

- `qiu2020firm` — FIRM. Only pp. 805–825 retrievable; consistent with OSDI 2020 but not asserted.
  arXiv:2008.08509 cited.
- `shahrad2020wild` — Serverless in the Wild. Only pp. 205–218 retrievable; consistent with
  USENIX ATC 2020 but not asserted. arXiv:2003.03423 cited. **This is the source of the Azure
  Functions 2019 trace, so the attribution matters — a human should confirm the ATC 2020 venue.**
- `wang2018peeking` — Peeking Behind the Curtains of Serverless Platforms. Only pp. 133–145
  retrievable, no DOI, no arXiv version found. Believed USENIX ATC 2018.

**Journal record absent from both indices; arXiv version cited instead.**

- `huang2022cleanrl` — CleanRL. Believed JMLR 2022; no journal record in OpenAlex or Crossref.
  arXiv:2111.08819 cited.
- `pineau2021improving` — NeurIPS reproducibility programme report. Believed JMLR 2021; no journal
  record found (only HAL, Papyrus and arXiv repository copies). arXiv:2003.12206 cited.

**Sub-track not asserted.**

- `towers2024gymnasium` — Crossref container is "Advances in Neural Information Processing Systems
  38" (2025), doi:10.52202/085713-4916. Whether this is the Datasets & Benchmarks track is not
  stated by the record. Also note the year is **2025**, not 2024 as the key suggests.

**One entry has neither DOI nor arXiv ID:**

- `colas2019hitchhiker` — "A Hitchhiker's Guide to Statistical Comparisons of Reinforcement
  Learning Algorithms". An OpenAlex work record exists (W2938030785, 2019, Cédric Colas, Olivier
  Sigaud, Pierre-Yves Oudeyer, hosted on arXiv) but carries no DOI and no arXiv identifier that we
  could extract. The only identifier in `refs.bib` is the OpenAlex ID. A human should locate the
  arXiv ID.

**Citation keys whose embedded year differs from the record year** (label only, not an error;
listed so nobody "fixes" them): `santos2024gwydion`→2025, `towers2024gymnasium`→2025,
`dutot2016batsim`→2017, `pineau2021improving`→2020, `huang2022cleanrl`→2021,
`huang2024cleanba`→2023, `patterson2024empirical`→2023, `chen2018survey`→2019.

---

## C. Comparative and descriptive claims not backed by a verified source

These appear in `related_work.tex` and/or `positioning.md`. Each is either marked inline or
carried in a footnote; none should reach camera-ready unbacked.

1. **"Neither prior effort reports per-constant calibration provenance or a gating mechanism of
   this kind."** (related_work.tex, footnote attached.) Basis: the retrieved abstracts and index
   records of RLScale-Bench (arXiv:2605.26418) and BatchBench (arXiv:2605.12272) do not mention
   per-constant provenance. **Absence from an abstract is not absence from an artifact.** To make
   this claim safely, a human must inspect both released artifacts — BatchBench states its
   implementation is unreleased, so for that one the claim can only be made about the design.

2. **Every `?` cell in the positioning comparison table.** These mean "not asserted by the record
   consulted", not "no". Converting any `?` to `no` in the paper requires downloading and
   inspecting the corresponding artifact. Not done here.

3. **"KubeGym is the first to gate a run as uncalibrated when a claims-relevant constant is a
   placeholder."** `[UNVERIFIED]` — this is a novelty claim over the whole benchmark literature and
   our search was targeted, not exhaustive. Recommend weakening to "we are not aware of a prior
   benchmark that…" or dropping the comparative form entirely.

4. **"Park was the first collection to put a common learning-oriented interface in front of twelve
   real systems problems."** The count (twelve) and the framing come from the retrieved record;
   the **"was the first"** ordering claim is `[UNVERIFIED]`. Recommend removing "first".

5. **Whether RLScale-Bench validates its workload patterns distributionally** is not stated in its
   abstract and its artifact was not inspected. `[UNVERIFIED]` — flagged inline in
   `positioning.md` §5.

6. **Whether gym-hpa replays a real public production trace.** Its abstract, as retrieved,
   describes complex microservice-based applications in Kubernetes but does not state a public
   trace source. Recorded as `?` in the table; do not assert either way.

---

## D. Things checked and found to be fine (recorded so they are not re-litigated)

- **Azure Functions 2019 trace attribution** resolves to Shahrad et al., arXiv:2003.03423, with the
  full ten-author list index-confirmed. Only the ATC 2020 venue string is inferred (§B).
- **ACM records with split title/subtitle** (AutoScale, CloudScale, Borg: the next generation,
  ns-3 meets OpenAI Gym, SeBS) initially failed title matching and were re-resolved by DOI against
  Crossref, which returned the canonical title, subtitle, venue, pages and full author list.
- **KIS-S has a peer-reviewed version**: IEEE IPCCC 2025, doi:10.1109/ipccc66453.2025.11304654,
  pp. 1–8, in addition to arXiv:2507.07932. Cite the IPCCC version.
- **Khilji et al. has a peer-reviewed version**: IEEE SOSE 2026, doi:10.1109/sose71128.2026.00015,
  pp. 51–62, in addition to arXiv:2606.16555.
- **vHive**: an earlier Crossref match returned the ACM *artifact* DOI (10.1145/3410279) rather than
  the paper. Corrected to the ASPLOS 2021 paper DOI 10.1145/3445814.3446714.
- **"Deep Reinforcement Learning that Matters"**: an initial fuzzy match returned an unrelated MIT
  Press book chapter. Corrected to AAAI 2018, doi:10.1609/aaai.v32i1.11694 (arXiv:1709.06560).
- **CleanRL**: an initial fuzzy match returned "CleanQRL" (IEEE QCE 2025). Corrected to
  arXiv:2111.08819.
