# Building this system from scratch

A phase by phase implementation plan for the LGA violence risk service, written
as the order you would actually build it in rather than the order the finished
repository reads in.

The guiding principle throughout is that each phase ends at a point where
something is verifiably true. You do not move to the next phase because the code
compiles; you move because you have printed a number and checked it. Roughly ten
phases, and the dependency chain between them is real: reversing any two of the
first five will cost you a rewrite.

## Dependency order at a glance

```
Phase 0  Define the prediction target
   |
Phase 1  Config, schema, storage         <- everything writes here
   |
Phase 2  Reference geography             <- every join key comes from here
   |
Phase 3  Avro contracts and the bus      <- fixes record shapes before 9 connectors exist
   |
Phase 4  Ingestion (UCDP first)          <- the first phase that produces data
   |
Phase 5  Feature panel and Hawkes        <- the analytical core
   |
Phase 6  Baselines, then the model       <- baselines before the model, not after
   |
   +---> Phase 7  Actor graph, brief, alerting
   +---> Phase 8  API, CLI, Shiny dashboard
   |
Phase 9  Spark path, scheduling, tests, figures, write-up
```

Phases 7 and 8 are independent of each other and can run in parallel if you are
splitting work across a group.

---

## Phase 0. Decide what you are predicting

**Time: half a day. Produces no code, and determines everything.**

Before opening an editor, write down and agree four things.

**The unit of analysis.** One LGA in one ISO week. Every row in the system is a
(lga_code, week_start) pair. Choosing a day instead would give you 774 x 4380
rows with a positive rate near 0.07%, which is too sparse to learn from. Choosing
a month would be useless operationally, because the brief promises a seven day
horizon.

**The label.** A banditry or kidnapping class event occurring in the LGA in the
seven days beginning on that Monday. Write it as an interval, `[W, W+7)`, and
keep that notation in the code comments, because half of the leakage bugs you
will write come from being vague about whether the boundary is inclusive.

**The feature cutoff.** Everything on the row must be computable from data
timestamped strictly before `W`. Not "before the event", not "in the previous
month", strictly before the Monday that opens the horizon.

**Which event classes are label and which are feature.** This is the decision
that pays off later. In this build, non-state conflict and one-sided violence
became the label class, while state-based violence, meaning military operations,
became a separate class used only as a predictor. A military operation in an
adjacent LGA is a leading indicator of displacement, not an instance of the thing
being predicted, and conflating the two would both inflate your metrics and
destroy the explainability story.

**Gate before Phase 1:** you can state the label in one sentence, including the
interval boundaries, without hedging.

---

## Phase 1. Project skeleton, configuration and storage

**Time: one day. Files: 10.**

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, `src` layout, pytest config |
| `requirements.txt` | Pinned lower bounds for every dependency |
| `.env.example`, `.gitignore` | Secret slots, and keeping data out of git |
| `config/settings.yaml` | Every tunable in one place, no secrets |
| `src/pau_risk/config.py` | Loads YAML, overlays environment, resolves paths |
| `src/pau_risk/logging_utils.py` | One logging format for every entry point |
| `sql/schema_postgres.sql` | The real target: PostGIS geometry, pgvector embeddings |
| `sql/schema_sqlite.sql` | Identical column names, geometry as WKT text |
| `src/pau_risk/storage/engine.py` | Backend probe, capability detection, DDL runner |
| `src/pau_risk/storage/repository.py` | Dialect-aware upserts and typed readers |

**Why the schema comes before the data.** The schema is the contract between the
nine connectors, the feature build, the model, the API and the dashboard. If you
let it emerge from whatever the first connector happens to return, you will
rewrite it when the second connector returns something different.

**The one design decision worth arguing about.** Write two schema files with
identical column names, one for PostgreSQL and one for SQLite, and put a
capability probe in `engine.py` that picks whichever is reachable. The brief
specifies PostgreSQL with PostGIS and pgvector, and also permits a file back end
as an alternative. Supporting both is not a compromise: it means the whole
pipeline runs on a laptop with no services installed, while the production target
stays exactly what the architecture says. The cost is that every query must be
written in the intersection of the two dialects, which in practice means using
the upsert helper in `repository.py` rather than writing raw `ON CONFLICT`.

**Verification gate:**

```bash
python -c "from pau_risk.storage import init_schema, table_counts; init_schema(); print(table_counts())"
```

Every table appears with a count of zero. Then set `DATABASE_URL` to a
PostgreSQL instance and confirm the same command works there.

---

## Phase 2. Reference geography, the fixed point of the system

**Time: one day. Files: 4.**

| File | Purpose |
|---|---|
| `src/pau_risk/reference/geography.py` | Download boundaries, build registry, derive adjacency, simplify for the map |
| `src/pau_risk/reference/calendar_ng.py` | Nigerian holidays, seasons, school terms |
| `src/pau_risk/reference/__init__.py` | Cached accessor so polygons load once per process |
| `scripts/build_reference.py` | One command that populates the registry |

**Get the boundaries from OCHA, not from a list of names.** The Common
Operational Dataset for Nigerian administrative boundaries carries all 774 admin
level 2 units with official P-codes, parent state, senatorial district, area and
centroid. The P-code scheme is the one GRID3 uses, so the adjacency graph you
derive from it is the GRID3 adjacency the brief asks for. Query the HDX API for
the dataset, take the `nga_admin2.geojson` member from the archive.

**Derive adjacency from shared borders, not centroid distance.** Contagion in
banditry follows roads across a shared boundary, not straight lines. Use a
shapely STRtree to find candidate pairs, compute the shared border length, then
row-normalise so each LGA's neighbour weights sum to one. Two details matter.
Buffer the polygons by about 0.01 degrees before intersecting, because the source
data has sliver gaps that would otherwise disconnect genuinely adjacent units.
And give riverine LGAs with no land border their nearest neighbours by distance,
otherwise the spatial term of the Hawkes kernel switches off for exactly the
areas where waterborne movement matters.

**Write a simplified copy of the polygons.** The source file is 5.6 MB and stalls
a browser. Simplifying to a 0.01 degree tolerance produces 686 KB, which is
indistinguishable at national and state zoom. Analysis reads the original, the
map reads the simplified copy. Do this now rather than discovering it in Phase 8.

**Calendar features are cheap and defensible.** Gregorian computus for Easter,
tabular Islamic calendar for the Eids, fixed dates for the national holidays, plus
a dry season flag and a school term flag. Widen the festival window by a day on
each side to absorb the uncertainty in the Islamic conversion, and say in the
docstring that you have done so.

**Verification gate:**

```bash
python scripts/build_reference.py
```

Expect exactly 774 LGAs, 37 states, 6 geopolitical zones, roughly 4,677 adjacency
edges at a mean degree near 6.0, and a minimum degree of at least 3. If the mean
degree is below 4 your buffer is too small. If it is above 9 it is too large.

---

## Phase 3. Avro contracts and the transport bus

**Time: half a day. Files: 5.**

| File | Purpose |
|---|---|
| `src/pau_risk/schemas/incident.avsc` | Located violent event |
| `src/pau_risk/schemas/document.avsc` | Unstructured report awaiting extraction |
| `src/pau_risk/schemas/chatter.avsc` | Short social post with a threat score |
| `src/pau_risk/stream/registry.py` | Versioned subjects, backward compatibility check |
| `src/pau_risk/stream/bus.py` | Confluent wire format over Kafka, or Avro files offline |

**Do this before writing any connector.** Nine sources returning nine different
shapes is nine refactors if you define the contract afterwards. Three record types
is the right number: everything that has a date and a place is an incident,
everything that is prose is a document, everything short and real-time is chatter.

**The schema registry does not need to be Confluent.** A JSON file holding one
immutable version chain per subject, with a compatibility check that rejects
removed fields and added fields without defaults, reproduces the contract that
matters. Producers only ever ask for a schema id, so swapping in the hosted
registry later replaces one class and touches nothing else.

**Make the offline path identical, not approximate.** When no broker answers,
write the same encoded payloads to Avro object container files, one per topic and
producer and date, and have `replay()` read them back in order. This is what lets
you exercise serialisation, validation and the consumer without standing up
Kafka. Silence the Kafka driver's retry logging, because a refused connection is
the expected path and printing it as an error is misleading.

**Verification gate:** encode a record, decode it, confirm the defaults were
applied and the date logical type round-tripped. Then confirm the compatibility
check raises on a removed field.

---

## Phase 4. Ingestion, starting with the source that unblocks everything

**Time: two to three days. Files: 8.**

| File | Purpose |
|---|---|
| `src/pau_risk/ingest/base.py` | Connector contract, readiness reporting, record builders |
| `src/pau_risk/ingest/ucdp.py` | UCDP GED bulk CSV and API |
| `src/pau_risk/ingest/acled.py` | ACLED read API |
| `src/pau_risk/nlp/extract.py` | Gazetteer, money, casualties, threat scoring |
| `src/pau_risk/ingest/web.py` | Polite fetching, robots, caching, HTML and PDF to text |
| `src/pau_risk/ingest/documents.py` | The five document producers |
| `src/pau_risk/ingest/social.py` | X and Telegram watchlists |
| `src/pau_risk/ingest/runner.py` | Runs the producers, then drains topics to the warehouse |

**Build in this order, and resist the temptation to do them alphabetically.**

1. `base.py`. The contract: a connector reports readiness, fetches a window, and
   returns records already shaped to an Avro schema. Readiness is the important
   part. A connector that lacks a key must say `needs_credentials` and be skipped
   with a logged reason, so a partial run is visible rather than looking like a
   quiet week.
2. `ucdp.py`. Do this second because it is the only source that gives you a
   training corpus without credentials. The versioned bulk CSV is public and
   covers 1989 to 2024. Everything in Phases 5 and 6 is blocked until it works.
3. `nlp/extract.py`. Needed before the document connectors, because a document is
   worthless until you know which LGA it refers to.
4. `web.py`, then `documents.py`, then `social.py`, then `acled.py`.
5. `runner.py` last, once you know what the connectors return.

**Trust coordinates over text.** UCDP has an `adm_2` field. It is inconsistently
spelled and often empty. Ignore it and do a point in polygon join against the
registry, falling back to nearest centroid within 40 km and leaving anything
further unassigned. On this dataset that locates 99.7% of events inside a
polygon. A connector that force-fits the remainder into the nearest unit is
manufacturing data.

**The gazetteer is where the extraction work actually is.** Build it from the
registry rather than a hard-coded list. Handle compound names such as
Wasagu/Danko in all three written forms, handle hyphens, and handle the roughly
one name in ten that is shared between states by resolving against a state
mention in the same text and returning a low-confidence candidate set when there
is none. Match money in both `N15 million` and `15,000,000 naira` forms, and
match casualty counts in words as well as digits, because "Two herders shot dead
in Plateau" is what a Nigerian headline actually looks like.

**Split producing from consuming.** Producers publish to the bus and stop. One
consumer reads the topics back and is the only component that writes to the
warehouse. A source that fails, rate-limits or lacks a key then cannot leave the
warehouse half written, and replaying a topic reconstructs the tables exactly
because every write is an upsert on a natural key.

**Verification gate:**

```bash
python -m pau_risk.cli ingest --since 2010-01-01 --until 2024-12-31
```

Check the percentage of events located inside a polygon, the count by source and
event class, and the year-by-year distribution. On this build: 6,875 events,
3,640 in the label class, spread evenly across 2013 to 2024. A year with ten
times its neighbours means your date parsing is wrong.

---

## Phase 5. The feature panel and the Hawkes process

**Time: three to four days. This is the hardest phase. Files: 3.**

| File | Purpose |
|---|---|
| `src/pau_risk/features/hawkes.py` | Multivariate Hawkes over the adjacency graph |
| `src/pau_risk/features/panel.py` | The LGA by week panel and every feature family |
| `tests/test_panel.py`, `tests/test_hawkes.py` | Written during this phase, not after |

**Write the leakage tests first.** This is the one place in the build where test
first genuinely pays. A window that accidentally includes the current week
produces plausible numbers and excellent metrics, and nothing about the code
looks wrong. Construct a tiny panel with events in known weeks and assert the
exact value each window should take.

**Enforce the cutoff structurally.** Do not use a rolling window and remember to
shift it. Build an exclusive prefix cumulative sum and take differences out of
it, so the sum over the `k` weeks ending immediately before week `w` is
`prefix[w] - prefix[w-k]` by construction. The current week cannot enter that
expression, which is a stronger guarantee than a convention you have to remember.

**The six feature families, in the order they are worth building:**

1. Own history: lagged counts and fatalities over 1, 2, 4, 8, 12, 26 and 52 weeks.
2. Neighbour history: the same windows applied to the adjacency-weighted sum over
   neighbours. This is how displacement into a quiet LGA becomes visible.
3. State operations: military activity counted separately, in the LGA and in its
   neighbours. This is the "active military operation in an adjacent LGA" driver
   the brief names.
4. Hawkes intensity: the fitted decomposition, described below.
5. Calendar: festival within the horizon, dry season, school term, week of year.
6. Attention and chatter: document mentions and social volume against the LGA's
   own recent baseline.

**Now the Hawkes process itself.** The model is

```
lambda_i(t) = mu_i
            + alpha_self * sum over past events in i of beta * exp(-beta * (t - s))
            + alpha_nb   * sum over neighbours j, weighted by w_ij, of the same
```

Four implementation decisions carry the whole thing.

*Parameterise by branching ratio and self share.* Set `alpha_self = rho * phi`
and `alpha_nb = rho * (1 - phi)`. Because neighbour weights are row-normalised,
every row of the branching matrix sums to `rho`, so stationarity is exactly
`rho < 1` and is enforced by construction rather than by a penalty. It also makes
the fitted values readable: `rho` is the share of events that are triggered by an
earlier event, and `phi` says how much of that contagion stays inside the LGA.
Optimise the four parameters through a sigmoid and a log, unconstrained, with
L-BFGS-B.

*Anchor the baseline, do not free it.* Setting `mu_i` free would be 774
parameters and the baseline would absorb the clustering the excitation is meant
to explain. Anchor it to each LGA's own Laplace-smoothed empirical rate and scale
it by a single fitted parameter. Four parameters across 774 dimensions.

*Evaluate the excitation sums by forward recursion, not the closed form.* The
closed form factorises into `exp(beta * s)`, which overflows once the observation
window is a few thousand days long at realistic decay rates. Walk the evaluation
times in order, decay the running sum by the gap, then add the source events that
fall in that gap. Use strict inequality so an event never excites itself or
anything sharing its timestamp.

*Fit on the training window only, then apply to everything.* Load training events,
fit, then reload the full event history for feature generation while keeping the
fitted parameters. The intensity at a date then conditions on everything
observable at that date without the parameters having seen the test period.

**Test the estimator by recovering parameters you generated.** This catches
errors that no amount of reading will. One warning from experience: the simulator
must be a joint Ogata thinning across all dimensions on one clock. Simulating
each LGA to the horizon in turn leaves the first LGA with no neighbour history to
be excited by, which silently removes the cross term you are trying to recover,
and you will spend an afternoon debugging a correct estimator.

**Gate features on train and serve skew, not on rarity.** A feature that fires on
half a percent of rows is not weak, it is describing a rare event, which is the
whole point. What must be excluded is a feature that is nearly empty across
training but populated in recent weeks, because its source started reporting part
way through. Compare training coverage against recent coverage explicitly and
drop only on that pattern, plus anything with zero variance.

**Verification gate:** the leakage tests pass, and the fitted parameters are
plausible. On this build: branching ratio 0.50, self share 0.82, half life 22
days. A branching ratio above 0.9 means your baseline is under-parameterised. A
half life under a day means the optimiser has run to a bound.

---

## Phase 6. Baselines first, then the model

**Time: two days. Files: 6.**

| File | Purpose |
|---|---|
| `src/pau_risk/models/metrics.py` | Precision and recall at k, calibration, lift |
| `src/pau_risk/models/baselines.py` | Persistence, historical rate, Hawkes alone |
| `src/pau_risk/models/train.py` | Split, fit, calibrate, evaluate, log, persist |
| `src/pau_risk/models/explain.py` | SHAP attribution with human-readable labels |
| `src/pau_risk/models/predict.py` | Score a week, tier it, rank it, write it |
| `scripts/tune_model.py` | Validation-only sweep |

**Write the metrics module before anything it will measure.** ROC AUC on a 0.49%
positive rate is close to useless on its own: a model can post 0.92 while filling
the top of the list with false alarms. The metric that matters is precision and
recall in the top k of each week, computed within weeks and then averaged,
because the ranking is used one week at a time. Pooling across weeks gives a
different and wrong answer.

**Then write the baselines, and run them, before you fit anything.** Four of
them: persistence (last week's incident list, which is current practice), a 52
week historical rate, the Hawkes intensity alone, and later gradient boosting
without the Hawkes features. You need to know what you have to beat. On this
build the Hawkes process on its own reaches recall at 20 of 0.345, which is
higher than gradient boosting on its own at 0.327, and discovering that after
building the full hybrid would have been much less useful than discovering it
before.

**Split strictly by time.** Train to end of 2021, validate on 2022, test on 2023
onward and touch it once. Adjacent weeks in the same LGA share almost all their
feature values, so a random split lets the model see the answer to a question it
is about to be asked.

**Calibrate on validation, and break the ties.** Fit isotonic regression on the
validation fold alone. Then note that isotonic is a step function: it maps whole
intervals of raw score onto one calibrated value, so inside a plateau the ordering
of 774 LGAs becomes arbitrary and measured recall at 20 falls even though the
probabilities are more honest. Add one millionth of the raw score to restore the
ordering within each plateau. On this build that recovered recall at 20 from
0.338 to 0.355 while moving no probability by more than one part in a million.

**Tune on validation only, and record the sweep.** A small grid over depth,
learning rate, positive class weight and minimum child weight is enough. Leaving
the positive class unweighted ranked better than up-weighting it, because the
objective already optimises area under the precision-recall curve and the
isotonic step restores the probability scale afterwards.

**Verification gate:** a comparison table where the hybrid beats every baseline
on both ranking and calibration, and a calibration plot whose points sit on the
diagonal. If the hybrid does not beat the Hawkes process alone, say so in the
write-up rather than hiding it.

---

## Phase 7. The three downstream products

**Time: two days. Files: 3. Can run in parallel with Phase 8.**

| File | Purpose |
|---|---|
| `src/pau_risk/graph/actors.py` | Actor to LGA, actor to state, and rivalry edges |
| `src/pau_risk/brief/generator.py` | Grounded prompt, template fallback |
| `src/pau_risk/alerting/job.py` | Threshold, cooldown, escalation, audit trail |

**The actor graph is built from coded fields, not inferred from text.** Weight
edges by an exponential recency decay with a one year half life, because a group
that last appeared in 2015 tells you much less than one that appeared in March.
Keep rivalry pairs separately: competition between armed groups over the same
territory is a reliable precursor of escalation against the civilians living there.

**The language model writes prose and nothing else.** Assemble every figure from
the warehouse first and pass it in as a structured context block. Instruct the
model to use only what it is given. Two reasons: a brief that invents a number is
worse than no brief because it will be acted on, and the context contains text
drawn from news headlines, which is untrusted input and must be delimited and
declared as data rather than instruction. Provide a deterministic template
fallback and label its output as such, so a reader always knows which they are
reading.

**The failure mode in alerting is repetition, not omission.** An LGA that stays
severe for six weeks should generate one alert, not six, or the recipient learns
to ignore the channel. Implement a cooldown, override it on escalation because a
tier increase is new information, write every decision to an audit table
including the suppressed ones, and default to a dry run because dispatch is an
outward-facing action.

---

## Phase 8. Service layer and dashboard

**Time: two to three days. Files: 4. Can run in parallel with Phase 7.**

| File | Purpose |
|---|---|
| `src/pau_risk/api/main.py` | Read endpoints, ingestion endpoints, GeoJSON for the map |
| `src/pau_risk/cli.py` | One subcommand per stage, plus `pipeline` |
| `app/R/data_access.R` | DBI reads, reticulate for the one write that needs Python |
| `app/app.R` | Choropleth, ranked table, drill-down, brief pane |

**Read through SQL, write through Python.** The natural division for the
reticulate binding is not "everything through Python". Reads are plain queries
that DBI handles cleanly against the same SQLite or PostgreSQL warehouse. The one
operation that genuinely needs Python is generating the brief, because that call
owns the model credential and the prompt construction. Splitting it this way also
means the dashboard still opens when the Python environment is absent.

**Give the map three fallbacks.** With `sf` installed, read simple features and
use `addPolygons`. Without it, read the simplified GeoJSON with `jsonlite` and use
`addGeoJSON`, injecting a per-feature style, which avoids a GDAL dependency for
what is only a rendering step. With no boundary file at all, fall back to
graduated circles on the centroids already in the registry.

**Arrange the screen around the sequence a council works through.** Map answers
where, ranked table answers which areas specifically, selecting one opens the
drivers, actors and recent incidents which answer why, and the brief pane holds
what goes into the minutes.

**Verification gate:** start the app, click the top row, and confirm the drivers,
actors and incident history all populate for that LGA. Then hit each API endpoint
and confirm the run id is identical across all of them.

---

## Phase 9. Scale-out path, scheduling, hardening and write-up

**Time: two days. Files: 6 plus tests.**

| File | Purpose |
|---|---|
| `src/pau_risk/features/spark_panel.py` | The window aggregations in Spark SQL |
| `scripts/weekly_job.sh` | Cron entry point with per-stage failure handling |
| `scripts/make_figures.py` | Evaluation figures and their written interpretation |
| `tests/test_guards.py` | Regression tests for the defensive controls |
| `README.md`, `docs/figure_notes.md` | The write-up |

**Prove the Spark path rather than asserting it.** Express the same window
aggregations with `rowsBetween(-n, -1)`, which excludes the current row by
construction and is the direct equivalent of the exclusive prefix sums. Then join
the Spark output against the pandas output and assert the maximum absolute
difference is zero across every column and all 484,524 rows. That comparison is
the evidence that the architecture scales without changing the answer.

**Separate the cron stages.** Run them as separate commands rather than one
pipeline call, so a failure in the brief does not prevent the alerts going out.
Retrain monthly rather than weekly: a panel that grows by 774 rows a week does
not change the fit, and retraining weekly makes week to week movements in the
score impossible to attribute because the model and the data have both moved.

**Interpret every figure in writing, next to the figure.** Generate the notes in
the same script that generates the plots so the two cannot drift apart.

---

## The traps, in the order they will bite you

These cost real time on this build. Each one produces working-looking output
rather than an error, which is why they are worth knowing in advance.

**1. An empty panel trains silently.** Running the pipeline from an empty
warehouse with a short ingestion window produced 484,524 rows with no positives.
Training completed, every metric returned as not a number, all 774 LGAs were
scored severe, and the alerting stage prepared to dispatch 774 alerts. Exit
status zero throughout. Fix it in three places: refuse to train below a minimum
positive count per fold, hold the alert channel if more than 5% of areas clear the
threshold in a week, and choose the ingestion window by inspecting the warehouse
so a cold start triggers a backfill.

**2. Isotonic calibration flattens your ranking.** Covered in Phase 6. Watch for
recall at k falling when you calibrate.

**3. A single-dimension Hawkes simulator does not test a multivariate model.**
Covered in Phase 5.

**4. Full-resolution polygons stall the browser.** 5.6 MB of GeoJSON parsed into
a nested R list will hang the dashboard with no error. Simplify at build time.

**5. RSS article pages are often client-rendered.** Fetching the article URL gave
bodies averaging 311 characters and one LGA match in fourteen. The feed summary is
authored server side and carries the lede, which is where the place names are.
Combining both raised matching to six in fourteen.

**6. MLflow's file store is in maintenance mode.** It raises on `set_experiment`
in current versions. Point the tracking URI at SQLite, and wrap the whole logging
block so tracking can never block a training run.

**7. MLflow metric names reject most punctuation.** Sanitise the model name
before using it as a metric prefix.

---

## Splitting this across a group of four

Phases 0 to 2 are shared and should be done together, because everyone joins on
`lga_code` and everyone reads `settings.yaml`. After that:

- One person owns Phases 3 and 4, ingestion and transport. This is the largest
  surface area and the most external-facing.
- One person owns Phase 5, the panel and the Hawkes process. This is the deepest
  work and the least parallelisable.
- One person owns Phase 6, metrics, baselines and the model, starting with the
  metrics module which the Phase 5 owner also needs.
- One person owns Phases 7 and 8, the actor graph, brief, alerting, API and
  dashboard, which can begin against a hand-written predictions table before the
  real model exists.

The interface between them is the warehouse schema from Phase 1 and the Avro
contracts from Phase 3, which is exactly why both come early.

## Total effort

Roughly 15 to 18 working days for one person, or 5 to 6 days elapsed for a group
of four with the split above. Phase 5 is the one to protect: if it is rushed, the
leakage will not be found and every number after it is fiction.
