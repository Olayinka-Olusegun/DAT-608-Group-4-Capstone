# Toward Proactive Security Planning

A predictive, LGA-level violence risk score for Nigerian state governments.

DAT608 Continuous Assessment, Project 8, Group 4.

Dashboard link: https://olusegvn.shinyapps.io/lga-violence-risk/

This repository is the working implementation of the architecture set out in the
project document. It ingests conflict data from nine sources, builds a weekly
panel over all 774 local government areas, fits a Hawkes process over the LGA
adjacency graph, feeds that into a calibrated gradient boosted classifier,
explains every score with SHAP, assembles a threat actor graph, drafts a security
council brief, and serves the result through an API and an R Shiny dashboard.

## Group 4

Work was allocated by area of the system, with each member tasked as follows.

| Member | Responsibility |
|---|---|
| **Olayinka Olusegun** | Tasked with the design and implementation of the modelling pipeline, comprising the nine source connectors, the Hawkes process fitted over the LGA adjacency graph, and the calibrated classifier with its SHAP explainability layer. Additionally responsible for the service API, the command line interface and the packaging of the project. |
| **Anjolaoluwa Olowokere** | Tasked with the configuration and operational tooling of the system, holding responsibility for every model and service parameter, including the temporal split boundaries, the risk tier thresholds and the alerting limits. Additionally responsible for the hyper-parameter selection procedure, the reference data build and the scheduled weekly run. |
| **Kelvin Obini** | Tasked with assuring the correctness of the system through a test suite that verifies the model cannot observe the outcome it is asked to predict, and that a degraded run cannot escalate into mass alerting. Additionally responsible for the alerting controls and for the technical documentation on which the reported findings depend. |
| **Elobike Nwosa** | Tasked with the design and development of the analyst-facing dashboard, covering the national choropleth, the ranked area listing and the drill-down into score drivers, armed actors and incident history. Additionally responsible for the data access layer that serves it. |
| **Edith Ofor** | Tasked with the acquisition, preparation and governance of all project data, including the official administrative boundaries, the derived adjacency graph and the georeferenced incident corpus used for training. Additionally responsible for the model artifacts and experiment records that make results reproducible. |
| **Eddidiong Obong** | Tasked with the deployment and release of the system to a publicly accessible environment, covering the container images, the cloud build pipeline and the live hosted dashboard. Additionally responsible for ensuring the deployed application is derived from source, so that it cannot diverge from the repository. |
| **Kayode Oluwalana** | Tasked with the design of the data warehouse schema that every other component reads from and writes to. Additionally responsible for maintaining it across two back ends under identical column definitions, so that queries execute unchanged on PostgreSQL and on the embedded alternative. |

---

## What it produces

For each of the 774 LGAs, once a week:

- a calibrated probability that a banditry or kidnapping event occurs in the next
  seven days, together with a risk tier of Low, Elevated, High or Severe;
- the five drivers that moved that score most, named in plain language;
- the armed actors recorded operating in that area and how recently;
- a national or state level narrative brief;
- an alert for any area that has newly reached High or Severe.

## Results on held-out data

Trained on 2013 to 2021, tuned on 2022, evaluated once on 2023 to 2024. The test
fold contains 81,270 LGA-weeks with 400 positives, a base rate of 0.49%.

| Model | ROC AUC | Average precision | Brier | Recall at 20 | Recall at 50 | Lift at 20 |
|---|---|---|---|---|---|---|
| Persistence, attacks last week | 0.549 | 0.018 | 0.0089 | 0.109 | 0.125 | 4.2 |
| Historical rate, past year | 0.811 | 0.045 | 0.1277 | 0.322 | 0.538 | 12.5 |
| Hawkes process alone | 0.910 | 0.067 | 0.0048 | 0.345 | 0.588 | 13.4 |
| XGBoost without Hawkes features | 0.883 | 0.063 | 0.0048 | 0.327 | 0.602 | 12.7 |
| Hybrid, uncalibrated | 0.923 | 0.070 | 0.0047 | 0.360 | 0.611 | 13.9 |
| **Hybrid, calibrated** | **0.923** | **0.068** | **0.0047** | **0.355** | **0.614** | **13.8** |

The operational reading: covering the top 20 LGAs, 2.6% of the country, reaches
36% of the areas attacked that week, against 11% for the current practice of
re-covering last week's incident list. Covering 50 reaches 61%. Both components
of the hybrid contribute, and the point process contributes more than the tree:
the Hawkes model on its own beats gradient boosting on its own, and the two
together beat either.

The fitted process is interpretable in its own right. The branching ratio is
0.50, so about half of recorded events are triggered by an earlier event rather
than arising independently. Of that triggered half, 82% stays inside the same LGA
and 18% crosses into a neighbour, which is the displacement effect the project
document describes, measured. The excitation decays with a half life of 22 days,
so the elevated period after an attack runs to roughly six weeks.

Figures and their interpretation are in [docs/figure_notes.md](docs/figure_notes.md),
generated by `scripts/make_figures.py`.

If you are rebuilding this from scratch, or want to understand why the pieces
are ordered the way they are, [docs/build_guide.md](docs/build_guide.md) is the
phase by phase plan, including the traps that cost time on this build.

---

## Data sources and their live status

The nine producers from the ingestion strategy are all implemented. Their status
below is what this environment could actually reach, and every connector reports
its own readiness at run time rather than failing silently.

| Producer | Type | Status on the reference run | Note |
|---|---|---|---|
| UCDP GED v25.1 | structured | **Live, 6,875 Nigerian events** | Public bulk CSV, 1989 to 2024. The training corpus. |
| ACLED | structured | Needs `ACLED_API_KEY` | Implemented against the read API, activates with a key. |
| Nextier | documents | **Live, 38 documents** | Conflict database commentary. |
| Press: Daily Trust, Vanguard, NPF | documents | **Live, 14 documents** | RSS plus article text. NPF blocks on TLS. |
| Humanitarian: HDX, ReliefWeb | documents | **Live, 20 documents** | Situation reports and displacement datasets. |
| NBS crime statistics | documents | **Live, 1 document** | Portal is intermittent; the connector retries on schedule. |
| SBM Intelligence | documents | Publisher returns 403 | Crawler implemented, the site blocks this client. |
| HumAngle | documents | Publisher rate limits | Feed reachable but throttled to 429. |
| Social: X, Telegram | chatter | Needs credentials | Watchlist search implemented for both platforms. |

Document counts depend on the window requested, since the feeds only expose
recent items; the numbers above are from a 45 day window. Six of the nine
producers returned data without any credential.

Geography is the OCHA Common Operational Dataset for Nigerian administrative
boundaries: all 774 admin level 2 units with official P-codes, the same coding
GRID3 uses. Adjacency is derived from shared borders, giving 4,677 weighted edges
at a mean degree of 6.04.

### The most important limitation

UCDP records events that produced fatalities. A kidnapping that ends in a ransom
payment and no deaths is therefore largely invisible to it, which is precisely
the event class this project targets. The consequence is visible in the output:
risk concentrates in Borno, Benue and Plateau, where violence is lethal, rather
than in the Zamfara and Katsina abduction economy that SBM Intelligence documents.
ACLED codes abduction as its own sub-event type with no fatality threshold, so
adding an ACLED key is the single change that would most improve this system. The
connector is already written for it; the model would need retraining on the wider
event base, not rebuilding.

---

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
```

Then the whole pipeline, which takes a few minutes on a laptop and needs no
credentials:

```bash
.venv/bin/python -m pau_risk.cli pipeline
```

Or one stage at a time:

```bash
.venv/bin/python -m pau_risk.cli reference
```

```bash
.venv/bin/python -m pau_risk.cli ingest --since 2010-01-01 --until 2024-12-31
```

```bash
.venv/bin/python -m pau_risk.cli features && .venv/bin/python -m pau_risk.cli train
```

```bash
.venv/bin/python -m pau_risk.cli score && .venv/bin/python -m pau_risk.cli brief
```

The API:

```bash
.venv/bin/python -m pau_risk.cli serve --port 8000
```

The dashboard, from the `app` directory:

```bash
Rscript -e 'shiny::runApp(".", port = 7788)'
```

Tests:

```bash
.venv/bin/python -m pytest tests -q
```

Credentials go in `.env`, copied from `.env.example`. Nothing in the pipeline
requires them; each connector that needs one reports that it is skipping and why.

---

## Deploying it

Two Cloud Run services, one for the API and one for the dashboard, both scaling
to zero. Full notes in [deploy/README.md](deploy/README.md).

Run the same topology locally first:

```bash
docker compose up --build
```

Then deploy, after authenticating gcloud yourself:

```bash
./deploy/deploy_cloud_run.sh YOUR_PROJECT_ID europe-west1
```

The images bake in the prebuilt warehouse, trained model and boundary files,
about 13 MB, so a container answers requests as soon as it starts. That also
means the deployed predictions are frozen at the week the image was built, and
the ingestion endpoints accept and extract a submission without retaining it.
Pointing `DATABASE_URL` at a Cloud SQL PostGIS instance is what turns the
deployment from a demonstration into a service that stays current.

---

## How the code maps onto the architecture

| Architecture element | Where it lives | What actually runs |
|---|---|---|
| Nine producers | `src/pau_risk/ingest/` | One connector class per source, each reporting its own readiness. |
| Kafka with Avro and a schema registry | `src/pau_risk/stream/` | Avro contracts in `schemas/*.avsc`, a versioned registry enforcing backward compatibility, Kafka transport with an Avro file sink when no broker is running. |
| Spark feature pipeline | `src/pau_risk/features/spark_panel.py` | Full Spark SQL implementation, verified to produce values identical to the pandas path on all 484,524 rows. |
| PostgreSQL with PostGIS and pgvector | `sql/schema_postgres.sql`, `src/pau_risk/storage/` | Used when `DATABASE_URL` is set; otherwise the identical schema runs on embedded SQLite, which the project document permits. |
| Hawkes process as a feature generator | `src/pau_risk/features/hawkes.py` | Multivariate exponential-kernel Hawkes over the adjacency graph, fitted by maximum likelihood. |
| XGBoost | `src/pau_risk/models/train.py` | Temporal split, isotonic calibration, MLflow logging. |
| SHAP explainability | `src/pau_risk/models/explain.py` | Per-LGA driver attribution stored alongside every prediction. |
| Threat actor graph | `src/pau_risk/graph/actors.py` | Actor to LGA, actor to state and rivalry edges with recency decay. |
| Claude brief | `src/pau_risk/brief/generator.py` | Grounded prompt, deterministic template fallback, both clearly labelled. |
| R Shiny with leaflet and reticulate | `app/` | Choropleth, ranked table, driver and actor drill-down, brief pane. |
| Alerting | `src/pau_risk/alerting/job.py` | Tier threshold, cooldown, escalation override, full audit trail, dry by default. |

Two deviations from the project document, both deliberate and both recorded here.
`PyPDF2` is archived, so its maintained continuation `pypdf` is used with the same
reader interface. The document's implementation section also names NCDC, NIMET and
Google Trends, which belong to a different problem; the connectors follow the data
sources actually listed in the ingestion strategy.

---

## Repository layout

```
config/            settings.yaml, alert recipients
sql/               warehouse schema for PostgreSQL and for SQLite
src/pau_risk/
  ingest/          the nine producers plus the warehouse consumer
  stream/          Avro schemas, schema registry, Kafka or file transport
  reference/       LGA registry, adjacency graph, Nigerian calendar
  nlp/             gazetteer matching, ransom and casualty extraction, threat scoring
  features/        weekly panel, Hawkes process, Spark equivalent
  models/          training, calibration, metrics, SHAP, scoring
  graph/           threat actor graph
  brief/           security council brief generation
  alerting/        alert decisions and dispatch
  api/             FastAPI service
app/               R Shiny dashboard
scripts/           reference build, hyper-parameter sweep, figures
tests/             58 tests covering leakage, the estimator, extraction and transport
docs/              figures and their interpretation
```

## A failure found in testing, and what was done about it

Running the pipeline from an empty warehouse with the default short ingestion
window produced a panel of 484,524 rows containing no events at all. Nothing
raised. Training completed, every metric returned as not a number, the scoring
stage assigned the severe tier to all 774 areas, and the alerting stage prepared
to dispatch 774 alerts. The exit status was zero throughout.

That is the worst class of bug an alerting system can have, because it produces
confident output rather than an error. Three guards were added in response, each
covered by a regression test in `tests/test_guards.py`.

Training now refuses a panel that cannot support the claim being made, requiring
a minimum number of positive weeks in each fold and naming the backfill command
in the error. The alerting job holds the channel entirely if more than 5% of
areas clear the threshold in one week, on the reasoning that such a pattern
indicates a broken input rather than a national emergency. And the ingestion
stage now chooses its window by inspecting the warehouse: an empty one triggers a
backfill across the panel period, while a populated one pulls only the recent
fortnight.

## Ethical note

This system ranks places, not people. It holds no personal data, and the score is
attached to an administrative unit rather than to any individual or group. It is
built as a prioritisation aid: an area outside the list is not assessed as safe,
only as carrying a lower modelled probability this week, and the brief says so
every time it is generated. The narrative layer is explicitly barred from
recommending deployments or use of force, because that decision belongs to the
council and carries consequences a model cannot weigh.
