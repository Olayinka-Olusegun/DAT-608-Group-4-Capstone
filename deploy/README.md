# Deployment

Two containers, deployed as two Cloud Run services.

| Service | Image | What it serves |
|---|---|---|
| `pau-risk-dashboard` | `deploy/Dockerfile.dashboard` | The R Shiny screen: choropleth, ranked areas, drivers, actors, brief |
| `pau-risk-api` | `deploy/Dockerfile.api` | FastAPI: predictions, drivers, actor graph, GeoJSON, brief generation, ingestion |

They are separate services rather than one container because they have almost
nothing in common at runtime. The API is a small Python process; the dashboard is
an R process holding a websocket per viewer. Splitting them means the dashboard
can scale on session count while the API scales on request rate, and a crash in
one does not take the other down.

## The one step nobody can do for you

The gcloud credentials on this machine are dead. The refresh token returns
`invalid_grant`, the application default credentials expired in 2023, and there
is no service account key anywhere on the machine. Every non-interactive path has
been checked and none of them work.

Re-issuing credentials means signing into Google in a browser with your password,
so it is yours to do:

```bash
gcloud auth login
```

## Then one command for everything

To create a brand new project and deploy into it:

```bash
./deploy/create_and_deploy.sh
```

That generates a project id of the form `pau-risk-<date>-<random>`, creates the
project, links your billing account, enables the three required APIs, waits for
the enablement to propagate, then hands over to the deployment script below. Pass
a project id as the first argument if you want to choose it yourself, and set
`PAU_RISK_BILLING_ACCOUNT` if you have more than one billing account and do not
want the first open one.

To deploy into a project that already exists:

```bash
./deploy/deploy_cloud_run.sh YOUR_PROJECT_ID europe-west1
```

Both scripts are idempotent. An existing project, repository or service is reused
rather than recreated, so re-running after a failure resumes instead of
duplicating.

## Verified before handover

Both images were built for `linux/amd64`, the architecture Cloud Run requires,
and both were run locally and smoke tested:

| Check | Result |
|---|---|
| API image builds | 1.44 GB |
| Dashboard image builds | 1.2 GB |
| API `/health` in container | 200, reports 774 LGAs and 6,875 incidents |
| Dashboard in container | 200, renders 774 choropleth polygons and the ranked table |

So the only thing standing between this repository and a live URL is the
authentication step above.

The script enables the three required APIs, creates an Artifact Registry
repository if it is missing, builds both images on Cloud Build, deploys both
services, and prints the URLs. It finishes with a health check against the API.

Expect roughly ten to fifteen minutes for the first run, most of it building the
R image. Subsequent deployments reuse cached layers and are quicker.

## Running the same topology locally first

```bash
docker compose up --build
```

Dashboard on http://localhost:7788 and API on http://localhost:8000/docs. Worth
doing before deploying, because it catches image problems without waiting on
Cloud Build.

## What is baked into the images

Each image ships a prebuilt warehouse, the trained model and the boundary files,
about 13 MB in total. The containers therefore answer requests the moment they
start, with no ingestion or training at boot.

This is the right trade for a demonstration and the wrong one for a service that
has to stay current. Two consequences are worth being explicit about.

**The predictions are frozen at the week the image was built.** Refreshing them
means rebuilding and redeploying. A production deployment would instead point
`DATABASE_URL` at a Cloud SQL PostgreSQL instance with PostGIS, run the weekly
job from Cloud Scheduler and Cloud Run Jobs, and let both services read whatever
the latest run wrote.

**Writes do not survive.** The `/ingest/document` and `/ingest/chatter` endpoints
publish onto the Avro file sink inside the container, which is ephemeral, so a
submitted report is accepted, extracted, and then lost when the instance is
recycled. The extraction result comes back in the response, so the endpoints
still demonstrate the pipeline, but nothing is retained. Pointing `DATABASE_URL`
at a real database stops this being true.

## Access control

The script deploys both services with `--allow-unauthenticated`, which is what
makes them reachable from a browser without a Google account. That is the point
of putting them online, and the content is aggregate model output over public
conflict data with no personal information in it.

To restrict access instead, drop `--allow-unauthenticated` from both deploy
commands and grant the invoker role to named accounts:

```bash
gcloud run services add-iam-policy-binding pau-risk-dashboard --region europe-west1 --member user:someone@example.com --role roles/run.invoker
```

## Cost

Both services set `--min-instances 0`, so they scale to zero and cost nothing
while idle. Charges are per request and per container second while serving, plus
a few cents a month for image storage in Artifact Registry. For coursework
traffic this sits inside the Cloud Run free tier.

To remove any ongoing charge once the assessment is marked:

```bash
gcloud run services delete pau-risk-dashboard pau-risk-api --region europe-west1
```

```bash
gcloud artifacts repositories delete pau-risk --location europe-west1
```

## Updating a deployed service

Rerun the same script. It rebuilds and rolls out a new revision, and Cloud Run
shifts traffic once the new revision passes its health check.

## Troubleshooting

Logs for either service:

```bash
gcloud run services logs read pau-risk-api --region europe-west1 --limit 100
```

A dashboard that loads but shows no data means the warehouse did not reach the
image. Check `.gcloudignore`: gcloud falls back to `.gitignore` when that file is
absent, and `.gitignore` excludes `data/`, which would strip the warehouse out of
the build context. The `.gcloudignore` in this repository exists to prevent that.

A build that fails on architecture means an image was built locally on Apple
silicon and pushed as arm64. Build through Cloud Build, as the script does, or
pass `--platform linux/amd64` to a local `docker build`.
