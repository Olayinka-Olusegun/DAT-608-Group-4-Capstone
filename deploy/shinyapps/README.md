# shinyapps.io deployment

A self-contained copy of the dashboard that runs on shinyapps.io, which is R only
and needs no billing account.

## What is in the bundle

| File | Size | Purpose |
|---|---|---|
| `app.R` | 331 lines | The dashboard, derived from `app/app.R` |
| `data_access.R` | 4 KB | Standalone reads, no Python |
| `data/warehouse.db` | 2.2 MB | Trimmed warehouse, seven tables |
| `data/nga_admin2_simplified.geojson` | 688 KB | Boundaries for the choropleth |

Total 2.7 MB, comfortably inside the free tier.

## Three differences from the local app

These are the only ways the deployed copy differs, and each is forced by the
platform rather than chosen.

**No Python.** shinyapps.io runs R alone, so the reticulate binding used locally
for brief generation cannot exist. The brief pane serves the brief stored by the
last pipeline run and says so on screen. Everything else the dashboard does was
already a database read.

**No port pinning.** The container build binds `0.0.0.0` on `$PORT` because Cloud
Run requires it. shinyapps.io allocates the socket itself, so this copy ends with
a plain `shinyApp(ui, server)`. That single line is the difference.

**A trimmed warehouse.** Only the seven tables the screen queries are shipped,
with incidents reduced to the label class and the displayed columns. 5.3 MB
becomes 2.2 MB, which shortens cold start since the file is read from local disk.

## Connect your account, once

This is the step that needs you. Open
<https://www.shinyapps.io/admin/#/tokens>, press **Show**, and run the
`setAccountInfo(...)` line it gives you in an R session:

```r
rsconnect::setAccountInfo(name = "yourname", token = "...", secret = "...")
```

That line carries your account secret. It belongs in your R session only, never
in this repository and never pasted into a chat. It writes the credential to
`~/.config/R/rsconnect`, after which the deploy script publishes without ever
handling it.

If you do not have a shinyapps.io account yet, create one at
<https://www.shinyapps.io> first. Signing up requires your email and a password,
so it is also yours to do.

## Then publish

```bash
Rscript deploy/shinyapps/deploy.R
```

The app is published as `lga-violence-risk` and the script prints the URL, which
will be `https://YOURNAME.shinyapps.io/lga-violence-risk/`. Pass a different name
as the first argument if you want one. Re-running replaces the existing app
rather than creating a second copy.

## Refreshing the data

The bundle carries a snapshot, so a new scoring run does not reach the deployed
app until the bundle is rebuilt and republished:

```bash
python -m pau_risk.cli score && python scripts/build_shinyapps_bundle.py
```

```bash
Rscript deploy/shinyapps/deploy.R
```

## Why leaflet is pinned to 2.1.2

The first two deployments failed on the server with:

    Error building image: Error building terra (1.9-34)

The chain is a hard one. leaflet 2.2.3 imports raster and sf, raster 3.5 and
later import terra, and terra fails to compile on the shinyapps.io build image.
Nothing in this app calls raster, sf or terra; they arrive purely as transitive
dependencies of leaflet.

The fix is to pin the two packages to versions from before that chain existed:

    leaflet 2.1.2   does not import sf
    raster  3.4-13  does not import terra

That removes terra, sf, s2, units, proxy and e1071 from the deployment, taking it
from 83 dependencies to 80 and, more importantly, from a failing build to one
that completes in about a minute. The map is unchanged: addGeoJSON, addLegend and
addProviderTiles all behave identically on 2.1.2.

Note this pin is expressed by what is installed locally, because rsconnect
records the versions in your library rather than resolving fresh ones. If you
upgrade leaflet or raster locally and redeploy, terra returns and the build
breaks again.

## A failure that reports itself as success

While debugging the above, `rsconnect::applications()` reported the app status as
`pending` for over half an hour, and the URL returned 404. Both builds had in fact
already failed. The real state only shows up in the task list:

```r
info <- rsconnect::accountInfo("olusegvn")
client <- rsconnect:::clientForAccount(info)
tasks <- client$listTasks(info$accountId)
```

If a deployment appears stuck, read the tasks rather than trusting the status
field or waiting longer.

## Verified locally before publishing

The bundle was run exactly as shinyapps.io runs it, from inside its own directory
with no repository around it:

| Check | Result |
|---|---|
| App starts standalone | Yes, on first try |
| Choropleth renders | 774 polygons |
| Ranked table | 12 rows, Kukawa Borno 0.157 Severe at rank 1 |
| Shiny errors | None |
