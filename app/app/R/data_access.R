# Data access for the dashboard.
#
# Two routes into the same warehouse are supported and the app picks whichever
# is available at start up.
#
# The direct route uses DBI against PostgreSQL, or against the embedded SQLite
# warehouse when no PostgreSQL instance is configured. This is the route the
# brief describes, where the Shiny app queries the predictions table itself.
#
# The reticulate route binds the Python scoring environment into the R session
# and calls the same repository functions the pipeline uses. It exists because
# the brief specifies reticulate binding the two languages inside one project,
# and because it removes any chance of the dashboard and the pipeline disagreeing
# about what a column means. It is preferred when the virtual environment is
# present, since it reuses tested code rather than duplicating SQL.

project_root <- function() {
  normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."), "..", ".."), mustWork = FALSE)
}

`%||%` <- function(x, y) if (is.null(x)) y else x

app_config <- function(root = getwd()) {
  list(
    root = root,
    sqlite_path = file.path(root, "data", "warehouse.db"),
    # The simplified layer is what the map draws; the full resolution file stays
    # on disk for the spatial joins the pipeline performs.
    geojson_path = file.path(root, "data", "reference", "nga_admin2_simplified.geojson"),
    geojson_full_path = file.path(root, "data", "reference", "nga_admin2.geojson"),
    venv_python = file.path(root, ".venv", "bin", "python"),
    src_path = file.path(root, "src"),
    database_url = Sys.getenv("DATABASE_URL", "")
  )
}

# ---------------------------------------------------------------- reticulate
init_python <- function(config) {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(NULL)
  if (!file.exists(config$venv_python)) return(NULL)
  ok <- tryCatch({
    reticulate::use_python(config$venv_python, required = TRUE)
    sys <- reticulate::import("sys")
    if (!(config$src_path %in% sys$path)) sys$path$insert(0L, config$src_path)
    TRUE
  }, error = function(e) FALSE)
  if (!isTRUE(ok)) return(NULL)
  tryCatch(reticulate::import("pau_risk.storage"), error = function(e) NULL)
}

# ----------------------------------------------------------------------- DBI
init_connection <- function(config) {
  if (!requireNamespace("DBI", quietly = TRUE)) return(NULL)
  if (nzchar(config$database_url) && requireNamespace("RPostgres", quietly = TRUE)) {
    parsed <- httr::parse_url(sub("^postgresql\\+psycopg", "postgresql", config$database_url))
    return(tryCatch(
      DBI::dbConnect(
        RPostgres::Postgres(),
        host = parsed$hostname, port = parsed$port %||% 5432,
        dbname = sub("^/", "", parsed$path),
        user = parsed$username, password = parsed$password
      ),
      error = function(e) NULL
    ))
  }
  if (!requireNamespace("RSQLite", quietly = TRUE)) return(NULL)
  if (!file.exists(config$sqlite_path)) return(NULL)
  tryCatch(
    DBI::dbConnect(RSQLite::SQLite(), config$sqlite_path, flags = RSQLite::SQLITE_RO),
    error = function(e) NULL
  )
}

# -------------------------------------------------------------------- reads
#
# Reads go through DBI, which is a plain query against the same tables the
# pipeline writes. The reticulate binding is kept for the one operation that
# genuinely needs Python, drafting the brief, because that call owns the model
# credential and the prompt construction. Reading through SQL and writing through
# Python keeps each side doing what it is better at, and means the dashboard
# still opens when the Python environment is not present.

fetch_predictions <- function(store) {
  if (is.null(store$connection)) {
    frame <- store$python$load_predictions()
    return(as.data.frame(frame, stringsAsFactors = FALSE))
  }
  DBI::dbGetQuery(store$connection, "
    SELECT p.run_id, p.lga_code, r.lga_name, r.state_name, r.zone,
           r.centre_lat, r.centre_lon, p.week_start, p.probability,
           p.risk_tier, p.rank_national, p.rank_state
    FROM predictions p
    JOIN lga_registry r ON r.lga_code = p.lga_code
    WHERE p.run_id = (SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1)
    ORDER BY p.probability DESC")
}

fetch_drivers <- function(store, lga_code) {
  if (is.null(store$connection)) {
    run_id <- store$python$latest_run_id()
    return(as.data.frame(store$python$load_drivers(run_id, lga_code), stringsAsFactors = FALSE))
  }
  DBI::dbGetQuery(store$connection, "
    SELECT driver_rank, feature_label, feature_value, shap_value
    FROM prediction_drivers
    WHERE run_id = (SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1)
      AND lga_code = ?
    ORDER BY driver_rank", params = list(lga_code))
}

fetch_actors <- function(store, lga_code) {
  if (!is.null(store$connection)) {
    return(DBI::dbGetQuery(store$connection, "
      SELECT actor, events, fatalities, first_seen, last_seen, weight
      FROM threat_actor_edges
      WHERE run_id = (SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1)
        AND target_kind = 'lga' AND target = ?
      ORDER BY weight DESC LIMIT 8", params = list(lga_code)))
  }
  data.frame()
}

fetch_incidents <- function(store, lga_code, limit = 12) {
  if (!is.null(store$connection)) {
    return(DBI::dbGetQuery(store$connection, "
      SELECT event_date, event_type, actor_primary, actor_secondary,
             fatalities, headline
      FROM incidents
      WHERE lga_code = ? AND event_class = 'banditry_kidnapping'
      ORDER BY event_date DESC LIMIT ?", params = list(lga_code, limit)))
  }
  data.frame()
}

fetch_brief <- function(store, scope = "national") {
  if (!is.null(store$connection)) {
    result <- DBI::dbGetQuery(store$connection, "
      SELECT content, generator, generated_at FROM security_briefs
      WHERE scope = ? ORDER BY generated_at DESC LIMIT 1", params = list(scope))
    if (nrow(result) > 0) return(result)
  }
  data.frame(
    content = "No brief has been generated yet. Run pau-risk brief.",
    generator = "none",
    generated_at = NA_character_
  )
}

fetch_run_metadata <- function(store) {
  if (!is.null(store$connection)) {
    return(DBI::dbGetQuery(store$connection, "
      SELECT run_id, created_at, model_version, horizon_days, train_end, metrics
      FROM model_runs ORDER BY created_at DESC LIMIT 1"))
  }
  data.frame()
}

# Generating a brief is a write, and it needs the model credential and the prompt
# construction, so it is never reimplemented in R. There are two ways to reach
# that code. In a container the dashboard has no Python runtime, so it calls the
# API service over HTTP. From a checkout it binds the local environment through
# reticulate. Both end up in exactly the same function.
generate_brief_via_api <- function(scope) {
  base_url <- Sys.getenv("PAU_RISK_API_URL", "")
  if (!nzchar(base_url) || !requireNamespace("httr", quietly = TRUE)) return(NULL)
  response <- tryCatch(
    httr::POST(
      paste0(base_url, "/brief/generate"),
      body = list(scope = scope), encode = "json", httr::timeout(120)
    ),
    error = function(e) NULL
  )
  if (is.null(response) || httr::status_code(response) >= 400) return(NULL)
  payload <- httr::content(response, as = "parsed", type = "application/json")
  payload$content
}

generate_brief <- function(store, scope = "national") {
  via_api <- generate_brief_via_api(scope)
  if (!is.null(via_api)) return(via_api)
  if (is.null(store$python)) {
    return(paste(
      "Brief generation is unavailable. Set PAU_RISK_API_URL to reach the API",
      "service, or run the app from a checkout with reticulate installed."
    ))
  }
  brief <- reticulate::import("pau_risk.brief")
  storage <- store$python
  run_id <- storage$latest_run_id()
  predictions <- storage$load_predictions(run_id)
  drivers <- storage$load_drivers(run_id)
  edges <- storage$read_sql(
    "SELECT actor, target_kind, target, events, fatalities, first_seen, last_seen, weight
     FROM threat_actor_edges WHERE run_id = :run_id",
    list(run_id = run_id)
  )
  pandas <- reticulate::import("pandas")
  week_start <- pandas$to_datetime(predictions$week_start$iloc[0L])$date()
  result <- brief$create_and_store(run_id, week_start, predictions, drivers, edges, scope)
  result$content
}

open_store <- function(config) {
  python <- init_python(config)
  connection <- init_connection(config)
  if (is.null(python) && is.null(connection)) {
    stop("No route to the warehouse. Build it with: python -m pau_risk.cli pipeline")
  }
  list(python = python, connection = connection, config = config)
}
