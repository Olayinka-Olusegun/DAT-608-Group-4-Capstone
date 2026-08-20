# Data access for the shinyapps.io deployment.
#
# This is the standalone variant of app/R/data_access.R. The difference is that
# shinyapps.io runs R only: there is no Python interpreter on the instance, so
# the reticulate binding used locally for brief generation cannot exist here.
# Rather than failing at the point of use, the brief pane serves the brief that
# was generated during the last pipeline run and stored in the warehouse, and
# says so. Everything else the screen does is a read, and reads were already
# going through DBI.
#
# The warehouse shipped with this bundle is a trimmed copy holding only the seven
# tables the dashboard queries, with the incident table reduced to the displayed
# columns and to the label class.

`%||%` <- function(x, y) if (is.null(x)) y else x

app_config <- function(root = getwd()) {
  list(
    root = root,
    sqlite_path = file.path(root, "data", "warehouse.db"),
    geojson_path = file.path(root, "data", "nga_admin2_simplified.geojson")
  )
}

open_store <- function(config) {
  if (!file.exists(config$sqlite_path)) {
    stop("The bundled warehouse is missing at ", config$sqlite_path)
  }
  connection <- DBI::dbConnect(
    RSQLite::SQLite(), config$sqlite_path, flags = RSQLite::SQLITE_RO
  )
  list(connection = connection, python = NULL, config = config)
}

latest_run <- function(store) {
  DBI::dbGetQuery(
    store$connection,
    "SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1"
  )$run_id[1]
}

fetch_predictions <- function(store) {
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
  DBI::dbGetQuery(store$connection, "
    SELECT driver_rank, feature_label, feature_value, shap_value
    FROM prediction_drivers
    WHERE run_id = (SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1)
      AND lga_code = ?
    ORDER BY driver_rank", params = list(lga_code))
}

fetch_actors <- function(store, lga_code) {
  DBI::dbGetQuery(store$connection, "
    SELECT actor, events, fatalities, first_seen, last_seen, weight
    FROM threat_actor_edges
    WHERE run_id = (SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1)
      AND target_kind = 'lga' AND target = ?
    ORDER BY weight DESC LIMIT 8", params = list(lga_code))
}

fetch_incidents <- function(store, lga_code, limit = 12) {
  DBI::dbGetQuery(store$connection, "
    SELECT event_date, event_type, actor_primary, actor_secondary,
           fatalities, headline
    FROM incidents
    WHERE lga_code = ? AND event_class = 'banditry_kidnapping'
    ORDER BY event_date DESC LIMIT ?", params = list(lga_code, limit))
}

fetch_brief <- function(store, scope = "national") {
  result <- DBI::dbGetQuery(store$connection, "
    SELECT content, generator, generated_at FROM security_briefs
    WHERE scope = ? ORDER BY generated_at DESC LIMIT 1", params = list(scope))
  if (nrow(result) > 0) return(result)
  data.frame(
    content = "No brief is stored for this scope.",
    generator = "none", generated_at = NA_character_
  )
}

fetch_run_metadata <- function(store) {
  DBI::dbGetQuery(store$connection, "
    SELECT run_id, created_at, model_version, horizon_days, train_end, metrics
    FROM model_runs ORDER BY created_at DESC LIMIT 1")
}

# Generating a new brief calls the Anthropic API from Python, which this
# instance does not have. The stored brief is served instead.
generate_brief <- function(store, scope = "national") {
  stored <- fetch_brief(store, "national")
  paste0(
    stored$content[1],
    "\n\n[This deployment serves the brief produced by the last pipeline run. ",
    "Drafting a new one runs in the Python service, which is not part of this ",
    "R-only deployment.]"
  )
}
