# Weekly LGA violence risk dashboard.
#
# The screen is arranged around the sequence a security council actually works
# through. The map answers where, at a glance and at the level the council
# allocates. The ranked table answers which areas specifically, ordered so that
# the top of the list is the day's agenda. Selecting an area opens the drivers,
# the armed actors recorded there and the recent incident history, which together
# answer why. The brief pane holds the narrative that goes into the minutes.
#
# The page is white and near-black. Colour is spent only where it carries
# meaning: the four risk tiers, running blue for quiet through to red for severe,
# and a single blue accent on the controls a person can act on. Nothing else is
# tinted, so a coloured element on this screen always means something.

library(shiny)
library(leaflet)
library(DT)

source(file.path("R", "data_access.R"), local = TRUE)

# The project root is explicit in a container and inferred when run from a
# checkout, where the app is normally started from inside app/.
resolve_root <- function() {
  from_env <- Sys.getenv("PAU_RISK_ROOT", "")
  if (nzchar(from_env)) return(normalizePath(from_env, mustWork = FALSE))
  parent <- normalizePath(file.path(getwd(), ".."), mustWork = FALSE)
  if (file.exists(file.path(parent, "data", "warehouse.db"))) return(parent)
  getwd()
}

CONFIG <- app_config(root = resolve_root())
STORE <- open_store(CONFIG)

TIER_LEVELS <- c("Low", "Elevated", "High", "Severe")
# Blue for quiet through to red for dangerous, with intensity rising alongside
# hue. Around 680 of the 774 areas sit in the lowest tier in a normal week, so a
# saturated blue at that end floods the map and leaves the handful of areas that
# matter competing with the background. Low is therefore pale enough to recede
# while the warm tiers advance. Hue still carries the ordering, which keeps the
# map readable for the commonest forms of colour blindness in a way a red to
# green ramp would not.
TIER_COLOURS <- c(
  Low = "#D8E7F3", Elevated = "#7FB0D5", High = "#F08C4B", Severe = "#C9302C"
)

# Boundary polygons are read once at start up as raw GeoJSON.
#
# There were three rendering paths here originally: simple features when sf was
# installed, raw GeoJSON when it was not, and centroid circles when the boundary
# file was missing. Only the first carried hover labels, so the deployed copy
# silently lost its tooltips because sf is not installed there.
#
# Two attempts to unify them are worth recording, because both dead ends are
# easy to walk into. Installing sf on the deployment pulls GDAL, GEOS and PROJ
# onto the build image, which is what made an earlier build fail. Converting the
# polygons to an sp frame instead looks like it avoids that, but leaflet's
# handler for sp objects calls into rgeos, which was archived from CRAN in 2023
# and cannot be installed at all.
#
# So the GeoJSON is drawn directly, which needs nothing beyond jsonlite, and the
# hover label and click handler are bound to each feature in a few lines of
# JavaScript once the widget has rendered. The label text is built in R and
# carried inside the feature properties, so the JavaScript stays a binding step
# and never becomes the place where content is decided.
boundaries_geojson <- NULL
if (file.exists(CONFIG$geojson_path) && requireNamespace("jsonlite", quietly = TRUE)) {
  boundaries_geojson <- tryCatch(
    jsonlite::fromJSON(CONFIG$geojson_path, simplifyVector = FALSE),
    error = function(e) NULL
  )
}

# The label answers "what am I looking at", so it leads with the place and
# follows with the two numbers a viewer would otherwise hunt for in the table.
label_html <- function(lga_name, state_name, risk_tier, probability) {
  detail <- if (is.null(risk_tier) || is.na(risk_tier)) {
    "not scored"
  } else {
    sprintf("%s &middot; %.3f over 7 days", risk_tier, probability)
  }
  sprintf(
    paste0(
      "<div style=\"font-size:13px;line-height:1.4\">",
      "<strong>%s</strong><br/>",
      "<span style=\"color:#5F6672\">%s</span><br/>%s</div>"
    ),
    lga_name, state_name, detail
  )
}

style_geojson <- function(payload, scores) {
  tier_by_code <- setNames(as.list(scores$risk_tier), scores$lga_code)
  prob_by_code <- setNames(as.list(scores$probability), scores$lga_code)

  payload$features <- lapply(payload$features, function(feature) {
    code <- feature$properties$adm2_pcode
    tier <- tier_by_code[[code]]
    probability <- prob_by_code[[code]]
    if (is.null(tier)) tier <- "Low"
    feature$properties$risk_tier <- tier
    feature$properties$lga_label <- label_html(
      feature$properties$adm2_name, feature$properties$adm1_name,
      tier, if (is.null(probability)) NA_real_ else probability
    )
    feature$properties$style <- list(
      fillColor = unname(TIER_COLOURS[[tier]]),
      fillOpacity = 0.85, color = "#FFFFFF", weight = 0.5
    )
    feature
  })
  payload
}

# Bound after render because the GeoJSON layers do not exist until then. The
# click sets the same input the sp and marker paths would have set, so the
# server side selection logic is untouched.
BIND_FEATURES <- "
function(el, x) {
  var map = this;
  map.eachLayer(function(layer) {
    var feature = layer.feature;
    if (!feature || !feature.properties || !feature.properties.lga_label) return;
    layer.bindTooltip(feature.properties.lga_label, {
      sticky: true, opacity: 0.97, className: 'lga-tooltip'
    });
    layer.on('mouseover', function() {
      layer.setStyle({ weight: 2, color: '#14161A', fillOpacity: 0.95 });
      layer.bringToFront();
    });
    layer.on('mouseout', function() {
      layer.setStyle(feature.properties.style);
    });
    layer.on('click', function() {
      Shiny.setInputValue(
        'map_shape_click',
        { id: feature.properties.adm2_pcode, nonce: Math.random() },
        { priority: 'event' }
      );
    });
  });
}
"

ui <- fluidPage(
  tags$head(tags$style(HTML("
    /* The page stays white and near-black. Colour is spent only where it
       carries meaning: the four risk tiers, and one blue accent for the things
       a person can act on. Corners are rounded and borders are hairlines rather
       than hard rules, which is most of what makes a dense screen feel
       approachable rather than administrative. */
    :root {
      --ink: #14161A;
      --muted: #5F6672;
      --paper: #FFFFFF;
      --hairline: #E6E8EC;
      --surface: #FAFBFC;
      --accent: #2F6FAF;
      --low: #4575B4;
      --elevated: #91BFDB;
      --high: #FC8D59;
      --severe: #D73027;
      --radius-lg: 14px;
      --radius-sm: 8px;
      --shadow: 0 1px 2px rgba(20,22,26,.05), 0 6px 18px rgba(20,22,26,.05);
    }

    body {
      background: var(--paper); color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                   Helvetica, Arial, sans-serif;
      font-size: 15px; line-height: 1.55;
    }
    .container-fluid { max-width: 1520px; }

    /* Headings keep the serif. It gives the page a considered voice, while the
       sans body face keeps the tables and numbers legible at small sizes. */
    h1, h2, h3, h4 {
      font-family: Georgia, 'Times New Roman', serif;
      font-weight: 600; letter-spacing: -0.01em; text-wrap: balance;
    }
    h2 { font-size: 27px; margin-bottom: 18px; }
    h4 { font-size: 17px; margin-top: 22px; margin-bottom: 10px; }

    .well, .panel {
      background: var(--paper); border: 1px solid var(--hairline);
      border-radius: var(--radius-lg); box-shadow: var(--shadow); padding: 18px;
    }

    /* Each headline figure carries a stripe in the colour of the tier it counts,
       so the eye links the number to the same colour it just read on the map. */
    .metric {
      position: relative; overflow: hidden; background: var(--paper);
      border: 1px solid var(--hairline); border-radius: var(--radius-lg);
      box-shadow: var(--shadow); padding: 14px 16px 14px 22px; margin-bottom: 12px;
    }
    .metric::before {
      content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
      background: var(--accent);
    }
    .metric-severe::before { background: var(--severe); }
    .metric-high::before   { background: var(--high); }
    .metric-total::before  { background: var(--low); }
    .metric .value {
      font-size: 30px; font-weight: 650; font-variant-numeric: tabular-nums;
      line-height: 1.1;
    }
    .metric .caption {
      font-size: 11px; text-transform: uppercase; letter-spacing: 0.09em;
      color: var(--muted); margin-top: 2px;
    }

    .map-frame {
      border: 1px solid var(--hairline); border-radius: var(--radius-lg);
      overflow: hidden; box-shadow: var(--shadow);
    }

    /* A tier reads as a shape as well as a word, so the ranked list can be
       scanned without reading every row. */
    .pill {
      display: inline-block; padding: 2px 10px; border-radius: 999px;
      font-size: 12px; font-weight: 600; white-space: nowrap;
    }
    .pill-Low      { background: rgba(69,117,180,.14);  color: #2C5A8F; }
    .pill-Elevated { background: rgba(145,191,219,.30); color: #276881; }
    .pill-High     { background: rgba(252,141,89,.22);  color: #9E4A1B; }
    .pill-Severe   { background: rgba(215,48,39,.16);   color: #A31C16; }

    table.dataTable thead th {
      border-bottom: 1px solid var(--hairline) !important;
      font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--muted); font-weight: 650;
    }
    table.dataTable tbody td {
      border-top: none !important; font-variant-numeric: tabular-nums;
      padding: 9px 10px;
    }
    table.dataTable tbody tr:hover { background: var(--surface); cursor: pointer; }
    table.dataTable tbody tr.selected {
      background: rgba(47,111,175,.10) !important; color: var(--ink) !important;
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .table > tbody > tr > td, .table > thead > tr > th {
      border-color: var(--hairline); padding: 8px 10px;
    }
    .table > thead > tr > th {
      font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--muted);
    }

    #make_brief {
      background: var(--accent); color: #FFFFFF; border: none;
      border-radius: var(--radius-sm); padding: 10px 18px;
      font-weight: 600; font-size: 14px;
    }
    #make_brief:hover, #make_brief:focus { background: #255C93; color: #FFFFFF; }

    .form-control, .selectize-input {
      border-radius: var(--radius-sm) !important;
      border: 1px solid var(--hairline) !important; box-shadow: none !important;
    }
    .irs-bar, .irs-single, .irs-from, .irs-to { background: var(--accent) !important; }
    :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

    pre.brief {
      white-space: pre-wrap; background: var(--surface);
      border: 1px solid var(--hairline); border-radius: var(--radius-lg);
      color: var(--ink); font-family: Georgia, serif; font-size: 14px;
      line-height: 1.65; padding: 18px;
    }
    /* The detail column is narrow, and the driver names are full sentences. Left
       to itself the table stretches past its column and collides with the panels
       underneath, so it is constrained here and allowed to scroll on its own
       rather than pushing the page sideways. */
    #drivers table, #actors table, #incidents table {
      font-size: 12.5px; width: auto; min-width: 100%;
    }
    #drivers table td, #actors table td, #incidents table td,
    #drivers table th, #actors table th, #incidents table th {
      padding: 6px 8px; vertical-align: top;
    }
    /* Headers are short labels and must never break mid-word, which is what a
       fixed table layout does to them in a column this narrow. Columns are sized
       to their content and the box scrolls instead. */
    #drivers table th, #actors table th, #incidents table th { white-space: nowrap; }
    #drivers table td:first-child { min-width: 130px; }
    #drivers table td:not(:first-child), #actors table td:not(:first-child),
    #incidents table td:not(:first-child) {
      white-space: nowrap; font-variant-numeric: tabular-nums;
    }
    #drivers, #actors, #incidents {
      overflow-x: auto; -webkit-overflow-scrolling: touch; padding-bottom: 2px;
    }

    .leaflet-tooltip.lga-tooltip {
      background: var(--paper); border: 1px solid var(--hairline);
      border-radius: var(--radius-sm); box-shadow: 0 4px 14px rgba(20,22,26,.12);
      padding: 8px 10px; color: var(--ink); font-weight: 400;
    }
    .leaflet-tooltip.lga-tooltip::before { border-top-color: var(--hairline); }
    .footnote { font-size: 12px; color: var(--muted); }
    hr { border-top: 1px solid var(--hairline); }
  "))),
  titlePanel("Seven day violence risk, Nigerian local government areas"),
  fluidRow(
    column(
      3,
      wellPanel(
        selectInput("state", "State", choices = c("All states"), selected = "All states"),
        checkboxGroupInput("tiers", "Risk tier", choices = TIER_LEVELS,
                           selected = c("Elevated", "High", "Severe")),
        sliderInput("top_n", "Areas listed", min = 10, max = 200, value = 40, step = 10),
        hr(),
        uiOutput("run_metadata")
      ),
      uiOutput("headline_metrics")
    ),
    column(
      6,
      div(class = "map-frame", leafletOutput("map", height = 470)),
      div(class = "footnote",
          "Blue marks the lowest risk and red the most severe. Areas without a score are unshaded."),
      hr(),
      h4("Ranked areas"),
      DTOutput("table")
    ),
    column(
      3,
      h4(textOutput("selected_title")),
      uiOutput("selected_summary"),
      h4("Why this score"),
      tableOutput("drivers"),
      h4("Actors recorded here"),
      tableOutput("actors"),
      h4("Recent incidents"),
      tableOutput("incidents")
    )
  ),
  hr(),
  fluidRow(
    column(
      12,
      h4("Security council brief"),
      actionButton("make_brief", "Draft brief for the current selection"),
      br(), br(),
      uiOutput("brief_source"),
      tags$pre(class = "brief", textOutput("brief_text"))
    )
  ),
  fluidRow(
    column(12, div(class = "footnote",
      "Probabilities are calibrated over a seven day horizon and are an aid to prioritisation, ",
      "not a forecast of individual incidents. An area outside the list is not assessed as safe."))
  )
)

server <- function(input, output, session) {
  predictions <- reactiveVal(fetch_predictions(STORE))
  selected_code <- reactiveVal(NULL)
  brief_content <- reactiveVal(NULL)

  observe({
    frame <- predictions()
    states <- sort(unique(frame$state_name))
    updateSelectInput(session, "state", choices = c("All states", states))
  })

  filtered <- reactive({
    frame <- predictions()
    if (!is.null(input$state) && input$state != "All states") {
      frame <- frame[frame$state_name == input$state, ]
    }
    if (length(input$tiers) > 0) {
      frame <- frame[frame$risk_tier %in% input$tiers, ]
    }
    frame[order(-frame$probability), ]
  })

  output$run_metadata <- renderUI({
    meta <- fetch_run_metadata(STORE)
    if (nrow(meta) == 0) return(tags$p("No model run recorded."))
    tagList(
      tags$p(tags$strong("Model run"), tags$br(), meta$run_id[1]),
      tags$p(tags$strong("Trained through"), tags$br(), meta$train_end[1]),
      tags$p(tags$strong("Horizon"), tags$br(), paste(meta$horizon_days[1], "days"))
    )
  })

  output$headline_metrics <- renderUI({
    frame <- predictions()
    counts <- table(factor(frame$risk_tier, levels = TIER_LEVELS))
    tagList(
      div(class = "metric metric-severe",
          div(class = "value", counts[["Severe"]]), div(class = "caption", "Severe areas")),
      div(class = "metric metric-high",
          div(class = "value", counts[["High"]]), div(class = "caption", "High risk areas")),
      div(class = "metric metric-total",
          div(class = "value", nrow(frame)), div(class = "caption", "Areas scored"))
    )
  })

  output$map <- renderLeaflet({
    frame <- predictions()
    base <- leaflet(options = leafletOptions(minZoom = 5)) |>
      addProviderTiles("CartoDB.PositronNoLabels") |>
      setView(lng = 8.7, lat = 9.1, zoom = 6)

    legend <- function(map) {
      addLegend(
        map, position = "bottomright", colors = unname(TIER_COLOURS),
        labels = TIER_LEVELS, title = "Risk tier", opacity = 0.9
      )
    }

    if (!is.null(boundaries_geojson)) {
      base |>
        addGeoJSON(style_geojson(boundaries_geojson, frame)) |>
        legend() |>
        htmlwidgets::onRender(BIND_FEATURES)
    } else {
      # Only reached when the boundary file is missing entirely. The same
      # information is shown as graduated circles on the LGA centroids, which are
      # already in the registry, so the app stays usable.
      base |>
        addCircleMarkers(
          data = frame,
          lng = ~centre_lon, lat = ~centre_lat, layerId = ~lga_code,
          radius = ~pmax(3, 22 * probability),
          fillColor = ~unname(TIER_COLOURS[risk_tier]), fillOpacity = 0.85,
          color = "#FFFFFF", weight = 0.6,
          label = ~lapply(
            mapply(label_html, lga_name, state_name, risk_tier, probability,
                   SIMPLIFY = FALSE),
            htmltools::HTML
          ),
          labelOptions = labelOptions(sticky = TRUE, className = "lga-tooltip")
        ) |>
        legend()
    }
  })

  output$table <- renderDT({
    frame <- head(filtered(), input$top_n)
    shown <- frame[, c("rank_national", "lga_name", "state_name",
                       "probability", "risk_tier", "rank_state")]
    # The tier is rendered as a pill so the list can be scanned by shape and
    # colour, matching the map, instead of being read word by word.
    shown$risk_tier <- sprintf(
      '<span class="pill pill-%s">%s</span>', shown$risk_tier, shown$risk_tier
    )
    datatable(
      shown,
      colnames = c("Rank", "Area", "State", "Probability", "Tier", "In state"),
      selection = "single", rownames = FALSE, escape = FALSE,
      options = list(
        pageLength = 12, dom = "tp",
        columnDefs = list(list(className = "dt-left", targets = c(1, 2, 4)))
      )
    ) |> formatRound("probability", 3)
  })

  observeEvent(input$table_rows_selected, {
    frame <- head(filtered(), input$top_n)
    if (length(input$table_rows_selected) == 1) {
      selected_code(frame$lga_code[input$table_rows_selected])
    }
  })

  observeEvent(input$map_shape_click, { selected_code(input$map_shape_click$id) })
  observeEvent(input$map_marker_click, { selected_code(input$map_marker_click$id) })

  selected_row <- reactive({
    code <- selected_code()
    if (is.null(code)) return(NULL)
    frame <- predictions()
    row <- frame[frame$lga_code == code, ]
    if (nrow(row) == 0) NULL else row[1, ]
  })

  output$selected_title <- renderText({
    row <- selected_row()
    if (is.null(row)) "Select an area" else paste0(row$lga_name, ", ", row$state_name)
  })

  output$selected_summary <- renderUI({
    row <- selected_row()
    if (is.null(row)) {
      return(tags$p("Choose an area on the map or in the table to see its drivers."))
    }
    tags$p(
      sprintf("Probability %.3f over seven days. Tier %s. Ranked %d nationally and %d within %s.",
              row$probability, row$risk_tier, row$rank_national, row$rank_state, row$state_name)
    )
  })

  output$drivers <- renderTable({
    row <- selected_row()
    if (is.null(row)) return(NULL)
    frame <- fetch_drivers(STORE, row$lga_code)
    if (nrow(frame) == 0) return(NULL)
    data.frame(
      Driver = frame$feature_label,
      Value = round(as.numeric(frame$feature_value), 3),
      Effect = ifelse(as.numeric(frame$shap_value) > 0, "raises", "lowers")
    )
  }, striped = FALSE, bordered = TRUE)

  output$actors <- renderTable({
    row <- selected_row()
    if (is.null(row)) return(NULL)
    frame <- fetch_actors(STORE, row$lga_code)
    if (nrow(frame) == 0) return(NULL)
    data.frame(
      Actor = frame$actor, Events = frame$events,
      Deaths = frame$fatalities, `Last seen` = frame$last_seen, check.names = FALSE
    )
  }, striped = FALSE, bordered = TRUE)

  output$incidents <- renderTable({
    row <- selected_row()
    if (is.null(row)) return(NULL)
    frame <- fetch_incidents(STORE, row$lga_code)
    if (nrow(frame) == 0) return(NULL)
    data.frame(
      Date = frame$event_date, Type = frame$event_type, Deaths = frame$fatalities
    )
  }, striped = FALSE, bordered = TRUE)

  observeEvent(input$make_brief, {
    scope <- if (is.null(input$state) || input$state == "All states") "national" else input$state
    withProgress(message = "Drafting the brief", value = 0.5, {
      brief_content(generate_brief(STORE, scope))
    })
  })

  output$brief_text <- renderText({
    if (!is.null(brief_content())) return(brief_content())
    fetch_brief(STORE, "national")$content[1]
  })

  output$brief_source <- renderUI({
    stored <- fetch_brief(STORE, "national")
    tags$p(class = "footnote",
           sprintf("Latest stored brief generated by %s.", stored$generator[1]))
  })
}

# Cloud Run and most container hosts hand the port in through the environment
# and require binding on all interfaces rather than loopback.
shinyApp(
  ui, server,
  options = list(
    host = Sys.getenv("SHINY_HOST", "0.0.0.0"),
    port = as.integer(Sys.getenv("PORT", "7788"))
  )
)
