#!/usr/bin/env Rscript
#
# Publish the dashboard to shinyapps.io.
#
#   Rscript deploy/shinyapps/deploy.R [APP_NAME]
#
# Before the first run you must connect your shinyapps.io account once. Open
# https://www.shinyapps.io/admin/#/tokens, press "Show", copy the whole
# setAccountInfo call, and run it in R. It looks like this:
#
#   rsconnect::setAccountInfo(name = "yourname", token = "...", secret = "...")
#
# That command carries your account secret, so it is yours to run and never
# belongs in this repository or in a chat window. It writes the credential to
# ~/.config/R/rsconnect, after which this script can publish without ever
# handling it.

app_name <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(app_name)) app_name <- "lga-violence-risk"

app_dir <- normalizePath(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))))
if (!nzchar(app_dir) || !file.exists(file.path(app_dir, "app.R"))) {
  app_dir <- normalizePath("deploy/shinyapps")
}

# The default client timeout is ten seconds, which is shorter than the server
# takes to acknowledge a build step. The first deployment failed on the client
# side with a curl timeout while the server was still building normally, so the
# poll window is widened here rather than left to look like a deployment failure.
options(rsconnect.http.timeout = 600)

if (!requireNamespace("rsconnect", quietly = TRUE)) {
  stop("rsconnect is not installed. Run: install.packages('rsconnect')")
}

accounts <- rsconnect::accounts()
if (is.null(accounts) || nrow(accounts) == 0) {
  cat(
    "\nNo shinyapps.io account is connected on this machine.\n\n",
    "Open https://www.shinyapps.io/admin/#/tokens, press Show, and run the\n",
    "setAccountInfo(...) line it gives you in an R session. Then re-run this\n",
    "script. The secret in that line is yours and should not be pasted into a\n",
    "chat or committed.\n\n",
    sep = ""
  )
  quit(status = 1)
}

account <- accounts$name[1]
cat(sprintf("Publishing %s to shinyapps.io as account '%s'\n", app_name, account))
cat(sprintf("Bundle: %s\n", app_dir))

# forceUpdate keeps re-runs idempotent: the same app name is replaced rather than
# refused. The data directory is included explicitly because rsconnect only walks
# code dependencies by default.
rsconnect::deployApp(
  appDir = app_dir,
  appName = app_name,
  appTitle = "Seven day violence risk, Nigerian LGAs",
  account = account,
  forceUpdate = TRUE,
  launch.browser = FALSE,
  logLevel = "normal"
)

url <- sprintf("https://%s.shinyapps.io/%s/", account, app_name)
cat("\nDeployed\n")
cat(sprintf("URL: %s\n", url))
