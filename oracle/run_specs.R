#!/usr/bin/env Rscript
# Golden generator: runs every YAML spec through real dplyr/tidyr and writes
# the result as parquet into tests/golden/. See docs/TESTING.md.
#
# Usage: Rscript oracle/run_specs.R [spec_dir] [golden_dir]

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(arrow)
  library(yaml)
})

args <- commandArgs(trailingOnly = TRUE)
spec_dir <- if (length(args) >= 1) args[1] else "tests/specs"
golden_dir <- if (length(args) >= 2) args[2] else "tests/golden"
dir.create(golden_dir, recursive = TRUE, showWarnings = FALSE)

cast_col <- function(values, type) {
  v <- unlist(lapply(values, function(x) if (is.null(x)) NA else x))
  switch(type,
    int = as.integer(v),
    float = as.double(v),
    str = as.character(v),
    bool = as.logical(v),
    date = as.Date(v),
    stop("unknown type: ", type)
  )
}

build_table <- function(data, types) {
  cols <- lapply(names(data), function(k) cast_col(data[[k]], types[[k]]))
  names(cols) <- names(data)
  as_tibble(cols)
}

apply_step <- function(df, step, env) {
  verb <- step$verb
  splice <- function(call_tpl) {
    eval(parse(text = sprintf(call_tpl, step$r)), envir = env)
  }
  env$df <- df
  if (verb == "filter") return(splice("dplyr::filter(df, %s)"))
  if (verb == "mutate") return(splice("dplyr::mutate(df, %s)"))
  if (verb == "summarize") {
    return(splice("dplyr::summarise(df, %s, .groups = 'drop_last')"))
  }
  if (verb == "arrange") return(splice("dplyr::arrange(df, %s)"))
  if (verb == "select") return(splice("dplyr::select(df, %s)"))
  if (verb == "rename") return(splice("dplyr::rename(df, %s)"))
  if (verb == "group_by") {
    return(dplyr::group_by(df, !!!syms(step$cols), .add = TRUE))
  }
  if (verb == "ungroup") return(dplyr::ungroup(df))
  if (verb == "distinct") {
    if (is.null(step$cols) || length(step$cols) == 0) {
      return(dplyr::distinct(df))
    }
    return(dplyr::distinct(df, !!!syms(step$cols)))
  }
  if (verb == "slice_head") return(dplyr::slice_head(df, n = step$rows))
  if (verb == "slice_min") {
    return(dplyr::slice_min(df, !!sym(step$order), n = step$rows))
  }
  if (verb == "slice_max") {
    return(dplyr::slice_max(df, !!sym(step$order), n = step$rows))
  }
  if (verb == "separate") {
    return(tidyr::separate(df, !!sym(step$column), into = unlist(step$into),
                           sep = step$sep, fill = "right", extra = "drop"))
  }
  if (verb == "unite") {
    return(tidyr::unite(df, !!step$new, dplyr::all_of(unlist(step$cols)),
                        sep = step$sep))
  }
  if (verb == "slice_tail") return(dplyr::slice_tail(df, n = step$rows))
  if (verb == "count") {
    if (is.null(step$cols)) return(dplyr::count(df))
    return(dplyr::count(df, !!!syms(step$cols)))
  }
  if (verb %in% c("left_join", "inner_join", "right_join", "full_join",
                  "semi_join", "anti_join")) {
    df2 <- env$df2
    fn <- get(verb, envir = asNamespace("dplyr"))
    return(fn(df, df2, by = step$by))
  }
  if (verb == "pivot_longer") {
    return(tidyr::pivot_longer(df, cols = all_of(step$cols),
                               names_to = step$names_to %||% "name",
                               values_to = step$values_to %||% "value"))
  }
  if (verb == "pivot_wider") {
    return(tidyr::pivot_wider(df, names_from = all_of(step$names_from),
                              values_from = all_of(step$values_from)))
  }
  stop("unknown verb: ", verb)
}

`%||%` <- function(a, b) if (is.null(a)) b else a

specs <- list.files(spec_dir, pattern = "\\.yaml$", recursive = TRUE,
                    full.names = TRUE)
cat(sprintf("oracle: %d specs, dplyr %s\n", length(specs),
            as.character(packageVersion("dplyr"))))

failures <- 0
for (path in specs) {
  spec <- yaml::read_yaml(path)
  env <- new.env()
  df <- build_table(spec$data, spec$types)
  if (!is.null(spec$data2)) env$df2 <- build_table(spec$data2, spec$types2)
  ok <- TRUE
  for (step in spec$chain) {
    df <- tryCatch(apply_step(df, step, env), error = function(e) {
      cat(sprintf("FAIL %s: %s\n", path, conditionMessage(e)))
      ok <<- FALSE
      NULL
    })
    if (!ok) break
  }
  if (!ok) { failures <- failures + 1; next }
  df <- dplyr::ungroup(df)
  rel <- sub("\\.yaml$", ".parquet", sub(paste0("^", spec_dir, "/?"), "", path))
  out <- file.path(golden_dir, rel)
  dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
  arrow::write_parquet(df, out)
}

meta <- list(dplyr = as.character(packageVersion("dplyr")),
             tidyr = as.character(packageVersion("tidyr")),
             r = R.version.string)
writeLines(yaml::as.yaml(meta), file.path(golden_dir, "_meta.yaml"))
if (failures > 0) quit(status = 1)
cat("oracle: all goldens written\n")
