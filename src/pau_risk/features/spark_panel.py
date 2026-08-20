"""The same weekly aggregation expressed as a Spark job.

The pandas implementation in :mod:`panel` is the default because the Nigerian
panel is 774 units by roughly 620 weeks, which fits comfortably in memory and
runs in seconds. The architecture in the brief nonetheless routes the producers
into a Spark feature pipeline, and that choice is right for the shape this
service takes in production rather than in assessment: a national deployment
consuming ACLED, UCDP, nine scraped feeds and a social firehose accumulates far
more raw records than events, the text extraction over documents is embarrassingly
parallel, and the same job has to run over a multi year backfill on a schedule.

This module implements the window aggregations in Spark SQL so that both paths
produce an identical panel. The window frames are expressed with
``rowsBetween(-n, -1)``, which excludes the current week by construction and is
the direct Spark equivalent of the exclusive prefix sums used in the pandas path.
The Hawkes intensity is not computed here: it is a single fitted model with four
parameters, so it is broadcast and joined rather than recomputed per partition.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..config import Settings, get_settings
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)


def spark_available() -> bool:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    return True


def get_session(app_name: str = "pau-lga-risk-features"):
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def build_event_panel(
    incidents: pd.DataFrame,
    registry: pd.DataFrame,
    adjacency: pd.DataFrame,
    weeks: Sequence[pd.Timestamp],
    lag_weeks: Sequence[int],
    neighbour_lag_weeks: Sequence[int],
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Return the count based feature block, computed with Spark.

    Only the event driven families are built here, which are the ones that scale
    with data volume. Calendar terms, Hawkes intensity and the static attributes
    are cheap and are attached by the caller.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    settings = settings or get_settings()
    session = get_session()
    LOGGER.info("spark session %s on %s", session.version, session.sparkContext.master)

    week_frame = pd.DataFrame({"week_start": pd.to_datetime(list(weeks))})
    week_frame["week_index"] = range(len(week_frame))

    events = incidents[incidents["lga_code"].notna()].copy()
    events["event_date"] = pd.to_datetime(events["event_date"])
    events["week_start"] = events["event_date"].dt.to_period("W-SUN").dt.start_time

    weekly = (
        events.assign(
            is_target=(events["event_class"] == "banditry_kidnapping").astype(int),
            is_operation=(events["event_class"] == "state_operation").astype(int),
        )
        .assign(
            target_fatalities=lambda frame: frame["is_target"] * frame["fatalities"].fillna(0)
        )
        .groupby(["lga_code", "week_start"], as_index=False)
        .agg(
            events=("is_target", "sum"),
            fatalities=("target_fatalities", "sum"),
            operations=("is_operation", "sum"),
        )
    )

    spark_weeks = session.createDataFrame(week_frame)
    spark_registry = session.createDataFrame(registry[["lga_code", "state_name"]])
    spark_weekly = session.createDataFrame(weekly) if not weekly.empty else None
    spark_adjacency = session.createDataFrame(
        adjacency[["lga_code", "neighbour_code", "weight"]]
    )

    skeleton = spark_registry.crossJoin(spark_weeks)
    if spark_weekly is not None:
        panel = skeleton.join(spark_weekly, on=["lga_code", "week_start"], how="left")
    else:
        panel = skeleton.withColumn("events", F.lit(0.0))
        panel = panel.withColumn("fatalities", F.lit(0.0)).withColumn("operations", F.lit(0.0))
    panel = panel.fillna({"events": 0.0, "fatalities": 0.0, "operations": 0.0})

    # Neighbour exposure: weight each neighbour's weekly count by the shared
    # border share, then sum back onto the focal LGA for the same week.
    neighbour = (
        panel.select(
            F.col("lga_code").alias("neighbour_code"),
            "week_index",
            F.col("events").alias("nb_events_raw"),
            F.col("fatalities").alias("nb_fatalities_raw"),
            F.col("operations").alias("nb_operations_raw"),
        )
        .join(spark_adjacency, on="neighbour_code", how="inner")
        .groupBy("lga_code", "week_index")
        .agg(
            F.sum(F.col("nb_events_raw") * F.col("weight")).alias("nb_events"),
            F.sum(F.col("nb_fatalities_raw") * F.col("weight")).alias("nb_fatalities"),
            F.sum(F.col("nb_operations_raw") * F.col("weight")).alias("nb_operations"),
        )
    )
    panel = panel.join(neighbour, on=["lga_code", "week_index"], how="left").fillna(
        {"nb_events": 0.0, "nb_fatalities": 0.0, "nb_operations": 0.0}
    )

    ordering = Window.partitionBy("lga_code").orderBy("week_index")
    for window in lag_weeks:
        panel = panel.withColumn(
            f"own_events_{window}w",
            F.sum("events").over(ordering.rowsBetween(-window, -1)),
        )
    for window in (4, 12, 52):
        panel = panel.withColumn(
            f"own_fatalities_{window}w",
            F.sum("fatalities").over(ordering.rowsBetween(-window, -1)),
        )
    for window in (4, 12):
        panel = panel.withColumn(
            f"own_ops_{window}w", F.sum("operations").over(ordering.rowsBetween(-window, -1))
        )
        panel = panel.withColumn(
            f"nb_ops_{window}w", F.sum("nb_operations").over(ordering.rowsBetween(-window, -1))
        )
    for window in neighbour_lag_weeks:
        panel = panel.withColumn(
            f"nb_events_{window}w", F.sum("nb_events").over(ordering.rowsBetween(-window, -1))
        )
    panel = panel.withColumn(
        "nb_fatalities_4w", F.sum("nb_fatalities").over(ordering.rowsBetween(-4, -1))
    )

    expanding = ordering.rowsBetween(Window.unboundedPreceding, -1)
    panel = panel.withColumn(
        "hist_rate", F.coalesce(F.avg("events").over(expanding), F.lit(0.0))
    )

    state_window = Window.partitionBy("state_name", "week_index")
    panel = panel.withColumn("state_events_week", F.sum("events").over(state_window))
    panel = panel.withColumn("state_ops_week", F.sum("operations").over(state_window))
    state_ordering = Window.partitionBy("state_name", "lga_code").orderBy("week_index")
    panel = panel.withColumn(
        "state_events_4w", F.sum("state_events_week").over(state_ordering.rowsBetween(-4, -1))
    )
    panel = panel.withColumn(
        "state_ops_4w", F.sum("state_ops_week").over(state_ordering.rowsBetween(-4, -1))
    )

    panel = panel.withColumn("label", (F.col("events") > 0).cast("int"))
    result = panel.drop("state_events_week", "state_ops_week").toPandas().fillna(0.0)
    session.stop()
    LOGGER.info("spark panel returned %d rows", len(result))
    return result
