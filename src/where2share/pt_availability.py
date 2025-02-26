import datetime
from typing import Iterable, Any

import pandas as pd
import geopandas as gpd

from pathlib import Path
from sqlalchemy import Engine

from where2share.common import wgs84, utm_32N
from where2share.gtfs_feed import GTFSFeed
from where2share.cy_stop_od import (
    compute_od_matrix_from_stops,
    compute_od_dict_from_stops,
)


def get_eligible_stops(
    region: str,
    engine: Engine,
    db_table: str = "gemeinden",
    region_is_ars: bool = False,
) -> pd.DataFrame:
    """
    Fetch all PT stops from the Database that are contained in the given region.

    Parameters
    ----------
    region
        Name of the region. Example: ``Dresden``.
    db_table
        Kind of the region. Can be 'gemeinden' or 'kreise'.
    region_is_ars
        If True, the region is an ARS (Amtlicher Regionalschlüssel) code.
        Otherwise, it is a name.

    Returns
    -------
    DataFrame with eligible stops
    """

    if db_table not in ["gemeinden", "kreise"]:
        raise ValueError("Invalid db_table")

    col = "amtlicher_regierungs_schluessel" if region_is_ars else "geografischer_name"

    sql = (
        "SELECT p.stop_id, p.geometry_wgs84, p.stop_name, p.stop_desc "
        "FROM pt_stops_projected p "
        f"JOIN {db_table} r "
        "ON ST_Intersects(p.geometry_wgs84, r.geometry) "
        f"WHERE r.{col} = '{region}';"
    )

    return (
        gpd.read_postgis(
            sql,
            con=engine,
            geom_col="geometry_wgs84",
            crs=wgs84,
        )
        .rename({"geometry_wgs84": "geometry"})
        .set_geometry("geometry", crs=wgs84)
        .to_crs(utm_32N)
    )


def compute_od_matrix_from_gtfs(
    gtfs_feed: Path | str | GTFSFeed,
    date: str | datetime.datetime = "now",
    return_type: str = "ndarray",
    stop_subset: Iterable[Any] | None = None,
) -> pd.DataFrame | dict:
    """
    Computes an origin-destination matrix for a given GTFS feed and date. The entries
    of the matrix are the number of trips between each pair of stops on the given date.

    Parameters
    ----------
    sample_gtfs_path
        Path to the GTFS feed as ZIP archive or GTFSFeed instance.
    date
        Date for which to compute the OD matrix.
    return_type
        If 'ndarray', return the OD matrix as a stop index + 2D numpy array. If 'dict',
        return the OD matrix as a dictionary. If 'dataframe', return the OD matrix as a
        pandas DataFrame.
    stop_subset
        Iterable of stop IDs. If not None, only consider the stops in this list.

    Returns
    -------
    Origin-destination matrix
    """
    if not isinstance(gtfs_feed, GTFSFeed):
        feed = GTFSFeed(gtfs_feed)
    else:
        feed = gtfs_feed
    date = pd.Timestamp(date)

    assert return_type in [
        "ndarray",
        "dict",
        "dataframe",
        "dataframe_from_dict",
    ], "Invalid return_type"

    eligible_stop_ids = set(feed.stops.stop_id.unique())

    if stop_subset is not None:
        eligible_stop_ids &= set(stop_subset)

    eligible_services = feed.calendar.query(f"start_date <= @date <= end_date")

    # eligible services for date
    eligible_services_date = set(
        eligible_services[
            eligible_services[date.day_name().lower()] == 1
        ].service_id.values
    )
    eligible_service_exceptions = feed.calendar_dates[
        feed.calendar_dates.date.isin([date.normalize()])
    ]

    eligible_services_date |= set(
        eligible_service_exceptions.query("exception_type == 1").service_id.values
    )  # added services

    eligible_services_date -= set(
        eligible_service_exceptions.query("exception_type == 2").service_id.values
    )  # removed services

    # eligible_services_date now contains all services that are active on the given
    # date
    # Now, get the trip_ids of every trip that is active on the given date
    eligible_trips_date = feed.trips.query(
        "service_id in @eligible_services_date"
    ).trip_id.values

    # schedule contains the stop times for all stops that are "eligible", i.e., in the
    # stop_subset if one was supplied, and that are associated with a trip that is
    # "eligible", i.e., active on the given date. The whole thing is indexed by trip ID
    # and stop sequence.

    schedule = (
        feed.stop_times.query(
            "stop_id in @eligible_stop_ids & trip_id in @eligible_trips_date"
        )
        .set_index(["trip_id", "stop_sequence"])
        .drop(
            [
                "arrival_time",
                "stop_headsign",
                "pickup_type",
                "drop_off_type",
                "shape_dist_traveled",
            ],
            axis=1,
            errors="ignore",
        )
        .sort_index()
    )

    assert schedule.index.levels[0].is_unique, "Duplicate trip IDs in schedule"

    stops_array = schedule["stop_id"].reset_index().to_numpy(dtype="U")

    match return_type:
        case "ndarray":
            stop_idx, od = compute_od_matrix_from_stops(stops_array)
            return stop_idx, od
        case "dict":
            return compute_od_dict_from_stops(stops_array)
        case "dataframe":
            stop_idx, od = compute_od_matrix_from_stops(stops_array)
            return pd.DataFrame(od, index=stop_idx, columns=stop_idx)
        case "dataframe_from_dict":
            return (
                pd.DataFrame(compute_od_dict_from_stops(stops_array))
                .T.sort_index(axis=0)
                .sort_index(axis=1)
                .fillna(0)
                .astype("i8")
            )
