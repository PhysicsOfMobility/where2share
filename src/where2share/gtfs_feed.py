import io
import zipfile

import pandas as pd
import geopandas as gpd

from pathlib import Path
from sqlalchemy import Engine


import logging

log = logging.getLogger(__name__)


class GTFSFeed:
    """
    Makes a GTFS feed available as pandas DataFrames.
    """

    def __init__(self, gtfs_path: Path | str):
        """
        Initializes a GTFS feed object.

        Parameters
        ----------
        gtfs_path
            Path to the GTFS feed as directory or ZIP archive.
        """
        self.gtfs_path = Path(gtfs_path)
        self.files = {}

    def _read_gtfs_file(self, table_name):
        log.info(f"Reading {table_name}.txt from GTFS feed...")
        fname = f"{table_name}.txt"
        if self.gtfs_path.is_dir():
            gtfs_file = pd.read_csv(self.gtfs_path / fname)
        else:
            gtfs_file = pd.read_csv(
                io.BytesIO(zipfile.ZipFile(self.gtfs_path, "r").read(fname))
            )

        date_cols = []
        match table_name:
            case "calendar":
                date_cols += ["start_date", "end_date"]
            case "calendar_dates":
                date_cols.append("date")

        for col in date_cols:
            gtfs_file[col] = pd.to_datetime(gtfs_file[col], format="%Y%m%d")

        return gtfs_file

    def _get_single_file(self, table_name):
        try:
            if table_name not in self.files:
                self.files[table_name] = self._read_gtfs_file(table_name)
            return self.files[table_name]
        except KeyError:
            raise ValueError(f"{table_name}.txt is not available in this GTFS feed.")

    @property
    def agency(self):
        return self._get_single_file("agency")

    @property
    def stops(self):
        return self._get_single_file("stops")

    @property
    def routes(self):
        return self._get_single_file("routes")

    @property
    def trips(self):
        return self._get_single_file("trips")

    @property
    def stop_times(self):
        return self._get_single_file("stop_times")

    @property
    def calendar(self):
        return self._get_single_file("calendar")

    @property
    def calendar_dates(self):
        return self._get_single_file("calendar_dates")

    @property
    def fare_attributes(self):
        return self._get_single_file("fare_attributes")

    @property
    def fare_rules(self):
        return self._get_single_file("fare_rules")

    @property
    def timeframes(self):
        return self._get_single_file("timeframes")

    @property
    def fare_media(self):
        return self._get_single_file("fare_media")

    @property
    def fare_products(self):
        return self._get_single_file("fare_products")

    @property
    def fare_leg_rules(self):
        return self._get_single_file("fare_leg_rules")

    @property
    def fare_transfer_rules(self):
        return self._get_single_file("fare_transfer_rules")

    @property
    def areas(self):
        return self._get_single_file("areas")

    @property
    def stop_areas(self):
        return self._get_single_file("stop_areas")

    @property
    def networks(self):
        return self._get_single_file("networks")

    @property
    def route_networks(self):
        return self._get_single_file("route_networks")

    @property
    def shapes(self):
        return self._get_single_file("shapes")

    @property
    def frequencies(self):
        return self._get_single_file("frequencies")

    @property
    def transfers(self):
        return self._get_single_file("transfers")

    @property
    def pathways(self):
        return self._get_single_file("pathways")

    @property
    def levels(self):
        return self._get_single_file("levels")

    @property
    def translations(self):
        return self._get_single_file("translations")

    @property
    def feed_info(self):
        return self._get_single_file("feed_info")

    @property
    def attributions(self):
        return self._get_single_file("attributions")

    def get_stops_as_gdf(self, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Returns the stops file as a GeoDataFrame.

        Parameters
        ----------
        crs
            Coordinate reference system of the stop locations' coordinates in the feed.

        Returns
        -------
        GeoDataFrame
        """
        return gpd.GeoDataFrame(
            self.stops.drop(["stop_lon", "stop_lat"], axis=1),
            geometry=gpd.points_from_xy(
                self.stops.stop_lon,
                self.stops.stop_lat,
                crs=crs,
            ),
        )


class DBGTFSFeed(GTFSFeed):
    def __init__(self, gtfs_path: Path | str, db_engine: Engine):
        """
        Initializes a GTFS feed object that can interface with a PostGIS database.

        Parameters
        ----------
        gtfs_path
            Path to the GTFS feed as ZIP archive.
        db_engine
            SQLAlchemy engine to connect to the database.
        """
        super().__init__(gtfs_path)
        self.engine = db_engine

    def push_stops(self, table: str = "pt_stops_projected", schema: str = "public"):
        gdf = self.get_stops_as_gdf().to_crs("EPSG:3035")
        with self.engine.connect() as conn:
            gdf.to_postgis(
                name=table,
                con=conn,
                if_exists="fail",
                schema=schema,
                chunksize=10000,
            )
