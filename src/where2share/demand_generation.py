import re
import os

from where2share.common import utm_32N

os.environ["USE_PYGEOS"] = "0"

import functools as ft
import numpy as np
import shapely as shp
import pandas as pd
import osmnx as ox
import networkx as nx

import geopandas as gpd
import graph_tool.all as gt

from collections import defaultdict
from scipy.spatial import Voronoi
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points
from shapely import to_wkb
from tqdm.auto import tqdm
from sqlalchemy import URL, create_engine, Engine, text
from pathlib import Path
from datetime import datetime

from where2share.cy_stop_od import expand_pt_od_matrix, get_nodes_for_stops
from where2share.gtfs_feed import DBGTFSFeed
from where2share.pt_availability import get_eligible_stops, compute_od_matrix_from_gtfs


import logging

log = logging.getLogger(__name__)


class DemandGenerator:
    """
    Base Class for the extraction of Data used for ridepooling simulations from a database.
    """

    trip_factor = 3.2
    """Number of trips per person per day across all RegioStar7 regions in germany"""

    average_trip_duration = 22
    """
    Average duration of a car trip in Germany (there is about 2 minutes of variation 
    between RegioStar7 regions)
    """

    def __init__(self, db_engine: str | URL | Engine):
        """
        Initializes a DemandGenerator object with the given database connection parameters.

        Parameters
        ----------
        db_url
            SQLAlchemy URL of the database to connect to.
        """

        if isinstance(db_engine, (URL, str)):
            db_engine = create_engine(db_engine)
        elif not isinstance(db_engine, Engine):
            raise TypeError(
                f"db_engine must be a sqlalchemy.Engine, not {type(db_engine)}"
            )

        self.engine = db_engine

        # Check connection
        dresden = gpd.read_postgis(
            "SELECT * FROM kreise WHERE geografischer_name='Dresden';",
            con=self.engine,
            geom_col="geometry",
        )
        assert len(dresden) == 1, "something is wrong with database/connection"

    @staticmethod
    def _cut(line, distance):
        # Cuts a line in two at a distance from its starting point
        if distance <= 0.0 or distance >= line.length:
            return [LineString(line)]
        coords = list(line.coords)
        for i, p in enumerate(coords):
            pd = line.project(Point(p))
            # Save space by checking whether a point already exists that matches the distance
            if pd == distance:
                return [LineString(coords[: i + 1]), LineString(coords[i:])]
        else:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:]),
            ]

    @classmethod
    def _ancillary_nodes(
        cls,
        nodes0: gpd.GeoDataFrame,
        edges0: gpd.GeoDataFrame,
        target_length: float = 250,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Splits edges into multiple shorter segments and introduces new nodes at the
        splitting points.

        This function iterates over all edges and if an edge is longer than the specified
        target length, it splits the edge into multiple segments. New nodes are
        introduced at the splitting points. The resulting nodes and edges are returned in
        the same format as the input nodes and edges.

        Parameters
        ----------
        nodes0
            A GeoDataFrame representing the original nodes. Each node is expected to have
            'geometry' attribute.
        edges0
            A GeoDataFrame representing the original edges. Each edge is expected to have
            'geometry', 'length', and 'reversed' attributes.
        target_length
            The target maximum length for edge segments. Defaults to 250.

        Returns
        -------
        nodes
            A GeoDataFrame of nodes, including the original nodes and the new nodes
            introduced at splitting points.
        edges
            A GeoDataFrame of edges, where edges longer than the target_length have been
            split into multiple shorter segments.
        """
        log.info(f"Adding ancillary nodes...")

        nodes = nodes0.copy().to_crs("EPSG:25832")
        edges = edges0.copy().to_crs("EPSG:25832")
        for (u, v, key), data in edges0.iterrows():

            geometry = data["geometry"]
            if geometry is None:
                n1 = nodes.loc[u]["geometry"]
                n2 = nodes.loc[v]["geometry"]
                geometry = LineString((n1, n2))
            heading = (
                nodes.loc[u]["geometry"].xy[0][0] - nodes.loc[v]["geometry"].xy[0][0]
            ) > 0
            length = data["length"]
            assert (
                np.abs((geometry.length / length) - 1) < 0.1
            ), f"Lengths do not match {length}, {geometry.length}"
            # length = geometry.length

            # hotfix without knowing what happens
            if type(data["reversed"]) is not bool:
                data["reversed"] = False
            data["reversed"] = False
            c = int(length // target_length)
            # keep track of lengths to figure out where things go wrong
            resulting_length = 0.0
            if c > 2:
                last_node = u
                new_node = ""
                for i in range(1, c) if not data["reversed"] else range(c - 1, 0, -1):
                    first_line = cls._cut(geometry, i / c * geometry.length)[0]
                    inbtw_point = (
                        Point(np.array(first_line.xy)[:, -1])
                        if not data["reversed"]
                        else Point(np.array(first_line.xy)[:, 0])
                    )
                    new_node = f"{u}_{v}_{i}"
                    nodes.loc[new_node] = dict(
                        x=inbtw_point.x + (1 if heading else -1),
                        y=inbtw_point.y + (1 if heading else -1),
                        geometry=Point(
                            inbtw_point.x + (1 if heading else -1),
                            inbtw_point.y + (1 if heading else -1),
                        ),
                    )

                    # Split the edge into two parts
                    # new_edge_1 = (last_node, new_node, key) if i==1 else (f"{u}_{v}_{i-1}", new_node, key)
                    new_data = data.copy()
                    new_data["reversed"] = data["reversed"]
                    new_data["length"] = length / c
                    new_data["travel_time"] = data["travel_time"] / c
                    resulting_length += length / c
                    new_data["geometry"] = (
                        cls._cut(first_line, (i - 1) / c * geometry.length)[-1]
                        if i != 1
                        else first_line
                    )
                    edges.loc[last_node, new_node, key] = new_data
                    last_node = new_node

                    # Remove old edge

                # new_edge_2 = (new_node, v, key)
                new_data = data.copy()
                new_data["reversed"] = False
                new_data["length"] = length / c
                new_data["travel_time"] = data["travel_time"] / c
                resulting_length += length / c
                new_data["geometry"] = cls._cut(
                    geometry, (c - 1) / c * geometry.length
                )[-1]
                edges.loc[new_node, v, key] = new_data
                edges.drop((u, v, key), axis=0, inplace=True)
                assert (
                    resulting_length - length
                ) ** 2 < 0.01, f"Lengths do not match {resulting_length} {length}"
        edges.crs = "EPSG:25832"
        nodes.crs = "EPSG:25832"
        return nodes, edges
        
    
    @ft.lru_cache(maxsize=10)
    def extract_graph(
        self, geographical_name: str, ars: bool = False, table_name: str = "kreise"
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Extracts a graph from a given geographical area, simplifies it, calculates
        travel times, and maps population data onto its nodes.

        This function reads boundaries, nodes, edges, and population data from a PostGIS
        database for a given geographical area. It then constructs a graph from the
        nodes and edges, simplifies it, and calculates edge speeds and travel times.
        Additionally, it maps population data onto the nodes of the graph.

        Parameters
        ----------
        geographical_name
            The name of the geographical area to extract the graph from.
        ars
            If True, the 'amtlicher_regierungs_schluessel' is used as the column name
            for the geographical area in the database. If False, 'geografischer_name'
            is used. Defaults to False.
        table_name
            The name of the database table to extract the graph from. Defaults to
            ``kreise``.

        Returns
        -------
        nodes
            A GeoDataFrame of nodes, updated with population data.
            Coordinates in UTM32N (EPSG:25832).
        edges
            A GeoDataFrame of edges, updated with travel times.
        """
        log.info(f"Extracting graph for {geographical_name} from {table_name}...")

        col_name = (
            "geografischer_name" if not ars else "amtlicher_regierungs_schluessel"
        )
        # Read data from PostGIS database
        boundary, nodes, edges, population = self._read_data(
            geographical_name, col_name, table=table_name
        )
        if not (len(nodes) and len(edges)):
            raise ValueError(
                f"No data found for {geographical_name} in {table_name} "
                f"({'' if ars else 'not'} using ARS)"
            )
        nodes, edges = self._preprocess_graph(nodes, edges)
        nodes, edges = self._ancillary_nodes(nodes, edges)
        nodes = self._map_population(nodes, edges, population)

        nodes = nodes.drop(labels=["x", "y"], axis=1)
        nodes["x"] = nodes.geometry.x
        nodes["y"] = nodes.geometry.y
        return nodes, edges
        
    @ft.lru_cache(maxsize=10)
    def extract_graph_from_geometry(
        self, g: str
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Extracts a graph from a given geometry, simplifies it, calculates
        travel times, and maps population data onto its nodes.

        This function reads boundaries, nodes, edges, and population data from a PostGIS
        database for a given geographical area. It then constructs a graph from the
        nodes and edges, simplifies it, and calculates edge speeds and travel times.
        Additionally, it maps population data onto the nodes of the graph.

        Parameters
        ----------
        g geometry
            geometry in wkb hex format including the srid
        ars
            If True, the 'amtlicher_regierungs_schluessel' is used as the column name
            for the geographical area in the database. If False, 'geografischer_name'
            is used. Defaults to False.
        table_name
            The name of the database table to extract the graph from. Defaults to
            ``kreise``.

        Returns
        -------
        nodes
            A GeoDataFrame of nodes, updated with population data.
            Coordinates in UTM32N (EPSG:25832).
        edges
            A GeoDataFrame of edges, updated with travel times.
        """
        log.info(f"Extracting graph for geometry...")

        # Read data from PostGIS database
        boundary, nodes, edges, population = self._read_data_from_geometry(
            g, 
        )
        if not (len(nodes) and len(edges)):
            raise ValueError(
                f"No data found"
            )
        nodes, edges = self._preprocess_graph(nodes, edges)
        nodes, edges = self._ancillary_nodes(nodes, edges)
        nodes = self._map_population(nodes, edges, population)

        nodes = nodes.drop(labels=["x", "y"], axis=1)
        nodes["x"] = nodes.geometry.x
        nodes["y"] = nodes.geometry.y
        return nodes, edges
        
        
    @staticmethod
    def get_graph_and_node_index(
        nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame
    ) -> tuple[nx.MultiDiGraph, dict]:
        """
        Creates a graph from the given nodes and edges and returns the graph and a
        mapping from nodes to their integer index.

        Parameters
        ----------
        nodes
            Nodes GeoDataFrame
        edges
            Edges GeoDataFrame

        Returns
        -------
        G
            A directed graph created from the given nodes and edges.
        n2i
            A mapping from node objects to their integer index.
        """
        G = ox.graph_from_gdfs(nodes, edges)
        n2i = {x: i for i, x in enumerate(G.nodes)}
        return G, n2i

    @staticmethod
    def _is_accessible(x: str) -> bool:
        """
        Checks whether the edge having ``x`` as a maximum speed is "accessible" for
        ridepooling, i.e. if its max speed is smaller than 70.

        Parameters
        ----------
        x
            The maximum speed of an edge.

        Returns
        -------
        True if maxspeed less then 70

        """
        return not (
            "70" in str(x)
            or "80" in str(x)
            or "90" in str(x)
            or "100" in str(x)
            or "110" in str(x)
            or "120" in str(x)
            or "130" in str(x)
            or "none" in str(x)
        )

    @classmethod
    def _voronoi_cells(
        cls, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Calculates and assigns Voronoi cells to a given set of nodes based on their edges.

        This function first determines the accessibility of each node based on the
        maximum speed of its incoming and outgoing edges. It then computes the Voronoi
        cells for the accessible nodes. The Voronoi cell of each node is added as a new
        attribute to the respective node.

        Parameters
        ----------
        nodes
            A GeoDataFrame representing the nodes. Each node is expected to have a
            'geometry' attribute.
        edges
            A GeoDataFrame representing the edges between nodes. Each edge is expected to
            have a 'maxspeed' attribute.

        Returns
        -------
        nodes
            The input GeoDataFrame of nodes, updated with a new 'voronoi' attribute
            representing the Voronoi cell for each node.
        """
        # Initialize an array to mark whether a node is accessible
        accessible = np.zeros(len(nodes), dtype=bool)

        # For each node, check if all its incoming and outgoing edges are accessible
        for i, n in enumerate(nodes.index):
            inedges = edges.xs(n, level=1)
            outedges = edges.xs(n, level=0)
            nedges = pd.concat([inedges, outedges])
            accessible[i] = nedges["maxspeed"].apply(cls._is_accessible).all()

        # Extract coordinates from the node geometries
        geom = nodes.geometry
        points = np.array(
            [[x, y] for x, y in geom.apply(lambda x: np.array(x.xy).reshape(-1))]
        )
        # Create a convex hull around the points and buffer it by 250 meters (distance people walk)
        hull = LineString(points).convex_hull.buffer(250)
        hull = Polygon(
            [hull.boundary.interpolate(x) for x in np.arange(0, hull.length, 250)]
        )
        x, y = hull.boundary.xy
        relevant_points = points[accessible]

        # Create an array to hold the relevant points and the convex hull boundary points
        more_points = np.zeros((len(relevant_points) + len(x), 2))
        more_points[: len(relevant_points), :] = relevant_points
        more_points[len(relevant_points) :, 0] = x
        more_points[len(relevant_points) :, 1] = y
        # Calculate the Voronoi cells for the points
        vor = Voronoi(more_points)
        # Initialize arrays to store the Voronoi cell polygons
        region_shapes = np.array(
            [Polygon([(0, 0), (0, 0), (0, 0)]) for i in range(len(points))]
        )
        relevant_region_shapes = np.array(
            [Polygon([(0, 0), (0, 0), (0, 0)]) for i in range(len(more_points))]
        )
        # For each Voronoi cell, calculate the polygon shape
        for i, pidx in enumerate(vor.point_region):
            region = vor.regions[pidx]
            coords = []
            for j, idx in enumerate(region):
                if idx != -1:
                    coords.append(vor.vertices[idx])
                else:
                    bound_point_1 = nearest_points(
                        hull, Point(vor.vertices[region[(idx - 1) % len(region)]])
                    )[0]
                    bound_point_2 = nearest_points(
                        hull, Point(vor.vertices[region[(idx + 1) % len(region)]])
                    )[0]
                    # This may be improved to include the entire span of the boundary
                    coords.append(bound_point_1)
                    coords.append(bound_point_2)
            poly = Polygon(coords)
            relevant_region_shapes[i] = poly

        # Assign the Voronoi cell polygons to the respective nodes
        region_shapes[accessible] = relevant_region_shapes[: len(relevant_points)]
        nodes["voronoi"] = [x for x in region_shapes]
        return nodes

    @classmethod
    def _map_population(
        cls,
        nodes: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        population: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """

        Parameters
        ----------
        nodes
        edges
        population

        Returns
        -------

        """
        log.info(f"Adding population data to nodes...")

        # Transform nodes to a new coordinate system and create Voronoi cells
        if not "maxspeed" in edges.columns:
            edges["maxspeed"] = 50.0  # Default value which would be imputed anyway
        nodes = cls._voronoi_cells(nodes.to_crs("EPSG:25832"), edges)

        # Transform population data to the same coordinate system
        pop_data = population.to_crs("EPSG:25832")

        # Assign population data's geometry to a new column called 'hexagon'
        pop_data["hexagon"] = pop_data.geometry

        # Join nodes and population data on the left, using their geometry
        jdf = gpd.sjoin(
            nodes.set_geometry("voronoi", crs="EPSG:25832"), pop_data, how="left"
        )

        # Define a function that projects population data onto nodes
        def project_population(x):
            # For each row in x, calculate the area of intersection between Voronoi cell and
            # hexagon, divide it by the area of the hexagon and multiply by the population.
            # If there's no hexagon, return 0.
            return np.sum(
                x.apply(
                    lambda row: (
                        shp.intersection(row["voronoi"], row["hexagon"]).area
                        / row["hexagon"].area
                        * row["population"]
                        if row["hexagon"]
                        else 0
                    ),
                    axis=1,
                )
            )

        # Apply the project_population function to each group of rows in jdf with the same index
        node_pop = jdf.groupby(jdf.index).apply(project_population)

        # Assign the resulting population data to a new column in nodes
        nodes["population"] = node_pop

        # Return the modified nodes data
        return nodes

    @staticmethod
    def _f_gamma(gamma: float, Dmat: np.ndarray, O: np.ndarray):
        L = len(Dmat)
        z_gamma = Dmat.copy()
        z_gamma[Dmat > 0] = Dmat[Dmat > 0] ** gamma
        z_gamma[Dmat <= 0] = 0
        lnz = Dmat.copy()
        lnz[Dmat > 0] = np.log(Dmat[Dmat > 0])
        lnz[Dmat <= 0] = 0

        Sz = np.tile(np.sum(z_gamma, axis=1), (L, 1)).T

        Szlnz2 = np.tile(np.sum((z_gamma * lnz**2), axis=1), (L, 1)).T
        Szlnz = np.tile(np.sum((z_gamma * lnz), axis=1), (L, 1)).T

        f = np.sum(O * Dmat * (z_gamma / Sz))
        df = np.sum(O * Dmat * (z_gamma * lnz / Sz - z_gamma * Szlnz / Sz**2))
        ddf = np.sum(
            O
            * Dmat
            * (
                z_gamma * lnz * (-Szlnz + lnz * Sz) / Sz**2
                + z_gamma * (-Szlnz2 + (Szlnz * lnz)) / Sz**2
                + z_gamma * (-2 * Szlnz * (-Szlnz + lnz * Sz)) / Sz**3
            )
        )
        return f, df, ddf

    def get_network_navigation(
        self, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, sigma: float = None
    ):
        """
        Distances are in minutes. Trips are normalized.
        """
        # Create a graph from the given nodes and edges
        G, n2i = self.get_graph_and_node_index(nodes=nodes, edges=edges)
        Dmat = np.ones((len(nodes), len(nodes)), dtype=np.float32) * -1
        Pmat = np.ones((len(nodes), len(nodes)), dtype=np.int32) * -1
        Amat = nx.adjacency_matrix(G, weight="travel_time", dtype=np.float32) / 60 * 1.5

        O = np.tile(self.trip_factor * nodes["population"].values, (len(Dmat), 1)).T

        G_gt = gt.Graph(directed=G.is_directed())

        # "Dict" [idx] = weight
        G_gt_weights = G_gt.new_edge_property("double")

        # mapping of nx vertices to gt indices
        vertices = {}
        v2n = {}
        for node in G.nodes:

            v = G_gt.add_vertex()
            vertices[node] = v
            v2n[int(v)] = node

        # mapping of nx edges to gt edge indices
        edges_ = defaultdict(lambda: defaultdict(dict))

        for src, dst, k, data in G.edges(data=True, keys=True):
            # Look up the vertex idxs from our vertices mapping and add edge.
            e = G_gt.add_edge(vertices[src], vertices[dst])
            edges_[src][dst][k] = e
            # Save weights in property map
            G_gt_weights[e] = data["travel_time"] / 60

        for n in G.nodes():  # tqdm(G.nodes(),desc='Distances', total=len(G.nodes)):
            src = vertices[n]
            D, P = gt.shortest_distance(
                G_gt, source=src, weights=G_gt_weights, pred_map=True
            )
            darr = D.get_array()
            parr = P.get_array()  # list(P)
            Dmat[n2i[n], [n2i[v2n[i]] for i in range(len(G.nodes))]] = darr.astype(
                np.float32
            )
            Pmat[n2i[n], [n2i[v2n[i]] for i in range(len(G.nodes))]] = parr.astype(
                np.float32
            )

        # Tmat,sigma, optimal_delay_factor, missing_travel_time = self._trip_matrix(nodes, Dmat)
        # Dmat*=optimal_delay_factor
        # Amat*=optimal_delay_factor
        Tmat = None
        if not sigma is None:
            Tmat = self.calc_trip_matrix(nodes, Dmat, sigma)

        # Compute Network Measures
        Onorm = O / np.sum(self.trip_factor * np.sum(nodes["population"]))

        network_measures = dict(
            average_distance=np.mean(Dmat) * (len(Dmat) / (len(Dmat) - 1)),
            average_weighted_distance=(
                np.sum(Dmat * Tmat) / np.sum(Tmat)
                if Tmat is not None
                else np.mean(Dmat) * (len(Dmat) / (len(Dmat) - 1))
            ),
            number_of_nodes=len(Dmat),
            number_of_edges=np.sum(Amat > 0),
            average_edge_length=np.mean(Amat),
            diameter=np.max(Dmat),
            length_of_network=np.sum(Amat),
            length_of_all_shortest_paths=np.sum(Dmat),
            pseudo_ellipse_volume_at_10=(
                self.pseudo_ellipse_volume(Dmat, Amat) if Tmat is not None else 0
            ),
            pseudo_sphere_volumes=(
                [
                    self.pseudo_sphere_volume(m, Dmat, Amat)
                    for m in list(np.linspace(1, 9, 9))
                    + list(np.linspace(10, 30, 6))
                    + list(np.linspace(30, 60, 3))
                ]
                if Tmat is not None
                else []
            ),
            r0_values=self.get_r0_grid(Dmat, Tmat) if Tmat is not None else {},
            taylor=(
                {
                    "sigma0": self._f_gamma(0, Dmat, Onorm),
                    "sigma-1": self._f_gamma(-1, Dmat, Onorm),
                    "sigma-2": self._f_gamma(-2, Dmat, Onorm),
                }
                if Tmat is None
                else {}
            ),
        )

        # Return the adjacency, distance, predecessor, and trip probability matrices
        return Amat, Dmat, Pmat, Tmat, network_measures

    def calc_trip_matrix(
        self, nodes: gpd.GeoDataFrame, Dmat, sigma: float
    ) -> np.ndarray:
        """
        Calculate the ridepooling OD matrix for a given set of nodes and distances
        using the gravity model.

        Parameters
        ----------
        nodes
            A GeoDataFrame containing the graph nodes.
        Dmat
            Shortest path distance matrix between all nodes.
        sigma
            Gravity model exponent.

        Returns
        -------
        Gravity model OD matrix

        """
        D_sigma = Dmat.copy().astype(np.float32)
        D_sigma[Dmat > 0] = (Dmat[Dmat > 0]) ** sigma
        D_sigma[Dmat <= 0] = 0
        O = np.tile(self.trip_factor * nodes["population"].values, (len(Dmat), 1)).T
        # O = np.ones_like(Dmat)
        # for i in range(len(O)):
        #     O[:,i]*=nodes['population'].values
        # _ = np.ones_like(Dmat)
        # for i in range(len(Dmat)):
        #     row = Dmat[i,:]
        #     _[i,:] = np.sum(D_sigma[i,:])
        Sz = np.tile(np.sum(D_sigma, axis=1), (len(Dmat), 1)).T
        grav = O * D_sigma / Sz
        return grav

    def get_raw_network_matrices(
        self, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, sigma: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Distances are in minutes. Trips are normalized.

        Parameters
        ----------
        nodes
            OSM nodes within the geographical area.
        edges
            OSM edges within the geographical area.
        sigma
            Gravity model exponent

        Returns
        -------
        Amat
            Adjacency matrix, with nonzero edge weights representing travel times on
            the edge minutes.
        Dmat
            Shortest path distance matrix between all nodes
        Pmat
            Shortest path predecessor matrix
        Tmat
            OD matrix from the gravity model
        """
        # Create a graph from the given nodes and edges
        G, n2i = self.get_graph_and_node_index(nodes=nodes, edges=edges)
        Dmat = np.ones((len(nodes), len(nodes)), dtype=np.float32) * -1
        Pmat = np.ones((len(nodes), len(nodes)), dtype=np.int32) * -1
        Amat = nx.adjacency_matrix(G, weight="travel_time", dtype=np.float32) / 60 * 1.5

        # O = np.tile(self.trip_factor*nodes['population'].values,(len(Dmat),1)).T

        G_gt = gt.Graph(directed=G.is_directed())

        # "Dict" [idx] = weight
        G_gt_weights = G_gt.new_edge_property("double")

        # mapping of nx vertices to gt indices
        vertices = {}
        v2n = {}
        for node in G.nodes:

            v = G_gt.add_vertex()
            vertices[node] = v
            v2n[int(v)] = node

        # mapping of nx edges to gt edge indices
        edges_ = defaultdict(lambda: defaultdict(dict))

        for src, dst, k, data in G.edges(data=True, keys=True):
            # Look up the vertex idxs from our vertices mapping and add edge.
            e = G_gt.add_edge(vertices[src], vertices[dst])
            edges_[src][dst][k] = e
            # Save weights in property map
            G_gt_weights[e] = data["travel_time"] / 60

        for n in G.nodes():  # tqdm(G.nodes(),desc='Distances', total=len(G.nodes)):
            src = vertices[n]
            D, P = gt.shortest_distance(
                G_gt, source=src, weights=G_gt_weights, pred_map=True
            )
            darr = D.get_array()
            parr = P.get_array()  # list(P)
            Dmat[n2i[n], [n2i[v2n[i]] for i in range(len(G.nodes))]] = darr.astype(
                np.float32
            )
            Pmat[n2i[n], [n2i[v2n[i]] for i in range(len(G.nodes))]] = parr.astype(
                np.float32
            )
        Tmat = self.calc_trip_matrix(nodes, Dmat, sigma)

        return Amat, Dmat, Pmat, Tmat

    @ft.lru_cache(maxsize=10)
    def get_raw_network_matrices_from_db(
        self, region: str, table: str, sigma: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the "raw network matrices" Amat, Dmat, Pmat, Tmat for the given region
        `region`, as represented in the database table `table`.

        Parameters
        ----------
        region
            "Geographical name" of the region to consider.
        table
            Name of the databse table to look up the region.
        sigma
            Gravity model exponent.

        Returns
        -------

        """
        nodes, edges = self.extract_graph(region, table_name=table)
        return self.get_raw_network_matrices(nodes, edges, sigma)

    def fetch_edges(
        self, geographical_name: str, column: str, table: str
    ) -> gpd.GeoDataFrame:
        """
        Fetches edges from the database that are within the geographical area specified by
        `geographical_name`.
        Parameters
        ----------
        geographical_name
        column
        table

        Returns
        -------

        """
        return gpd.read_postgis(
            f"SELECT a.* FROM graph_edges a JOIN {table} b "
            f"ON ST_Intersects(a.geometry, b.geometry) "
            f"WHERE b.{column}='{geographical_name}';",
            con=self.engine,
            geom_col="geometry",
        )

    def _read_data(
        self, geographical_name: str, column: str, table: str
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Reads boundaries, nodes, edges, and population data from the PostGIS database
        for the geographical area specified by `geographical_name`.

        Parameters
        ----------
        geographical_name
            The name of the geographical area to extract the data for.
        column
            Column to use to identify the geographical area. Suitable values are
            likely 'geografischer_name' or 'amtlicher_regierungsschluessel'.
        table
            The name of the table to extract the data from. Suitable values are
            likely 'kreise' or 'pendel_zonen'.

        Returns
        -------
        boundary
            A GeoDataFrame representing the boundary of the geographical area.
        nodes
            A GeoDataFrame representing the OSM nodes within the geographical area.
        edges
             A GeoDataFrame representing the OSM edges within the geographical area.
        population
            A GeoDataFrame representing the population within the geographical area.

        """
        log.info(
            f"Retrieving data for '{geographical_name}' by column '{column}' "
            f"from table '{table}'..."
        )

        if table == "pendel_zonen":
            column = "name"

        boundary = gpd.read_postgis(
            f"SELECT * FROM {table} WHERE {column}='{geographical_name}';",
            con=self.engine,
            geom_col="geometry",
        )

        nodes = gpd.read_postgis(
            f"SELECT a.* FROM graph_nodes a JOIN {table} b "
            f"ON ST_Intersects(a.geometry, b.geometry) "
            f"WHERE b.{column}='{geographical_name}';",
            con=self.engine,
            geom_col="geometry",
        )

        edges = self.fetch_edges(
            geographical_name=geographical_name, column=column, table=table
        )
        sql = f"""
        WITH b_one AS (
          SELECT geometry
          FROM {table}
          WHERE {column} = %s
        )
        SELECT a.*
        FROM kontur_de a
        JOIN (
          SELECT ST_Transform(
                   ST_Buffer(ST_Transform(geometry, 25832), 250), 4326
                 ) AS geom_buf_4326
          FROM b_one
        ) b
        ON ST_Intersects(a.geometry, b.geom_buf_4326);
        """
        population = gpd.read_postgis(
            # f"SELECT a.* FROM kontur_de a JOIN {table} b "
            # f"""ON ST_Intersects(
            #     a.geometry, 
            #     ST_Transform(ST_Buffer(ST_Transform(b.geometry,25832),250),4326),
            #     ) """
            # f"WHERE b.{column}='{geographical_name}';",
            sql,
            con=self.engine,
            geom_col="geometry",
        )

        return boundary, nodes, edges, population

    def _read_data_from_geometry(
            self, g: str,
        ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
            """
            Reads nodes, edges, and population data from the PostGIS database
            for the geometry specified.
    
            Parameters
            ----------
            g
                geometry in wkb hex format including the srid.
            column
                Column to use to identify the geographical area. Suitable values are
                likely 'geografischer_name' or 'amtlicher_regierungsschluessel'.
            table
                The name of the table to extract the data from. Suitable values are
                likely 'kreise' or 'pendel_zonen'.
    
            Returns
            -------
            boundary
                A GeoDataFrame representing the boundary of the geographical area.
            nodes
                A GeoDataFrame representing the OSM nodes within the geographical area.
            edges
                 A GeoDataFrame representing the OSM edges within the geographical area.
            population
                A GeoDataFrame representing the population within the geographical area.
    
            """
            log.info(
                f"Retrieving data from geometry "
                f"..."
            )
    

    
            wkb = g #to_wkb(g,hex=True, include_srid=True)
    
            nodes = gpd.read_postgis(
                text("""
                    SELECT a.* FROM graph_nodes a
                    WHERE ST_Intersects(
                    a.geometry,
                    ST_GeomFromWKB(decode( :wkb, 'hex'))
                    );
                    """),
                con=self.engine,
                params={"wkb":wkb},
                geom_col="geometry",
            )

            edge_sql = text("""
            SELECT a.* FROM graph_edges a
            WHERE ST_Intersects(
                    a.geometry,
                    ST_GeomFromWKB(decode( :wkb, 'hex'))
                    );
            """)
            edges = gpd.read_postgis(
                edge_sql,
                params={"wkb":wkb},
                con=self.engine,
                geom_col="geometry",
            )
    
            # edges = self.fetch_edges(
            #     geographical_name=geographical_name, column=column, table=table
            # )
            sql = text(f"""
            
            SELECT a.*
            FROM kontur_de a
            WHERE ST_Intersects(
                a.geometry,
                ST_Transform(ST_Buffer(ST_Transform(ST_GeomFromWKB(decode( :wkb, 'hex')),25832),250),4326)
            );
            """)
            population = gpd.read_postgis(
                sql,
                con=self.engine,
                params={"wkb":wkb},
                geom_col="geometry",
            )
    
            return g, nodes, edges, population
    def _preprocess_graph(
        self, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Preprocesses nodes and edges to create a simplified graph with edge speeds
        and travel times.

        This function takes in GeoDataFrames of nodes and edges, filters out any
        edges that do not connect existing nodes, and removes any columns from edges
        that have all NaN values. The function then creates a graph from the nodes
        and edges, simplifies it, finds the largest strongly connected component,
        and adds edge speeds and travel times. Finally, it re-projects the nodes and
        edges to a different coordinate system (EPSG:25832) and returns them.

        Parameters
        ----------
        nodes
            A GeoDataFrame representing the nodes. Each node is expected to have an
            'osmid' attribute.
        edges
            A GeoDataFrame representing the edges. Each edge is expected to
             have 'u', 'v', and 'key' attributes.

        Returns
        -------
        nodes
            The input GeoDataFrame of nodes, reprojected to 'EPSG:25832'.
        edges
            The input GeoDataFrame of edges, reprojected to 'EPSG:25832', with
            calculated speeds and travel times.
        """
        log.info(f"Preprocessing graph...")
        edges = edges[
            edges["u"].isin(nodes["osmid"]) & (edges["v"].isin(nodes["osmid"]))
        ]
        edges = edges.dropna(axis=1, how="all")
        edges = edges.loc[edges.groupby(["u", "v"]).length.idxmin()]
        edges = edges.set_index(["u", "v", "key"])
        nodes = nodes.set_index("osmid")
        G = ox.graph_from_gdfs(
            nodes,
            edges,
        )
        G = ox.simplify_graph(G)
        G = G.subgraph(max(nx.strongly_connected_components(G), key=len))
        G = ox.add_edge_speeds(G, fallback=50)
        G = ox.add_edge_travel_times(G)
        nodes, edges = ox.graph_to_gdfs(G)
        edges.crs = "WGS 84"
        nodes.crs = "WGS 84"
        nodes = nodes.to_crs("EPSG:25832")
        edges = edges.to_crs("EPSG:25832")
        return nodes, edges

    def locally_best_trip_matrix(self, nodes, D, gravity=True, sigma=0.0, offset=1.0):
        if gravity:
            D_ = (D.copy()).astype(np.float32) + offset
            O = np.ones_like(D)
            for i in range(len(O)):
                O[:, i] *= nodes["population"].values
            # O = nodes['population'].values # The raw population.

            # Population at each node needs to be weighted with the number of trips
            # people at these locations do. One problem is, that this is obviously
            # not the number of trips starting at this location. A solution might
            # involve the updating of the number of people at a location and their
            # remaining trips:
            #
            # M is the transition matrix:
            # O' = M * O
            # O = sum O_k , where O_k is the vector of people traveling k times per day
            # The population distribution
            # Simple gravity model in which only the number of trip origins is
            # constrained
            #
            # Mobilität in Deutschland suggests that the average travel time for car
            # travel is about 25 Minutes for every region. Thus, we aim to find an
            # exponent which matches this value by optimizing. This has drawbacks,
            # however. It is nonsensical to have an exponent larger than 0
            # corresponding to an all to all travel pattern. We cut of the exponent
            # at 0. One may change the travel delay factor, i.e. the factor that
            # translates optimal network distance according to speed limits,
            # to the actual distance accounting for traffic. Because traffic excerts
            # its effect mostly in high density areas, where travel is likely global,
            # we first optimize the exponent, and afterwards optimize the travel factor.

            c_star = (
                self.average_trip_duration
            )  # Trips in Germany take around 22(2) minutes (across RegioStar7 Regions there is little variation (2 minutes))
            sigma_array = np.array(
                [-2, -1.5, -1.0, -0.5, 0.0]
            )  # Plausible values for the exponent
            c_array = []
            C_sigma = []

            for i, sigma in enumerate(sigma_array):
                grav = self.trip_factor * O * D_ ** (sigma) / np.sum(D_**sigma, axis=1)
                normalization = np.sum(grav)

                c = np.sum(grav * D) / normalization
                C_sigma.append((c - c_star) ** 2)
                c_array.append(c)

            sigma = sigma_array[np.argmin(C_sigma)]
            c = c_array[np.argmin(C_sigma)]
            delay_factor_bounds = [1.0, 2.0]
            optimal_delay_factor = c_star / c / 2

            if optimal_delay_factor < delay_factor_bounds[0]:
                optimal_delay_factor = 1.0
            elif optimal_delay_factor > delay_factor_bounds[1]:
                optimal_delay_factor = 2.0

            grav = self.trip_factor * O * D_ ** (sigma) / np.sum(D_**sigma, axis=1)
            missing_travel_time = c_star - optimal_delay_factor * c
            # grav = grav/np.sum(grav) # do not normalize for the number of trips to match
            return grav, sigma, optimal_delay_factor, missing_travel_time

    def pseudo_ellipse_volume(self, Dmat, Amat):
        n_samples = 200
        allowed_delay = 10
        volume_per_distance = np.zeros(n_samples)
        i = 0
        Adense = Amat.todense()
        for u, v in np.random.choice(
            np.arange(len(Dmat)), size=(n_samples, 2), replace=True
        ):
            if u != v:
                okay_nodes = (Dmat[u, :] + Dmat[:, v]) <= (Dmat[u, v] + allowed_delay)
                volume_per_distance[i] = (
                    np.sum(Adense[okay_nodes]) / 2 / allowed_delay / Dmat[u, v]
                )
            i += 1
        return np.mean(volume_per_distance), np.std(volume_per_distance) / np.sqrt(
            n_samples
        )

    def pseudo_sphere_volume(self, m, Dmat, Amat) -> tuple[float, float]:
        """
        Calculate the volume of the pseudo-sphere around a given node.

        Parameters
        ----------
        m
        Dmat
        Amat

        Returns
        -------

        """
        volume_per_distance = np.zeros(len(Dmat))
        Adense = Amat.todense()
        volume_per_distance = np.sum((Adense * (Dmat <= m).astype(float)), axis=1) / 2
        return np.mean(volume_per_distance), np.std(volume_per_distance) / np.sqrt(
            len(Dmat)
        )

    @staticmethod
    def _get_volume(D: np.ndarray, c: float) -> np.ndarray:
        """Calculate the number of nodes within radius c"""
        N = len(D)
        V = np.einsum("ji->i", (D <= c).astype(int)) / N
        return V

    @staticmethod
    def _get_r0(T: np.ndarray, V: np.ndarray, B: int) -> np.ndarray:
        """Calculate the average default rejection rate"""
        N = len(T)
        r0 = np.einsum("i,i", T, (1 - V) ** B)
        return r0

    def get_r0_grid(self, Dmat, Tmat):
        """Calculate a grid of default rejection values for different constraints and fleet sizes"""
        r0_mods = {"r0_values": {}}
        # origins = np.sum(Tmat, axis=1)
        trips = np.sum(Tmat)
        md = np.mean(Dmat)
        Ts = np.einsum("ij->i", Tmat) / trips
        for Bi in 2 ** np.arange(0, 14, 1):
            for c in np.concatenate(
                [
                    np.linspace(0.0 * md, 0.5 * md, 9)[:-1],
                    np.linspace(0.5 * md, 1.0 * md, 5)[:-1],
                    np.linspace(1 * md, 4.0 * md, 5),
                ]
            ):
                V = self._get_volume(Dmat, c)
                r0 = self._get_r0(Ts, V, Bi)
                if c in r0_mods["r0_values"].keys():
                    r0_mods["r0_values"][c].update({int(Bi): r0})
                else:
                    r0_mods["r0_values"][c] = {int(Bi): r0}
        return r0_mods["r0_values"]

    def get_kreise_list(self):
        df = None
        with self.engine.connect() as conn:
            df = gpd.read_postgis(
                "SELECT * FROM kreise;", con=conn, geom_col="geometry"
            )
        return df

    def get_gemeinde_list(self) -> gpd.GeoDataFrame:
        """

        Returns
        -------
        gemeinden
            Gemeinden
        """
        df = None
        with self.engine.connect() as conn:
            df = gpd.read_postgis(
                "SELECT * FROM gemeinden;", con=conn, geom_col="geometry"
            )
        return df


class PTDemandGenerator(DemandGenerator):
    """
    Demand generator that removes demand between nodes that are likely serviced by a
    direct PT service.
    """

    def __init__(self, gtfs_path: str | Path, db_engine: str | URL | Engine):
        """

        Parameters
        ----------
        gtfs_path
            Path to the GTFS feed as a ZIP file.
        db_engine
            SQLAlchemy URL of the database to connect to.
        """
        self._feed = DBGTFSFeed(gtfs_path=gtfs_path, db_engine=db_engine)

        super().__init__(db_engine)

    @property
    def feed(self) -> DBGTFSFeed:
        """
        GTFS feed object.
        """
        return self._feed

    @ft.lru_cache(maxsize=10)
    def get_raw_network_matrices_without_pt_demand(
        self,
        region: str,
        table: str,
        date: datetime | str = "now",
        pt_stop_radius: float = 250,
        sigma: float = -0.54,
        n_trips_per_day: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the "raw network matrices" Amat, Dmat, Pmat, Tmat for the given region
        `region`, as represented in the database table `table`, but with removing the
        demand between nodes that are likely serviced by a direct PT service.

        For each node pair in the network, the demand predicted by the gravity model
        will be nulled if both nodes are within a radius of `pt_stop_radius` around a
        PT stop, and a direct PT service runs on this relation at least once on the
        GTFS service day specified by the `date` argument.


        Parameters
        ----------
        region
            "Geographical name" of the region to consider.
        table
            Name of the database table to look up the region.
        date
            Date for which to compute the PT stop OD matrix. Defaults to "now", i.e.,
            the current day.
        pt_stop_radius
            Radius around PT stops to consider "close" to a PT stop in metres.
        sigma
            Gravity model exponent.
        n_trips_per_day
            Minimum number of trips per day for a node pair to be considered as serviced
            by PT and therefore removed from the ridepooling demand.

        Returns
        -------
        Amat
            Adjacency matrix, with nonzero edge weights representing travel times on
            the edge minutes.
        Dmat
            Shortest path distance matrix between all nodes
        Pmat
            Shortest path predecessor matrix
        Tmat
            OD matrix from the gravity model, with all cells whose corresponding nodes
            are likely serviced by PT set to zero.
        network_measures
            Dictionary of network measures, including average distance, number of nodes,
            number of edges, average edge length, diameter, length of network, length of
            all shortest paths, pseudo-ellipse volume at 10 minutes, pseudo-sphere
            volumes, r0 values, and Taylor series coefficients.
        """
        log.info(f"Fetching all PT stops for {region=} from {table=}...")
        stops = get_eligible_stops(
            region, engine=self.engine, db_table=table
        )  # in UTM coordinates

        log.info(f"Computing the PT stop OD matrix...")
        stop_idx, pt_od = compute_od_matrix_from_gtfs(
            gtfs_feed=self._feed,
            date=date,
            return_type="ndarray",
            stop_subset=stops.stop_id.unique(),
        )
        # breakpoint()
        log.info(
            f'Extracting the street graph from the "f"database for {region=} from '
            f"{table=}..."
        )
        nodes, edges = self.extract_graph(
            geographical_name=region, table_name=table
        )  # coordinates are in UTM

        log.info(f"Constructing the street graph object...")
        G, node_idx = self.get_graph_and_node_index(nodes=nodes, edges=edges)

        log.info(
            f"Computing which network nodes are within a radius of "
            f"{pt_stop_radius:_} m around PT stops..."
        )

        # (PT) stops array
        astops = np.c_[stops.stop_id, stops.geometry.x, stops.geometry.y]

        # (OSM + ancillary) nodes array
        anodes = np.c_[nodes.index, nodes.geometry.x, nodes.geometry.y]

        nodes_for_stops = get_nodes_for_stops(
            astops, anodes, radius=float(pt_stop_radius)
        )

        log.info(
            f"Expanding the PT stop OD matrix to an OD matrix of corresponding "
            f"close network nodes..."
        )
        od = expand_pt_od_matrix(
            nodes_for_stops=nodes_for_stops,
            node_idx=node_idx,
            pt_od=pt_od,
            stop_idx=stop_idx,
        )

        log.info(f"Computing the demand matrices according to the gravity model...")
        Amat, Dmat, Pmat, Tmat = self.get_raw_network_matrices(
            nodes=nodes, edges=edges, sigma=sigma
        )
        assert Tmat.shape == od.shape

        log.info(
            f"Setting the demand of nodes that are close to PT stops and are "
            f"serviced by PT to zero"
        )
        Tmat *= (od < n_trips_per_day).astype("i8")

        # Compute Network Measures
        trip_origins = np.sum(Tmat, axis=1)
        # trip_origins = np.tile(self.trip_factor*nodes['population'].values,(len(Dmat),1)).T
        trip_origins_normalized = trip_origins / np.sum(trip_origins)

        network_measures = dict(
            average_distance=np.mean(Dmat) * (len(Dmat) / (len(Dmat) - 1)),
            average_weighted_distance=(
                np.sum(Dmat * Tmat) / np.sum(Tmat)
                if Tmat is not None
                else np.mean(Dmat) * (len(Dmat) / (len(Dmat) - 1))
            ),
            number_of_nodes=len(Dmat),
            number_of_edges=np.sum(Amat > 0),
            average_edge_length=np.mean(Amat),
            diameter=np.max(Dmat),
            length_of_network=np.sum(Amat),
            length_of_all_shortest_paths=np.sum(Dmat),
            pseudo_ellipse_volume_at_10=self.pseudo_ellipse_volume(Dmat, Amat),
            pseudo_sphere_volumes=[
                self.pseudo_sphere_volume(m, Dmat, Amat)
                for m in list(np.linspace(1, 9, 9))
                + list(np.linspace(10, 30, 6))
                + list(np.linspace(30, 60, 3))
            ],
            r0_values=self.get_r0_grid(Dmat, Tmat),
            taylor={
                "sigma0": self._f_gamma(0, Dmat, trip_origins_normalized),
                "sigma-1": self._f_gamma(-1, Dmat, trip_origins_normalized),
                "sigma-2": self._f_gamma(-2, Dmat, trip_origins_normalized),
            },
        )

        return Amat, Dmat, Pmat, Tmat, network_measures
