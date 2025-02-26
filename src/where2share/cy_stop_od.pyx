import numpy as np
cimport numpy as cnp

cnp.import_array()
DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t

def compute_od_dict_from_stops(cnp.ndarray stops):
    """
    Compute an "OD dictionary" from a list of PT stops.

    Returns a dictionary with pickup stop IDs as keys and dictionaries as values,
    where the keys are dropoff stop IDs and the values are the number of trips
    contained in the supplied stops array.

    Parameters
    ----------
    stops
        Numpy ndarray containing N stops with three columns, i.e, shape (N, 3).

        The first column needs to contain the stop ID, the second column the GTFS
        stop sequence value, i.e., the position of the stop within the trip, and the
        third column the stop_id of the stop.

    Returns
    -------
    dict
        OD dictionary.

    """
    # stoplist list
    sll = np.split(
        stops[:, 2:].squeeze(),
        np.unique(stops[:, 0], return_index=True)[1][1:]
    )

    res = {}
    for sl in sll:
        for i, pickup in enumerate(sl):
            if pickup not in res:
                res[pickup] = {}
            for dropoff in sl[i + 1:]:
                if dropoff in res[pickup]:
                    res[pickup][dropoff] += 1
                else:
                    res[pickup][dropoff] = 1
    return res

def compute_od_matrix_from_stops(cnp.ndarray stops):
    """
    Compute an OD matrix from a list of PT stops.
    Returns a PT OD matrix as a numpy array and a list of stop IDs.

    The OD matrix is a square matrix with the number of rows and columns equal to the
    number of unique stops in the supplied stops array. The value of each cell in the
    matrix is the number of trips from the stop corresponding to the row to the stop
    corresponding to the column.

    The list of stop IDs is a list of the unique stop IDs in the supplied stops array,
    ordered in the same order as the rows and columns of the OD matrix so that the
    stop IDs of the matrix entries can be reconstructed.


    Parameters
    ----------
    stops
        Numpy ndarray containing N stops with three columns, i.e, shape (N, 3).

        The first column needs to contain the ``trip_id``, the second column the GTFS
        ``stop_sequence`` value, i.e., the position of the stop within the trip, and the
        third column the ``stop_id`` of the stop.

    Returns
    -------
    list
        List of stop IDs.
    numpy.ndarray
        PT OD matrix.

    """
    # stoplist list
    sll = np.split(
        stops[:, 2:].squeeze(),
        np.unique(stops[:, 0], return_index=True)[1][1:]
    )


    uniq_stops = np.unique(stops[:, 2])
    n_uniq_stops = len(uniq_stops)

    stop_idx = {}
    for i in range(n_uniq_stops):
        stop_idx[uniq_stops[i]] = i

    od = np.zeros((n_uniq_stops, n_uniq_stops), dtype=DTYPE)

    for sl in sll:
        sl_len = len(sl)
        for i_pu in range(sl_len):
            for i_do in range(i_pu + 1, sl_len):
                od[stop_idx[sl[i_pu]], stop_idx[sl[i_do]]] += 1

    return list(stop_idx), od

def get_nodes_for_stops(cnp.ndarray stops, cnp.ndarray nodes, float radius=250):
    """
    Get the nodes that are within a certain radius of each stop.

    Returns a dictionary with the stop id as key and a list of node ids for the nodes
    that are "near" the stop as value.

    Parameters
    ----------
    stops
        Numpy ndarray containing N stops with three columns, i.e, shape (N, 3).

        The first column needs to contain the stop ID, the second column the x
        coordinate (easting, longitude), and the third column the y coordinate (
        northing, latitude) of the stop.
    nodes
        Numpy ndarray containing N nodes with three columns, i.e, shape (N, 3).

        The first column needs to contain the node ID, the second column the x
        coordinate (easting, longitude), and the third column the y coordinate (
        northing, latitude) of the node.
    radius
        The integer radius within which nodes are considered to be "near" a stop (
        inclusive). Mind the units! If the coordinates are in meters, the radius will
        be interpreted as meters as well (i.e., no projection will be performed, just
        Euclidean distance computation).

    Returns
    -------
    dict
        Dictionary with PT stop id as key and list of network node ids as value.

    """
    n_stops = len(stops)
    n_nodes = len(nodes)

    nodes_has_pt = np.full(n_nodes, False)
    sqradius = radius ** 2

    nodes_for_stops = {}
    for i in range(n_stops):
        stop_nodes = []
        for j in range(n_nodes):
            if np.sum((stops[i, 1:3] - nodes[j, 1:3]) ** 2) <= sqradius:
                stop_nodes.append(nodes[j, 0])
        nodes_for_stops[stops[i, 0]] = stop_nodes
    return nodes_for_stops

def expand_pt_od_matrix(
        dict nodes_for_stops,
        dict node_idx,
        cnp.ndarray pt_od,
        list stop_idx):
    """
    Expand the PT OD matrix to a node OD matrix.

    The entries of the matrix denote the number of trips from the ith node to the jth
    node. The matrix is a square matrix with the number of rows and columns equal to
    the number of unique nodes in the supplied nodes array.

    Parameters
    ----------
    nodes_for_stops
        Dictionary with PT stop id as key and list of network node ids as value.
        May be generated by `get_nodes_for_stops`.
    node_idx
        Dictionary with network node ids as keys and desired position in the expanded
        OD matrix as value. May be generated by `get_graph_and_node_index`.
    pt_od
        PT OD matrix. May be generated by `compute_od_matrix_from_stops`.
    stop_idx
        Ordered list of stop IDs to match the stops in the PT OD matrix. May be
        generated by `compute_od_matrix_from_stops`.

    Returns
    -------
    numpy.ndarray
        Node OD matrix.

    """
    n_stops = len(pt_od)
    assert n_stops == pt_od.shape[1]
    assert n_stops == len(stop_idx)

    n_nodes = len(node_idx)

    od = np.zeros((n_nodes, n_nodes), dtype='i8')

    for i_pu_stop in range(n_stops):
        id_pu_stop = stop_idx[i_pu_stop]
        node_ids_pu = nodes_for_stops[id_pu_stop]
        # if id_pu_stop == "de:03158:503:1:11" and id_do_stop == "de:03158:3061:1:2":
        #     print(f"{i_pu_stop}th pick-up stop with id {id_pu_stop} has node_ids_pu {node_ids_pu}")

        for i_do_stop in range(n_stops):
            id_do_stop = stop_idx[i_do_stop]
            node_ids_do = nodes_for_stops[id_do_stop]

            n_trips = pt_od[i_pu_stop, i_do_stop]

            # if (
            #         (id_pu_stop == "de:03158:503:1:11" and id_do_stop == "de:03158:3061:1:2") or
            #         (id_do_stop == "de:03158:503:1:11" and id_pu_stop == "de:03158:3061:1:2")
            # ):
            #     print(f"{i_do_stop}th drop-off stop with id {id_do_stop} has node_ids_do {node_ids_do}")
            #     print(f"We have {n_trips} trips between stops {id_pu_stop} and {id_do_stop} and are expanding")


            # print(f"{i_do_stop}th drop-off stop with id {id_do_stop}")

            # print(f"Expanding...")
            for node_id_pu in node_ids_pu:
                for node_id_do in node_ids_do:
                    node_idx_pu = node_idx[node_id_pu]
                    node_idx_do = node_idx[node_id_do]
                    # if id_pu_stop == "de:03158:503:1:11" and id_do_stop == "de:03158:3061:1:2":
                    #     print(f"PU node idx {node_idx_pu}, DO node idx {node_idx_do} with {n_trips} trips")
                    #     if node_idx_pu == 10 and node_idx_do== 164:
                    #         print("<----------------------\n")
                    #     else:
                    #         print("\n")
                    # if (node_idx_pu == 10 and node_idx_do== 164) or (node_idx_pu == 164 and node_idx_do== 10):
                    #     print(f"PU node idx {node_idx_pu}, DO node idx {node_idx_do} with {n_trips} trips, "
                    #           f"stop ids {id_pu_stop} and {id_do_stop}")

                    od[node_idx_pu, node_idx_do] += n_trips

    return od
