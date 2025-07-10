import networkx as nx
from networkx.utils import py_random_state


def _zipf_rv_below(gamma, xmin, threshold, seed):
    """Returns a random value chosen from the bounded Zipf distribution.

    Repeatedly draws values from the Zipf distribution until the
    threshold is met, then returns that value.
    """
    result = nx.utils.zipf_rv(gamma, xmin, seed)
    while result > threshold:
        result = nx.utils.zipf_rv(gamma, xmin, seed)
    return result


def _powerlaw_sequence(
    gamma, low, high, length_condition, valid_condition, max_iters, seed
):
    """Returns a list of numbers obeying a constrained power law distribution.

    ``gamma`` and ``low`` are the parameters for the Zipf distribution.

    ``high`` is the maximum allowed value for values draw from the Zipf
    distribution. For more information, see :func:`_zipf_rv_below`.

    ``condition`` and ``length`` are Boolean-valued functions on
    lists. While generating the list, random values are drawn and
    appended to the list until ``length`` is satisfied by the created
    list. Once ``condition`` is satisfied, the sequence generated in
    this way is returned.

    ``max_iters`` indicates the number of times to generate a list
    satisfying ``length``. If the number of iterations exceeds this
    value, :exc:`~networkx.exception.ExceededMaxIterations` is raised.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    """
    for _ in range(max_iters):
        seq = []
        # Keep drawing until we meet length_condition(seq)
        while not length_condition(seq):
            val = _zipf_rv_below(gamma, low, high, seed)
            seq.append(val)

        # Once we have enough elements, check if the sequence passes valid_condition
        if valid_condition(seq):
            return seq
    raise nx.ExceededMaxIterations(
        "Could not create power law sequence under constraints"
    )


def _hurwitz_zeta(x, q, tolerance):
    """The Hurwitz zeta function, or the Riemann zeta function of two arguments.

    ``x`` must be greater than one and ``q`` must be positive.

    This function repeatedly computes subsequent partial sums until
    convergence, as decided by ``tolerance``.
    """
    z = 0
    z_prev = -float("inf")
    k = 0
    while abs(z - z_prev) > tolerance:
        z_prev = z
        z += 1 / ((k + q) ** x)
        k += 1
    return z


def _generate_min_degree(gamma, average_degree, max_degree, tolerance, max_iters):
    """Returns a minimum degree from the given average degree."""
    # Defines zeta function whether or not Scipy is available
    try:
        from scipy.special import zeta
    except ImportError:

        def zeta(x, q):
            return _hurwitz_zeta(x, q, tolerance)

    min_deg_top = max_degree
    min_deg_bot = 1
    min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
    itrs = 0
    mid_avg_deg = 0
    while abs(mid_avg_deg - average_degree) > tolerance:
        if itrs > max_iters:
            raise nx.ExceededMaxIterations("Could not match average_degree")
        mid_avg_deg = 0
        for x in range(int(min_deg_mid), max_degree + 1):
            mid_avg_deg += (x ** (-gamma + 1)) / zeta(gamma, min_deg_mid)
        if mid_avg_deg > average_degree:
            min_deg_top = min_deg_mid
            min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
        else:
            min_deg_bot = min_deg_mid
            min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
        itrs += 1
    # return int(min_deg_mid + 0.5)
    return round(min_deg_mid)


@py_random_state(13)
def signed_LFR_benchmark_graph(
    n,
    tau1,
    tau2,
    mu,
    P_minus=0.2,
    P_plus=0.9,
    average_degree=None,
    min_degree=None,
    max_degree=None,
    min_community=None,
    max_community=None,
    tol=1.0e-7,
    max_iters=500,
    seed=None,
):
    """
    Generates a signed LFR benchmark graph with partial positivity/negativity.
    *P_plus* = fraction of edges within a community that are assigned weight +1
              (the remainder can be negative or skipped, depending on your design)
    *P_minus* = fraction of edges across different communities that are assigned weight -1
              (the remainder can be positive or skipped)
    """

    # == 1) Validate input / generate degree sequence (the usual LFR steps) ==
    if not tau1 > 1:
        raise nx.NetworkXError("tau1 must be > 1")
    if not tau2 >= 1:
        raise nx.NetworkXError("tau2 must be >= 1")
    if not (0 <= mu <= 1):
        raise nx.NetworkXError("mu must be in [0, 1]")

    if max_degree is None:
        max_degree = n
    elif not (0 < max_degree <= n):
        raise nx.NetworkXError("max_degree must be in (0, n]")

    # Exactly one of min_degree or average_degree must be set
    if not ((min_degree is None) ^ (average_degree is None)):
        raise nx.NetworkXError(
            "Must assign exactly one of min_degree and average_degree"
        )

    # Possibly compute min_degree from average_degree
    if min_degree is None:
        min_degree = _generate_min_degree(
            tau1, average_degree, max_degree, tol, max_iters
        )

    # Generate the degree sequence (deg_seq)
    deg_seq = _generate_powerlaw_degs(n, tau1, min_degree, max_degree, max_iters, seed)

    # Generate community sizes
    if min_community is None:
        min_community = min(deg_seq)
    if max_community is None:
        max_community = max(deg_seq)

    comms = _generate_powerlaw_comms(
        n, tau2, min_community, max_community, max_iters, seed
    )

    # Assign nodes to communities
    max_iters *= 10 * n
    communities = _generate_communities(deg_seq, comms, mu, max_iters, seed)
    # Filter out any empty sets
    communities = [c for c in communities if len(c) > 0]

    # == 2) Build the graph ==
    G = nx.Graph()
    G.add_nodes_from(range(n))
    all_nodes = set(range(n))

    # For each community c, for each node u in c:
    for c_index, c in enumerate(communities):
        for u in c:
            # Store ground-truth label
            G.nodes[u]["community"] = c_index

            # (A) Intra-community edges:
            #     We want node u to have roughly deg_seq[u]*(1 - mu) edges inside c.
            while G.degree(u) < round(deg_seq[u] * (1 - mu)):
                v = seed.choice(list(c))  # pick a node within the same community
                if v == u:
                    continue
                if not G.has_edge(u, v):
                    # Use P_plus to decide if edge is positive or negative
                    if seed.random() < P_plus:
                        G.add_edge(u, v, weight=1)
                    else:
                        # continue
                        G.add_edge(u, v, weight=-1)

            # (B) Inter-community edges:
            #     We want the rest of node u's edges to reach deg_seq[u].
            while G.degree(u) < deg_seq[u]:
                candidates = list(all_nodes - c)  # outside community c
                v = seed.choice(candidates)
                if v == u:
                    continue
                if not G.has_edge(u, v):
                    # Use P_minus to decide if edge is negative or positive
                    if seed.random() < P_minus:
                        G.add_edge(u, v, weight=-1)
                    else:
                        # Maybe make it positive, or skip, or something else
                        G.add_edge(u, v, weight=1)

    return G


@py_random_state(5)
def _generate_powerlaw_degs(n, tau1, low, high, max_iters, seed=None):
    """
    Generate a valid powerlaw degree sequence of length n (exponent = tau1).
    Ensures the sum of degrees is even (so a simple undirected graph is possible).
    """

    def length_condition(seq):
        return len(seq) >= n

    def valid_condition(seq):
        # Ensure the total sum is even
        return sum(seq) % 2 == 0

    seq = _powerlaw_sequence(
        gamma=tau1,
        low=low,
        high=high,
        length_condition=length_condition,
        valid_condition=valid_condition,
        max_iters=max_iters,
        seed=seed,
    )
    # If we grabbed more than needed, slice to exactly n
    seq = seq[:n]

    return seq


@py_random_state(5)
def _generate_powerlaw_comms(n, tau2, low, high, max_iters, seed=None):
    """
    Generate a powerlaw distribution of community sizes that sums exactly to n.
    In other words, keep drawing community sizes from a powerlaw until the sum = n.
    """

    if tau2 == 1:
        # Special case: truncated power-law with exponent 1
        # Compute weights w_k = 1/k for k in [low, high]
        weights = [1 / k for k in range(low, high + 1)]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]

        comm_sizes = []
        while sum(comm_sizes) < n:
            # Draw a community size from the finite distribution
            chosen = seed.choices(range(low, high + 1), weights=probs, k=1)[0]
            comm_sizes.append(chosen)

        # If sum is greater than n, adjust last community size
        total = sum(comm_sizes)
        if total > n:
            excess = total - n
            comm_sizes[-1] -= excess
            if comm_sizes[-1] <= 0:
                # If last community size becomes zero or negative, remove it
                comm_sizes.pop()

        # At this point, sum(comm_sizes) == n
        return comm_sizes

    def length_condition(comm_sizes):
        # We keep drawing until sum(comm_sizes) >= n
        return sum(comm_sizes) >= n

    def valid_condition(comm_sizes):
        # We want sum(comm_sizes) == n
        return sum(comm_sizes) == n

    comm_sizes = _powerlaw_sequence(
        gamma=tau2,
        low=low,
        high=high,
        length_condition=length_condition,
        valid_condition=valid_condition,
        max_iters=max_iters,
        seed=seed,
    )

    return comm_sizes


def _generate_communities(deg_seq, comm_sizes, mu, max_iters, seed):
    """
    Assign each node to exactly one community, using the approach from NetworkX’s LFR:
      - deg_seq: the desired degrees of each node
      - comm_sizes: a list of community sizes summing to len(deg_seq)
      - mu: fraction of edges that are inter-community

    Returns a list of sets, each set is a community of node IDs.
    """

    n = len(deg_seq)
    # We'll create `result`, each entry is a set of node IDs in that community
    result = [set() for _ in comm_sizes]

    free = list(range(n))  # the set of nodes not yet assigned
    seed.shuffle(free)

    # Each node v wants 'intra_degree' ~ deg_seq[v]*(1-mu) inside its community
    # We attempt to place v in one of the available communities if that community can accommodate it

    for i in range(max_iters):
        if not free:  # we assigned everyone
            return result

        v = free.pop()
        # pick a random community index
        c_idx = seed.choice(range(len(comm_sizes)))
        desired_intra_degree = round(deg_seq[v] * (1 - mu))

        # If the chosen community can hold v:
        if desired_intra_degree < comm_sizes[c_idx]:
            result[c_idx].add(v)
        else:
            # put v back in free, try again
            free.append(v)

        # If the chosen community got too big, pop one out
        if len(result[c_idx]) > comm_sizes[c_idx]:
            removed = result[c_idx].pop()
            free.append(removed)

    # If we exit the loop, we couldn't assign communities properly
    raise nx.ExceededMaxIterations(
        "Could not assign communities; try adjusting min_community or parameters."
    )
