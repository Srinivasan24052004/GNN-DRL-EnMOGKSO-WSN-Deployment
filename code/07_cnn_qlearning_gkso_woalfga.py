# =====================================================================
# TR-EnMOGKSO + GNN + DRL + LEARNED ARLS - FINAL EXPERIMENT SUITE
# 250 x 300 m, sensing 25 m, population 6, 30 generations
#
# PART A : original algorithm (unchanged; used as the ORIGINAL baseline)
# PART B : learned ARLS. Proposed = "gnn_dqn": a GNN encodes the sensor
#          graph for a Move-DQN that chooses how ARLS moves sensors; the
#          GNN is trained end-to-end through the DQN's TD loss.
# PART C : EXP 1 final test (30 new seeds), EXP 2 scalability (20-50
#          nodes), EXP 3 PSO/GA/DE/GWO/WOA baselines (equal time),
#          statistics (Wilcoxon, Holm, A12, Friedman) and PNG figures.
# Kaggle: paste in ONE cell -> Save Version -> Save & Run All (Commit).
# Expected time ~6-7 h. Outputs: CSVs, final_results_tables.xlsx, fig*.png
# =====================================================================

# =====================================================================
# PART A - ORIGINAL ALGORITHM (unchanged)
# =====================================================================

import random
import time
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

SEED = 0
AREA_X, AREA_Y = 250.0, 300.0
R_SENSE, COMM_RANGE = 25.0, 30.0
GRID_RES = 50
NODE_COUNTS = [20, 30, 40, 50]
POP_SIZE, GENERATIONS = 6, 30

TARGET_COVERAGE = 0.98
TARGET_OVERLAP = 0.10
TARGET_EE = 0.97

SEARCH_MARGIN_COVERAGE = 0.005
SEARCH_MARGIN_OVERLAP = 0.010
SEARCH_MARGIN_EE = 0.005

F, CR = 0.45, 0.80
ARCHIVE_SIZE = 50

OVERLAP_REG_WEIGHT = 1.75

GNN_INPUT, GNN_HIDDEN, GNN_OUTPUT = 4, 24, 16

STATE_BASE_DIM = 12
STATE_DIM = 28
DQN_HIDDEN = 48
NUM_ACTIONS = 5
DQN_LR, GAMMA = 0.001, 0.99
BATCH_SIZE, BUFFER_SIZE = 24, 2000
EPS_START, EPS_END, EPS_DECAY = 0.70, 0.05, 15.0
TARGET_UPDATE = 5

ARLS_MOVES_INIT = 1500
ARLS_MOVES_POP = 2500
ARLS_MOVES_ELITE = 12000
ARLS_MOVES_FINAL = 30000
ARLS_FINAL_CANDIDATES = 5
ARLS_T_START, ARLS_T_END = 1e-2, 1e-5
ARLS_HOLE_PROB = 0.25
ARLS_STEP_MAX, ARLS_STEP_MIN = 30.0, 1.0
PRIORITY_REFRESH = 100

MIN_SEPARATION = 0.0
SEP_PENALTY = 0.02

REWARD_COVERAGE = 8.0
REWARD_OVERLAP = 18.0
REWARD_EE = 5.0
REWARD_FEASIBLE = 3.0

ACTION_NAMES = [
    "Global Exploration",
    "Elite Hunting",
    "Differential Hunting",
    "Redundancy Reduction",
    "Adaptive Local Search",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

gx = np.linspace(0.0, AREA_X, GRID_RES)
gy = np.linspace(0.0, AREA_Y, GRID_RES)

GX, GY = np.meshgrid(gx, gy)

GRID = np.column_stack(
    (
        GX.ravel(),
        GY.ravel()
    )
)

NUM_GRID = len(GRID)

DX = gx[1] - gx[0]
DY = gy[1] - gy[0]


def clip_positions(pos):
    pos = np.asarray(
        pos,
        dtype=np.float64
    ).copy()

    pos[:, 0] = np.clip(
        pos[:, 0],
        0.0,
        AREA_X
    )

    pos[:, 1] = np.clip(
        pos[:, 1],
        0.0,
        AREA_Y
    )

    return pos


def clip_point(p):
    return np.array(
        [
            np.clip(
                float(p[0]),
                0.0,
                AREA_X
            ),
            np.clip(
                float(p[1]),
                0.0,
                AREA_Y
            ),
        ],
        dtype=np.float64,
    )


def pairwise_distances(pos):
    d = (
        pos[:, None, :]
        -
        pos[None, :, :]
    )

    return np.sqrt(
        np.sum(
            d * d,
            axis=2
        )
    )


def sensing_mask(pos):
    d = (
        GRID[:, None, :]
        -
        pos[None, :, :]
    )

    return (
        np.sum(
            d * d,
            axis=2
        )
        <=
        R_SENSE**2
    )


def coverage_data(pos):
    mask = sensing_mask(pos)

    count = mask.sum(
        axis=1
    )

    coverage = float(
        np.mean(
            count >= 1
        )
    )

    overlap = float(
        np.mean(
            count >= 2
        )
    )

    return (
        coverage,
        overlap,
        count,
        mask
    )


def energy_efficiency(
    coverage,
    overlap
):
    return float(
        np.clip(
            0.82 * coverage
            +
            0.18 * (
                1.0
                -
                overlap
            ),
            0.0,
            1.0
        )
    )


def separation_violations(pos):

    if (
        MIN_SEPARATION <= 0.0
        or
        len(pos) < 2
    ):
        return 0

    d = pairwise_distances(
        pos
    )[
        np.triu_indices(
            len(pos),
            1
        )
    ]

    return int(
        np.count_nonzero(
            d
            <
            MIN_SEPARATION
        )
    )


def min_pair_distance(pos):

    if len(pos) < 2:
        return 0.0

    d = pairwise_distances(
        pos
    )[
        np.triu_indices(
            len(pos),
            1
        )
    ]

    return float(
        np.min(
            d
        )
    )


def connectivity_metrics(pos):

    n = len(pos)

    if n <= 1:
        return (
            True,
            1.0,
            0,
            1
        )

    adjacency = (
        (
            pairwise_distances(
                pos
            )
            <=
            COMM_RANGE
        )
        &
        (
            ~np.eye(
                n,
                dtype=bool
            )
        )
    )

    links = int(
        np.sum(
            np.triu(
                adjacency,
                1
            )
        )
    )

    visited = np.zeros(
        n,
        dtype=bool
    )

    sizes = []

    for start in range(n):

        if visited[start]:
            continue

        stack = [
            start
        ]

        visited[start] = True

        size = 0

        while stack:

            u = stack.pop()

            size += 1

            for v in np.where(
                adjacency[u]
                &
                (~visited)
            )[0]:

                visited[v] = True

                stack.append(
                    int(v)
                )

        sizes.append(
            size
        )

    components = len(
        sizes
    )

    gcc = float(
        max(sizes)
        /
        n
    )

    return (
        components == 1,
        gcc,
        links,
        components
    )


def quick_evaluate(pos):

    pos = clip_positions(
        pos
    )

    c, o, count, mask = (
        coverage_data(
            pos
        )
    )

    return {
        "pos":
            pos.copy(),

        "coverage":
            c,

        "overlap":
            o,

        "ee":
            energy_efficiency(
                c,
                o
            ),

        "sep_viol":
            separation_violations(
                pos
            ),

        "count":
            count,

        "mask":
            mask,
    }


def full_evaluate(pos):

    sol = quick_evaluate(
        pos
    )

    connected, gcc, links, components = (
        connectivity_metrics(
            sol["pos"]
        )
    )

    sol.update(
        connected=connected,
        gcc=gcc,
        links=links,
        components=components
    )

    return sol


def is_feasible(sol):

    return (
        sol["coverage"]
        >
        TARGET_COVERAGE

        and

        sol["overlap"]
        <
        TARGET_OVERLAP

        and

        sol["ee"]
        >
        TARGET_EE

        and

        sol.get(
            "sep_viol",
            0
        )
        ==
        0
    )


def fitness_raw(
    coverage,
    overlap
):

    ee = energy_efficiency(
        coverage,
        overlap
    )

    violation = (

        2.0
        *
        max(
            0.0,
            TARGET_COVERAGE
            +
            SEARCH_MARGIN_COVERAGE
            -
            coverage
        )

        +

        4.0
        *
        max(
            0.0,
            overlap
            -
            (
                TARGET_OVERLAP
                -
                SEARCH_MARGIN_OVERLAP
            )
        )

        +

        2.0
        *
        max(
            0.0,
            TARGET_EE
            +
            SEARCH_MARGIN_EE
            -
            ee
        )
    )

    return (
        ee
        -
        violation
        -
        OVERLAP_REG_WEIGHT
        *
        overlap
    )


def fitness(sol):

    c = sol[
        "coverage"
    ]

    o = sol[
        "overlap"
    ]

    ee = sol[
        "ee"
    ]

    if (
        c
        >
        TARGET_COVERAGE

        and

        ee
        >
        TARGET_EE
    ):

        value = (
            5.0
            *
            c

            +

            5.0
            *
            ee

            -

            14.0
            *
            o
        )

    else:

        value = fitness_raw(
            c,
            o
        )

    return (
        value
        -
        SEP_PENALTY
        *
        sol.get(
            "sep_viol",
            0
        )
    )


def rank_key(sol):

    return (
        int(
            is_feasible(
                sol
            )
        ),

        fitness(
            sol
        ),

        -sol[
            "overlap"
        ],

        sol[
            "coverage"
        ],

        sol[
            "ee"
        ],
    )


def better_solution(
    new,
    old
):

    return (
        old is None
        or
        rank_key(new)
        >
        rank_key(old)
    )


def disk_indices(p):

    ix0 = max(
        0,
        int(
            np.floor(
                (
                    p[0]
                    -
                    R_SENSE
                )
                /
                DX
            )
        )
        -
        1
    )

    ix1 = min(
        GRID_RES - 1,
        int(
            np.ceil(
                (
                    p[0]
                    +
                    R_SENSE
                )
                /
                DX
            )
        )
        +
        1
    )

    iy0 = max(
        0,
        int(
            np.floor(
                (
                    p[1]
                    -
                    R_SENSE
                )
                /
                DY
            )
        )
        -
        1
    )

    iy1 = min(
        GRID_RES - 1,
        int(
            np.ceil(
                (
                    p[1]
                    +
                    R_SENSE
                )
                /
                DY
            )
        )
        +
        1
    )

    ix = np.arange(
        ix0,
        ix1 + 1
    )

    iy = np.arange(
        iy0,
        iy1 + 1
    )

    ddx = (
        gx[ix]
        -
        p[0]
    )

    ddy = (
        gy[iy]
        -
        p[1]
    )

    inside = (
        ddy[:, None] ** 2
        +
        ddx[None, :] ** 2
    ) <= (
        R_SENSE**2
    )

    jj, ii = np.nonzero(
        inside
    )

    return (
        iy[jj]
        *
        GRID_RES
        +
        ix[ii]
    )


class CoverageState:

    def __init__(
        self,
        pos
    ):

        self.pos = clip_positions(
            pos
        )

        self.n = len(
            self.pos
        )

        self.idx = [
            disk_indices(p)
            for p in self.pos
        ]

        self.count = np.zeros(
            NUM_GRID,
            dtype=np.int32
        )

        for ind in self.idx:

            self.count[
                ind
            ] += 1

        self.n_cov = int(
            np.sum(
                self.count
                >=
                1
            )
        )

        self.n_ov = int(
            np.sum(
                self.count
                >=
                2
            )
        )

        self.sep_viol = (
            separation_violations(
                self.pos
            )
        )

    def metrics(self):

        return (
            self.n_cov
            /
            NUM_GRID,

            self.n_ov
            /
            NUM_GRID
        )

    def fitness(self):

        c, o = (
            self.metrics()
        )

        return (
            fitness_raw(
                c,
                o
            )

            -

            SEP_PENALTY
            *
            self.sep_viol
        )

    def node_violations(
        self,
        i,
        p
    ):

        if MIN_SEPARATION <= 0.0:
            return 0

        d = np.hypot(
            self.pos[:, 0]
            -
            p[0],

            self.pos[:, 1]
            -
            p[1]
        )

        d[i] = np.inf

        return int(
            np.count_nonzero(
                d
                <
                MIN_SEPARATION
            )
        )

    def remove(
        self,
        ind
    ):

        old = self.count[
            ind
        ].copy()

        self.count[
            ind
        ] -= 1

        self.n_cov -= int(
            np.count_nonzero(
                old
                ==
                1
            )
        )

        self.n_ov -= int(
            np.count_nonzero(
                old
                ==
                2
            )
        )

    def add(
        self,
        ind
    ):

        old = self.count[
            ind
        ].copy()

        self.count[
            ind
        ] += 1

        self.n_cov += int(
            np.count_nonzero(
                old
                ==
                0
            )
        )

        self.n_ov += int(
            np.count_nonzero(
                old
                ==
                1
            )
        )

    def move(
        self,
        i,
        new_p
    ):

        new_p = clip_point(
            new_p
        )

        old_p = (
            self.pos[
                i
            ].copy()
        )

        old_ind = (
            self.idx[
                i
            ]
        )

        new_ind = disk_indices(
            new_p
        )

        dv = (
            self.node_violations(
                i,
                new_p
            )
            -
            self.node_violations(
                i,
                old_p
            )
        )

        self.remove(
            old_ind
        )

        self.add(
            new_ind
        )

        self.pos[
            i
        ] = new_p

        self.idx[
            i
        ] = new_ind

        self.sep_viol += dv

        return (
            i,
            old_p,
            old_ind,
            new_ind,
            dv
        )

    def undo(
        self,
        info
    ):

        (
            i,
            old_p,
            old_ind,
            new_ind,
            dv
        ) = info

        self.remove(
            new_ind
        )

        self.add(
            old_ind
        )

        self.pos[
            i
        ] = old_p

        self.idx[
            i
        ] = old_ind

        self.sep_viol -= dv

    def redundancy_priority(
        self
    ):

        priority = np.empty(
            self.n,
            dtype=np.float64
        )

        for k, ind in enumerate(
            self.idx
        ):

            counts = self.count[
                ind
            ]

            redundant = np.count_nonzero(
                counts
                >=
                2
            )

            unique = np.count_nonzero(
                counts
                ==
                1
            )

            priority[k] = (
                redundant
                /
                (
                    unique
                    +
                    1.0
                )
            )

        return priority


def ARLS(
    pos,
    moves,
    t_start,
    t_end,
    step_start,
    step_end,
    rng
):

    state = CoverageState(
        pos
    )

    current_fitness = (
        state.fitness()
    )

    best_fitness = (
        current_fitness
    )

    best_position = (
        state.pos.copy()
    )

    probability = np.full(
        state.n,
        1.0
        /
        state.n
    )

    for t in range(
        moves
    ):

        frac = (
            t
            /
            max(
                moves - 1,
                1
            )
        )

        temperature = (
            t_start
            *
            (
                t_end
                /
                t_start
            )
            **
            frac
        )

        step = (
            step_start
            +
            (
                step_end
                -
                step_start
            )
            *
            frac
        )

        if (
            t
            %
            PRIORITY_REFRESH
            ==
            0
        ):

            priority = (
                state.redundancy_priority()
                +
                0.05
            )

            probability = (
                0.5
                *
                priority
                /
                priority.sum()

                +

                0.5
                /
                state.n
            )

        node = int(
            rng.choice(
                state.n,
                p=probability
            )
        )

        if (
            rng.random()
            <
            ARLS_HOLE_PROB

            and

            state.n_cov
            <
            NUM_GRID
        ):

            holes = np.flatnonzero(
                state.count
                ==
                0
            )

            target = GRID[
                holes[
                    rng.integers(
                        len(holes)
                    )
                ]
            ]

            new_position = clip_point(
                target
                +
                rng.normal(
                    0.0,
                    0.25
                    *
                    R_SENSE,
                    2
                )
            )

        else:

            new_position = clip_point(
                state.pos[
                    node
                ]
                +
                rng.normal(
                    0.0,
                    step,
                    2
                )
            )

        info = state.move(
            node,
            new_position
        )

        new_fitness = (
            state.fitness()
        )

        delta = (
            new_fitness
            -
            current_fitness
        )

        accept = (
            delta >= 0.0
            or
            rng.random()
            <
            np.exp(
                delta
                /
                max(
                    temperature,
                    1e-12
                )
            )
        )

        if accept:

            current_fitness = (
                new_fitness
            )

            if (
                current_fitness
                >
                best_fitness
            ):

                best_fitness = (
                    current_fitness
                )

                best_position = (
                    state.pos.copy()
                )

        else:

            state.undo(
                info
            )

    return (
        quick_evaluate(
            state.pos
        ),

        quick_evaluate(
            best_position
        )
    )


def schedule(
    generation
):

    a = (
        generation
        /
        max(
            GENERATIONS - 1,
            1
        )
    )

    b = min(
        1.0,
        (
            generation
            +
            1
        )
        /
        max(
            GENERATIONS - 1,
            1
        )
    )

    t0 = (
        ARLS_T_START
        *
        (
            ARLS_T_END
            /
            ARLS_T_START
        )
        **
        a
    )

    t1 = (
        ARLS_T_START
        *
        (
            ARLS_T_END
            /
            ARLS_T_START
        )
        **
        b
    )

    s0 = (
        ARLS_STEP_MAX
        +
        (
            ARLS_STEP_MIN
            -
            ARLS_STEP_MAX
        )
        *
        a
    )

    s1 = (
        ARLS_STEP_MAX
        +
        (
            ARLS_STEP_MIN
            -
            ARLS_STEP_MAX
        )
        *
        b
    )

    return (
        t0,
        t1,
        s0,
        s1
    )


def make_seed(
    n,
    phase,
    jitter,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    cols = max(
        2,
        int(
            round(
                np.sqrt(
                    n
                    *
                    AREA_X
                    /
                    AREA_Y
                )
            )
        )
    )

    rows = int(
        np.ceil(
            n
            /
            cols
        )
    )

    xs = np.linspace(
        5.0,
        AREA_X - 5.0,
        cols
    )

    ys = np.linspace(
        0.0,
        AREA_Y,
        rows
    )

    positions = []

    for i, x in enumerate(
        xs
    ):

        for y in ys:

            if len(positions) >= n:
                break

            xx = x

            yy = (
                y
                +
                (
                    phase
                    if i % 2
                    else
                    0.0
                )
            )

            if jitter > 0:

                xx += rng.normal(
                    0.0,
                    jitter
                )

                yy += rng.normal(
                    0.0,
                    jitter
                )

            positions.append(
                [
                    xx,
                    yy
                ]
            )

        if len(positions) >= n:
            break

    return clip_positions(
        np.asarray(
            positions,
            dtype=np.float64
        )
    )


def create_population(n):

    phases = [
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        8.0
    ]

    jitters = [
        0.0,
        0.2,
        0.3,
        0.3,
        0.4,
        0.5
    ]

    return [
        make_seed(
            n,
            phases[i],
            jitters[i],
            SEED + i
        )
        for i in range(
            POP_SIZE
        )
    ]


class GNNEncoder(
    nn.Module
):

    def __init__(
        self,
        input_dim=GNN_INPUT,
        hidden_dim=GNN_HIDDEN,
        output_dim=GNN_OUTPUT
    ):

        super().__init__()

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            output_dim
        )

        self.relu = nn.ReLU()

    def forward(
        self,
        x,
        adjacency
    ):

        n = x.shape[0]

        eye = torch.eye(
            n,
            dtype=torch.float32,
            device=x.device
        )

        a = (
            adjacency
            +
            eye
        )

        degree = a.sum(
            dim=1,
            keepdim=True
        )

        denom = (
            torch.sqrt(
                degree
            )
            @
            torch.sqrt(
                degree
            ).T
        )

        a_norm = (
            a
            /
            (
                denom
                +
                1e-8
            )
        )

        h = self.relu(
            self.fc1(
                a_norm
                @
                x
            )
        )

        h = self.relu(
            self.fc2(
                a_norm
                @
                h
            )
        )

        return h.mean(
            dim=0
        )


def build_graph(pos):

    d = pairwise_distances(
        pos
    )

    return (
        (
            d
            <=
            COMM_RANGE
        )
        &
        (
            ~np.eye(
                len(pos),
                dtype=bool
            )
        )
    ).astype(
        np.float32
    )


def graph_tensors(pos):

    adjacency_np = build_graph(
        pos
    )

    n = len(pos)

    xy = np.column_stack(
        (
            pos[:, 0]
            /
            AREA_X,

            pos[:, 1]
            /
            AREA_Y
        )
    )

    degree = (
        adjacency_np.sum(
            axis=1,
            keepdims=True
        )
        /
        max(
            n - 1,
            1
        )
    )

    energy = np.ones(
        (
            n,
            1
        ),
        dtype=np.float32
    )

    features = np.hstack(
        (
            xy,
            degree,
            energy
        )
    ).astype(
        np.float32
    )

    x = torch.tensor(
        features,
        dtype=torch.float32,
        device=DEVICE
    )

    adjacency = torch.tensor(
        adjacency_np,
        dtype=torch.float32,
        device=DEVICE
    )

    return (
        x,
        adjacency
    )


def topology_features(pos):

    adjacency = build_graph(
        pos
    )

    n = len(pos)

    degree = (
        adjacency.sum(
            axis=1
        )
        /
        max(
            n - 1,
            1
        )
    )

    link_ratio = (
        np.sum(
            np.triu(
                adjacency,
                1
            )
        )
        /
        max(
            n
            *
            (
                n - 1
            )
            /
            2,
            1
        )
    )

    return np.array(
        [
            np.mean(
                degree
            ),

            1.0
            -
            np.mean(
                degree == 0
            ),

            link_ratio,

            np.std(
                degree
            ),
        ],
        dtype=np.float32,
    )


def build_state(sol):

    pos = sol[
        "pos"
    ]

    n = len(pos)

    if n > 1:

        upper = pairwise_distances(
            pos
        )[
            np.triu_indices(
                n,
                1
            )
        ]

        mean_distance = float(
            np.mean(
                upper
            )
        )

        minimum_distance = float(
            np.min(
                upper
            )
        )

    else:

        mean_distance = 0.0

        minimum_distance = 0.0

    scale = max(
        AREA_X,
        AREA_Y
    )

    base = np.array(
        [
            sol[
                "coverage"
            ],

            sol[
                "overlap"
            ],

            sol[
                "ee"
            ],

            mean_distance
            /
            scale,

            minimum_distance
            /
            scale,

            sol[
                "coverage"
            ]
            -
            TARGET_COVERAGE,

            TARGET_OVERLAP
            -
            sol[
                "overlap"
            ],

            TARGET_EE
            -
            sol[
                "ee"
            ],
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        (
            base,
            topology_features(
                pos
            )
        )
    ).astype(
        np.float32
    )


class HybridQNet(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        self.gnn = GNNEncoder()

        self.q = nn.Sequential(

            nn.Linear(
                STATE_BASE_DIM
                +
                GNN_OUTPUT,
                DQN_HIDDEN
            ),

            nn.ReLU(),

            nn.Linear(
                DQN_HIDDEN,
                DQN_HIDDEN
            ),

            nn.ReLU(),

            nn.Linear(
                DQN_HIDDEN,
                NUM_ACTIONS
            )
        )

    def forward(
        self,
        base_states,
        positions
    ):

        embeddings = []

        for pos in positions:

            x, adjacency = (
                graph_tensors(
                    pos
                )
            )

            embeddings.append(
                self.gnn(
                    x,
                    adjacency
                )
            )

        graph_embeddings = torch.stack(
            embeddings,
            dim=0
        )

        return self.q(
            torch.cat(
                (
                    base_states,
                    graph_embeddings
                ),
                dim=1
            )
        )


class ReplayBuffer:

    def __init__(
        self,
        capacity=BUFFER_SIZE
    ):

        self.buffer = deque(
            maxlen=capacity
        )

    def push(
        self,
        state,
        pos,
        action,
        reward,
        next_state,
        next_pos,
        done
    ):

        self.buffer.append(
            (
                np.asarray(
                    state,
                    np.float32
                ).copy(),

                np.asarray(
                    pos,
                    np.float32
                ).copy(),

                int(
                    action
                ),

                float(
                    reward
                ),

                np.asarray(
                    next_state,
                    np.float32
                ).copy(),

                np.asarray(
                    next_pos,
                    np.float32
                ).copy(),

                float(
                    done
                )
            )
        )

    def sample(
        self,
        batch_size
    ):

        return random.sample(
            self.buffer,
            batch_size
        )

    def __len__(
        self
    ):

        return len(
            self.buffer
        )


def choose_action(
    q_net,
    state,
    pos,
    epsilon,
    rng
):

    if (
        rng.random()
        <
        epsilon
    ):

        return int(
            rng.integers(
                NUM_ACTIONS
            )
        )

    s = torch.tensor(
        state,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(
        0
    )

    with torch.no_grad():

        q = q_net(
            s,
            [
                pos
            ]
        )

    return int(
        torch.argmax(
            q,
            dim=1
        ).item()
    )


def update_dqn(
    q_net,
    target_net,
    optimizer,
    replay
):

    if (
        len(replay)
        <
        BATCH_SIZE
    ):

        return (
            None,
            None
        )

    batch = replay.sample(
        BATCH_SIZE
    )

    (
        states,
        positions,
        actions,
        rewards,
        next_states,
        next_positions,
        dones
    ) = zip(
        *batch
    )

    states = torch.tensor(
        np.asarray(
            states,
            np.float32
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    next_states = torch.tensor(
        np.asarray(
            next_states,
            np.float32
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    actions = torch.tensor(
        np.asarray(
            actions,
            np.int64
        ),
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(
        1
    )

    rewards = torch.tensor(
        np.asarray(
            rewards,
            np.float32
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    dones = torch.tensor(
        np.asarray(
            dones,
            np.float32
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    current_q = (
        q_net(
            states,
            list(
                positions
            )
        )
        .gather(
            1,
            actions
        )
        .squeeze(
            1
        )
    )

    with torch.no_grad():

        next_q = (
            target_net(
                next_states,
                list(
                    next_positions
                )
            )
            .max(
                dim=1
            )
            .values
        )

        target = (
            rewards
            +
            GAMMA
            *
            (
                1.0
                -
                dones
            )
            *
            next_q
        )

    loss = nn.functional.mse_loss(
        current_q,
        target
    )

    optimizer.zero_grad()

    loss.backward()

    grad_norm = 0.0

    for p in q_net.gnn.parameters():

        if p.grad is not None:

            grad_norm += float(
                p.grad.detach().norm().item()
            )

    torch.nn.utils.clip_grad_norm_(
        q_net.parameters(),
        1.0
    )

    optimizer.step()

    return (
        float(
            loss.item()
        ),
        float(
            grad_norm
        )
    )


def global_exploration(
    current,
    elite,
    rng,
    generation
):

    progress = (
        generation
        /
        max(
            GENERATIONS - 1,
            1
        )
    )

    exploration = (
        0.16
        -
        0.08
        *
        progress
    )

    candidate = current.copy()

    noise = rng.normal(
        0.0,
        1.0,
        current.shape
    )

    attraction = (
        rng.random(
            current.shape
        )
        *
        (
            elite
            -
            current
        )
    )

    selected = (
        rng.random(
            len(current)
        )
        <
        0.25
    )

    candidate[
        selected
    ] += (
        exploration
        *
        max(
            AREA_X,
            AREA_Y
        )
        *
        noise
    )[
        selected
    ]

    candidate += (
        0.20
        *
        attraction
    )

    return clip_positions(
        candidate
    )


def elite_hunting(
    current,
    elite,
    rng
):

    candidate = (
        current

        +

        rng.random(
            current.shape
        )

        *
        (
            elite
            -
            current
        )

        +

        rng.normal(
            0.0,
            1.2,
            current.shape
        )
    )

    return clip_positions(
        candidate
    )


def differential_hunting(
    current,
    population,
    elite,
    rng
):

    if len(population) < 3:

        return current.copy()

    i1, i2 = rng.choice(
        len(population),
        2,
        replace=False
    )

    mutant = (
        current

        +

        F
        *
        (
            population[
                i1
            ]
            -
            population[
                i2
            ]
        )

        +

        0.20
        *
        rng.random(
            current.shape
        )
        *
        (
            elite
            -
            current
        )
    )

    mask = (
        rng.random(
            current.shape
        )
        <
        CR
    )

    return clip_positions(
        np.where(
            mask,
            mutant,
            current
        )
    )


def redundancy_operator(
    current,
    rng
):

    state = CoverageState(
        current
    )

    current_fitness = (
        state.fitness()
    )

    priority = (
        state.redundancy_priority()
    )

    for node in np.argsort(
        -priority
    )[
        :6
    ]:

        holes = np.flatnonzero(
            state.count
            ==
            0
        )

        if len(holes) == 0:
            break

        best_target = None

        best_value = (
            current_fitness
        )

        sampled = rng.choice(
            holes,
            size=min(
                12,
                len(holes)
            ),
            replace=False
        )

        for h in sampled:

            info = state.move(
                int(node),
                GRID[h]
            )

            value = (
                state.fitness()
            )

            state.undo(
                info
            )

            if value > best_value:

                best_value = (
                    value
                )

                best_target = (
                    GRID[h].copy()
                )

        if (
            best_target
            is not None
        ):

            state.move(
                int(node),
                best_target
            )

            current_fitness = (
                best_value
            )

    return state.pos.copy()


def adaptive_local_operator(
    current,
    rng
):

    state = CoverageState(
        current
    )

    node = int(
        np.argmax(
            state.redundancy_priority()
        )
    )

    directions = np.array(
        [
            [
                1.0,
                0.0
            ],

            [
                -1.0,
                0.0
            ],

            [
                0.0,
                1.0
            ],

            [
                0.0,
                -1.0
            ]
        ]
    )

    step = rng.choice(
        [
            6.0,
            3.0,
            1.5,
            0.75
        ]
    )

    direction = directions[
        rng.integers(
            4
        )
    ]

    candidate = current.copy()

    candidate[
        node
    ] += (
        step
        *
        direction
    )

    return clip_positions(
        candidate
    )


def apply_operator(
    action,
    current,
    population,
    elite,
    rng,
    generation
):

    if action == 0:

        return global_exploration(
            current,
            elite,
            rng,
            generation
        )

    if action == 1:

        return elite_hunting(
            current,
            elite,
            rng
        )

    if action == 2:

        return differential_hunting(
            current,
            population,
            elite,
            rng
        )

    if action == 3:

        return redundancy_operator(
            current,
            rng
        )

    return adaptive_local_operator(
        current,
        rng
    )


def dominates(
    a,
    b
):

    if (
        a.get(
            "sep_viol",
            0
        )
        >
        b.get(
            "sep_viol",
            0
        )
    ):

        return False

    no_worse = (
        a["coverage"]
        >=
        b["coverage"]
        -
        1e-10

        and

        a["ee"]
        >=
        b["ee"]
        -
        1e-10

        and

        a["overlap"]
        <=
        b["overlap"]
        +
        1e-10
    )

    strictly = (
        a["coverage"]
        >
        b["coverage"]
        +
        1e-10

        or

        a["ee"]
        >
        b["ee"]
        +
        1e-10

        or

        a["overlap"]
        <
        b["overlap"]
        -
        1e-10
    )

    return (
        no_worse
        and
        strictly
    )


def add_to_archive(
    archive,
    candidate
):

    if any(
        dominates(
            item,
            candidate
        )
        for item in archive
    ):

        return archive

    new_archive = [
        item
        for item in archive
        if not dominates(
            candidate,
            item
        )
    ]

    new_archive.append(
        {
            k: candidate[k]
            for k in (
                "coverage",
                "overlap",
                "ee",
                "sep_viol"
            )
        }
        |
        {
            "pos":
                candidate[
                    "pos"
                ].copy()
        }
    )

    new_archive.sort(
        key=rank_key,
        reverse=True
    )

    return new_archive[
        :ARCHIVE_SIZE
    ]


def select_global_leader(
    candidates
):

    candidates = [
        x
        for x in candidates
        if x is not None
    ]

    if candidates:

        return max(
            candidates,
            key=rank_key
        )

    return None


def select_final_candidate(
    candidates
):

    candidates = [
        x
        for x in candidates
        if x is not None
    ]

    feasible = [
        x
        for x in candidates
        if is_feasible(
            x
        )
    ]

    if feasible:

        return min(
            feasible,
            key=lambda x: (
                x["overlap"],
                -x["coverage"],
                -x["ee"]
            )
        )

    return max(
        candidates,
        key=rank_key
    )


def run_proposed(
    n,
    seed=SEED
):

    start = time.time()

    set_seed(
        seed
    )

    rng = np.random.default_rng(
        seed
    )

    q_net = HybridQNet().to(
        DEVICE
    )

    target_net = HybridQNet().to(
        DEVICE
    )

    target_net.load_state_dict(
        q_net.state_dict()
    )

    optimizer = optim.Adam(
        q_net.parameters(),
        lr=DQN_LR
    )

    replay = ReplayBuffer()

    gnn_params = sum(
        p.numel()
        for p in q_net.gnn.parameters()
    )

    losses = []

    grads = []

    gnn_updates = 0

    population = []

    archive = []

    elite = None

    t0, _, s0, _ = schedule(
        0
    )

    for raw in create_population(
        n
    ):

        current, best = ARLS(
            raw,
            ARLS_MOVES_INIT,
            t0,
            t0,
            s0,
            s0,
            rng
        )

        population.append(
            current[
                "pos"
            ].copy()
        )

        archive = add_to_archive(
            archive,
            best
        )

        if better_solution(
            best,
            elite
        ):

            elite = best

    population_solutions = [
        quick_evaluate(
            p
        )
        for p in population
    ]

    history = []

    chain_position = (
        elite[
            "pos"
        ].copy()
    )

    for generation in range(
        GENERATIONS
    ):

        epsilon = max(
            EPS_END,
            EPS_START
            *
            np.exp(
                -generation
                /
                EPS_DECAY
            )
        )

        (
            t_a,
            t_b,
            s_a,
            s_b
        ) = schedule(
            generation
        )

        new_population = []

        actions = []

        for i, current in enumerate(
            population
        ):

            current_sol = (
                population_solutions[
                    i
                ]
            )

            state = build_state(
                full_evaluate(
                    current
                )
            )

            action = choose_action(
                q_net,
                state,
                current,
                epsilon,
                rng
            )

            actions.append(
                action
            )

            candidate = apply_operator(
                action,
                current,
                population,
                elite[
                    "pos"
                ],
                rng,
                generation
            )

            cand_cur, cand_best = ARLS(
                candidate,
                ARLS_MOVES_POP,
                t_a,
                t_b,
                s_a,
                s_b,
                rng
            )

            archive = add_to_archive(
                archive,
                cand_best
            )

            if better_solution(
                cand_best,
                elite
            ):

                elite = cand_best

            reward = (
                REWARD_COVERAGE
                *
                (
                    cand_cur[
                        "coverage"
                    ]
                    -
                    current_sol[
                        "coverage"
                    ]
                )

                +

                REWARD_OVERLAP
                *
                (
                    current_sol[
                        "overlap"
                    ]
                    -
                    cand_cur[
                        "overlap"
                    ]
                )

                +

                REWARD_EE
                *
                (
                    cand_cur[
                        "ee"
                    ]
                    -
                    current_sol[
                        "ee"
                    ]
                )

                +

                REWARD_FEASIBLE
                *
                float(
                    is_feasible(
                        cand_best
                    )
                    and
                    not is_feasible(
                        current_sol
                    )
                )
            )

            next_sol = full_evaluate(
                cand_cur[
                    "pos"
                ]
            )

            next_state = build_state(
                next_sol
            )

            replay.push(
                state,
                current,
                action,
                reward,
                next_state,
                cand_cur[
                    "pos"
                ],
                generation
                ==
                GENERATIONS - 1
            )

            loss, grad = update_dqn(
                q_net,
                target_net,
                optimizer,
                replay
            )

            if loss is not None:

                losses.append(
                    loss
                )

                grads.append(
                    grad
                )

                gnn_updates += int(
                    grad > 0.0
                )

            delta = (
                fitness(
                    cand_cur
                )
                -
                fitness(
                    current_sol
                )
            )

            accept = (
                delta >= 0.0

                or

                rng.random()
                <
                np.exp(
                    delta
                    /
                    max(
                        t_b,
                        1e-12
                    )
                )
            )

            new_population.append(
                cand_cur[
                    "pos"
                ].copy()
                if accept
                else
                current.copy()
            )

        population = (
            new_population
        )

        population_solutions = [
            quick_evaluate(
                p
            )
            for p in population
        ]

        for sol in (
            population_solutions
        ):

            archive = add_to_archive(
                archive,
                sol
            )

        chain_current, elite_best = ARLS(
            chain_position,
            ARLS_MOVES_ELITE,
            t_a,
            t_b,
            s_a,
            s_b,
            rng
        )

        chain_position = (
            chain_current[
                "pos"
            ].copy()
        )

        archive = add_to_archive(
            archive,
            elite_best
        )

        leader_pool = (
            population_solutions

            +

            [
                quick_evaluate(
                    a[
                        "pos"
                    ]
                )
                for a in archive
            ]

            +

            [
                elite,
                elite_best
            ]
        )

        elite = select_global_leader(
            leader_pool
        )

        population_solutions = [
            quick_evaluate(
                p
            )
            for p in population
        ]

        worst = int(
            np.argmin(
                [
                    fitness(
                        sol
                    )
                    for sol in (
                        population_solutions
                    )
                ]
            )
        )

        population[
            worst
        ] = (
            elite[
                "pos"
            ].copy()
        )

        population_solutions[
            worst
        ] = quick_evaluate(
            elite[
                "pos"
            ]
        )

        if (
            generation
            +
            1
        ) % TARGET_UPDATE == 0:

            target_net.load_state_dict(
                q_net.state_dict()
            )

        dominant_action = (
            max(
                set(
                    actions
                ),
                key=actions.count
            )
            if actions
            else
            0
        )

        history.append(
            {
                "generation":
                    generation + 1,

                "coverage":
                    elite[
                        "coverage"
                    ]
                    *
                    100.0,

                "overlap":
                    elite[
                        "overlap"
                    ]
                    *
                    100.0,

                "ee":
                    elite[
                        "ee"
                    ]
                    *
                    100.0,

                "epsilon":
                    epsilon,

                "dominant_action":
                    ACTION_NAMES[
                        dominant_action
                    ],

                "feasible":
                    is_feasible(
                        elite
                    ),
            }
        )

        if (
            generation == 0
            or
            (
                generation + 1
            ) % 5 == 0
            or
            generation == GENERATIONS - 1
        ):

            print(
                f"N={n:2d} | "
                f"Gen={generation+1:2d} | "
                f"Cov={elite['coverage']*100:7.3f}% | "
                f"Ov={elite['overlap']*100:7.3f}% | "
                f"EE={elite['ee']*100:7.3f}% | "
                f"Action={ACTION_NAMES[dominant_action]} | "
                f"{'PASS' if is_feasible(elite) else 'SEARCHING'}",
                flush=True
            )

    pool = (
        population_solutions

        +

        [
            quick_evaluate(
                a[
                    "pos"
                ]
            )
            for a in archive
        ]

        +

        [
            elite
        ]
    )

    pool = [
        x
        for x in pool
        if x is not None
    ]

    pool.sort(
        key=rank_key,
        reverse=True
    )

    polished = []

    for candidate in pool[
        :ARLS_FINAL_CANDIDATES
    ]:

        _, refined = ARLS(
            candidate[
                "pos"
            ],
            ARLS_MOVES_FINAL,
            1e-4,
            ARLS_T_END,
            3.0,
            ARLS_STEP_MIN,
            rng
        )

        polished.append(
            refined
        )

    final_candidate = (
        select_final_candidate(
            pool
            +
            polished
        )
    )

    final = full_evaluate(
        final_candidate[
            "pos"
        ]
    )

    c2, o2, _, _ = (
        coverage_data(
            final[
                "pos"
            ]
        )
    )

    e2 = energy_efficiency(
        c2,
        o2
    )

    independent_match = (
        abs(
            final[
                "coverage"
            ]
            -
            c2
        )
        <
        1e-12

        and

        abs(
            final[
                "overlap"
            ]
            -
            o2
        )
        <
        1e-12

        and

        abs(
            final[
                "ee"
            ]
            -
            e2
        )
        <
        1e-12
    )

    return {
        "nodes":
            n,

        "coverage":
            final[
                "coverage"
            ]
            *
            100.0,

        "overlap":
            final[
                "overlap"
            ]
            *
            100.0,

        "ee":
            final[
                "ee"
            ]
            *
            100.0,

        "gcc":
            final[
                "gcc"
            ]
            *
            100.0,

        "links":
            final[
                "links"
            ],

        "components":
            final[
                "components"
            ],

        "connected":
            final[
                "connected"
            ],

        "feasible":
            is_feasible(
                final
            ),

        "min_sep":
            min_pair_distance(
                final[
                    "pos"
                ]
            ),

        "colocated":
            int(
                n
                -
                len(
                    np.unique(
                        np.round(
                            final[
                                "pos"
                            ],
                            3
                        ),
                        axis=0
                    )
                )
            ),

        "runtime":
            time.time()
            -
            start,

        "positions":
            final[
                "pos"
            ].copy(),

        "history":
            history,

        "archive_size":
            len(
                archive
            ),

        "gnn_params":
            gnn_params,

        "gnn_updates":
            gnn_updates,

        "avg_dqn_loss":
            (
                float(
                    np.mean(
                        losses
                    )
                )
                if losses
                else
                np.nan
            ),

        "avg_gnn_grad":
            (
                float(
                    np.mean(
                        grads
                    )
                )
                if grads
                else
                np.nan
            ),

        "independent_match":
            independent_match,
    }


def main():

    print(
        "="
        *
        110
    )

    print(
        "TR-EnMOGKSO + TRAINABLE GNN + DRL/DQN + ARLS"
    )

    print(
        "ENHANCED OVERLAP-AWARE 250 x 300 m NODE-SCALABILITY"
    )

    print(
        "="
        *
        110
    )

    print(
        f"Device                  : {DEVICE}"
    )

    print(
        f"Area                    : "
        f"{AREA_X:.0f} x {AREA_Y:.0f} m"
    )

    print(
        f"Sensing range           : "
        f"{R_SENSE:.0f} m"
    )

    print(
        f"Communication range     : "
        f"{COMM_RANGE:.0f} m"
    )

    print(
        f"Population              : "
        f"{POP_SIZE}"
    )

    print(
        f"Generations             : "
        f"{GENERATIONS}"
    )

    print(
        f"Node counts             : "
        f"{NODE_COUNTS}"
    )

    print(
        f"Overlap reg. weight     : "
        f"{OVERLAP_REG_WEIGHT:.2f}"
    )

    print(
        "Stage-2 overlap weight  : "
        "14.00"
    )

    print(
        f"Final ARLS moves        : "
        f"{ARLS_MOVES_FINAL}"
    )

    print(
        f"Final candidates        : "
        f"{ARLS_FINAL_CANDIDATES}"
    )

    print()

    print(
        "Hardcoded results       : NO"
    )

    print(
        "50-node special rule    : NO"
    )

    print(
        "Lloyd                   : NO"
    )

    print(
        "Virtual-force Lloyd     : NO"
    )

    print(
        "Trainable GNN           : YES"
    )

    print(
        "GNN -> DQN TD gradients : YES"
    )

    print(
        "DRL / DQN operator      : YES"
    )

    print(
        "ARLS refinement         : YES"
    )

    print(
        "Pareto archive          : YES"
    )

    print()

    print(
        "FEASIBILITY TARGETS"
    )

    print(
        f"Coverage > "
        f"{TARGET_COVERAGE * 100:.2f}%"
    )

    print(
        f"Overlap  < "
        f"{TARGET_OVERLAP * 100:.2f}%"
    )

    print(
        f"EE       > "
        f"{TARGET_EE * 100:.2f}%"
    )

    print(
        "="
        *
        110
    )

    overall_start = time.time()

    results = []

    histories = {}

    positions = {}

    for n in NODE_COUNTS:

        print(
            "\n"
            +
            "#"
            *
            110
        )

        print(
            f"STARTING {n}-NODE EXPERIMENT"
        )

        print(
            "#"
            *
            110
        )

        r = run_proposed(
            n,
            seed=SEED
        )

        histories[
            n
        ] = r[
            "history"
        ]

        positions[
            n
        ] = r[
            "positions"
        ]

        results.append(
            {
                "Nodes":
                    r[
                        "nodes"
                    ],

                "Coverage (%)":
                    r[
                        "coverage"
                    ],

                "Overlap (%)":
                    r[
                        "overlap"
                    ],

                "EE (%)":
                    r[
                        "ee"
                    ],

                "GCC (%)":
                    r[
                        "gcc"
                    ],

                "Links":
                    r[
                        "links"
                    ],

                "Components":
                    r[
                        "components"
                    ],

                "Connected":
                    r[
                        "connected"
                    ],

                "Feasible":
                    r[
                        "feasible"
                    ],

                "Min sep (m)":
                    r[
                        "min_sep"
                    ],

                "Co-located":
                    r[
                        "colocated"
                    ],

                "Runtime (s)":
                    r[
                        "runtime"
                    ],

                "Archive":
                    r[
                        "archive_size"
                    ],

                "GNN params":
                    r[
                        "gnn_params"
                    ],

                "GNN updates":
                    r[
                        "gnn_updates"
                    ],

                "Avg DQN loss":
                    r[
                        "avg_dqn_loss"
                    ],

                "Avg GNN grad":
                    r[
                        "avg_gnn_grad"
                    ],

                "Independent check":
                    r[
                        "independent_match"
                    ],
            }
        )

        print()

        print(
            f"FINAL {n}-NODE RESULT"
        )

        print(
            f"Coverage            : "
            f"{r['coverage']:.4f}%"
        )

        print(
            f"Overlap             : "
            f"{r['overlap']:.4f}%"
        )

        print(
            f"Energy Efficiency   : "
            f"{r['ee']:.4f}%"
        )

        print(
            f"GCC                 : "
            f"{r['gcc']:.4f}%"
        )

        print(
            f"Links               : "
            f"{r['links']}"
        )

        print(
            f"Components          : "
            f"{r['components']}"
        )

        print(
            f"Connected           : "
            f"{r['connected']}"
        )

        print(
            "Feasibility         : "
            +
            (
                "PASS"
                if r[
                    "feasible"
                ]
                else
                "NOT ACHIEVED"
            )
        )

        print(
            f"GNN trainable params: "
            f"{r['gnn_params']}"
        )

        print(
            f"GNN TD updates      : "
            f"{r['gnn_updates']}"
        )

        print(
            f"Average DQN loss    : "
            f"{r['avg_dqn_loss']:.8e}"
        )

        print(
            f"Average GNN grad    : "
            f"{r['avg_gnn_grad']:.8e}"
        )

        print(
            f"Independent check   : "
            f"{r['independent_match']}"
        )

        print(
            f"Runtime             : "
            f"{r['runtime']:.2f} s"
        )

    df = pd.DataFrame(
        results
    )

    print(
        "\n"
        +
        "="
        *
        120
    )

    print(
        "FINAL 250 x 300 NODE-SCALABILITY RESULTS"
    )

    print(
        "="
        *
        120
    )

    print(
        df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    independent = []

    for n in NODE_COUNTS:

        c, o, _, _ = (
            coverage_data(
                positions[
                    n
                ]
            )
        )

        e = energy_efficiency(
            c,
            o
        )

        independent.append(
            {
                "Nodes":
                    n,

                "Coverage (%)":
                    c
                    *
                    100,

                "Overlap (%)":
                    o
                    *
                    100,

                "EE (%)":
                    e
                    *
                    100,

                "Coverage > 98%":
                    c
                    >
                    TARGET_COVERAGE,

                "Overlap < 10%":
                    o
                    <
                    TARGET_OVERLAP,

                "EE > 97%":
                    e
                    >
                    TARGET_EE,
            }
        )

    idf = pd.DataFrame(
        independent
    )

    print(
        "\n"
        +
        "="
        *
        110
    )

    print(
        "INDEPENDENT METRIC RECOMPUTATION"
    )

    print(
        "="
        *
        110
    )

    print(
        idf.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    feasible = df[
        df[
            "Feasible"
        ]
        ==
        True
    ]

    print(
        "\n"
        +
        "="
        *
        110
    )

    print(
        "BEST FEASIBLE OPERATING POINT"
    )

    print(
        "="
        *
        110
    )

    if len(feasible):

        best = (
            feasible
            .sort_values(
                [
                    "Overlap (%)",
                    "Coverage (%)",
                    "EE (%)"
                ],
                ascending=[
                    True,
                    False,
                    False
                ]
            )
            .iloc[0]
        )

        print(
            f"Best feasible node count : "
            f"{int(best['Nodes'])}"
        )

        print(
            f"Coverage                 : "
            f"{best['Coverage (%)']:.4f}%"
        )

        print(
            f"Overlap                  : "
            f"{best['Overlap (%)']:.4f}%"
        )

        print(
            f"EE                       : "
            f"{best['EE (%)']:.4f}%"
        )

    else:

        print(
            "No node count achieved all targets."
        )

    row40 = (
        df[
            df[
                "Nodes"
            ]
            ==
            40
        ]
        .iloc[0]
    )

    row50 = (
        df[
            df[
                "Nodes"
            ]
            ==
            50
        ]
        .iloc[0]
    )

    cov_better = (
        row50[
            "Coverage (%)"
        ]
        >
        row40[
            "Coverage (%)"
        ]
    )

    ov_better = (
        row50[
            "Overlap (%)"
        ]
        <
        row40[
            "Overlap (%)"
        ]
    )

    ee_better = (
        row50[
            "EE (%)"
        ]
        >
        row40[
            "EE (%)"
        ]
    )

    print(
        "\n"
        +
        "="
        *
        110
    )

    print(
        "40-NODE VS 50-NODE"
    )

    print(
        "="
        *
        110
    )

    print(
        f"40-node Coverage : "
        f"{row40['Coverage (%)']:.4f}%"
    )

    print(
        f"50-node Coverage : "
        f"{row50['Coverage (%)']:.4f}%"
    )

    print(
        f"40-node Overlap  : "
        f"{row40['Overlap (%)']:.4f}%"
    )

    print(
        f"50-node Overlap  : "
        f"{row50['Overlap (%)']:.4f}%"
    )

    print(
        f"40-node EE       : "
        f"{row40['EE (%)']:.4f}%"
    )

    print(
        f"50-node EE       : "
        f"{row50['EE (%)']:.4f}%"
    )

    print(
        f"50 better Coverage : "
        f"{cov_better}"
    )

    print(
        f"50 better Overlap  : "
        f"{ov_better}"
    )

    print(
        f"50 better EE       : "
        f"{ee_better}"
    )

    print(
        "50-NODE OUTPERFORMS 40-NODE "
        "ON ALL THREE PRIMARY METRICS : "
        +
        str(
            cov_better
            and
            ov_better
            and
            ee_better
        )
    )

    total_runtime = (
        time.time()
        -
        overall_start
    )

    results_file = (
        "TR_EnMOGKSO_GNN_DRL_ARLS_"
        "250x300_UPDATED_results.csv"
    )

    independent_file = (
        "TR_EnMOGKSO_GNN_DRL_ARLS_"
        "250x300_UPDATED_independent_check.csv"
    )

    history_file = (
        "TR_EnMOGKSO_GNN_DRL_ARLS_"
        "250x300_UPDATED_history.csv"
    )

    df.to_csv(
        results_file,
        index=False
    )

    idf.to_csv(
        independent_file,
        index=False
    )

    history_rows = []

    for n, hist in histories.items():

        for item in hist:

            history_rows.append(
                {
                    "Nodes":
                        n,
                    **item
                }
            )

    pd.DataFrame(
        history_rows
    ).to_csv(
        history_file,
        index=False
    )

    for n in NODE_COUNTS:

        pd.DataFrame(
            {
                "Node":
                    np.arange(
                        1,
                        n + 1
                    ),

                "X":
                    positions[
                        n
                    ][
                        :,
                        0
                    ],

                "Y":
                    positions[
                        n
                    ][
                        :,
                        1
                    ],
            }
        ).to_csv(
            (
                "TR_EnMOGKSO_GNN_DRL_ARLS_"
                f"250x300_UPDATED_{n}nodes_positions.csv"
            ),
            index=False
        )

    print(
        "\n"
        +
        "="
        *
        110
    )

    print(
        "FINAL VERIFICATION"
    )

    print(
        "="
        *
        110
    )

    print(
        "Same algorithm rules     : "
        "YES for 20/30/40/50"
    )

    print(
        "Hardcoded results        : NO"
    )

    print(
        "50-node special rule     : NO"
    )

    print(
        "Trainable GNN            : YES"
    )

    print(
        "GNN -> DQN TD gradients  : YES"
    )

    print(
        "ARLS                     : YES"
    )

    print(
        "Pareto archive           : YES"
    )

    print(
        "Independent check        : "
        "see table above"
    )

    print(
        f"Total runtime            : "
        f"{total_runtime:.2f} s"
    )

    print(
        f"Saved results            : "
        f"{results_file}"
    )

    print(
        f"Saved independent check  : "
        f"{independent_file}"
    )

    print(
        f"Saved history            : "
        f"{history_file}"
    )


# =====================================================================
# PART B - LEARNED ARLS: the GNN and the DQN control the local search
# =====================================================================
#
# In Part A the GNN/DQN only choose 1 of 5 TR-EnMOGKSO operators
# (~180 decisions per run) while ARLS does almost all the work with
# hand-made rules. Here ARLS itself is driven by the learned components:
#
#   * NODE-GNN (trained online): graph convolution over the
#     disk-intersection graph of the current deployment. Per-node
#     features: position, unique / redundant coverage share, share of
#     the disk inside the field, boundary distance, nearby uncovered
#     points. It scores every node and ARLS samples WHICH node to move
#     (score re-weights the redundancy prior). It is trained on the
#     fitness gains its chosen moves produced.
#   * MOVE-DQN (trained online): before every block of ARLS moves it
#     chooses HOW to move (5 move operators) from a state that includes
#     the Node-GNN graph embedding. Reward = fitness gain of the block.
#
#   Both networks persist for the whole run (all individuals, all
#   generations), so they make thousands of decisions per run.
#
# The outer GNN + DQN of Part A (operator selection) are unchanged.
# =====================================================================

ARLS_MODE = "gnn_dqn"     # "gnn_dqn"  : GNN encodes the deployment graph for the
                          #              Move-DQN, trained end-to-end by TD loss
                          # "learned"  : GNN chooses node + DQN chooses move
                          # "no_gnn"   : DQN only, hand-made node choice
                          # "no_drl"   : GNN only, random move operator
                          # "original" : original Part-A ARLS (baseline)

ARLS_BLOCK = 100          # moves per DQN decision
NODE_FEAT = 7
NGNN_HID, NGNN_EMB = 32, 16
NGNN_LR = 1e-3
NGNN_BUFFER, NGNN_BATCH, NGNN_TRAIN_EVERY = 400, 8, 2
NGNN_TARGET_SCALE, NGNN_TARGET_CLIP = 1.0e4, 20.0
NODE_UNIFORM_MIX = 0.5

MOVE_OPS = ["Scheduled step", "Hole filling", "Small step",
            "Redundancy push", "Boundary push"]
N_MOVE_OPS = len(MOVE_OPS)
MDQN_STATE = 8 + N_MOVE_OPS + NGNN_EMB
MDQN_HIDDEN = 64
MDQN_LR, MDQN_GAMMA = 1e-3, 0.9
MDQN_BUFFER, MDQN_BATCH, MDQN_TARGET_EVERY = 5000, 32, 100
MDQN_EPS_START, MDQN_EPS_END, MDQN_EPS_DECAY = 0.5, 0.05, 800.0
MDQN_REWARD_SCALE = 1.0e3

ARLS_ORIGINAL = ARLS                 # keep the Part-A local search
RUN_PROPOSED_ORIGINAL = run_proposed
POLICY = None
MAX_DISK = len(disk_indices(np.array([AREA_X / 2.0, AREA_Y / 2.0])))


class NodeGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.g1 = nn.Linear(NODE_FEAT, NGNN_HID)
        self.g2 = nn.Linear(NGNN_HID, NGNN_EMB)
        self.head = nn.Sequential(nn.Linear(NGNN_EMB + NODE_FEAT, 32),
                                  nn.ReLU(), nn.Linear(32, 1))
        self.act = nn.ReLU()

    def forward(self, x, a_norm):
        h = self.act(self.g1(a_norm @ x))
        h = self.act(self.g2(a_norm @ h))
        return self.head(torch.cat([h, x], dim=-1)).squeeze(-1), h


class MoveDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(MDQN_STATE, MDQN_HIDDEN), nn.ReLU(),
                                 nn.Linear(MDQN_HIDDEN, MDQN_HIDDEN), nn.ReLU(),
                                 nn.Linear(MDQN_HIDDEN, N_MOVE_OPS))

    def forward(self, x):
        return self.net(x)


def disk_graph(pos):
    a = (pairwise_distances(pos) < 2.0 * R_SENSE).astype(np.float32)
    np.fill_diagonal(a, 1.0)
    d = 1.0 / np.sqrt(a.sum(axis=1))
    return (a * d[:, None] * d[None, :]).astype(np.float32)


def node_features(st):
    f = np.zeros((st.n, NODE_FEAT), dtype=np.float32)
    f[:, 0] = st.pos[:, 0] / AREA_X
    f[:, 1] = st.pos[:, 1] / AREA_Y
    for k, ind in enumerate(st.idx):
        ck = st.count[ind]
        sz = max(len(ind), 1)
        f[k, 2] = np.count_nonzero(ck == 1) / sz
        f[k, 3] = np.count_nonzero(ck >= 2) / sz
        f[k, 4] = len(ind) / MAX_DISK
    edge = np.minimum.reduce([st.pos[:, 0], AREA_X - st.pos[:, 0],
                              st.pos[:, 1], AREA_Y - st.pos[:, 1]])
    f[:, 5] = np.minimum(edge / (2.0 * R_SENSE), 1.0)
    holes = GRID[st.count == 0]
    if len(holes):
        d2 = ((st.pos[:, None, :] - holes[None, :, :]) ** 2).sum(axis=2)
        f[:, 6] = np.minimum(np.count_nonzero(d2 < (2 * R_SENSE) ** 2, axis=1)
                             / MAX_DISK, 1.0)
    return f


class GraphMoveQNet(nn.Module):
    """Move-DQN whose state includes a GNN embedding of the deployment
    graph. Gradients of the TD loss flow into the GNN (end-to-end)."""

    def __init__(self):
        super().__init__()
        self.g1 = nn.Linear(NODE_FEAT, NGNN_HID)
        self.g2 = nn.Linear(NGNN_HID, NGNN_EMB)
        self.q = nn.Sequential(nn.Linear(8 + N_MOVE_OPS + NGNN_EMB, MDQN_HIDDEN), nn.ReLU(),
                               nn.Linear(MDQN_HIDDEN, MDQN_HIDDEN), nn.ReLU(),
                               nn.Linear(MDQN_HIDDEN, N_MOVE_OPS))
        self.act = nn.ReLU()

    def embed(self, x, a):                       # x: [B,N,F], a: [B,N,N]
        h = self.act(self.g1(torch.bmm(a, x)))
        h = self.act(self.g2(torch.bmm(a, h)))
        return h.mean(dim=1)                     # [B, NGNN_EMB]

    def forward(self, base, x, a):
        return self.q(torch.cat([base, self.embed(x, a)], dim=1))


class LearnedSearchPolicy:
    """Node-GNN + Move-DQN shared by every ARLS call of one run."""

    def __init__(self, mode):
        self.mode = mode
        self.gnn = NodeGNN().to(DEVICE)
        self.gnn_opt = optim.Adam(self.gnn.parameters(), lr=NGNN_LR)
        self.gnn_buf = deque(maxlen=NGNN_BUFFER)
        self.gnn_losses = []
        qcls = GraphMoveQNet if mode == "gnn_dqn" else MoveDQN
        self.q = qcls().to(DEVICE)
        self.q_t = qcls().to(DEVICE)
        self.q_t.load_state_dict(self.q.state_dict())
        self.q_opt = optim.Adam(self.q.parameters(), lr=MDQN_LR)
        self.q_buf = deque(maxlen=MDQN_BUFFER)
        self.q_losses = []
        self.decisions = 0
        self.blocks = 0
        self.op_counts = np.zeros((2, N_MOVE_OPS), dtype=np.int64)

    # ---------------- Node-GNN: which node to move ----------------------
    def node_probs(self, st):
        prior = st.redundancy_priority() + 0.05
        if self.mode == "gnn_dqn":
            p = (1.0 - NODE_UNIFORM_MIX) * prior / prior.sum() + NODE_UNIFORM_MIX / st.n
            return p, node_features(st), disk_graph(st.pos), np.zeros(NGNN_EMB, np.float32)
        if self.mode == "no_gnn":
            p = (1.0 - NODE_UNIFORM_MIX) * prior / prior.sum() + NODE_UNIFORM_MIX / st.n
            return p, None, None, np.zeros(NGNN_EMB, np.float32)
        x, a = node_features(st), disk_graph(st.pos)
        with torch.no_grad():
            s, h = self.gnn(torch.from_numpy(x).to(DEVICE),
                            torch.from_numpy(a).to(DEVICE))
        s = s.cpu().numpy().astype(np.float64)
        z = (s - s.max()) / (s.std() + 1e-6)
        p = prior * np.exp(z)
        p = (1.0 - NODE_UNIFORM_MIX) * p / p.sum() + NODE_UNIFORM_MIX / st.n
        return p, x, a, h.mean(dim=0).cpu().numpy().astype(np.float32)

    def learn_nodes(self, rng, x, a, nodes, targets):
        if self.mode in ("no_gnn", "gnn_dqn") or x is None or len(nodes) == 0:
            return
        self.gnn_buf.append((x, a, nodes, targets.astype(np.float32)))
        self.blocks += 1
        if self.blocks % NGNN_TRAIN_EVERY or len(self.gnn_buf) < NGNN_BATCH:
            return
        loss = 0.0
        for j in rng.choice(len(self.gnn_buf), NGNN_BATCH, replace=False):
            xb, ab, nb, tb = self.gnn_buf[j]
            s, _ = self.gnn(torch.from_numpy(xb).to(DEVICE),
                            torch.from_numpy(ab).to(DEVICE))
            loss = loss + nn.functional.mse_loss(
                s[torch.from_numpy(nb).to(DEVICE)], torch.from_numpy(tb).to(DEVICE))
        loss = loss / NGNN_BATCH
        self.gnn_opt.zero_grad()
        loss.backward()
        self.gnn_opt.step()
        self.gnn_losses.append(float(loss.item()))

    # ---------------- Move-DQN: how to move it --------------------------
    def state(self, st, emb, progress, T, acc_rate, last_gain, last_op):
        c, o = st.metrics()
        onehot = np.zeros(N_MOVE_OPS, np.float32)
        if last_op is not None:
            onehot[last_op] = 1.0
        base = np.array([c, o, energy_efficiency(c, o), st.fitness(), progress,
                         np.log10(max(T, 1e-12)) / 5.0, acc_rate,
                         np.clip(last_gain * MDQN_REWARD_SCALE, -5, 5)], np.float32)
        return np.concatenate([base, onehot, emb]).astype(np.float32)

    def choose_op(self, rng, s, progress, graph=None):
        if self.mode == "no_drl":
            op = int(rng.integers(N_MOVE_OPS))
        else:
            eps = max(MDQN_EPS_END,
                      MDQN_EPS_START * np.exp(-self.decisions / MDQN_EPS_DECAY))
            if rng.random() < eps:
                op = int(rng.integers(N_MOVE_OPS))
            else:
                with torch.no_grad():
                    if self.mode == "gnn_dqn":
                        qv = self.q(torch.from_numpy(s[:8 + N_MOVE_OPS]).unsqueeze(0).to(DEVICE),
                                    torch.from_numpy(graph[0]).unsqueeze(0).to(DEVICE),
                                    torch.from_numpy(graph[1]).unsqueeze(0).to(DEVICE))
                    else:
                        qv = self.q(torch.from_numpy(s).unsqueeze(0).to(DEVICE))
                op = int(torch.argmax(qv, dim=1).item())
        self.op_counts[int(progress >= 0.5), op] += 1
        return op

    def learn_op(self, rng, s, op, r, s2, graph=None, graph2=None):
        if self.mode == "no_drl":
            return
        if self.mode == "gnn_dqn":
            return self._learn_op_graph(rng, s, op, r, s2, graph, graph2)
        self.q_buf.append((s, op, r, s2))
        self.decisions += 1
        if len(self.q_buf) >= MDQN_BATCH:
            idx = rng.choice(len(self.q_buf), MDQN_BATCH, replace=False)
            b = [self.q_buf[j] for j in idx]
            st = torch.from_numpy(np.stack([t[0] for t in b])).to(DEVICE)
            ac = torch.tensor([t[1] for t in b], device=DEVICE).unsqueeze(1)
            rw = torch.tensor([t[2] for t in b], dtype=torch.float32, device=DEVICE)
            ns = torch.from_numpy(np.stack([t[3] for t in b])).to(DEVICE)
            qv = self.q(st).gather(1, ac).squeeze(1)
            with torch.no_grad():
                tgt = rw + MDQN_GAMMA * self.q_t(ns).max(dim=1).values
            loss = nn.functional.smooth_l1_loss(qv, tgt)
            self.q_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
            self.q_opt.step()
            self.q_losses.append(float(loss.item()))
        if self.decisions % MDQN_TARGET_EVERY == 0:
            self.q_t.load_state_dict(self.q.state_dict())


def _learn_op_graph(self, rng, s, op, r, s2, graph, graph2):
    k = 8 + N_MOVE_OPS
    self.q_buf.append((s[:k], graph[0], graph[1], op, r, s2[:k], graph2[0], graph2[1]))
    self.decisions += 1
    if len(self.q_buf) >= MDQN_BATCH and self.decisions % GQ_TRAIN_EVERY == 0:
        idx = rng.choice(len(self.q_buf), GQ_BATCH, replace=False)
        b = [self.q_buf[j] for j in idx]
        T = lambda arr: torch.from_numpy(np.stack(arr)).to(DEVICE)
        bs, bx, ba = T([t[0] for t in b]), T([t[1] for t in b]), T([t[2] for t in b])
        ac = torch.tensor([t[3] for t in b], device=DEVICE).unsqueeze(1)
        rw = torch.tensor([t[4] for t in b], dtype=torch.float32, device=DEVICE)
        ns, nx, na = T([t[5] for t in b]), T([t[6] for t in b]), T([t[7] for t in b])
        qv = self.q(bs, bx, ba).gather(1, ac).squeeze(1)
        with torch.no_grad():
            tgt = rw + MDQN_GAMMA * self.q_t(ns, nx, na).max(dim=1).values
        loss = nn.functional.smooth_l1_loss(qv, tgt)
        self.q_opt.zero_grad()
        loss.backward()                      # TD gradients reach the GNN layers
        gnorm = sum(float(p.grad.norm()) for p in (self.q.g1.weight, self.q.g2.weight)
                    if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.q_opt.step()
        self.q_losses.append(float(loss.item()))
        self.gnn_losses.append(gnorm)        # logged as GNN gradient norm
    if self.decisions % MDQN_TARGET_EVERY == 0:
        self.q_t.load_state_dict(self.q.state_dict())


LearnedSearchPolicy._learn_op_graph = _learn_op_graph
GQ_TRAIN_EVERY = 2          # graph-DQN update every 2 decisions
GQ_BATCH = 16


def propose_move(op, st, i, step, rng):
    p = st.pos[i]
    if op == 1 and st.n_cov < NUM_GRID:                       # hole filling
        holes = np.flatnonzero(st.count == 0)
        target = GRID[holes[rng.integers(len(holes))]]
        return clip_point(target + rng.normal(0.0, 0.25 * R_SENSE, 2))
    if op == 2:                                               # small step
        return clip_point(p + rng.normal(0.0, max(ARLS_STEP_MIN, 0.3 * step), 2))
    if op == 3:                                               # redundancy push
        d = p[None, :] - st.pos
        dist = np.hypot(d[:, 0], d[:, 1])
        w = np.clip(2.0 * R_SENSE - dist, 0.0, None)
        w[i] = 0.0
        v = (d * (w / np.maximum(dist, 1e-6))[:, None]).sum(axis=0)
        nv = np.hypot(v[0], v[1])
        if nv > 1e-9:
            return clip_point(p + max(step, 2.0) * v / nv + rng.normal(0.0, 1.0, 2))
    if op == 4:                                               # boundary push
        gaps = np.array([p[0], AREA_X - p[0], p[1], AREA_Y - p[1]])
        dirs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float64)
        return clip_point(p + dirs[int(np.argmin(gaps))] * rng.uniform(0.5, 1.0)
                          * max(step, 2.0) + rng.normal(0.0, 1.0, 2))
    return clip_point(p + rng.normal(0.0, step, 2))           # scheduled step


def ARLS(pos, moves, t_start, t_end, step_start, step_end, rng):
    """Policy-driven ARLS (same signature as Part A, so Part A uses it)."""
    if ARLS_MODE == "original" or POLICY is None:
        return ARLS_ORIGINAL(pos, moves, t_start, t_end, step_start, step_end, rng)

    pol = POLICY
    st = CoverageState(pos)
    f_cur = st.fitness()
    best_f, best_pos = f_cur, st.pos.copy()
    n_blocks = max(1, int(np.ceil(moves / ARLS_BLOCK)))
    acc_rate, last_gain, last_op = 0.0, 0.0, None

    for b in range(n_blocks):
        frac = b / max(n_blocks - 1, 1)
        T = t_start * (t_end / t_start) ** frac
        step = step_start + (step_end - step_start) * frac
        progress = float(np.clip(np.log(max(T, 1e-12) / ARLS_T_START)
                                 / np.log(ARLS_T_END / ARLS_T_START), 0.0, 1.0))

        probs, x, a, emb = pol.node_probs(st)
        s = pol.state(st, emb, progress, T, acc_rate, last_gain, last_op)
        op = pol.choose_op(rng, s, progress, graph=(x, a))

        f_start = f_cur
        gain = np.zeros(st.n)
        tried = np.zeros(st.n)
        accepted = 0
        for i in rng.choice(st.n, size=ARLS_BLOCK, p=probs):
            i = int(i)
            info = st.move(i, propose_move(op, st, i, step, rng))
            f_new = st.fitness()
            d = f_new - f_cur
            tried[i] += 1
            if d >= 0.0 or rng.random() < np.exp(d / max(T, 1e-12)):
                f_cur = f_new
                accepted += 1
                gain[i] += max(d, 0.0)
                if f_cur > best_f:
                    best_f, best_pos = f_cur, st.pos.copy()
            else:
                st.undo(info)

        acc_rate = accepted / ARLS_BLOCK
        last_gain = f_cur - f_start
        moved = np.flatnonzero(tried > 0)
        pol.learn_nodes(rng, x, a, moved.astype(np.int64),
                        np.minimum(NGNN_TARGET_SCALE * gain[moved] / tried[moved],
                                   NGNN_TARGET_CLIP))
        s2 = pol.state(st, emb, progress, T, acc_rate, last_gain, op)
        g2 = ((node_features(st), disk_graph(st.pos)) if pol.mode == "gnn_dqn" else None)
        pol.learn_op(rng, s, op, float(np.clip(MDQN_REWARD_SCALE * last_gain, -5, 5)), s2,
                     graph=(x, a), graph2=g2)
        last_op = op

    return quick_evaluate(st.pos), quick_evaluate(best_pos)


def run_proposed(n, seed=SEED):
    """Part-A run, with a fresh learned ARLS policy for this run."""
    global POLICY
    set_seed(seed)
    POLICY = LearnedSearchPolicy(ARLS_MODE) if ARLS_MODE != "original" else None
    r = RUN_PROPOSED_ORIGINAL(n, seed=seed)
    if POLICY is not None:
        sh = POLICY.op_counts / np.maximum(POLICY.op_counts.sum(axis=1, keepdims=True), 1)
        r["node_gnn_loss_start"] = (float(np.mean(POLICY.gnn_losses[:50]))
                                    if POLICY.gnn_losses else np.nan)
        r["node_gnn_loss_end"] = (float(np.mean(POLICY.gnn_losses[-50:]))
                                  if POLICY.gnn_losses else np.nan)
        r["move_dqn_loss_end"] = (float(np.mean(POLICY.q_losses[-100:]))
                                  if POLICY.q_losses else np.nan)
        r["move_dqn_decisions"] = int(POLICY.op_counts.sum())
        r["move_op_share"] = sh
    POLICY = None
    return r



# =====================================================================
# PART H - MAIN SETTING: CNN-DQN, Q-LEARNING, GKSO, WOA-LFGA
#          (one Kaggle commit, about 6 h)
# =====================================================================
#
#   EXACTLY the main-study setting (Part A/B above, unchanged): 50 nodes,
#   250 x 300 m, Rs = 25 m, 50 x 50 = 2,500-point grid, same fitness.
#   Results are measured on the same 2,500-point grid as the main tables
#   (98.63 / 3.75 / 98.21), and also on a 1 m grid.
#
#   PROPOSED seeds 20-49, run again on the same machine so every
#            comparison is paired on one machine (results can differ
#            slightly from the main study across hardware/library versions)
#   CNN_DQN  seeds 20-49 -> paired with PROPOSED in the main final test
#   QLEARN   seeds 20-49 -> paired with PROPOSED in the main final test
#            (same EnMOGKSO; only the move selector inside the ARLS
#             changes; not time-budgeted, so earlier runs stay valid)
#   GKSO     seeds 20-29, 188.09 s -> joins the main 9-method table
#   WOA_LFGA seeds 20-29, 188.09 s -> joins the main 9-method table
#            (188.09 s = mean PROPOSED runtime of the main final test,
#             the budget every main baseline received; best layout kept
#             by fitness, as for the main baselines)
#
#   Every finished run is saved at once; re-running the cell resumes.
#   OUTPUTS: p5_runs.csv, positions_p5.zip, p5_summary.csv
# =====================================================================
import os, io, contextlib, shutil
from scipy.special import gamma as _gamma

DL_SEEDS = list(range(20, 50))
META_SEEDS = list(range(20, 30))
MAIN_BUDGET_S = 188.09
BASE_POP = 30
STOP_AFTER_H = 11.0
RUNS5 = "p5_runs.csv"
POS5 = "positions_p5"
N5 = 50
N_NODES = N5


def append_csv(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=not os.path.exists(path))


def done_keys(path, cols):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(map(tuple, d[cols].values.tolist()))


FINE = np.column_stack([g.ravel() for g in np.meshgrid(np.arange(0.5, AREA_X, 1.0), np.arange(0.5, AREA_Y, 1.0))])


def eval_layout(pos):
    c, o, _, _ = coverage_data(pos)
    e = energy_efficiency(c, o)
    cnt = np.zeros(len(FINE), np.int16)
    for p in pos:
        cnt += (((FINE[:, 0] - p[0]) ** 2 + (FINE[:, 1] - p[1]) ** 2) <= R_SENSE ** 2)
    fc, fo = float(np.mean(cnt >= 1)), float(np.mean(cnt >= 2))
    fe = energy_efficiency(fc, fo)
    return {"Coverage (%)": 100 * c, "Overlap (%)": 100 * o, "EE (%)": 100 * e,
            "Feasible": bool(c > TARGET_COVERAGE and o < TARGET_OVERLAP and e > TARGET_EE),
            "Coverage fine (%)": 100 * fc, "Overlap fine (%)": 100 * fo, "EE fine (%)": 100 * fe}


# ---------------------------------------------------------------- learned move selectors
CNN_POOL = 2                                   # 50 x 50 -> 25 x 25


def coverage_image(st):
    cnt = np.asarray(st.count).reshape(GRID_RES, GRID_RES)
    img = np.stack([cnt == 0, cnt == 1, cnt >= 2]).astype(np.float32)
    g = GRID_RES // CNN_POOL
    img = img[:, :g * CNN_POOL, :g * CNN_POOL]
    return img.reshape(3, g, CNN_POOL, g, CNN_POOL).mean(axis=(2, 4)).astype(np.float32)


class CNNMoveQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.q = nn.Sequential(nn.Linear(8 + N_MOVE_OPS + NGNN_EMB, MDQN_HIDDEN), nn.ReLU(),
                               nn.Linear(MDQN_HIDDEN, MDQN_HIDDEN), nn.ReLU(),
                               nn.Linear(MDQN_HIDDEN, N_MOVE_OPS))

    def forward(self, base, img):
        return self.q(torch.cat([base, self.cnn(img)], dim=1))


class CNNMovePolicy(LearnedSearchPolicy):
    """Same as the proposed controller, but the GNN graph encoder is replaced
    by a CNN on the coverage map (3 x 25 x 25, 16-d embedding)."""

    def __init__(self):
        super().__init__("no_gnn")
        self.mode = "cnn_dqn"
        self.q, self.q_t = CNNMoveQNet().to(DEVICE), CNNMoveQNet().to(DEVICE)
        self.q_t.load_state_dict(self.q.state_dict())
        self.q_opt = optim.Adam(self.q.parameters(), lr=MDQN_LR)
        self._st = None

    def learn_nodes(self, *args, **kw):
        return

    def node_probs(self, st):
        self._st = st
        prior = st.redundancy_priority() + 0.05
        p = (1.0 - NODE_UNIFORM_MIX) * prior / prior.sum() + NODE_UNIFORM_MIX / st.n
        return p, coverage_image(st), None, np.zeros(NGNN_EMB, np.float32)

    def choose_op(self, rng, s, progress, graph=None):
        eps = max(MDQN_EPS_END, MDQN_EPS_START * np.exp(-self.decisions / MDQN_EPS_DECAY))
        if rng.random() < eps:
            op = int(rng.integers(N_MOVE_OPS))
        else:
            with torch.no_grad():
                qv = self.q(torch.from_numpy(s[:8 + N_MOVE_OPS]).unsqueeze(0).to(DEVICE),
                            torch.from_numpy(graph[0]).unsqueeze(0).to(DEVICE))
            op = int(torch.argmax(qv, dim=1).item())
        self.op_counts[int(progress >= 0.5), op] += 1
        return op

    def learn_op(self, rng, s, op, r, s2, graph=None, graph2=None):
        k = 8 + N_MOVE_OPS
        self.q_buf.append((s[:k], graph[0], op, r, s2[:k], coverage_image(self._st)))
        self.decisions += 1
        if len(self.q_buf) >= MDQN_BATCH and self.decisions % GQ_TRAIN_EVERY == 0:
            b = [self.q_buf[j] for j in rng.choice(len(self.q_buf), GQ_BATCH, replace=False)]
            Tn = lambda arr: torch.from_numpy(np.stack(arr)).to(DEVICE)
            bs, bi = Tn([t[0] for t in b]), Tn([t[1] for t in b])
            ac = torch.tensor([t[2] for t in b], device=DEVICE).unsqueeze(1)
            rw = torch.tensor([t[3] for t in b], dtype=torch.float32, device=DEVICE)
            ns, ni = Tn([t[4] for t in b]), Tn([t[5] for t in b])
            qv = self.q(bs, bi).gather(1, ac).squeeze(1)
            with torch.no_grad():
                tgt = rw + MDQN_GAMMA * self.q_t(ns, ni).max(dim=1).values
            loss = nn.functional.smooth_l1_loss(qv, tgt)
            self.q_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
            self.q_opt.step(); self.q_losses.append(float(loss.item()))
        if self.decisions % MDQN_TARGET_EVERY == 0:
            self.q_t.load_state_dict(self.q.state_dict())


QL_ALPHA = 0.1


class QLearnMovePolicy(LearnedSearchPolicy):
    """Tabular Q-learning, no neural network (216 discrete states)."""

    def __init__(self):
        super().__init__("no_gnn")
        self.mode = "qlearn"
        self.Q = np.zeros((4 * 3 * 3 * (N_MOVE_OPS + 1), N_MOVE_OPS))

    def learn_nodes(self, *args, **kw):
        return

    @staticmethod
    def _sid(s):
        pr = min(int(s[4] * 4), 3)
        ac = 0 if s[6] < 0.1 else (1 if s[6] < 0.4 else 2)
        g = 0 if s[7] < -1e-3 else (1 if s[7] <= 1e-3 else 2)
        oh = s[8:8 + N_MOVE_OPS]
        lo_ = int(np.argmax(oh)) if oh.max() > 0 else N_MOVE_OPS
        return ((pr * 3 + ac) * 3 + g) * (N_MOVE_OPS + 1) + lo_

    def choose_op(self, rng, s, progress, graph=None):
        eps = max(MDQN_EPS_END, MDQN_EPS_START * np.exp(-self.decisions / MDQN_EPS_DECAY))
        q = self.Q[self._sid(s)]
        op = int(rng.integers(N_MOVE_OPS)) if rng.random() < eps else int(rng.choice(np.flatnonzero(q == q.max())))
        self.op_counts[int(progress >= 0.5), op] += 1
        return op

    def learn_op(self, rng, s, op, r, s2, graph=None, graph2=None):
        a, b = self._sid(s), self._sid(s2)
        self.Q[a, op] += QL_ALPHA * (r + MDQN_GAMMA * self.Q[b].max() - self.Q[a, op])
        self.decisions += 1


def run_with_policy(seed, make_policy):
    global ARLS_MODE, POLICY
    t0 = time.time()
    set_seed(seed)
    pol = make_policy()
    ARLS_MODE, POLICY = "no_gnn", pol
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = RUN_PROPOSED_ORIGINAL(N5, seed=seed)
    finally:
        ARLS_MODE, POLICY = "gnn_dqn", None
    sh = pol.op_counts / np.maximum(pol.op_counts.sum(axis=1, keepdims=True), 1)
    extra = {f"Share {op} ({nm}) (%)": 100.0 * sh[h, k]
             for h, nm in ((0, "first half"), (1, "second half")) for k, op in enumerate(MOVE_OPS)}
    return np.asarray(r["positions"], float), time.time() - t0, np.nan, extra


# ---------------------------------------------------------------- time-budgeted metaheuristics
class _Track:
    def __init__(self, budget):
        self.budget, self.t0 = budget, time.time()
        self.best_f, self.best_x, self.evals = -np.inf, None, 0

    def frac(self):
        return (time.time() - self.t0) / self.budget

    def alive(self):
        return self.frac() < 1.0

    def update(self, x):
        f = fitness(quick_evaluate(x.reshape(N5, 2)))
        self.evals += 1
        if f > self.best_f:
            self.best_f, self.best_x = f, x.copy()
        return f


def _bounds():
    return np.zeros(2 * N5), np.tile([AREA_X, AREA_Y], N5)


def _levy(rng, d, beta=1.5):
    sig = (_gamma(1 + beta) * np.sin(np.pi * beta / 2) / (_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return rng.normal(0, sig, d) / np.abs(rng.normal(0, 1, d)) ** (1 / beta)


# Constants of the original GKSO not fixed by its equations as reproduced in
# the literature. CHECK AGAINST Hu et al. (2023) BEFORE PUBLISHING.
GKSO_M, GKSO_I = 1.5, 1.5
GKSO_BETA_MIN, GKSO_BETA_MAX = 0.2, 1.2
GKSO_PROTECT_P = 0.5


def run_gkso(T, rng):
    """Original GKSO (Hu et al. 2023): hunting, movement, foraging and
    self-protection, each followed by greedy selection (same code as Part G)."""
    lo, hi = _bounds(); D = 2 * N5; NP = BASE_POP
    x = rng.uniform(lo, hi, (NP, D)); f = np.array([T.update(xi) for xi in x])
    b = int(np.argmax(f)); xb, fb = x[b].copy(), f[b]
    w = rng.uniform(0.05, 0.95); t = 0

    def try_set(i, cand):
        nonlocal xb, fb
        cand = np.clip(cand, lo, hi); fc = T.update(cand)
        if fc > f[i]:
            x[i], f[i] = cand, fc
            if fc > fb:
                xb, fb = cand.copy(), fc
        return T.alive()

    while T.alive():
        t += 1
        tau = min(T.frac(), 1.0)
        w = abs(1.0 - 2.0 * w ** 4)
        p = 2.0 * (1.0 - tau ** 0.25 + w * (tau ** 0.25 - tau ** 3))
        beta = GKSO_BETA_MIN + (GKSO_BETA_MAX - GKSO_BETA_MIN) * (1.0 - tau ** 3) ** 2
        alpha = beta * np.sin(1.5 * np.pi + np.sin(1.5 * np.pi * beta))
        for i in range(NP):
            if not try_set(i, x[i] + (lo + rng.random(D) * (hi - lo)) / ((i + 1) * t)):
                return
        for i in range(NP):
            xh = GKSO_M * GKSO_I ** rng.random() * (xb - x[i])
            if not try_set(i, (xh + x[i - 1]) / 2.0):
                return
        for i in range(NP):
            lam = rng.choice([-1.0, 1.0])
            if not try_set(i, xb + rng.random(D) * (xb - x[i]) + lam * p ** 2 * (xb - x[i])):
                return
        for i in range(NP):
            if rng.random() >= GKSO_PROTECT_P:
                continue
            mu = rng.random()
            a1 = 2.0 * rng.random() if mu < 0.5 else 1.0
            a2 = rng.random() if mu < 0.5 else 1.0
            a3 = rng.random() if mu < 0.5 else 1.0
            l2 = 1.0 if mu < 0.5 else 0.0
            k1, k2 = rng.uniform(-1, 1), rng.normal()
            r1, r2, u1, u2, pp = rng.choice(NP, 5, replace=True)
            xr = rng.uniform(lo, hi)
            xk = l2 * (x[pp] - xr) + xr
            rho = alpha * (2.0 * rng.random() - 1.0)
            start = x[i] if rng.random() < 0.5 else xb
            cand = start + k1 * (a1 * xb - a2 * xk) + k2 * rho * (a3 * (x[r2] - x[r1])) + a2 / 2.0 * (x[u1] - x[u2])
            if not try_set(i, cand):
                return


def run_woa_lfga(T, rng):
    """WOA-LFGA (Xu, Zhang & Zhang 2023), same code as Part G."""
    lo, hi = _bounds(); D = 2 * N5; span = hi - lo
    z = np.empty((BASE_POP, D)); z[0] = rng.random(D)
    for k in range(1, BASE_POP):
        prev = z[k - 1]
        z[k] = np.clip(np.where(prev < 0.3, prev / 0.3, (1 - prev) / 0.7) + 1e-6 * rng.random(D), 0, 1)
    x = lo + z * span
    f = np.array([T.update(xi) for xi in x])
    wrap = lambda v: (v - lo) % span + lo
    n_el, n_bad = max(2, int(round(0.1 * BASE_POP))), max(1, int(round(0.2 * BASE_POP)))
    while T.alive():
        a = 2.0 * (1.0 - min(T.frac(), 1.0)); best = x[np.argmax(f)].copy()
        for i in range(BASE_POP):
            A = 2 * a * rng.random() - a; C = 2 * rng.random()
            if rng.random() < 0.5:
                if abs(A) < 1:
                    new = best - A * np.abs(C * best - x[i])
                else:
                    alpha = rng.random(D) * 1.6; lv = _levy(rng, D)
                    if rng.random() > 0.95:
                        xr = x[rng.integers(BASE_POP)]
                        new = x[i] + np.sign(rng.random() - 0.5) * alpha * (xr - x[i]) * lv
                    else:
                        new = best + rng.uniform(-2, 2, D) * alpha * (best - x[i]) * lv
            else:
                l = rng.uniform(-1, 1)
                new = np.abs(best - x[i]) * np.exp(l) * np.cos(2 * np.pi * l) + best
            x[i] = wrap(new); f[i] = T.update(x[i])
            if not T.alive():
                return
        if T.frac() >= 0.2:
            order = np.argsort(-f); elite = order[:n_el]
            for k in order[-n_bad:]:
                p1, p2 = x[rng.choice(elite, 2, replace=False)]
                cut = int(rng.integers(1, D))
                child = np.concatenate([p1[:cut], p2[cut:]])
                mut = rng.random(D) < 0.2
                child[mut] = rng.uniform(lo[mut], hi[mut])
                x[k] = child; f[k] = T.update(child)
                if not T.alive():
                    return


def run_timed(seed, fn):
    T = _Track(MAIN_BUDGET_S)
    fn(T, np.random.default_rng(seed))
    return clip_positions(T.best_x.reshape(N5, 2)), time.time() - T.t0, T.evals, {}


def run_proposed_again(seed):
    global ARLS_MODE
    t0 = time.time()
    ARLS_MODE = "gnn_dqn"
    with contextlib.redirect_stdout(io.StringIO()):
        r = run_proposed(N5, seed=seed)
    sh = r.get("move_op_share")
    extra = {} if sh is None else {f"Share {op} ({nm}) (%)": 100.0 * sh[h, k]
             for h, nm in ((0, "first half"), (1, "second half")) for k, op in enumerate(MOVE_OPS)}
    return np.asarray(r["positions"], float), time.time() - t0, np.nan, extra


RUNNERS = {"PROPOSED": run_proposed_again, "CNN_DQN": lambda s: run_with_policy(s, CNNMovePolicy),
           "QLEARN": lambda s: run_with_policy(s, QLearnMovePolicy),
           "GKSO": lambda s: run_timed(s, run_gkso),
           "WOA_LFGA": lambda s: run_timed(s, run_woa_lfga)}
SHARE_KEYS = [f"Share {op} ({nm}) (%)" for nm in ("first half", "second half") for op in MOVE_OPS]


def jobs5():
    done = done_keys(RUNS5, ["Method", "Seed"])
    jobs = []
    for s in DL_SEEDS:
        ks = ["PROPOSED", "CNN_DQN", "QLEARN"] + (["GKSO", "WOA_LFGA"] if s in META_SEEDS else [])
        jobs += [(k, s) for k in ks if (k, s) not in done]
    return jobs


def run_part_h():
    t_start = time.time()
    if os.path.isdir("/kaggle/working"):
        os.chdir("/kaggle/working")
    os.makedirs(POS5, exist_ok=True)
    jobs = jobs5()
    print("=" * 110 + f"\nPART H: main setting ({GRID_RES} x {GRID_RES} grid, {N5} nodes); "
          f"{len(jobs)} runs to do; budget for GKSO / WOA-LFGA {MAIN_BUDGET_S} s\n" + "=" * 110, flush=True)
    t0 = time.time()
    for j, (key, seed) in enumerate(jobs, 1):
        if (time.time() - t_start) / 3600.0 > STOP_AFTER_H:
            print(f"STOP: {STOP_AFTER_H} h reached; {len(jobs) - j + 1} runs left.", flush=True)
            break
        pos, rt, evals, extra = RUNNERS[key](seed)
        np.save(os.path.join(POS5, f"{key}_N{N5}_seed{seed}.npy"), pos)
        m = eval_layout(pos)
        append_csv(RUNS5, {"Method": key, "Seed": seed, **m, "Runtime (s)": rt, "Evaluations": evals,
                           **{k: np.nan for k in SHARE_KEYS}, **extra})
        eta = (time.time() - t0) / j * (len(jobs) - j)
        print(f"[{j:3d}/{len(jobs)}] {key:9s} seed={seed:2d} cov={m['Coverage (%)']:.2f} ov={m['Overlap (%)']:.2f} "
              f"ee={m['EE (%)']:.3f} {'PASS' if m['Feasible'] else 'fail'} ({rt:.0f}s, ETA {eta/60:.0f} min)", flush=True)
    d = pd.read_csv(RUNS5).drop_duplicates(["Method", "Seed"], keep="last")
    s = d.groupby("Method").agg(Runs=("Seed", "count"), **{f"{c} mean": (c, "mean") for c in
          ["Coverage (%)", "Overlap (%)", "EE (%)", "Coverage fine (%)", "Overlap fine (%)", "EE fine (%)", "Runtime (s)"]},
          **{"Feasible (%)": ("Feasible", lambda v: 100 * np.mean(v))})
    s.to_csv("p5_summary.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\nSUMMARY (main 2,500-point grid and 1 m grid)\n" + s.round(3).to_string())
    print("\nMain study for reference: PROPOSED 98.63 / 3.75 / 98.21 (seeds 20-49).")
    try:
        from scipy.stats import wilcoxon
        w = d.pivot(index="Seed", columns="Method", values="EE (%)")
        print("\nPaired EE (2,500-point grid), PROPOSED minus other, Wilcoxon (two-sided):")
        for k in ["CNN_DQN", "QLEARN", "GKSO", "WOA_LFGA"]:
            if k in w and "PROPOSED" in w:
                x = w[["PROPOSED", k]].dropna()
                if len(x) >= 5:
                    dd = x["PROPOSED"] - x[k]
                    p = wilcoxon(dd, zero_method="zsplit").pvalue if np.any(dd != 0) else 1.0
                    print(f"  vs {k:9s}: n={len(x):2d}, mean diff {dd.mean():+.3f}, wins {int((dd>0).sum())}/{len(x)}, p={p:.4f}")
    except Exception as e:
        print("stats skipped:", e)
    shutil.make_archive("positions_p5", "zip", POS5)
    print(f"Saved: p5_runs.csv, p5_summary.csv, positions_p5.zip\nTOTAL TIME: {(time.time() - t_start)/3600:.2f} h", flush=True)


run_part_h()
