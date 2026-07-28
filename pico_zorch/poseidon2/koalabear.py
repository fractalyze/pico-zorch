# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Pico's Poseidon2-KoalaBear-16 hash stack, on zorch's agnostic engine.

Round constants are Plonky3's `default_koalabear_poseidon2_16` (canonical
u32), the parameterization Pico's proving backend instantiates; copied from
upstream Plonky3 p3_commit=4318eba062fd1cbca3dbe98904ad18ad950f3b49. Pico
v2.0.0 vendors its own fork at brevis-network/Plonky3@7fbe1908; TODO:
byte-match these constants against that fork's generation.

The Merkle stack mirrors the shapes Plonky3's `MerkleTreeMmcs` composes for
this field: `PaddingFreeSponge<_, 16, 8, 8>` row leaves and
`TruncatedPermutation<_, 2, 8, 16>` node compression.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np

from zk_dtypes import koalabear_mont as F
from zorch.commit.merkle import MerkleTree
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.params import Poseidon2Params
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

_WIDTH, _ER, _IR, _ALPHA = 16, 4, 20, 3

# Canonical-u32 constants from the pinned Plonky3 koala-bear poseidon2-16.
_EXTERNAL_INITIAL = [
    [
        2128964168,
        288780357,
        316938561,
        2126233899,
        426817493,
        1714118888,
        1045008582,
        1738510837,
        889721787,
        8866516,
        681576474,
        419059826,
        1596305521,
        1583176088,
        1584387047,
        1529751136,
    ],
    [
        1863858111,
        1072044075,
        517831365,
        1464274176,
        1138001621,
        428001039,
        245709561,
        1641420379,
        1365482496,
        770454828,
        693167409,
        757905735,
        136670447,
        436275702,
        525466355,
        1559174242,
    ],
    [
        1030087950,
        869864998,
        322787870,
        267688717,
        948964561,
        740478015,
        679816114,
        113662466,
        2066544572,
        1744924186,
        367094720,
        1380455578,
        1842483872,
        416711434,
        1342291586,
        1692058446,
    ],
    [
        1493348999,
        1113949088,
        210900530,
        1071655077,
        610242121,
        1136339326,
        2020858841,
        1019840479,
        678147278,
        1678413261,
        1361743414,
        61132629,
        1209546658,
        64412292,
        1936878279,
        1980661727,
    ],
]

_EXTERNAL_TERMINAL = [
    [
        1139268644,
        630873441,
        669538875,
        462500858,
        876500520,
        1214043330,
        383937013,
        375087302,
        636912601,
        307200505,
        390279673,
        1999916485,
        1518476730,
        1606686591,
        1410677749,
        1581191572,
    ],
    [
        1004269969,
        143426723,
        1747283099,
        1016118214,
        1749423722,
        66331533,
        1177761275,
        1581069649,
        1851371119,
        852520128,
        1499632627,
        1820847538,
        150757557,
        884787840,
        619710451,
        1651711087,
    ],
    [
        505263814,
        212076987,
        1482432120,
        1458130652,
        382871348,
        417404007,
        2066495280,
        1996518884,
        902934924,
        582892981,
        1337064375,
        1199354861,
        2102596038,
        1533193853,
        1436311464,
        2012303432,
    ],
    [
        839997195,
        1225781098,
        2011967775,
        575084315,
        1309329169,
        786393545,
        995788880,
        1702925345,
        1444525226,
        908073383,
        1811535085,
        1531002367,
        1635653662,
        1585100155,
        867006515,
        879151050,
    ],
]

_INTERNAL_RC = [
    1423960925,
    2101391318,
    1915532054,
    275400051,
    1168624859,
    1141248885,
    356546469,
    1165250474,
    1320543726,
    932505663,
    1204226364,
    1452576828,
    1774936729,
    926808140,
    1184948056,
    1186493834,
    843181003,
    185193011,
    452207447,
    510054082,
]

_INTERNAL_DIAG = [
    2130706431,
    1,
    2,
    1065353217,
    3,
    4,
    1065353216,
    2130706430,
    2130706429,
    2122383361,
    1864368129,
    2130706306,
    8323072,
    266338304,
    133169152,
    127,
]


def koalabear16_params() -> Poseidon2Params:
    internal_rc = np.zeros((_IR, _WIDTH), dtype=np.int64)
    internal_rc[:, 0] = np.array(_INTERNAL_RC, dtype=np.int64)
    return Poseidon2Params(
        width=_WIDTH,
        dtype=F,
        alpha=_ALPHA,
        external_rounds=_ER,
        internal_rounds=_IR,
        external_constants_initial=fnp.array(_EXTERNAL_INITIAL, dtype=F),
        external_constants_terminal=fnp.array(_EXTERNAL_TERMINAL, dtype=F),
        internal_constants=fnp.array(internal_rc, dtype=F),
        internal_diag=fnp.array(_INTERNAL_DIAG, dtype=F),
    )


def koalabear16_perm() -> Poseidon2:
    """The Poseidon2-KoalaBear-16 permutation Pico's hash stack is built on."""
    return Poseidon2(koalabear16_params())


def koalabear16_merkle(
    rate: int = 8, out: int = 8, arity: int = 2, chunk: int = 8
) -> tuple[Sponge, Compression, MerkleTree]:
    """`(sponge, compressor, tree)` in Plonky3's MerkleTreeMmcs shapes."""
    perm = koalabear16_perm()
    sponge = Sponge(perm, SpongeParams(rate=rate, out=out))
    comp = Compression(perm, CompressionParams(arity=arity, chunk=chunk))
    return sponge, comp, MerkleTree(sponge, comp)
