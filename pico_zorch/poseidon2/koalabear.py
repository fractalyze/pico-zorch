# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Pico's Poseidon2-KoalaBear-16 hash stack, on zorch's agnostic engine.

Round constants are Pico's own RC_16_30 table — NOT Plonky3's default
instance — reduced with from_wrapped_u32 and split per
pico_poseidon2kb_init(): rows 0..4 initial external, rows 4..24 column 0
internal, rows 24..28 terminal external (rows 28..30 unused). Canonical
values below; source of truth is
https://github.com/brevis-network/pico/blob/v2.0.0/vm/src/primitives/mod.rs
and the byte-match against golden/ pins them.

The Merkle stack mirrors the shapes Plonky3's `MerkleTreeMmcs` composes for
this field: `PaddingFreeSponge<_, 16, 8, 8>` row leaves and
`TruncatedPermutation<_, 2, 8, 16>` node compression
(brevis-network/Plonky3@7fbe1908, the fork Pico v2.0.0 vendors).
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

# _ER counts one external half (4 initial + 4 terminal full rounds).
_WIDTH, _ER, _IR, _ALPHA = 16, 4, 20, 3

_P = 2130706433  # KoalaBear: 2^31 - 2^24 + 1

# The internal-layer diagonal [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8,
# 1/2^24, -1/2^8, -1/8, -1/16, -1/2^24] in canonical form, using
# p - 1 = 127 * 2^24, so -1/2^n = (p-1) >> n (the fork's INTERNAL_DIAG_MONTY_16
# construction in koala-bear/src/poseidon2.rs).
_INTERNAL_DIAG = [
    _P - 2,
    1,
    2,
    (_P + 1) >> 1,
    3,
    4,
    (_P - 1) >> 1,
    _P - 3,
    _P - 4,
    _P - ((_P - 1) >> 8),
    _P - ((_P - 1) >> 3),
    _P - 127,
    (_P - 1) >> 8,
    (_P - 1) >> 3,
    (_P - 1) >> 4,
    127,
]

_EXTERNAL_INITIAL = [
    [2110014213, 1834258172, 59956341, 602290050, 640767983, 1273192703, 1716033721, 1606702601, 1629166855, 1466015491, 1498308946, 713668661, 911757408, 1969905919, 1979238293, 1794341933],
    [1576153071, 759122502, 1036959013, 1131812921, 1080754908, 1949408060, 893583089, 2019677373, 997898123, 580640471, 1146913827, 842931656, 548879852, 1477848281, 1444941483, 81826002],
    [27673397, 1563933798, 1440025885, 184445025, 467944927, 1396647410, 1575877922, 1173146968, 137125468, 765010148, 633675867, 2037803363, 442683395, 1895729703, 541515871, 1783382863],
    [511150051, 905036909, 1542089893, 245668751, 2025460432, 201609705, 286217151, 1962769130, 388865749, 949993437, 631295399, 1244250808, 606038199, 1052034398, 73007766, 441497720],
]

_EXTERNAL_TERMINAL = [
    [229095348, 669525034, 879650602, 1035997899, 1210110952, 1018506770, 668761744, 1479380761, 1536021911, 358993854, 579904113, 1301438367, 1494809376, 199241497, 1927597676, 459457801],
    [1688530738, 1580733335, 313275084, 75564132, 649367796, 498033244, 809417226, 2014500394, 1441571576, 648901076, 1098718697, 1424913749, 93709442, 1108922178, 1515566129, 1804479751],
    [820046587, 1393386250, 535112142, 101075586, 672377010, 1920315467, 1913164407, 2029526876, 498565387, 384320012, 1981614152, 1001118340, 217111764, 90290953, 1772368609, 449253662],
    [1414224440, 225847443, 939375845, 95643305, 1307865609, 1182150076, 615850007, 1863868773, 803582265, 1331270426, 772319366, 1482092434, 1772266066, 1741635435, 1530411808, 84217151],
]

_INTERNAL_RC = [
    1196780786, 36046858, 1374600958, 1747514347, 766236642, 1648402910,
    1418914503, 1286942262, 859661334, 1548195514, 104929687, 61203351,
    1502431934, 130182653, 1175952680, 1564209905, 1165280628, 1170945922,
    792169465, 2066954820,
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
    """The Poseidon2-KoalaBear-16 permutation Pico's whole stack rides on."""
    return Poseidon2(koalabear16_params())


def koalabear16_merkle(
    rate: int = 8, out: int = 8, arity: int = 2, chunk: int = 8
) -> tuple[Sponge, Compression, MerkleTree]:
    """`(sponge, compressor, tree)` in Plonky3's MerkleTreeMmcs shapes."""
    perm = koalabear16_perm()
    sponge = Sponge(perm, SpongeParams(rate=rate, out=out))
    comp = Compression(perm, CompressionParams(arity=arity, chunk=chunk))
    return sponge, comp, MerkleTree(sponge, comp)
