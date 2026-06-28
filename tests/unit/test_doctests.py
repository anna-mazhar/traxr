"""Doctests on the curated public modules (the SDK-polish gate)."""

import doctest

import pytest

import traxr.metrics.manifest
import traxr.perturb.matrix
import traxr.results
import traxr.scoring
import traxr.trace.registry

CURATED_MODULES = [
    traxr.scoring,
    traxr.results,
    traxr.metrics.manifest,
    traxr.perturb.matrix,
    traxr.trace.registry,
]


@pytest.mark.parametrize("module", CURATED_MODULES, ids=lambda m: m.__name__)
def test_module_doctests(module):
    failures, _ = doctest.testmod(module, verbose=False)
    assert failures == 0
