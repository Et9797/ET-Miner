"""Tests for the synthetic dataset generator (CPU-only)."""

import json

import numpy as np
import pytest

from et_miner.synthetic import (
    PRESETS,
    SynthSpec,
    check_preset_purpose,
    estimate_level_sizes,
    generate_csr,
    generate_transactions,
)


def _small_spec(**overrides):
    base = dict(
        name="t",
        n_rows=5_000,
        vocab_size=300,
        zipf_a=1.0,
        row_len_mean=8,
        row_len_max=30,
        min_support=0.02,
        seed=7,
    )
    base.update(overrides)
    return SynthSpec(**base)


class TestGenerateCSR:
    def test_deterministic(self):
        spec = _small_spec()
        a = generate_csr(spec)
        b = generate_csr(spec)
        np.testing.assert_array_equal(a.indptr, b.indptr)
        np.testing.assert_array_equal(a.indices, b.indices)

    def test_seed_changes_output(self):
        a = generate_csr(_small_spec(seed=1))
        b = generate_csr(_small_spec(seed=2))
        assert not (len(a.indices) == len(b.indices) and np.array_equal(a.indices, b.indices))

    def test_rows_unique_and_sorted(self):
        """The tier-vs-oracle contract: every row strictly increasing."""
        data = generate_csr(_small_spec(motif_count=2, motif_size=4, motif_penetration=0.05))
        for r in range(data.n_rows):
            row = data.indices[data.indptr[r] : data.indptr[r + 1]]
            assert np.all(np.diff(row) > 0), f"row {r} not strictly increasing: {row}"

    def test_items_within_vocab(self):
        data = generate_csr(_small_spec())
        assert data.indices.min() >= 0
        assert data.indices.max() < data.n_cols

    def test_row_length_cap_without_motifs(self):
        spec = _small_spec(row_len_max=12)
        data = generate_csr(spec)
        lens = np.diff(data.indptr)
        # Dedup can only shorten background rows, never lengthen them.
        assert lens.max() <= 12

    def test_zipf_head_is_popular(self):
        data = generate_csr(_small_spec(n_rows=20_000))
        counts = np.bincount(data.indices, minlength=data.n_cols)
        assert counts[:10].mean() > 10 * counts[100:200].mean()

    def test_planted_rows_contain_motifs(self):
        spec = _small_spec(motif_count=2, motif_size=4, motif_penetration=0.05)
        data = generate_csr(spec)
        assert len(data.planted) == 2
        for motif, planted_count in data.planted:
            assert planted_count == int(round(0.05 * spec.n_rows))
            motif_arr = np.array(motif)
            # Realized support: rows containing ALL motif items must be >= planted.
            hits = 0
            for r in range(data.n_rows):
                row = data.indices[data.indptr[r] : data.indptr[r + 1]]
                if np.isin(motif_arr, row).all():
                    hits += 1
            assert hits >= planted_count

    def test_motifs_disjoint(self):
        data = generate_csr(_small_spec(motif_count=3, motif_size=5, motif_penetration=0.02))
        seen: set[int] = set()
        for motif, _ in data.planted:
            assert not (seen & set(motif))
            seen |= set(motif)

    def test_skew_clusters_nnz_in_front(self):
        spec = _small_spec(n_rows=10_000, skew_frac=0.2, skew_mult=4.0)
        data = generate_csr(spec)
        lens = np.diff(data.indptr)
        n_skew = 2_000
        assert lens[:n_skew].mean() > 2.0 * lens[n_skew:].mean()


class TestTransactionsFrame:
    def test_frame_matches_csr(self):
        spec = _small_spec()
        df, data = generate_transactions(spec)
        assert df.height == spec.n_rows
        first = df["items"][0].to_list()
        np.testing.assert_array_equal(np.array(first), data.indices[data.indptr[0] : data.indptr[1]])


class TestPresets:
    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_purpose_checks_pass(self, name):
        check_preset_purpose(PRESETS[name])

    def test_stress_k2_estimate_exceeds_100m_pairs(self):
        est = estimate_level_sizes(PRESETS["stress_k2"])
        assert est["expected_k2_candidates"] > 100_000_000

    def test_smoke_reaches_k4_and_is_ea_tractable(self):
        """Calibrated in-session: efficient-apriori mines the smoke preset in
        under a second, reaching K=6 with ~700 itemsets. Guard the shape so a
        drive-by preset change cannot silently blow the CI/oracle budget."""
        spec = PRESETS["smoke"]
        est = estimate_level_sizes(spec)
        assert est["expected_k1_survivors"] <= 300  # EA cost driver
        assert est["expected_k2_candidates"] <= 50_000
        # Motifs guarantee depth: 5-item motifs above min_count => K>=5 exists.
        assert spec.motif_size >= 5
        assert int(round(spec.motif_penetration * spec.n_rows)) >= spec.min_count

    def test_vacuous_motif_spec_rejected(self):
        bad = _small_spec(motif_count=1, motif_size=4, motif_penetration=0.001, min_support=0.02)
        with pytest.raises(AssertionError, match="vacuously"):
            check_preset_purpose(bad)


class TestCLI:
    def test_writes_parquet_and_sidecar(self, tmp_path):
        from et_miner.synthetic import _main

        assert _main(["--preset", "smoke", "--out", str(tmp_path)]) == 0
        assert (tmp_path / "smoke.parquet").exists()
        sidecar = json.loads((tmp_path / "smoke.json").read_text())
        assert sidecar["spec"]["name"] == "smoke"
        assert len(sidecar["planted"]) == PRESETS["smoke"].motif_count
        assert sidecar["nnz"] > 0
