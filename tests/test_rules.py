"""Tests for generate_rules() and Rule dataclass."""

import polars as pl

from et_miner import generate_rules
from et_miner.rules import Rule


class TestRuleDataclass:
    """Tests for Rule dataclass."""

    def test_rule_attributes(self):
        """Test Rule has all expected attributes."""
        rule = Rule(
            lhs=[1, 2],
            rhs=[3],
            support=0.5,
            confidence=0.8,
            lift=1.2
        )

        assert rule.lhs == [1, 2]
        assert rule.rhs == [3]
        assert rule.support == 0.5
        assert rule.confidence == 0.8
        assert rule.lift == 1.2

    def test_rule_equality(self):
        """Test Rule equality comparison."""
        rule1 = Rule([1], [2], 0.5, 0.8, 1.0)
        rule2 = Rule([1], [2], 0.5, 0.8, 1.0)
        rule3 = Rule([1], [3], 0.5, 0.8, 1.0)

        assert rule1 == rule2
        assert rule1 != rule3


class TestGenerateRulesBasic:
    """Basic functionality tests for generate_rules()."""

    def test_generate_rules_basic(self, known_frequent_itemsets):
        """Test basic rule generation."""
        rules = generate_rules(known_frequent_itemsets, min_confidence=0.5)

        assert isinstance(rules, list)
        assert len(rules) > 0
        assert all(isinstance(r, Rule) for r in rules)

    def test_generate_rules_no_overlap(self, known_frequent_itemsets):
        """Test that lhs and rhs have no overlap."""
        rules = generate_rules(known_frequent_itemsets, min_confidence=0.0)

        for rule in rules:
            overlap = set(rule.lhs) & set(rule.rhs)
            assert len(overlap) == 0, f"Rule {rule.lhs} -> {rule.rhs} has overlap"

    def test_generate_rules_union_in_itemsets(self, known_frequent_itemsets):
        """Test that lhs ∪ rhs exists in frequent itemsets."""
        rules = generate_rules(known_frequent_itemsets, min_confidence=0.0)
        itemsets = {tuple(sorted(row["itemset"])) for row in known_frequent_itemsets.iter_rows(named=True)}

        for rule in rules:
            union = tuple(sorted(rule.lhs + rule.rhs))
            assert union in itemsets, f"Union {union} not in frequent itemsets"


class TestGenerateRulesFiltering:
    """Tests for confidence filtering in generate_rules()."""

    def test_generate_rules_high_confidence_filters(self, known_frequent_itemsets):
        """Test that high min_confidence filters rules."""
        rules_low = generate_rules(known_frequent_itemsets, min_confidence=0.0)
        rules_high = generate_rules(known_frequent_itemsets, min_confidence=0.9)

        assert len(rules_high) < len(rules_low)

    def test_generate_rules_no_rules_high_confidence(self, known_frequent_itemsets):
        """Test that very high min_confidence returns empty list."""
        rules = generate_rules(known_frequent_itemsets, min_confidence=1.1)

        assert rules == []


class TestGenerateRulesMetrics:
    """Tests for confidence and lift calculations."""

    def test_confidence_calculation(self):
        """Test confidence is calculated correctly."""
        # Simple case: [1] -> [2] from itemset [1,2]
        # support([1,2]) = 0.4, support([1]) = 0.6
        # confidence = 0.4 / 0.6 = 0.6667
        itemsets = pl.DataFrame({
            "itemset": [[1], [2], [1, 2]],
            "support": [0.6, 0.8, 0.4]
        })

        rules = generate_rules(itemsets, min_confidence=0.0)

        # Find rule [1] -> [2]
        rule = next((r for r in rules if r.lhs == [1] and r.rhs == [2]), None)
        assert rule is not None

        expected_confidence = 0.4 / 0.6
        assert abs(rule.confidence - expected_confidence) < 0.001

    def test_lift_calculation(self):
        """Test lift is calculated correctly."""
        # lift = confidence / support(rhs)
        itemsets = pl.DataFrame({
            "itemset": [[1], [2], [1, 2]],
            "support": [0.6, 0.8, 0.4]
        })

        rules = generate_rules(itemsets, min_confidence=0.0)

        rule = next((r for r in rules if r.lhs == [1] and r.rhs == [2]), None)
        assert rule is not None

        expected_confidence = 0.4 / 0.6
        expected_lift = expected_confidence / 0.8
        assert abs(rule.lift - expected_lift) < 0.001

    def test_all_rules_have_valid_metrics(self, known_frequent_itemsets):
        """Test all rules have valid metric values."""
        rules = generate_rules(known_frequent_itemsets, min_confidence=0.0)

        for rule in rules:
            assert 0.0 <= rule.support <= 1.0
            assert 0.0 <= rule.confidence <= 1.0
            assert rule.lift >= 0.0
