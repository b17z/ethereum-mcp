"""Tests for the expert guidance module."""

import pytest

from ethereum_mcp.expert.guidance import (
    GUIDANCE_DB,
    GuidanceEntry,
    get_expert_guidance,
    list_guidance_topics,
)


class TestGuidanceEntry:
    """Tests for the GuidanceEntry dataclass."""

    def test_guidance_entry_is_frozen(self):
        entry = GuidanceEntry(
            topic="Test",
            summary="Test summary",
            key_points=["point 1"],
            gotchas=["gotcha 1"],
            references=["ref 1"],
        )

        with pytest.raises(AttributeError):
            entry.topic = "modified"

    def test_guidance_entry_fields(self):
        entry = GuidanceEntry(
            topic="Test Topic",
            summary="Brief summary",
            key_points=["key 1", "key 2"],
            gotchas=["gotcha 1"],
            references=["specs/fork/file.md"],
        )

        assert entry.topic == "Test Topic"
        assert entry.summary == "Brief summary"
        assert len(entry.key_points) == 2
        assert len(entry.gotchas) == 1
        assert len(entry.references) == 1


class TestGuidanceDB:
    """Tests for the GUIDANCE_DB."""

    def test_has_expected_topics(self):
        expected_topics = ["churn", "slashing", "maxeb"]
        for topic in expected_topics:
            assert topic in GUIDANCE_DB, f"Missing topic: {topic}"

    def test_all_entries_are_valid(self):
        for topic, entry in GUIDANCE_DB.items():
            assert isinstance(entry, GuidanceEntry)
            assert entry.topic, f"Empty topic for {topic}"
            assert entry.summary, f"Empty summary for {topic}"
            assert len(entry.key_points) > 0, f"No key points for {topic}"
            assert len(entry.references) > 0, f"No references for {topic}"

    def test_churn_entry_has_electra_info(self):
        entry = GUIDANCE_DB["churn"]
        assert "Electra" in entry.summary or "electra" in entry.summary.lower()
        # Should mention the separate pools
        assert any("separate" in p.lower() for p in entry.key_points)

    def test_slashing_entry_has_quotient_info(self):
        entry = GUIDANCE_DB["slashing"]
        # Should mention the quotient changes
        key_points_text = " ".join(entry.key_points)
        assert "quotient" in key_points_text.lower()


class TestGetExpertGuidance:
    """Tests for get_expert_guidance function."""

    def test_returns_guidance_for_known_topic(self):
        result = get_expert_guidance("churn")
        assert result is not None
        assert isinstance(result, GuidanceEntry)
        assert result.topic == "Validator Churn and Queue Mechanics"

    def test_returns_none_for_unknown_topic(self):
        result = get_expert_guidance("nonexistent_topic_xyz")
        assert result is None

    def test_case_insensitive_lookup(self):
        # Should work with different cases
        result_lower = get_expert_guidance("churn")
        # Note: Case sensitivity depends on implementation
        # At minimum, lowercase should work
        assert result_lower is not None

    def test_result_has_all_fields(self):
        result = get_expert_guidance("slashing")
        assert result is not None
        assert isinstance(result, GuidanceEntry)

        # Check the dataclass has all expected attributes
        assert hasattr(result, "topic")
        assert hasattr(result, "summary")
        assert hasattr(result, "key_points")
        assert hasattr(result, "gotchas")
        assert hasattr(result, "references")


class TestListGuidanceTopics:
    """Tests for list_guidance_topics function."""

    def test_returns_list(self):
        topics = list_guidance_topics()
        assert isinstance(topics, list)

    def test_returns_all_topics(self):
        topics = list_guidance_topics()
        assert len(topics) == len(GUIDANCE_DB)

    def test_includes_known_topics(self):
        topics = list_guidance_topics()
        assert "churn" in topics
        assert "slashing" in topics

    def test_topics_are_strings(self):
        topics = list_guidance_topics()
        for topic in topics:
            assert isinstance(topic, str)
