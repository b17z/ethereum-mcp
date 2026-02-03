"""Tests for the pydantic validation models."""

import pytest
from pydantic import ValidationError

from ethereum_mcp.models import (
    ClientListInput,
    ClientLookupInput,
    ConstantLookupInput,
    EipSearchInput,
    FunctionAnalysisInput,
    GuidanceInput,
    SearchInput,
)


class TestSearchInput:
    """Tests for SearchInput validation."""

    def test_valid_search(self):
        result = SearchInput(query="slashing penalty", fork="electra", limit=10)
        assert result.query == "slashing penalty"
        assert result.fork == "electra"
        assert result.limit == 10

    def test_defaults(self):
        result = SearchInput(query="test")
        assert result.fork is None
        assert result.limit == 5

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            SearchInput(query="")

    def test_query_too_long_rejected(self):
        with pytest.raises(ValidationError):
            SearchInput(query="x" * 1001)

    def test_invalid_fork_rejected(self):
        with pytest.raises(ValidationError):
            SearchInput(query="test", fork="invalid_fork")

    def test_fork_normalized_to_lowercase(self):
        result = SearchInput(query="test", fork="ELECTRA")
        assert result.fork == "electra"

    def test_limit_must_be_positive(self):
        with pytest.raises(ValidationError):
            SearchInput(query="test", limit=0)

    def test_limit_must_not_exceed_max(self):
        with pytest.raises(ValidationError):
            SearchInput(query="test", limit=51)


class TestConstantLookupInput:
    """Tests for ConstantLookupInput validation."""

    def test_valid_constant(self):
        result = ConstantLookupInput(constant_name="MAX_EFFECTIVE_BALANCE")
        assert result.constant_name == "MAX_EFFECTIVE_BALANCE"

    def test_lowercase_constant_rejected(self):
        with pytest.raises(ValidationError):
            ConstantLookupInput(constant_name="max_effective_balance")

    def test_mixed_case_constant_rejected(self):
        with pytest.raises(ValidationError):
            ConstantLookupInput(constant_name="MaxEffectiveBalance")

    def test_empty_constant_rejected(self):
        with pytest.raises(ValidationError):
            ConstantLookupInput(constant_name="")

    def test_fork_validation(self):
        result = ConstantLookupInput(constant_name="MAX_EFFECTIVE_BALANCE", fork="deneb")
        assert result.fork == "deneb"


class TestFunctionAnalysisInput:
    """Tests for FunctionAnalysisInput validation."""

    def test_valid_function(self):
        result = FunctionAnalysisInput(function_name="process_slashings")
        assert result.function_name == "process_slashings"

    def test_uppercase_function_rejected(self):
        with pytest.raises(ValidationError):
            FunctionAnalysisInput(function_name="ProcessSlashings")

    def test_function_with_leading_underscore(self):
        result = FunctionAnalysisInput(function_name="_internal_function")
        assert result.function_name == "_internal_function"

    def test_empty_function_rejected(self):
        with pytest.raises(ValidationError):
            FunctionAnalysisInput(function_name="")


class TestGuidanceInput:
    """Tests for GuidanceInput validation."""

    def test_valid_topic(self):
        result = GuidanceInput(topic="slashing")
        assert result.topic == "slashing"

    def test_empty_topic_rejected(self):
        with pytest.raises(ValidationError):
            GuidanceInput(topic="")

    def test_topic_too_long_rejected(self):
        with pytest.raises(ValidationError):
            GuidanceInput(topic="x" * 51)


class TestClientLookupInput:
    """Tests for ClientLookupInput validation."""

    def test_valid_client(self):
        result = ClientLookupInput(name="Geth")
        assert result.name == "geth"  # Normalized to lowercase

    def test_whitespace_stripped(self):
        result = ClientLookupInput(name="  lighthouse  ")
        assert result.name == "lighthouse"

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            ClientLookupInput(name="")


class TestClientListInput:
    """Tests for ClientListInput validation."""

    def test_valid_execution_layer(self):
        result = ClientListInput(layer="execution")
        assert result.layer == "execution"

    def test_valid_consensus_layer(self):
        result = ClientListInput(layer="consensus")
        assert result.layer == "consensus"

    def test_none_layer_allowed(self):
        result = ClientListInput(layer=None)
        assert result.layer is None

    def test_invalid_layer_rejected(self):
        with pytest.raises(ValidationError):
            ClientListInput(layer="both")

    def test_layer_normalized_to_lowercase(self):
        result = ClientListInput(layer="EXECUTION")
        assert result.layer == "execution"


class TestEipSearchInput:
    """Tests for EipSearchInput validation."""

    def test_valid_search(self):
        result = EipSearchInput(query="staking", eip_number="4844", limit=10)
        assert result.query == "staking"
        assert result.eip_number == "4844"
        assert result.limit == 10

    def test_defaults(self):
        result = EipSearchInput(query="test")
        assert result.eip_number is None
        assert result.limit == 5

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            EipSearchInput(query="")

    def test_non_numeric_eip_rejected(self):
        with pytest.raises(ValidationError):
            EipSearchInput(query="test", eip_number="EIP-4844")

    def test_numeric_eip_accepted(self):
        result = EipSearchInput(query="test", eip_number="1559")
        assert result.eip_number == "1559"
