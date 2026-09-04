"""Tests for repairing JSON-encoded fields in structured LLM output."""

from __future__ import annotations

from contextmine_core.analyzer.extractors.rules import ExtractionOutput
from contextmine_core.research.llm.structured import (
    coerce_json_string_fields,
    field_expects_structure,
    repair_structured_payload,
)
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

# The exact shape observed from Anthropic tool calling during a sync: the
# `rules` array arrived as a JSON string, so validation rejected the response
# and the extracted business rules were silently dropped.
ANTHROPIC_STRING_ARRAY_PAYLOAD = {
    "rules": (
        '[\n  {\n    "name": "Legacy series must stay sorted",\n'
        '    "description": "Series entries are ordered before rendering",\n'
        '    "category": "invariant",\n'
        '    "severity": "error",\n'
        '    "natural_language": "Series must be sorted before rendering",\n'
        '    "evidence_snippet": "if (!isSorted(series)) throw new Error()",\n'
        '    "start_line": 1137,\n    "end_line": 1139\n  }\n]'
    ),
    "reasoning": "Analyzed the series helper for ordering guarantees.",
}


class TestCoerceJsonStringFields:
    def test_decodes_json_string_array_into_list(self) -> None:
        coerced = coerce_json_string_fields(ANTHROPIC_STRING_ARRAY_PAYLOAD, ExtractionOutput)
        result = ExtractionOutput.model_validate(coerced)

        assert len(result.rules) == 1
        assert result.rules[0].name == "Legacy series must stay sorted"
        assert result.rules[0].start_line == 1137
        assert result.rules[0].end_line == 1139

    def test_leaves_string_fields_untouched(self) -> None:
        """A string field keeps its literal value even when it looks like JSON."""
        payload = {"rules": [], "reasoning": '["not", "a", "list"]'}

        coerced = coerce_json_string_fields(payload, ExtractionOutput)

        assert coerced["reasoning"] == '["not", "a", "list"]'
        assert ExtractionOutput.model_validate(coerced).reasoning == '["not", "a", "list"]'

    def test_leaves_wellformed_payloads_unchanged(self) -> None:
        payload = {
            "rules": [
                {
                    "name": "n",
                    "description": "d",
                    "category": "validation",
                    "severity": "error",
                    "natural_language": "nl",
                    "evidence_snippet": "s",
                    "start_line": 1,
                    "end_line": 2,
                }
            ],
            "reasoning": "r",
        }

        assert coerce_json_string_fields(payload, ExtractionOutput) == payload

    def test_leaves_malformed_json_for_validation_to_report(self) -> None:
        """Broken output must still surface as a validation error, not be hidden."""
        payload = {"rules": "[{unclosed", "reasoning": "r"}

        coerced = coerce_json_string_fields(payload, ExtractionOutput)

        assert coerced["rules"] == "[{unclosed"

    def test_returns_non_mapping_input_unchanged(self) -> None:
        assert coerce_json_string_fields(["a"], ExtractionOutput) == ["a"]
        assert coerce_json_string_fields(None, ExtractionOutput) is None

    def test_decodes_nested_model_and_dict_fields(self) -> None:
        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            inner: Inner
            mapping: dict[str, int]
            label: str

        payload = {"inner": '{"value": 5}', "mapping": '{"a": 1}', "label": "plain"}

        result = Outer.model_validate(coerce_json_string_fields(payload, Outer))

        assert result.inner.value == 5
        assert result.mapping == {"a": 1}
        assert result.label == "plain"

    def test_decodes_optional_structured_field(self) -> None:
        class WithOptional(BaseModel):
            items: list[int] | None = Field(default=None)

        coerced = coerce_json_string_fields({"items": "[1, 2, 3]"}, WithOptional)

        assert WithOptional.model_validate(coerced).items == [1, 2, 3]


class TestFieldExpectsStructure:
    def test_recognises_structured_annotations(self) -> None:
        assert field_expects_structure(list[int])
        assert field_expects_structure(dict[str, int])
        assert field_expects_structure(list[str] | None)

    def test_rejects_plain_scalar_annotations(self) -> None:
        assert not field_expects_structure(str)
        assert not field_expects_structure(int)
        assert not field_expects_structure(str | None)


def _tool_call_message(arguments: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ExtractionOutput",
                "args": arguments,
                "id": "toolu_test",
                "type": "tool_call",
            }
        ],
    )


class TestRepairStructuredPayload:
    def test_recovers_response_whose_array_arrived_as_json_string(self) -> None:
        repaired = repair_structured_payload(
            _tool_call_message(ANTHROPIC_STRING_ARRAY_PAYLOAD), ExtractionOutput
        )

        assert repaired is not None
        assert len(repaired.rules) == 1
        assert repaired.rules[0].name == "Legacy series must stay sorted"

    def test_returns_none_when_nothing_needed_repairing(self) -> None:
        """A payload that was already fine is left to the normal error path."""
        assert (
            repair_structured_payload(
                _tool_call_message({"rules": [], "reasoning": "r"}), ExtractionOutput
            )
            is None
        )

    def test_returns_none_for_unusable_payload(self) -> None:
        assert (
            repair_structured_payload(
                _tool_call_message({"rules": "[{unclosed", "reasoning": "r"}), ExtractionOutput
            )
            is None
        )

    def test_returns_none_without_tool_calls(self) -> None:
        assert repair_structured_payload(AIMessage(content=""), ExtractionOutput) is None
        assert repair_structured_payload(None, ExtractionOutput) is None
