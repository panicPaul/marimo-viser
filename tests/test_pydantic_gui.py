from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field

import marimo_3dv.gui.pydantic as pgui
from marimo_3dv import form_gui, json_gui
from marimo_3dv.gui.pydantic import (
    _DIRECT_JSON_EDITOR_KEY,
    PydanticGui,
    PydanticJsonGui,
)


class _RequiredModel(BaseModel):
    title: str
    count: int = Field(ge=0, le=10)


class _InnerModel(BaseModel):
    enabled: bool = True
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class _OuterModel(BaseModel):
    title: str = "viewer"
    inner: _InnerModel = _InnerModel()


class _TwoTabOuterModel(BaseModel):
    left: _InnerModel = _InnerModel()
    right: _InnerModel = _InnerModel()


class _ArrayModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    weights: Float[np.ndarray, "2 2"] = np.zeros((2, 2))


class _TensorModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    weights: Float[torch.Tensor, "3"] = torch.zeros(3)  # noqa: UP037


class _PathModel(BaseModel):
    source: Path = Path("README.md")


class _FallbackModel(BaseModel):
    when: date


class _ListFallbackModel(BaseModel):
    save_at_steps: list[int]


class _MagnitudeSliderModel(BaseModel):
    large: int = Field(default=100, ge=0, le=1000)
    small: float = Field(default=0.01, ge=0.0, le=1.0)


class _MagnitudeNumberModel(BaseModel):
    integer_zero: int = 0
    integer_ten: int = 10
    float_small: float = 0.01


class _HelpTextModel(BaseModel):
    count: int = Field(1, description="Visible field description.")
    ratio: float = 0.1
    """Visible attribute docstring."""


class _NestedHelpInnerModel(BaseModel):
    enabled: bool = True


class _NestedHelpOuterModel(BaseModel):
    inner: _NestedHelpInnerModel = _NestedHelpInnerModel()
    """Outer nested helper text."""


class _ArgsDocModel(BaseModel):
    """Model with tyro-style field notes.

    Args:
        count: Docstring fallback text.
        ratio: This should lose to the Field description.
    """

    count: int = 1
    ratio: float = Field(0.1, description="Field description wins.")


class _InlineCommentModel(BaseModel):
    count: int = 1  # Inline comment help text.


class _StringListFallbackModel(BaseModel):
    tags: list[str] = ["alpha", "beta"]


class _NestedStringListModel(BaseModel):
    groups: list[list[str]] = [["alpha"], ["beta"]]


class _NestedFallbackOuterModel(BaseModel):
    title: str = "viewer"
    inner: _NestedStringListModel = _NestedStringListModel()


@pytest.fixture
def notebook_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pgui.mo, "running_in_notebook", lambda: True)


def test_form_gui_returns_submit_gated_form(
    notebook_mode: None,
) -> None:
    generated = form_gui(_RequiredModel)

    assert generated.value is None
    assert isinstance(generated.element, PydanticGui)
    assert generated.validate({"title": "demo", "count": 4}) is None


def test_form_gui_can_return_live_pydantic_gui(
    notebook_mode: None,
) -> None:
    generated = form_gui(_RequiredModel, live_update=True)

    assert isinstance(generated, PydanticGui)
    assert "border: 1px solid #f2c94c" in generated.text
    assert generated.value == _RequiredModel(title="", count=0)
    assert (
        generated.validate_frontend_value({"title": "demo", "count": 4}) is None
    )


def test_json_gui_returns_submit_gated_json_editor(
    notebook_mode: None,
) -> None:
    generated = json_gui(_RequiredModel)

    assert generated.value is None
    assert isinstance(generated.element, PydanticJsonGui)
    assert generated.validate('{"title": "demo", "count": 4}') is None


def test_form_gui_uses_tyro_in_script_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _RequiredModel(title="cli", count=3)

    monkeypatch.setattr(pgui.mo, "running_in_notebook", lambda: False)
    monkeypatch.setattr(pgui.tyro, "cli", lambda *args, **kwargs: expected)

    generated = form_gui(_RequiredModel)

    assert generated.value == expected
    assert generated.element is None
    assert generated.validate is None


def test_json_gui_uses_tyro_default_in_script_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_cli(*args: object, **kwargs: object) -> _RequiredModel:
        captured.update(kwargs)
        return kwargs["default"]  # type: ignore[index]

    monkeypatch.setattr(pgui.mo, "running_in_notebook", lambda: False)
    monkeypatch.setattr(pgui.tyro, "cli", _fake_cli)

    generated = json_gui(
        _RequiredModel,
        value={"title": "seeded", "count": 2},
    )

    assert generated.value == _RequiredModel(title="seeded", count=2)
    assert captured["default"] == _RequiredModel(title="seeded", count=2)


def test_pydantic_json_gui_clone_uses_internal_state() -> None:
    generated = PydanticJsonGui(_RequiredModel)
    generated._update('{"title": "demo", "count": 4}')

    cloned = generated._clone()

    assert isinstance(cloned, PydanticJsonGui)
    assert cloned.value == _RequiredModel(title="demo", count=4)


def test_required_fields_without_defaults_initialize_and_validate() -> None:
    generated = PydanticGui(_RequiredModel, include_json_editor=False)

    assert generated.value == _RequiredModel(title="", count=0)
    assert generated.validate_frontend_value(generated._value_frontend) is None
    assert generated._last_payload == {"title": "", "count": 0}


def test_nested_models_use_nested_tabs() -> None:
    generated = PydanticGui(_OuterModel, include_json_editor=False)

    assert isinstance(generated.elements["inner"], PydanticGui)
    assert "marimo-tabs" in generated.text


def test_numpy_arrays_round_trip_through_matrix_updates() -> None:
    generated = PydanticGui(_ArrayModel, include_json_editor=False)
    frontend_value = dict(generated._value_frontend)
    frontend_value["weights"] = [[1.0, 2.0], [3.0, 4.0]]

    generated._update(frontend_value)

    assert isinstance(generated.value.weights, np.ndarray)
    assert generated.value.weights.shape == (2, 2)
    assert np.allclose(
        generated.value.weights, np.array([[1.0, 2.0], [3.0, 4.0]])
    )


def test_torch_arrays_round_trip_through_matrix_updates() -> None:
    generated = PydanticGui(_TensorModel, include_json_editor=False)
    frontend_value = dict(generated._value_frontend)
    frontend_value["weights"] = [[1.0], [2.0], [3.0]]

    generated._update(frontend_value)

    assert isinstance(generated.value.weights, torch.Tensor)
    assert generated.value.weights.shape == (3,)
    assert torch.allclose(
        generated.value.weights, torch.tensor([1.0, 2.0, 3.0])
    )


def test_path_fields_use_file_browser_and_nonexistent_defaults_fall_back() -> (
    None
):
    generated = PydanticGui(
        _PathModel,
        value={"source": Path("/definitely/not/here/file.txt")},
        include_json_editor=False,
    )

    assert type(generated.elements["source"]).__name__ == "file_browser"
    assert generated.elements["source"]._component_args["initial-path"] == str(
        Path.cwd()
    )


def test_numeric_slider_step_is_inferred_from_default_magnitude() -> None:
    generated = PydanticGui(_MagnitudeSliderModel, include_json_editor=False)

    assert generated.elements["large"]._component_args["step"] == 10
    assert generated.elements["small"]._component_args["step"] == 0.001


def test_numeric_number_step_stays_whole_for_integers() -> None:
    generated = PydanticGui(_MagnitudeNumberModel, include_json_editor=False)

    assert generated.elements["integer_zero"]._component_args["step"] == 1
    assert generated.elements["integer_ten"]._component_args["step"] == 1
    assert generated.elements["float_small"]._component_args["step"] == 0.001


def test_form_gui_renders_field_help_text_from_description_and_docstring() -> (
    None
):
    generated = PydanticGui(_HelpTextModel, include_json_editor=False)

    assert "Visible field description." in generated.text
    assert "Visible attribute docstring." in generated.text


def test_form_gui_prefers_field_description_over_docstring_args() -> None:
    generated = PydanticGui(_ArgsDocModel, include_json_editor=False)

    assert "Docstring fallback text." in generated.text
    assert "Field description wins." in generated.text
    assert "This should lose to the Field description." not in generated.text


def test_form_gui_renders_inline_comment_help_text_via_tyro() -> None:
    generated = PydanticGui(_InlineCommentModel, include_json_editor=False)

    assert "Inline comment help text." in generated.text


def test_json_gui_renders_nested_help_text() -> None:
    generated = PydanticJsonGui(_NestedHelpOuterModel)

    assert "Outer nested helper text." in generated.text


def test_fallback_text_fields_allow_pydantic_autocast() -> None:
    generated = PydanticGui(_FallbackModel, include_json_editor=False)
    frontend_value = dict(generated._value_frontend)
    frontend_value["when"] = "2026-04-07"

    generated._update(frontend_value)

    assert generated.value == _FallbackModel(when=date(2026, 4, 7))


def test_fallback_text_fields_parse_json_literals_when_needed() -> None:
    generated = PydanticGui(_ListFallbackModel, include_json_editor=False)
    frontend_value = dict(generated._current_frontend_value())
    frontend_value["save_at_steps"] = "[]"

    generated._update(frontend_value)

    assert generated.value == _ListFallbackModel(save_at_steps=[])


def test_submit_validation_accepts_structured_text_field_defaults(
    notebook_mode: None,
) -> None:
    generated = form_gui(_StringListFallbackModel)

    assert (
        generated.validate(generated.element._current_frontend_value()) is None
    )


def test_submit_validation_accepts_nested_structured_text_field_defaults(
    notebook_mode: None,
) -> None:
    generated = form_gui(_NestedFallbackOuterModel)

    assert (
        generated.validate(generated.element._current_frontend_value()) is None
    )


def test_json_gui_parses_nested_models() -> None:
    generated = PydanticJsonGui(_OuterModel)

    frontend_value = dict(generated._current_frontend_value())
    frontend_value[_DIRECT_JSON_EDITOR_KEY] = '{\n  "title": "json"\n}'
    frontend_value["inner"] = '{\n  "enabled": false,\n  "threshold": 0.8\n}'

    generated._update(frontend_value)

    assert generated.value == _OuterModel(
        title="json",
        inner=_InnerModel(enabled=False, threshold=0.8),
    )


def test_nested_json_gui_uses_nested_tabs() -> None:
    generated = PydanticJsonGui(_OuterModel)

    assert isinstance(generated.elements["inner"], PydanticJsonGui)
    assert "marimo-tabs" in generated.text


def test_partial_nested_updates_do_not_require_full_frontend_payload() -> None:
    generated = PydanticGui(_OuterModel, include_json_editor=False)
    nested = generated.elements["inner"]

    generated._update({"inner": {"threshold": 0.8}})

    assert isinstance(nested, PydanticGui)
    assert generated.value == _OuterModel(
        title="viewer",
        inner=_InnerModel(enabled=True, threshold=0.8),
    )


def test_inactive_nested_tabs_keep_live_state_on_submit() -> None:
    generated = PydanticGui(_TwoTabOuterModel, include_json_editor=False)
    left = generated.elements["left"]

    assert isinstance(left, PydanticGui)

    pgui._set_local_frontend_value(left.elements["threshold"], 0.8)

    generated._update({"right": {"enabled": False}})

    assert generated.value == _TwoTabOuterModel(
        left=_InnerModel(enabled=True, threshold=0.8),
        right=_InnerModel(enabled=False, threshold=0.5),
    )
