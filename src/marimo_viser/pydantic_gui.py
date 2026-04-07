"""Generate marimo forms from Pydantic models."""

from __future__ import annotations

import asyncio
import ast
import html
import inspect
import json
import math
import textwrap
import tokenize
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import UnionType
from typing import Any, Generic, Literal, TypeVar, get_args, get_origin

import annotated_types
import marimo as mo
import numpy as np
import torch
from jaxtyping import AbstractArray
from marimo._plugins.core.web_component import JSONType
from marimo._plugins.ui._core.ui_element import UIElement
from marimo._runtime.commands import UpdateUIElementCommand
from marimo._runtime.context import ContextNotInitializedError, get_context
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

ModelT = TypeVar("ModelT", bound=BaseModel)

_MAX_MATRIX_CELLS = 100
_DEFAULT_SLIDER_STEPS = 100
_FORM_TAB = "Form"
_JSON_TAB = "JSON"
_JSON_EDITOR_KEY = "__json_editor__"
_TABS_KEY = "__tabs__"
_DIRECT_JSON_EDITOR_KEY = "__direct_json_editor__"


@dataclass(frozen=True)
class _NumericBounds:
    lower: int | float | None
    upper: int | float | None
    step: int | float | None
    strict_lower: bool
    strict_upper: bool


@dataclass(frozen=True)
class _ArrayShape:
    ndim: int
    fixed_shape: tuple[int, ...] | None


@dataclass(frozen=True)
class _FieldSpec:
    model_cls: type[BaseModel]
    name: str
    annotation: Any
    info: FieldInfo
    is_nested_model: bool

    def label(self) -> str:
        return self.info.title or self.name.replace("_", " ").capitalize()

    def help_text(self) -> str | None:
        if self.info.description:
            return self.info.description
        docstring_help = _docstring_help_for_field(self.model_cls, self.name)
        if docstring_help:
            return docstring_help
        return _attribute_docstring_for_field(self.model_cls, self.name)

    def parse_frontend_value(
        self,
        element: UIElement[Any, Any],
        frontend_value: JSONType,
        *,
        update_children: bool,
    ) -> Any:
        if isinstance(element, PydanticGui):
            payload, _ = element._payload_from_frontend(  # noqa: SLF001
                frontend_value,
                update_children=update_children,
                force_json=False,
            )
            return payload

        if update_children and element._value_frontend != frontend_value:  # noqa: SLF001
            element._update(frontend_value)  # noqa: SLF001
            python_value = element._value  # noqa: SLF001
        else:
            python_value = element._convert_value(frontend_value)  # noqa: SLF001
        return self.to_model_value(python_value)

    def to_model_value(self, value: Any) -> Any:
        if self.is_nested_model:
            return value
        if self.annotation is Path:
            return _coerce_path_value(self.info, value)
        if _is_array_annotation(self.annotation):
            return _coerce_array_value(self.annotation, value)
        if _uses_text_fallback(self.annotation) and isinstance(value, str):
            return _maybe_parse_json_text(value)
        return value


class PydanticGui(UIElement[dict[str, JSONType], ModelT | None], Generic[ModelT]):
    """Internal marimo UI element for Pydantic-backed forms."""

    _name = "marimo-dict"

    def __init__(
        self,
        model_cls: type[ModelT],
        *,
        value: ModelT | dict[str, Any] | None = None,
        label: str = "",
        include_json_editor: bool = True,
        on_change: Any | None = None,
    ) -> None:
        self._model_cls = model_cls
        self._label = label
        self._include_json_editor = include_json_editor
        self._last_active_tab = _FORM_TAB if include_json_editor else ""
        self._last_json_error: str | None = None
        self._last_payload = _resolve_initial_payload(model_cls, value)

        field_specs, field_elements, form_layout = _build_model_gui(
            model_cls=model_cls,
            payload=self._last_payload,
        )
        self._field_specs = field_specs
        self._field_elements = field_elements
        self._field_names = list(field_elements)

        elements: dict[str, UIElement[Any, Any]] = dict(field_elements)
        layout: Any = form_layout
        self._json_editor: UIElement[Any, Any] | None = None
        self._tabs: UIElement[Any, Any] | None = None

        if include_json_editor:
            json_editor = mo.ui.code_editor(
                value=_payload_to_json(self._last_payload),
                language="json",
                show_copy_button=True,
                debounce=False,
                label="",
            )
            tabs = mo.ui.tabs(
                {
                    _FORM_TAB: form_layout,
                    _JSON_TAB: json_editor,
                },
                value=_FORM_TAB,
                label="",
            )
            elements[_JSON_EDITOR_KEY] = json_editor
            elements[_TABS_KEY] = tabs
            self._json_editor = json_editor
            self._tabs = tabs
            layout = tabs

        self._elements = elements
        super().__init__(
            component_name=self._name,
            initial_value={
                name: element._initial_value_frontend  # noqa: SLF001
                for name, element in self._elements.items()
            },
            label=label,
            args={
                "element-ids": {
                    element._id: name for name, element in self._elements.items()  # noqa: SLF001
                }
            },
            slotted_html=layout.text,
            on_change=on_change,
        )
        for name, element in self._elements.items():
            element._register_as_view(parent=self, key=name)  # noqa: SLF001

    @property
    def elements(self) -> dict[str, UIElement[Any, Any]]:
        return self._elements

    def _clone(self) -> PydanticGui[ModelT]:
        return type(self)(
            self._model_cls,
            value=self._last_payload,
            label=self._label,
            include_json_editor=self._include_json_editor,
            on_change=self._on_change,
        )

    def _convert_value(self, value: dict[str, JSONType]) -> ModelT | None:
        self._apply_non_field_partials(value)
        merged_value = self._merged_frontend_value(value)
        source_tab = self._source_tab_name(merged_value, value)
        payload, json_error = self._payload_from_frontend(
            merged_value,
            update_children=True,
            force_json=source_tab == _JSON_TAB,
        )
        self._last_payload = payload

        if self._include_json_editor:
            active_tab = self._active_tab_name(merged_value)
            if source_tab == _FORM_TAB:
                self._sync_json_editor(payload)
            elif json_error is None:
                self._sync_field_controls(payload)
            self._last_active_tab = active_tab
            self._last_json_error = json_error

        if json_error is not None:
            return None
        return _validate_payload(self._model_cls, payload)

    def _payload_from_frontend(
        self,
        value: dict[str, JSONType],
        *,
        update_children: bool,
        force_json: bool,
    ) -> tuple[dict[str, Any], str | None]:
        payload = self._field_payload_from_frontend(
            value,
            update_children=update_children,
        )
        if not self._include_json_editor:
            return payload, None

        active_tab = self._active_tab_name(value)
        should_use_json = (
            force_json
            or active_tab == _JSON_TAB
            or self._last_active_tab == _JSON_TAB
        )
        if not should_use_json:
            return payload, None
        return self._merge_json_payload(value[_JSON_EDITOR_KEY], payload)

    def _field_payload_from_frontend(
        self,
        value: dict[str, JSONType],
        *,
        update_children: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in self._field_names:
            spec = self._field_specs[name]
            frontend_value = value.get(
                name,
                self._field_elements[name]._value_frontend,  # noqa: SLF001
            )
            payload[name] = spec.parse_frontend_value(
                self._field_elements[name],
                frontend_value,
                update_children=update_children,
            )
        return payload

    def _merged_frontend_value(
        self,
        value: dict[str, JSONType],
    ) -> dict[str, JSONType]:
        merged: dict[str, JSONType] = self._current_frontend_value()
        merged.update(value)
        return merged

    def _current_frontend_value(self) -> dict[str, JSONType]:
        current: dict[str, JSONType] = {}
        for name, element in self._elements.items():
            current[name] = element._value_frontend  # noqa: SLF001
        return current

    def _apply_non_field_partials(self, value: dict[str, JSONType]) -> None:
        if self._include_json_editor and self._json_editor is not None:
            if _JSON_EDITOR_KEY in value:
                _set_local_frontend_value(
                    self._json_editor,
                    value[_JSON_EDITOR_KEY],
                )
        if self._include_json_editor and self._tabs is not None:
            if _TABS_KEY in value:
                _set_local_frontend_value(
                    self._tabs,
                    value[_TABS_KEY],
                )

    def _merge_json_payload(
        self,
        json_text: JSONType,
        base_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        if not isinstance(json_text, str):
            return base_payload, "JSON editor value must be a string."

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as exc:
            return base_payload, f"json: {exc.msg}"

        if not isinstance(parsed, dict):
            return base_payload, "json: top-level JSON value must be an object."

        merged = dict(base_payload)
        for name in self._field_names:
            if name in parsed:
                merged[name] = _merge_json_value(merged.get(name), parsed[name])
        return merged, None

    def validate_frontend_value(
        self, value: dict[str, JSONType] | None
    ) -> str | None:
        if value is None:
            return None

        merged_value = self._merged_frontend_value(value)

        payload, json_error = self._payload_from_frontend(
            merged_value,
            update_children=False,
            force_json=True,
        )
        if json_error is not None:
            return json_error

        if self._include_json_editor:
            active_tab = self._active_tab_name(merged_value)
            if active_tab == _JSON_TAB:
                self._sync_field_controls(payload)
            else:
                self._sync_json_editor(payload)

        try:
            self._model_cls.model_validate(payload)
        except ValidationError as exc:
            return _format_validation_error(exc)
        return None

    def get_model(self) -> ModelT | None:
        """Return the current validated model value, if any."""
        return self.value

    def _sync_json_editor(self, payload: dict[str, Any]) -> None:
        if self._json_editor is None:
            return
        json_text = _payload_to_json(payload)
        self._sync_elements([(self._json_editor, json_text)])

    def _active_tab_name(self, value: dict[str, JSONType]) -> str:
        if not self._include_json_editor or self._tabs is None:
            return _FORM_TAB
        raw_value = value.get(_TABS_KEY, self._tabs._value_frontend)  # noqa: SLF001
        return self._tabs._convert_value(raw_value)  # noqa: SLF001

    def _source_tab_name(
        self,
        merged_value: dict[str, JSONType],
        incoming_value: dict[str, JSONType],
    ) -> str:
        active_tab = self._active_tab_name(merged_value)
        if (
            _TABS_KEY in incoming_value
            and active_tab != self._last_active_tab
            and self._last_active_tab
        ):
            return self._last_active_tab
        return active_tab

    def _sync_field_controls(self, payload: dict[str, Any]) -> None:
        updates: list[tuple[UIElement[Any, Any], JSONType]] = []
        for name in self._field_names:
            element = self._field_elements[name]
            self._collect_sync_updates(
                self._field_specs[name],
                element,
                payload.get(name),
                updates,
            )
        self._sync_elements(updates)

    def _collect_sync_updates(
        self,
        spec: _FieldSpec,
        element: UIElement[Any, Any],
        value: Any,
        updates: list[tuple[UIElement[Any, Any], JSONType]],
    ) -> None:
        if isinstance(element, PydanticGui):
            nested_payload = value if isinstance(value, dict) else {}
            nested_frontend = element._frontend_value_from_payload(nested_payload)
            _set_local_frontend_value(element, nested_frontend)
            for child_name in element._field_names:
                element._collect_sync_updates(
                    element._field_specs[child_name],
                    element._field_elements[child_name],
                    nested_payload.get(child_name),
                    updates,
                )
            return

        frontend_value = _frontend_value_for_element(
            spec,
            element,
            value,
        )
        updates.append((element, frontend_value))

    def _frontend_value_from_payload(
        self, payload: dict[str, Any]
    ) -> dict[str, JSONType]:
        frontend_value: dict[str, JSONType] = {}
        for name in self._field_names:
            frontend_value[name] = _frontend_value_for_element(
                self._field_specs[name],
                self._field_elements[name],
                payload.get(name),
            )

        if self._include_json_editor and self._json_editor is not None and self._tabs is not None:
            frontend_value[_JSON_EDITOR_KEY] = _payload_to_json(payload)
            frontend_value[_TABS_KEY] = self._tabs._value_frontend  # noqa: SLF001
        return frontend_value

    def _sync_elements(
        self,
        updates: list[tuple[UIElement[Any, Any], JSONType]],
    ) -> None:
        deduped: dict[str, tuple[UIElement[Any, Any], JSONType]] = {}
        for element, frontend_value in updates:
            deduped[element._id] = (element, frontend_value)  # noqa: SLF001

        for element, frontend_value in deduped.values():
            _set_local_frontend_value(element, frontend_value)

        try:
            ctx = get_context()
            kernel = ctx._kernel  # noqa: SLF001
            loop = asyncio.get_running_loop()
        except (ContextNotInitializedError, RuntimeError, AttributeError):
            return

        command = UpdateUIElementCommand.from_ids_and_values(
            [
                (element._id, frontend_value)  # noqa: SLF001
                for element, frontend_value in deduped.values()
            ]
        )
        loop.create_task(kernel.set_ui_element_value(command))


class PydanticJsonGui(UIElement[Any, ModelT | None], Generic[ModelT]):
    """Internal JSON editor for Pydantic-backed forms."""

    _name = "marimo-dict"

    def __init__(
        self,
        model_cls: type[ModelT],
        *,
        value: ModelT | dict[str, Any] | None = None,
        label: str = "",
        on_change: Any | None = None,
    ) -> None:
        self._model_cls = model_cls
        self._label = label
        self._initial_payload = _resolve_initial_payload(model_cls, value)
        self._last_payload = self._initial_payload
        self._last_error: str | None = None
        self._field_specs: dict[str, _FieldSpec] = {}
        self._field_names: list[str] = []
        self._direct_field_names: list[str] = []
        self._elements: dict[str, UIElement[Any, Any]] = {}
        self._editor: UIElement[Any, Any] | None = None
        self._tabs: UIElement[Any, Any] | None = None
        self._composite_mode = any(
            _is_model_type(info.annotation) for info in model_cls.model_fields.values()
        )

        if not self._composite_mode:
            editor = mo.ui.code_editor(
                value=_payload_to_json(self._initial_payload),
                language="json",
                show_copy_button=True,
                debounce=False,
                label=label,
            )
            self._editor = editor
            self._elements = {_DIRECT_JSON_EDITOR_KEY: editor}
            super().__init__(
                component_name=editor._args.component_name,  # noqa: SLF001
                initial_value=editor._args.initial_value,  # noqa: SLF001
                label=label,
                args=editor._component_args,  # noqa: SLF001
                slotted_html=editor._args.slotted_html,  # noqa: SLF001
                on_change=on_change,
            )
            return

        (
            self._field_specs,
            self._field_names,
            self._direct_field_names,
            self._elements,
            layout,
        ) = _build_model_json_gui(
            model_cls=model_cls,
            payload=self._initial_payload,
        )
        self._editor = self._elements.get(_DIRECT_JSON_EDITOR_KEY)
        self._tabs = self._elements.get(_TABS_KEY)
        super().__init__(
            component_name="marimo-dict",
            initial_value={
                name: element._initial_value_frontend  # noqa: SLF001
                for name, element in self._elements.items()
            },
            label=label,
            args={
                "element-ids": {
                    element._id: name for name, element in self._elements.items()  # noqa: SLF001
                }
            },
            slotted_html=layout.text,
            on_change=on_change,
        )
        for name, element in self._elements.items():
            element._register_as_view(parent=self, key=name)  # noqa: SLF001

    @property
    def elements(self) -> dict[str, UIElement[Any, Any]]:
        return self._elements

    def _clone(self) -> PydanticJsonGui[ModelT]:
        return type(self)(
            self._model_cls,
            value=self._last_payload,
            label=self._label,
            on_change=self._on_change,
        )

    def _convert_value(self, value: Any) -> ModelT | None:
        if self._composite_mode:
            self._apply_non_field_partials(value)
            merged_value = self._merged_frontend_value(value)
            payload, error = self._payload_from_frontend(
                merged_value,
                update_children=True,
            )
        else:
            payload, error = _json_text_to_payload(value)
        if error is None:
            self._last_payload = payload
        self._last_error = error
        if error is not None:
            return None
        return _validate_payload(self._model_cls, payload)

    def validate_frontend_value(self, value: Any | None) -> str | None:
        if value is None:
            return None
        if self._composite_mode:
            merged_value = self._merged_frontend_value(value)
            payload, error = self._payload_from_frontend(
                merged_value,
                update_children=False,
            )
        else:
            payload, error = _json_text_to_payload(value)
        if error is not None:
            return error
        try:
            self._model_cls.model_validate(payload)
        except ValidationError as exc:
            return _format_validation_error(exc)
        return None

    def _payload_from_frontend(
        self,
        value: dict[str, JSONType],
        *,
        update_children: bool,
    ) -> tuple[dict[str, Any], str | None]:
        payload: dict[str, Any] = {}

        if self._editor is not None:
            editor_value = value.get(
                _DIRECT_JSON_EDITOR_KEY,
                self._editor._value_frontend,  # noqa: SLF001
            )
            direct_payload, error = _json_text_to_payload(editor_value)
            if error is not None:
                return {}, error
            unexpected_keys = sorted(set(direct_payload) - set(self._direct_field_names))
            if unexpected_keys:
                return (
                    {},
                    "json: nested model fields must be edited in their own tabs.",
                )
            if update_children and self._editor._value_frontend != editor_value:  # noqa: SLF001
                _set_local_frontend_value(self._editor, editor_value)
            payload.update(direct_payload)

        for name in self._field_names:
            spec = self._field_specs[name]
            if not spec.is_nested_model:
                continue
            element = self._elements[name]
            assert isinstance(element, PydanticJsonGui)
            frontend_value = value.get(name, element._value_frontend)  # noqa: SLF001
            if element._composite_mode:
                if not isinstance(frontend_value, dict):
                    return {}, f"{name}: Expected an object."
                nested_payload, error = element._payload_from_frontend(
                    element._merged_frontend_value(frontend_value),
                    update_children=update_children,
                )
            else:
                nested_payload, error = _json_text_to_payload(frontend_value)
            if error is not None:
                return {}, f"{name}.{error}"
            if update_children and element._value_frontend != frontend_value:  # noqa: SLF001
                _set_local_frontend_value(element, frontend_value)
            payload[name] = nested_payload
        return payload, None

    def _merged_frontend_value(
        self,
        value: dict[str, JSONType],
    ) -> dict[str, JSONType]:
        merged = self._current_frontend_value()
        merged.update(value)
        return merged

    def _current_frontend_value(self) -> dict[str, JSONType]:
        if not self._composite_mode:
            assert self._editor is not None
            return {_DIRECT_JSON_EDITOR_KEY: self._editor._value_frontend}  # noqa: SLF001
        return {
            name: element._value_frontend  # noqa: SLF001
            for name, element in self._elements.items()
        }

    def _apply_non_field_partials(self, value: Any) -> None:
        if not self._composite_mode or not isinstance(value, dict):
            return
        if self._tabs is not None and _TABS_KEY in value:
            _set_local_frontend_value(self._tabs, value[_TABS_KEY])


def form_gui(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    label: str = "",
    submit_label: str = "Submit",
):
    """Create a submit-gated marimo UI from a Pydantic model."""
    model_gui = PydanticGui(
        model_cls,
        value=value,
        label=label,
        include_json_editor=False,
    )
    form = model_gui.form(
        label=label,
        submit_button_label=submit_label,
        validate=model_gui.validate_frontend_value,
    )
    if isinstance(form.element, PydanticGui):
        form.validate = form.element.validate_frontend_value
    return form


def json_gui(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    label: str = "",
    submit_label: str = "Submit",
):
    """Create a submit-gated JSON editor for a Pydantic model."""
    model_gui = PydanticJsonGui(model_cls, value=value, label=label)
    form = model_gui.form(
        label=label,
        submit_button_label=submit_label,
        validate=model_gui.validate_frontend_value,
    )
    if isinstance(form.element, PydanticJsonGui):
        form.validate = form.element.validate_frontend_value
    return form


def _build_model_gui(
    *,
    model_cls: type[BaseModel],
    payload: dict[str, Any],
) -> tuple[dict[str, _FieldSpec], dict[str, UIElement[Any, Any]], UIElement[Any, Any]]:
    field_specs: dict[str, _FieldSpec] = {}
    elements: dict[str, UIElement[Any, Any]] = {}
    direct_controls: list[Any] = []
    nested_tabs: dict[str, Any] = {}

    for name, info in model_cls.model_fields.items():
        annotation = info.annotation
        spec = _FieldSpec(
            model_cls=model_cls,
            name=name,
            annotation=annotation,
            info=info,
            is_nested_model=_is_model_type(annotation),
        )
        field_value = payload[name]
        element = _build_field_element(spec, field_value)
        field_specs[name] = spec
        elements[name] = element
        if spec.is_nested_model:
            nested_tabs[spec.label()] = _with_help_text(element, spec.help_text())
        else:
            direct_controls.append(_with_help_text(element, spec.help_text()))

    if nested_tabs:
        direct_controls.append(mo.ui.tabs(nested_tabs))

    if not direct_controls:
        layout = mo.md("")
    elif len(direct_controls) == 1:
        layout = direct_controls[0]
    else:
        layout = mo.vstack(direct_controls)

    return field_specs, elements, layout


def _build_model_json_gui(
    *,
    model_cls: type[BaseModel],
    payload: dict[str, Any],
) -> tuple[
    dict[str, _FieldSpec],
    list[str],
    list[str],
    dict[str, UIElement[Any, Any]],
    UIElement[Any, Any],
]:
    field_specs: dict[str, _FieldSpec] = {}
    field_names: list[str] = []
    direct_field_names: list[str] = []
    elements: dict[str, UIElement[Any, Any]] = {}
    direct_controls: list[Any] = []
    nested_tabs: dict[str, Any] = {}

    direct_payload: dict[str, Any] = {}
    for name, info in model_cls.model_fields.items():
        spec = _FieldSpec(
            model_cls=model_cls,
            name=name,
            annotation=info.annotation,
            info=info,
            is_nested_model=_is_model_type(info.annotation),
        )
        field_specs[name] = spec
        field_names.append(name)
        if spec.is_nested_model:
            nested_payload = payload[name] if isinstance(payload[name], dict) else {}
            nested_editor = PydanticJsonGui(
                info.annotation,
                value=nested_payload,
                label="",
            )
            elements[name] = nested_editor
            nested_tabs[spec.label()] = _with_help_text(
                nested_editor,
                spec.help_text(),
            )
        else:
            direct_field_names.append(name)
            direct_payload[name] = payload[name]

    if direct_field_names:
        editor = mo.ui.code_editor(
            value=_payload_to_json(direct_payload),
            language="json",
            show_copy_button=True,
            debounce=False,
            label="",
        )
        elements[_DIRECT_JSON_EDITOR_KEY] = editor
        direct_controls.append(editor)

    if nested_tabs:
        tabs = mo.ui.tabs(nested_tabs, label="")
        elements[_TABS_KEY] = tabs
        direct_controls.append(tabs)

    if not direct_controls:
        layout = mo.md("")
    elif len(direct_controls) == 1:
        layout = direct_controls[0]
    else:
        layout = mo.vstack(direct_controls)

    return field_specs, field_names, direct_field_names, elements, layout


def _build_field_element(
    spec: _FieldSpec,
    value: Any,
) -> UIElement[Any, Any]:
    annotation = spec.annotation
    label = spec.label()

    if _is_union_type(annotation):
        raise NotImplementedError(
            f"Union fields are not supported yet: {spec.name}"
        )

    if annotation is bool:
        return mo.ui.checkbox(value=bool(value), label=label)

    if annotation is str:
        return mo.ui.text(value=str(value), label=label)

    if annotation is Path:
        return mo.ui.file_browser(
            initial_path=_initial_browser_path(value),
            selection_mode="file",
            multiple=False,
            label=label,
        )

    if annotation in (int, float):
        return _build_numeric_element(annotation, spec.info, value, label)

    if _is_literal_type(annotation):
        options = list(get_args(annotation))
        return mo.ui.dropdown(options=options, value=value, label=label)

    if _is_enum_type(annotation):
        options = list(annotation)
        return mo.ui.dropdown(options=options, value=value, label=label)

    if _is_model_type(annotation):
        nested_payload = value if isinstance(value, dict) else {}
        return PydanticGui(
            annotation,
            value=nested_payload,
            label="",
            include_json_editor=False,
        )

    if _is_array_annotation(annotation):
        return _build_array_element(annotation, value, label)

    return mo.ui.text(value=_text_value(value), label=label)


def _build_numeric_element(
    annotation: type[int] | type[float],
    info: FieldInfo,
    value: int | float,
    label: str,
) -> UIElement[Any, Any]:
    bounds = _numeric_bounds(info)
    if bounds.lower is not None and bounds.upper is not None:
        start, stop = _slider_limits(annotation, bounds)
        step = bounds.step
        if step is None:
            inferred_step = _infer_numeric_step(annotation, value)
            if inferred_step is None:
                if annotation is int:
                    step = 1
                else:
                    step = max((stop - start) / _DEFAULT_SLIDER_STEPS, 1e-6)
            else:
                step = inferred_step
        return mo.ui.slider(
            start=start,
            stop=stop,
            step=step,
            value=value,
            label=label,
        )

    step = bounds.step
    if step is None:
        inferred_step = _infer_numeric_step(annotation, value)
        if inferred_step is not None:
            step = inferred_step
        elif annotation is int:
            step = 1
    return mo.ui.number(
        start=bounds.lower,
        stop=bounds.upper,
        step=step,
        value=value,
        label=label,
    )


def _build_array_element(
    annotation: Any,
    value: Any,
    label: str,
) -> UIElement[Any, Any]:
    matrix_value = _normalize_matrix_value(annotation, value)
    total_cells = _matrix_total_cells(matrix_value)
    if total_cells > _MAX_MATRIX_CELLS:
        raise ValueError(
            f"Array field {label!r} has {total_cells} cells, which exceeds the "
            f"supported limit of {_MAX_MATRIX_CELLS}."
        )
    return mo.ui.matrix(value=matrix_value, label=label)


def _with_help_text(
    element: UIElement[Any, Any],
    help_text: str | None,
) -> Any:
    if not help_text:
        return element
    return mo.hstack(
        [
            element,
            mo.md(
                (
                    "<span style="
                    '"color: var(--mo-foreground-muted, #6b7280);'
                    " font-style: italic;"
                    ' font-size: 0.875em;">'
                    f"{html.escape(help_text)}"
                    "</span>"
                )
            ),
        ],
        align="start",
        justify="start",
    )


def _resolve_initial_payload(
    model_cls: type[BaseModel],
    value: BaseModel | dict[str, Any] | None,
) -> dict[str, Any]:
    raw: dict[str, Any]
    if isinstance(value, BaseModel):
        raw = value.model_dump()
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        raw = {}

    payload: dict[str, Any] = {}
    for name, info in model_cls.model_fields.items():
        if name in raw:
            field_value = raw[name]
            if _is_model_type(info.annotation):
                payload[name] = _resolve_initial_payload(info.annotation, field_value)
            else:
                payload[name] = field_value
        else:
            payload[name] = _initial_field_value(name, info)
    return payload


def _initial_field_value(name: str, info: FieldInfo) -> Any:
    if not info.is_required():
        default = info.get_default(call_default_factory=True)
        if _is_model_type(info.annotation):
            return _resolve_initial_payload(info.annotation, default)
        return default

    annotation = info.annotation
    if _is_union_type(annotation):
        raise NotImplementedError(
            f"Union fields are not supported yet: {name}"
        )
    if annotation is bool:
        return False
    if annotation is str:
        return ""
    if annotation is Path:
        return ""
    if annotation is int:
        bounds = _numeric_bounds(info)
        lower = bounds.lower if bounds.lower is not None else 0
        return int(lower) + (1 if bounds.strict_lower else 0)
    if annotation is float:
        bounds = _numeric_bounds(info)
        lower = bounds.lower if bounds.lower is not None else 0.0
        if bounds.strict_lower:
            step = bounds.step if bounds.step is not None else 0.1
            return float(lower) + float(step)
        return float(lower)
    if _is_literal_type(annotation):
        options = get_args(annotation)
        if not options:
            raise ValueError(f"Literal field {name!r} does not define any options.")
        return options[0]
    if _is_enum_type(annotation):
        options = list(annotation)
        if not options:
            raise ValueError(f"Enum field {name!r} does not define any options.")
        return options[0]
    if _is_model_type(annotation):
        return _resolve_initial_payload(annotation, None)
    if _is_array_annotation(annotation):
        return _default_array_value(annotation)
    return ""


def _validate_payload(
    model_cls: type[ModelT],
    payload: dict[str, Any],
) -> ModelT | None:
    try:
        return model_cls.model_validate(payload)
    except ValidationError:
        return None


def _numeric_bounds(info: FieldInfo) -> _NumericBounds:
    lower: int | float | None = None
    upper: int | float | None = None
    step: int | float | None = None
    strict_lower = False
    strict_upper = False

    for metadata in info.metadata:
        if isinstance(metadata, annotated_types.Ge):
            lower = metadata.ge
        elif isinstance(metadata, annotated_types.Gt):
            lower = metadata.gt
            strict_lower = True
        elif isinstance(metadata, annotated_types.Le):
            upper = metadata.le
        elif isinstance(metadata, annotated_types.Lt):
            upper = metadata.lt
            strict_upper = True
        elif isinstance(metadata, annotated_types.MultipleOf):
            step = metadata.multiple_of

    return _NumericBounds(
        lower=lower,
        upper=upper,
        step=step,
        strict_lower=strict_lower,
        strict_upper=strict_upper,
    )


def _infer_numeric_step(
    annotation: type[int] | type[float],
    value: int | float,
) -> int | float | None:
    if value == 0:
        return 1 if annotation is int else None

    magnitude = abs(float(value))
    exponent = math.floor(math.log10(magnitude)) - 1

    if annotation is int:
        return max(1, 10**exponent)
    return 10.0**exponent


@lru_cache(maxsize=None)
def _attribute_docstrings_for_model(
    model_cls: type[BaseModel],
) -> dict[str, str]:
    try:
        source = inspect.getsource(model_cls)
    except (OSError, TypeError, tokenize.TokenError):
        return {}

    try:
        module_ast = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return {}

    class_node = next(
        (
            node
            for node in ast.walk(module_ast)
            if isinstance(node, ast.ClassDef) and node.name == model_cls.__name__
        ),
        None,
    )
    if class_node is None:
        return {}

    docs: dict[str, str] = {}
    body = class_node.body
    for index, node in enumerate(body[:-1]):
        if not isinstance(node, ast.AnnAssign):
            continue
        target = node.target
        if not isinstance(target, ast.Name):
            continue
        next_node = body[index + 1]
        if not (
            isinstance(next_node, ast.Expr)
            and isinstance(next_node.value, ast.Constant)
            and isinstance(next_node.value.value, str)
        ):
            continue
        docs[target.id] = inspect.cleandoc(next_node.value.value)
    return docs


def _attribute_docstring_for_field(
    model_cls: type[BaseModel],
    field_name: str,
) -> str | None:
    return _attribute_docstrings_for_model(model_cls).get(field_name)


@lru_cache(maxsize=None)
def _docstring_args_for_model(
    model_cls: type[BaseModel],
) -> dict[str, str]:
    doc = inspect.getdoc(model_cls)
    if not doc:
        return {}

    lines = doc.splitlines()
    args_start: int | None = None
    for index, line in enumerate(lines):
        if line.strip() in {"Args:", "Arguments:"}:
            args_start = index + 1
            break
    if args_start is None:
        return {}

    result: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    for raw_line in lines[args_start:]:
        if not raw_line.strip():
            if current_name is not None and current_lines:
                current_lines.append("")
            continue

        indent = len(raw_line) - len(raw_line.lstrip())
        stripped = raw_line.strip()

        if indent == 0:
            break

        if ":" in stripped:
            candidate_name, candidate_text = stripped.split(":", 1)
            field_name = candidate_name.strip()
            if field_name.isidentifier():
                if current_name is not None:
                    result[current_name] = "\n".join(current_lines).strip()
                current_name = field_name
                current_lines = [candidate_text.strip()]
                continue

        if current_name is not None:
            current_lines.append(stripped)

    if current_name is not None:
        result[current_name] = "\n".join(current_lines).strip()

    return {key: value for key, value in result.items() if value}


def _docstring_help_for_field(
    model_cls: type[BaseModel],
    field_name: str,
) -> str | None:
    return _docstring_args_for_model(model_cls).get(field_name)


def _slider_limits(
    annotation: type[int] | type[float],
    bounds: _NumericBounds,
) -> tuple[int | float, int | float]:
    assert bounds.lower is not None
    assert bounds.upper is not None

    if annotation is int:
        start = int(bounds.lower) + (1 if bounds.strict_lower else 0)
        stop = int(bounds.upper) - (1 if bounds.strict_upper else 0)
        return start, stop

    start = float(bounds.lower)
    stop = float(bounds.upper)
    step = float(bounds.step) if bounds.step is not None else max(
        (stop - start) / _DEFAULT_SLIDER_STEPS,
        1e-6,
    )
    if bounds.strict_lower:
        start += step
    if bounds.strict_upper:
        stop -= step
    return start, stop


def _is_literal_type(annotation: Any) -> bool:
    return get_origin(annotation) is Literal


def _is_union_type(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin is UnionType or origin is getattr(__import__("typing"), "Union")


def _is_model_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _is_enum_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, Enum)


def _is_array_annotation(annotation: Any) -> bool:
    if annotation in (np.ndarray, torch.Tensor):
        return True
    return isinstance(annotation, type) and issubclass(annotation, AbstractArray)


def _uses_text_fallback(annotation: Any) -> bool:
    return not (
        annotation in (str, bool, int, float, Path)
        or _is_literal_type(annotation)
        or _is_enum_type(annotation)
        or _is_model_type(annotation)
        or _is_array_annotation(annotation)
    )


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    try:
        return json.dumps(_jsonify(value))
    except TypeError:
        return str(value)


def _maybe_parse_json_text(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in '[{"' and stripped not in {"true", "false", "null"}:
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _coerce_path_value(info: FieldInfo, value: Any) -> Path | str:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value) if value else ""
    if value:
        selected = value[0]
        path = getattr(selected, "path", None)
        if isinstance(path, Path):
            return path
        if isinstance(selected, dict):
            maybe_path = selected.get("path")
            if isinstance(maybe_path, str):
                return Path(maybe_path)
    if not info.is_required():
        default = info.get_default(call_default_factory=True)
        if isinstance(default, Path):
            return default
    return ""


def _initial_browser_path(value: Any) -> Path:
    if isinstance(value, Path):
        if value.exists():
            return value if value.is_dir() else value.parent
        return Path.cwd()
    if isinstance(value, str) and value:
        path = Path(value)
        if path.exists():
            return path if path.is_dir() else path.parent
    return Path.cwd()


def _frontend_value_for_element(
    spec: _FieldSpec,
    element: UIElement[Any, Any],
    value: Any,
) -> JSONType:
    if isinstance(element, PydanticGui):
        payload = value if isinstance(value, dict) else {}
        return element._frontend_value_from_payload(payload)  # noqa: SLF001

    annotation = spec.annotation
    if annotation is Path:
        return _file_browser_frontend_value(value)
    if _is_array_annotation(annotation):
        return _normalize_matrix_value(annotation, value)
    if _is_enum_type(annotation) or _is_literal_type(annotation):
        return [] if value is None else [_dropdown_key(value)]
    if annotation in (bool, int, float, str):
        return value
    return _text_value(value)


def _dropdown_key(value: Any) -> str:
    return str(value)


def _file_browser_frontend_value(value: Any) -> list[dict[str, Any]]:
    path: Path | None = None
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value:
        path = Path(value)
    if path is None or not path.exists():
        return []
    return [
        {
            "id": str(path),
            "name": path.name,
            "path": str(path),
            "is_directory": path.is_dir(),
        }
    ]


def _set_local_frontend_value(
    element: UIElement[Any, Any],
    frontend_value: JSONType,
) -> None:
    element._value_frontend = frontend_value  # noqa: SLF001
    try:
        element._value = element._convert_value(frontend_value)  # noqa: SLF001
    except Exception:
        pass


def _merge_json_value(current: Any, incoming: Any) -> Any:
    if isinstance(current, dict) and isinstance(incoming, dict):
        merged = dict(current)
        for key, value in incoming.items():
            merged[key] = _merge_json_value(merged.get(key), value)
        return merged
    return incoming


def _json_text_to_payload(json_text: str) -> tuple[dict[str, Any], str | None]:
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return {}, f"json: {exc.msg}"
    if not isinstance(parsed, dict):
        return {}, "json: top-level JSON value must be an object."
    return parsed, None


def _payload_to_json(payload: dict[str, Any]) -> str:
    return json.dumps(_jsonify(payload), indent=2, sort_keys=True)


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _array_shape(annotation: Any, value: Any | None = None) -> _ArrayShape:
    if isinstance(annotation, type) and issubclass(annotation, AbstractArray):
        fixed_sizes: list[int] = []
        for dim in annotation.dims:
            size = getattr(dim, "size", None)
            if size is None:
                return _ArrayShape(ndim=len(annotation.dims), fixed_shape=None)
            fixed_sizes.append(int(size))
        return _ArrayShape(
            ndim=len(annotation.dims),
            fixed_shape=tuple(fixed_sizes),
        )

    if value is not None:
        array = _to_numpy_array(value)
        if array.ndim in (1, 2):
            return _ArrayShape(ndim=array.ndim, fixed_shape=tuple(array.shape))

    raise ValueError(
        "Array fields without a fixed shape need a default value or explicit "
        "initial value."
    )


def _default_array_value(annotation: Any) -> np.ndarray | torch.Tensor:
    shape = _array_shape(annotation)
    if shape.fixed_shape is None:
        raise ValueError(
            "Array fields without a fixed shape need a default value or "
            "explicit initial value."
        )
    data = np.zeros(shape.fixed_shape, dtype=np.float64)
    return _coerce_array_value(annotation, data)


def _normalize_matrix_value(annotation: Any, value: Any) -> Any:
    array = _to_numpy_array(value)
    shape = _array_shape(annotation, value)
    if shape.ndim not in (1, 2):
        raise NotImplementedError(
            "Only 1D and 2D arrays are supported by the matrix widget."
        )
    if shape.fixed_shape is not None and tuple(array.shape) != shape.fixed_shape:
        raise ValueError(
            f"Expected array shape {shape.fixed_shape}, got {tuple(array.shape)}."
        )
    return array.tolist()


def _coerce_array_value(annotation: Any, value: Any) -> np.ndarray | torch.Tensor:
    array = _to_numpy_array(value)
    shape = _array_shape(annotation, array)
    if shape.ndim not in (1, 2):
        raise NotImplementedError(
            "Only 1D and 2D arrays are supported by the matrix widget."
        )
    if shape.fixed_shape is not None and tuple(array.shape) != shape.fixed_shape:
        raise ValueError(
            f"Expected array shape {shape.fixed_shape}, got {tuple(array.shape)}."
        )

    if annotation is torch.Tensor or (
        isinstance(annotation, type) and issubclass(annotation, AbstractArray)
        and annotation.array_type is torch.Tensor
    ):
        dtype = torch.float32 if array.dtype.kind == "f" else torch.int64
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        return torch.tensor(array.tolist(), dtype=dtype)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    return np.asarray(array)


def _to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _matrix_total_cells(value: Any) -> int:
    array = np.asarray(value)
    return int(array.size)


def _format_validation_error(exc: ValidationError) -> str:
    first_error = exc.errors()[0]
    location = ".".join(str(part) for part in first_error["loc"])
    return f"{location}: {first_error['msg']}"
