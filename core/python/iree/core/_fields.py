# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helpers for matching JSON data against the fields of dataclasses."""

import dataclasses
import typing


def unknown_keys(data_class: type, data: object, prefix: str = "") -> list[str]:
    """Return the keys in ``data`` that match no field of ``data_class``.

    Looks into nested dataclasses and lists of them, and returns the keys with
    their location, for example ``conditional_flags[0].wen``.
    """
    if not isinstance(data, dict):
        return []
    hints = typing.get_type_hints(data_class)
    unknown = [f"{prefix}{key}" for key in data if key not in hints]
    for name, hint in hints.items():
        value = data.get(name)
        if dataclasses.is_dataclass(hint) and isinstance(hint, type):
            unknown += unknown_keys(hint, value, f"{prefix}{name}.")
        elif typing.get_origin(hint) is list and isinstance(value, list):
            (item_type,) = typing.get_args(hint)
            if dataclasses.is_dataclass(item_type) and isinstance(item_type, type):
                for index, item in enumerate(value):
                    unknown += unknown_keys(
                        item_type, item, f"{prefix}{name}[{index}]."
                    )
    return unknown
