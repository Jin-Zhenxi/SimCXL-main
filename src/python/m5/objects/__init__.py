# Copyright (c) 2010 The Hewlett-Packard Development Company
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import importlib
import importlib.abc
import importlib.util
import os
import sys

loader_state = getattr(__spec__, "loader_state", None)
allow_source_fallback = (
    os.environ.get("M5_SOURCE_OBJECTS_FALLBACK", "0") == "1"
)

try:
    import m5.internal.params as _m5_internal_params
except Exception:
    _m5_internal_params = None


def _fallback_roots():
    env_roots = os.environ.get("M5_SOURCE_OBJECTS_FALLBACK_ROOTS", "")
    if env_roots:
        roots = [
            os.path.abspath(root)
            for root in env_roots.split(os.pathsep)
            if root
        ]
        if roots:
            return roots

    module_file = globals().get("__file__") or getattr(
        __spec__, "origin", None
    )
    if not module_file:
        module_file = os.path.join(
            os.getcwd(), "src", "python", "m5", "objects", "__init__.py"
        )

    return [
        os.path.abspath(
            os.path.join(
                os.path.dirname(module_file), "..", "..", "..", "..", "src"
            )
        )
    ]


def _build_source_fallback_map():
    fallback_modules = {}
    for scan_root in _fallback_roots():
        for root, _, files in os.walk(scan_root):
            if os.path.abspath(root).startswith(
                os.path.abspath(os.path.join(scan_root, "python"))
            ):
                continue
            for entry in sorted(files):
                if not entry.endswith(".py") or entry == "__init__.py":
                    continue
                mod_name = entry[:-3]
                if not mod_name.isidentifier():
                    continue
                path = os.path.join(root, entry)
                full_module = f"m5.objects.{mod_name}"
                fallback_modules.setdefault(full_module, path)
    return fallback_modules


def _install_source_fallback(fallback_modules):
    class _ObjectsSourceFallbackFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            file_path = fallback_modules.get(fullname)
            if not file_path:
                return None
            return importlib.util.spec_from_file_location(fullname, file_path)

    if not any(
        isinstance(finder, _ObjectsSourceFallbackFinder)
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _ObjectsSourceFallbackFinder())


def _module_has_backing_params(module_name, module_obj):
    if _m5_internal_params is None:
        return True

    module_basename = module_name.rsplit(".", 1)[-1]
    simobject_cls = getattr(module_obj, module_basename, None)
    if simobject_cls is None:
        return True

    params_name = f"{module_basename}Params"
    return hasattr(_m5_internal_params, params_name)


def _export_module_symbols(module_obj):
    export_names = getattr(module_obj, "__all__", None)
    if export_names is None:
        export_names = [
            name for name in module_obj.__dict__ if not name.startswith("_")
        ]
    for name in export_names:
        globals()[name] = getattr(module_obj, name)


missing_modules = set()

if loader_state:
    for module in loader_state:
        if not module.startswith("m5.objects."):
            continue
        try:
            module_obj = importlib.import_module(module)
        except ModuleNotFoundError:
            missing_modules.add(module)
            continue
        except Exception:
            continue

        if not _module_has_backing_params(module, module_obj):
            continue

        _export_module_symbols(module_obj)

if allow_source_fallback and (missing_modules or not loader_state):
    fallback_modules = _build_source_fallback_map()
    _install_source_fallback(fallback_modules)

    for full_module in sorted(missing_modules):
        try:
            module_obj = importlib.import_module(full_module)
        except Exception:
            continue

        if not _module_has_backing_params(full_module, module_obj):
            continue

        _export_module_symbols(module_obj)


def _ensure_required_source_objects():
    required_modules = (
        "m5.objects.RedirectPath",
        "m5.objects.ThermalModel",
    )
    installed_fallback = False

    for full_module in required_modules:
        class_name = full_module.rsplit(".", 1)[-1]
        if class_name in globals():
            continue

        if not installed_fallback:
            fallback_modules = _build_source_fallback_map()
            _install_source_fallback(fallback_modules)
            installed_fallback = True

        try:
            module_obj = importlib.import_module(full_module)
        except Exception:
            continue

        _export_module_symbols(module_obj)


_ensure_required_source_objects()
