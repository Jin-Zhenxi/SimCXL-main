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

    return [
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..", "src"
            )
        )
    ]


if loader_state:
    for module in loader_state:
        if module.startswith("m5.objects."):
            exec(f"from {module} import *")
elif allow_source_fallback:
    fallback_modules = {}
    preload_modules = {}
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
                try:
                    with open(path, encoding="utf-8") as src_file:
                        src = src_file.read()
                except OSError:
                    continue
                is_simobject_like = "cxx_header" in src or "cxx_class" in src
                is_enum_like = "enum_name" in src and "class " in src
                if is_simobject_like or is_enum_like:
                    preload_modules.setdefault(full_module, path)

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

    imported = set()
    progress = True
    while progress:
        progress = False
        for full_module in sorted(preload_modules):
            if full_module in imported or full_module in sys.modules:
                continue
            try:
                module = importlib.import_module(full_module)
            except Exception:
                continue
            imported.add(full_module)
            exec(f"from {full_module} import *")
            progress = True
