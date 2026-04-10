# Copyright (c) 2005 The Regents of The University of Michigan
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

# Import useful subpackages of M5, but *only* when run as an m5
# script.  This is mostly to keep backward compatibility with existing
# scripts while allowing new SCons code to operate properly.

try:
    # Try to import a native module
    import _m5.core

    # Try to grab something from it in case demandimport is being used
    _m5.core.curTick
    in_gem5 = True
except ImportError:
    # The import failed, we're being called from the build system
    in_gem5 = False

if not in_gem5:
    import collections
    import os
    import sys
    import types

    def _config_bool(name, default=False):
        config_root = ""
        if extra_pkg_root:
            config_root = os.path.abspath(
                os.path.join(extra_pkg_root, "..", "config")
            )
        header_path = (
            os.path.join(config_root, f"{name.lower()}.hh")
            if config_root
            else ""
        )
        if header_path and os.path.exists(header_path):
            try:
                with open(header_path, encoding="utf-8") as header_file:
                    text = header_file.read()
                if f"#define {name} 1" in text:
                    return True
                if f"#define {name} 0" in text:
                    return False
            except OSError:
                pass
        return default

    extra_pkg_root = os.environ.get("M5_BUILD_PYTHON_DIR", "")
    if extra_pkg_root:
        extra_m5_dir = os.path.join(extra_pkg_root, "m5")
        if os.path.isdir(extra_m5_dir) and extra_m5_dir not in __path__:
            __path__.append(extra_m5_dir)

    if "m5.defines" not in sys.modules:
        defines_module = types.ModuleType("m5.defines")
        fallback_build_env = collections.defaultdict(lambda: False)
        fallback_build_env.update(
            {
                "BUILD_ISA": os.environ.get("M5_BUILD_ISA", "X86"),
                "TARGET_ISA": os.environ.get("M5_TARGET_ISA", "x86"),
                "PROTOCOL": os.environ.get("M5_PROTOCOL", "MI_example"),
                "USE_SYSTEMC": False,
                "HAVE_TUNTAP": _config_bool("HAVE_TUNTAP", False),
                "HAVE_PROTOBUF": _config_bool("HAVE_PROTOBUF", False),
            }
        )
        defines_module.buildEnv = fallback_build_env
        sys.modules["m5.defines"] = defines_module

if in_gem5:
    from . import (
        SimObject,
        core,
        defines,
        objects,
        params,
        stats,
    )

    if defines.buildEnv["USE_SYSTEMC"]:
        from . import systemc
        from . import tlm
    from . import util
    from .event import *
    from .main import main
    from .simulate import *
