# -*- coding: UTF-8 -*-
# Copyright (c) 2018, Dirk Gütlin & Thomas Hartmann
# All rights reserved.
#
# This file is part of the pymatreader Project, see:
# https://gitlab.com/obob/pymatreader
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read `.mat` files disregarding of the underlying file version."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import scipy.io

try:
    from scipy.io.matlab import matfile_version
except ImportError:
    from scipy.io.matlab.miobase import get_matfile_version as matfile_version

from .utils import _hdf5todict, _import_h5py, _parse_scipy_mat_dict

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ['read_mat']


def read_mat(
    filename: str | Path,
    variable_names: Iterable[str] | None = None,
    ignore_fields: Iterable[str] | None = None,
    uint16_codec: str | None = None,
) -> dict[str, Any]:
    """Read .mat files of version <7.3 or 7.3 and return the contained data structure.

    Parameters
    ----------
    filename: str | Path
        Path and filename of the .mat file containing the data.
    variable_names: list of strings, optional
        Reads only the data contained in the specified dict key or
        variable name. Default is None.
    ignore_fields: list of strings, optional
        Ignores every dict key/variable name specified in the list within the
        entire structure. Only works for .mat files v 7.3. Default is [].
    uint16_codec : str | None
        If your file contains non-ascii characters, sometimes reading
        it may fail and give rise to error message stating that "buffer is
        too small". ``uint16_codec`` allows to specify what codec (for example:
        'latin1' or 'utf-8') should be used when reading character arrays and
        can therefore help you solve this problem.

    Returns
    -------
    dict
        A structure of nested dictionaries, with variable names as keys and
        variable data as values.
    """
    if not Path(filename).exists():
        raise OSError(f'The file {filename} does not exist.')

    ignore_fields = [] if ignore_fields is None else list(ignore_fields)

    try:
        with Path(filename).open('rb') as fid:  # avoid open file warnings on error
            mjv, _ = matfile_version(fid)
            extra_kwargs = {}
            if mjv == 1:
                extra_kwargs['uint16_codec'] = uint16_codec

            raw_data = scipy.io.loadmat(
                fid,
                struct_as_record=True,
                squeeze_me=True,
                mat_dtype=False,
                variable_names=variable_names,
                **extra_kwargs,
            )
        return _parse_scipy_mat_dict(raw_data)
    except NotImplementedError:
        ignore_fields.append('#refs#')
        h5py = _import_h5py()
        with h5py.File(filename, 'r') as hdf5_file:
            return _hdf5todict(hdf5_file, variable_names=variable_names, ignore_fields=ignore_fields)
