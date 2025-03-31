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

"""Utility functions for pymatreader."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    import h5py

try:
    from scipy.io.matlab import MatlabFunction, MatlabOpaque
except ImportError:  # scipy < 1.8
    from scipy.io.matlab.mio5 import MatlabFunction
    from scipy.io.matlab.mio5_params import MatlabOpaque


standard_matlab_classes = (
    'char',
    'cell',
    'float',
    'double',
    'int',
    'int8',
    'int16',
    'int32',
    'int64',
    'uint',
    'uint8',
    'uint16',
    'logical',
    'uint32',
    'uint64',
    'struct',
    'unknown',
)


def _import_h5py() -> h5py:
    try:
        import h5py
    except Exception as exc:
        raise ImportError('h5py is required to read MATLAB files >= v7.3 ' f'({exc})')
    return h5py


def _hdf5todict(
    hdf5_object: h5py.Dataset | h5py.Group,
    variable_names: Iterable | None = None,
    ignore_fields: Iterable | None = None,
) -> dict:
    """
    Recursively converts a hdf5 object to a python dictionary, converting all types as well.

    Parameters
    ----------
    hdf5_object: Union[h5py.Group, h5py.Dataset]
        Object to convert. Can be a h5py File, Group or Dataset
    variable_names: iterable, optional
        Tuple or list of variables to include. If set to none, all
        variable are read.
    ignore_fields: iterable, optional
        Tuple or list of fields to ignore. If set to none, all fields will
        be read.

    Returns
    -------
    dict
        Python dictionary
    """
    h5py = _import_h5py()

    if isinstance(hdf5_object, h5py.Group):
        return _handle_hdf5_group(hdf5_object, variable_names=variable_names, ignore_fields=ignore_fields)

    elif isinstance(hdf5_object, h5py.Dataset):
        result = _handle_hdf5_dataset(hdf5_object)
        if not isinstance(result, dict):
            raise TypeError('Unknown type in hdf5 file')
        return result

    raise TypeError('Unknown type in hdf5 file')


def _handle_hdf5_list(
    hdf5_object: list | types.GeneratorType | h5py.Dataset | h5py.Group,
    variable_names: Iterable | None = None,
    ignore_fields: Iterable | None = None,
) -> np.ndarray | list | str | int | float | complex | dict | None:
    h5py = _import_h5py()

    if isinstance(hdf5_object, h5py.Group):
        return _handle_hdf5_group(hdf5_object, variable_names=variable_names, ignore_fields=ignore_fields)

    elif isinstance(hdf5_object, h5py.Dataset):
        return _handle_hdf5_dataset(hdf5_object)
    elif isinstance(hdf5_object, (list, types.GeneratorType)):
        return [_handle_hdf5_list(item) for item in hdf5_object]

    raise TypeError('Unknown type in hdf5 file')


def _handle_hdf5_group(
    hdf5_object: h5py.Dataset, variable_names: Iterable | None = None, ignore_fields: Iterable | None = None
) -> dict:
    all_keys = set(hdf5_object.keys())
    if ignore_fields:
        all_keys = all_keys - set(ignore_fields)

    if variable_names:
        all_keys = all_keys & set(variable_names)

    return_dict = {}

    for key in all_keys:
        return_dict[key] = _handle_hdf5_list(hdf5_object[key], variable_names=None, ignore_fields=ignore_fields)

    return return_dict


def _handle_hdf5_dataset(hdf5_object: h5py.Dataset) -> np.ndarray | int | float | complex | str | list | dict | None:
    h5py = _import_h5py()

    data: np.ndarray

    if 'MATLAB_empty' in hdf5_object.attrs:  # noqa SIM108
        data = np.empty((0,))
    else:
        # this used to be just hdf5_object.value, but this is deprecated
        data = hdf5_object[()]

    matlab_class = hdf5_object.attrs.get('MATLAB_class', b'unknown').decode()

    if matlab_class not in standard_matlab_classes:
        warn(
            'Complex objects (like classes) are not supported. '
            'They are imported on a best effort base '
            'but your mileage will vary.'
        )

    if matlab_class == 'string':
        warn(
            'pymatreader cannot import Matlab string variables. '
            'Please convert these variables to char arrays in Matlab.'
        )
        return None

    if data.dtype == np.dtype('object'):
        data_list = [hdf5_object.file[cur_data] for cur_data in data.flatten()]

        if len(data_list) == 1 and matlab_class == 'cell':
            if isinstance(data_list[0], h5py.Group):
                return _handle_hdf5_group(data_list[0])

            matlab_class = data_list[0].attrs.get('MATLAB_class', matlab_class).decode()

            return _assign_types(data_list[0][()], matlab_class)

        return _assign_types(_handle_hdf5_list(data_list), matlab_class)

    return _assign_types(data, matlab_class)


def _convert_string_hdf5(values: np.ndarray) -> str | np.ndarray:
    if values.size > 1:
        return ''.join(chr(c) for c in values.flatten())
    else:
        try:
            return chr(int(values))
        except TypeError:
            return np.array([])


def _assign_types(
    values: np.ndarray | np.float64 | dict | list | str | int | float | complex | None, matlab_class: str
) -> np.ndarray | int | float | complex | str | list | dict | None:
    """Private function, which assigns correct types to h5py extracted values from _browse_dataset()."""
    if matlab_class == 'char' and isinstance(values, np.ndarray):
        values = np.squeeze(values).T
        return _handle_hdf5_strings(values)
    elif isinstance(values, np.ndarray):
        return _handle_ndarray(values)
    elif isinstance(values, np.float64):
        return float(values)
    else:
        return values


def _handle_ndarray(values: np.ndarray) -> np.ndarray | int | float | complex:
    """Handle conversion of ndarrays."""
    values = np.squeeze(values).T
    if values.dtype.names == ('real', 'imag'):
        values = np.array(values.view(complex))

    if values.size == 1:
        return values.item()
    else:
        return values


def _handle_hdf5_strings(values: np.ndarray) -> str | np.ndarray | list[str | np.ndarray]:
    if values.ndim in (0, 1):
        return _convert_string_hdf5(values)
    elif values.ndim == 2:  # noqa PLR2004
        return [_convert_string_hdf5(cur_val) for cur_val in values]
    else:
        raise RuntimeError('String arrays with more than 2 dimensions' 'are not supported at the moment.')


def _parse_scipy_mat_dict(data: dict) -> dict:
    """
    Parse a scipy.io.matlab.mio5_params.mat_struct dictionary.

    Parameters
    ----------
    data: dict
        data to be parsed

    Returns
    -------
    dict
        parsed data
    """
    for key in data:
        data[key] = _check_for_scipy_mat_struct(data[key])

    return data


def _check_for_scipy_mat_struct(data: dict | np.ndarray | MatlabOpaque) -> dict | np.ndarray | list | None:
    """
    Check all entries of data for occurrences of scipy.io.matlab.mio5_params.mat_struct and convert them.

    Parameters
    ----------
    data: any
        data to be checked

    Returns
    -------
    object
        checked and converted data
    """
    if isinstance(data, dict):
        return _parse_scipy_mat_dict(data)

    if isinstance(data, MatlabOpaque):
        try:
            if data[0][2] == b'string':
                warn(
                    'pymatreader cannot import Matlab string variables. '
                    'Please convert these variables to char arrays '
                    'in Matlab.'
                )
                return None
        except IndexError:
            pass
        warn(
            'Complex objects (like classes) are not supported. '
            'They are imported on a best effort base '
            'but your mileage will vary.'
        )

    if isinstance(data, np.ndarray):
        return _handle_scipy_ndarray(data)

    return data


def _handle_scipy_ndarray(data: np.ndarray | MatlabFunction) -> np.ndarray | list:
    if data.dtype == np.dtype('object') and not isinstance(data, MatlabFunction):
        as_list = []
        for element in data:
            as_list.append(_check_for_scipy_mat_struct(element))
        data = as_list
    elif isinstance(data.dtype.names, tuple):
        data = _todict_from_np_struct(data)
        data = _check_for_scipy_mat_struct(data)

    if isinstance(data, np.ndarray):
        data = np.array(data)

    return data


def _todict_from_np_struct(data: np.ndarray) -> dict[str, np.ndarray | int | float | str | list]:
    data_dict: dict[str, np.ndarray | int | float | str | list] = {}

    for cur_field_name in data.dtype.names:
        try:
            n_items = len(data[cur_field_name])
            cur_list = []

            for idx in np.arange(n_items):
                cur_value = data[cur_field_name].item(idx)
                cur_value = _check_for_scipy_mat_struct(cur_value)
                cur_list.append(cur_value)

            data_dict[cur_field_name] = cur_list
        except TypeError:
            cur_value = data[cur_field_name].item(0)
            cur_value = _check_for_scipy_mat_struct(cur_value)
            data_dict[cur_field_name] = cur_value

    return data_dict
