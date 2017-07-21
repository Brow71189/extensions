#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:36:16 2017

@author: mittelberger2
"""

#extratags = [(50838, 'I', 2, IJMD.bytecounts, True), (50839, 'B', len(IJMD.metadata), struct.unpack('>'+'B'*len(IJMD.metadata), IJMD.metadata))]

import struct
import numpy as np

# Constants
HEADER_SIZE = 64
HEADER2_SIZE = 64
VERSION = 227
# Offsets
VERSION_OFFSET = 4
TYPE = 6
TOP = 8
LEFT = 10
BOTTOM = 12
RIGHT = 14
N_COORDINATES = 16
STROKE_WIDTH = 34
STROKE_COLOR = 40
OPTIONS = 50
HEADER2_OFFSET = 60
COORDINATES = 64
# Options
SUB_PIXEL_RESOLUTION = 128
# Types
roi_types = {
             'polygon': 0,
             'rect': 1,
             'oval': 2,
             'line': 3,
             'freeline': 4,
             'polyline': 5,
             'noRoi': 6,
             'freehand': 7,
             'traced': 8,
             'angle': 9,
             'point': 10
             }
# Field data types
dtypes = {
          VERSION: '>h',
          TYPE: '<h', # For some weird reason 'type' is saved as little endian in contrast to all other fields
          TOP: '>h',
          LEFT: '>h',
          BOTTOM: '>h',
          RIGHT: '>h',
          N_COORDINATES: '>h',
          STROKE_WIDTH: '>h',
          STROKE_COLOR: '>3s',
#          OPTIONS: '>h',
          HEADER2_OFFSET: '>i',
          COORDINATES: ('>h', '>f') # save coordinates as 'ints' or 'floats' depending on subpixel_resolution being enabled
          }

class default_dict(dict):
    def __init__(self, mapping, **elements):
        super().__init__(mapping, **elements)
        self._default_value = None

    def set_default_value(self, default_value):
        self._default_value = default_value

    def __getitem__(self, key):
        if not self.__contains__(key) and self._default_value is not None:
            return self._default_value
        else:
            return super().__getitem__(key)

dtypes = default_dict(dtypes)
dtypes.set_default_value('>h')

class IJMetadata(object):
    def __init__(self):
        self._rois = []
        self._overlays = []
        self._labels = []
        self._info = []
        self._luts = []
        self._ranges = []

    @property
    def bytecounts(self):
        bytecounts = tuple()
        header_size = self.ntypes * 8 + 4
        bytecounts += (header_size,)

        for md_type in [self._rois, self._overlays, self._labels, self._info, self._luts, self._ranges]:
            for properties in md_type:
                bytecounts += (len(properties['bytestring']),)

        return bytecounts

    @property
    def metadata(self):
        metadata_bytestring = b'IJIJ'
        if len(self._rois) > 0:
            metadata_bytestring += b'roi ' + struct.pack('>i', len(self._rois))
        if len(self._overlays) > 0:
            metadata_bytestring += b'over' + struct.pack('>i', len(self._overlays))
        for md_type in [self._rois, self._overlays, self._labels, self._info, self._luts, self._ranges]:
            for properties in md_type:
                metadata_bytestring += properties['bytestring']
        return metadata_bytestring

    @property
    def ntypes(self):
        ntypes = 0
        if len(self._rois) > 0:
            ntypes += 1
        if len(self._overlays) > 0:
            ntypes += 1
        if len(self._labels) > 0:
            ntypes += 1
        if len(self._info) > 0:
            ntypes += 1
        if len(self._luts) > 0:
            ntypes += 1
        if len(self._ranges) > 0:
            ntypes += 1
        return ntypes

    def _add_data(self, offset, data):
        assert offset != COORDINATES
        if np.iterable(offset):
            assert len(offset) == len(data)
        else:
            offset = (offset,)
            data = (data,)
        for i in range(len(offset)):
            self.roi_bytes[offset[i]:offset[i]+struct.calcsize(dtypes[offset[i]])] = struct.pack(dtypes[offset[i]],
                                                                                                 data[i])

    def _add_roi_or_overlay(self, md_type: str, properties: dict, roi_type='point'):
        """
        md_type must be a 4 character imagej type string ('roi ' or 'over')
        """
        assert type(properties.get('points')) == list
        npoints = len(properties.get('points'))
        float_size = 8
        roi_name_size = 0
        roi_props_size = 0
        counters_size = 0

        self.roi_bytes = bytearray(b'\x00'*(HEADER_SIZE + HEADER2_SIZE + npoints*4 + float_size + roi_name_size +
                                            roi_props_size + counters_size))
        print(len(self.roi_bytes))

        self.roi_bytes[:4] = b'Iout'
        self._add_data(VERSION_OFFSET, VERSION)
        self._add_data(TYPE, roi_types[roi_type])

        points = np.array(properties.get('points'), dtype=np.float32)

        top = int(np.amin(points[:, 0]))
        left = int(np.amin(points[:, 1]))
        bottom = int(np.amax(points[:, 0]))
        right = int(np.amax(points[:, 1]))
        if right == left:
            right += 1
        if bottom == top:
            bottom += 1
        self._add_data((TOP, LEFT, BOTTOM, RIGHT), (top, left, bottom, right))

        self._add_data(N_COORDINATES, npoints)
        self._add_data(STROKE_WIDTH, 1)
        if md_type == 'over':
            self._add_data(STROKE_COLOR, bytes([255, 255, 255]))
        self._add_data(OPTIONS, SUB_PIXEL_RESOLUTION)
        self._add_data(HEADER2_OFFSET, HEADER_SIZE + 4*npoints + float_size)
        if properties.get('subpixel_resulution') or roi_type=='point':
            coordinates_format = dtypes[COORDINATES][1]
            base1 = COORDINATES + 4*npoints
            base2 = base1 + 4*npoints
            for i in range(npoints):
                self.roi_bytes[base1+4*i:base1+4*i+4] = struct.pack(coordinates_format, points[i, 1])
                self.roi_bytes[base2+4*i:base2+4*i+4] = struct.pack(coordinates_format, points[i, 0])
        else:
            coordinates_format = dtypes[COORDINATES][0]
            base1 = COORDINATES + 2*npoints
            base2 = base1 + 2*npoints
            for i in range(npoints):
                self.roi_bytes[base1+2*i:base1+2*i+2] = struct.pack(coordinates_format, int(points[i, 1]))
                self.roi_bytes[base2+2*i:base2+2*i+2] = struct.pack(coordinates_format, int(points[i, 0]))

        properties['bytestring'] = bytes(self.roi_bytes)
        if md_type == 'roi ':
            self._rois.append(properties)
        elif md_type == 'over':
            self._overlays.append(properties)
        else:
            raise ValueError('"md_type" must be one of ("roi ", "over)')

        delattr(self, 'roi_bytes')

    def add_roi(self, roi_properties: dict, roi_type='point'):
        """
        roi_properties (dict):
            points: list of (y, x) tuples in pixels, MUST be a list even if only one pair of coordinates
            position: position of roi in a stack (only required for stacks), optional
            subpixel_resolution: True/False, optional
        """
        self._add_roi_or_overlay('roi ', roi_properties, roi_type=roi_type)

    def add_overlay(self, overlay_properties: dict, overlay_type='point'):
        pass

    def add_labels(self, label_properties: dict):
        raise NotImplementedError

    def add_info(self, info_properties: dict):
        raise NotImplementedError

    def add_luts(self, luts_properties: dict):
        raise NotImplementedError

    def add_ranges(self, ranges_properties: dict):
        raise NotImplementedError