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
X1 = 18
Y1 = 22
X2 = 26
Y2 = 30
XD = 18
YD = 22
WIDTHD = 26
HEIGHTD = 30
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
          X1: '>f',
          Y1: '>f',
          X2: '>f',
          Y2: '>f',
          XD: '>f',
          YD: '>f',
          WIDTHD: '>f',
          HEIGHTD: '>f',
          N_COORDINATES: '>h',
          STROKE_WIDTH: '>h',
          STROKE_COLOR: '>3s',
#          OPTIONS: '>h',
          HEADER2_OFFSET: '>i',
          COORDINATES: ('>h', '>f') # save coordinates as 'ints' or 'floats' depending on subpixel_resolution being enabled
          }

class default_dict(dict):
    def __init__(self, *mapping, **elements):
        super().__init__(*mapping, **elements)
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
        self._extras = []

    @property
    def bytecounts(self):
        bytecounts = tuple()
        header_size = self.ntypes * 8 + 4
        bytecounts += (header_size,)

        for md_type in [self._rois, self._overlays, self._labels, self._info, self._luts, self._ranges, self._extras]:
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
        if len(self._info) > 0:
            metadata_bytestring += b'info' + struct.pack('>i', len(self._info))
        for extra_tag in self._extras:
            metadata_bytestring += extra_tag['type'] + struct.pack('>i', 1)

        for md_type in [self._rois, self._overlays, self._labels, self._info, self._luts, self._ranges, self._extras]:
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
        ntypes += len(self._extras)
        return ntypes

    @property
    def tifffile_extratags(self):
        metadata = self.metadata
        return [(50838, 'I', len(self.bytecounts), self.bytecounts, True), (50839, 'B', len(metadata),
                 struct.unpack('>'+'B'*len(metadata), metadata))]

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
        assert roi_type in roi_types, 'Unknown roi type'
        assert type(properties.get('points')) == list
        npoints = len(properties.get('points'))
        float_size = 0
        roi_name_size = 0
        roi_props_size = 0
        counters_size = 0

        if roi_type in ['point', 'rect', 'line', 'oval']:
            properties['subpixel_resolution'] = True
        if roi_type in ['rect', 'oval']:
            assert npoints == 4
            npoints = 0
        if roi_type == 'line':
            assert npoints == 2
            npoints = 0

        float_size = npoints*8

        self.roi_bytes = bytearray(b'\x00'*(HEADER_SIZE + HEADER2_SIZE + npoints*4 + float_size + roi_name_size +
                                            roi_props_size + counters_size))

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

        if roi_type in ['rect', 'oval']:
            self._add_data(XD, left)
            self._add_data(YD, top)
            self._add_data(HEIGHTD, bottom-top)
            self._add_data(WIDTHD, right-left)

        if roi_type == 'line':
            self._add_data(X1, points[0, 1])
            self._add_data(Y1, points[0, 0])
            self._add_data(X2, points[1, 1])
            self._add_data(Y2, points[1, 0])

        if npoints > 0:
            if properties.get('subpixel_resolution'):
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
            points: list of (y, x) tuples in pixels, MUST be a list even if only one pair of coordinates.
                    For a rectangle it must be the four corners of the rectangle, for a line the two endpoints and
                    for an oval the corners of the bounding rectangle.
            position: position of roi in a stack (only required for stacks), optional
            subpixel_resolution: True/False, optional
        """
        self._add_roi_or_overlay('roi ', roi_properties, roi_type=roi_type)

    def add_overlay(self, overlay_properties: dict, overlay_type='point'):
        """
        overlay_properties (dict):
            points: list of (y, x) tuples in pixels, MUST be a list even if only one pair of coordinates.
                    For a rectangle it must be the four corners of the rectangle, for a line the two endpoints and
                    for an oval the corners of the bounding rectangle.
            position: position of overlay in a stack (only required for stacks), optional
            subpixel_resolution: True/False, optional
        """
        self._add_roi_or_overlay('over', overlay_properties, roi_type=overlay_type)

    def add_labels(self, label_properties: dict):
        raise NotImplementedError

    def add_info(self, info_properties: dict):
        """
        info_properties (dict):
            text: string that contains the text to write to info tag
            encoding: encoding of the text, optional, defaults to 'ASCII'
        """

        text = info_properties.get('text')
        encoding = info_properties.get('encoding', 'ASCII')
        if text is None:
            return

        info_properties['bytestring'] = struct.pack('>' + 'H'*len(text), *bytes(text, encoding))
        self._info.append(info_properties)

    def add_luts(self, luts_properties: dict):
        raise NotImplementedError

    def add_ranges(self, ranges_properties: dict):
        raise NotImplementedError

    def add_extra_metadata(self, extra_properties: dict):
        """
        extra_properties (dict):
            bytes: bytestring that is to be written to tag
            type: string of length 4 that is used in Imagej to identify the metadata type.
                  Cannot be one of the predefined types ('info', 'labl', 'rang', 'luts', 'plot', 'over', 'roi ')
        """
        md_type = str(extra_properties.get('type', ''))
        assert len(md_type) == 4
        assert md_type not in ('info', 'labl', 'rang', 'luts', 'plot', 'over', 'roi ')

        bytestring = extra_properties.get('bytes')
        if bytestring is None:
            return

        extra_properties['bytestring'] = bytes(bytestring)
        extra_properties['type'] = bytes(md_type, 'ASCII')
        self._extras.append(extra_properties)