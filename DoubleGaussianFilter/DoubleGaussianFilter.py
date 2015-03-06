"""
    Double Gaussian Filter.

    Implemented as an operation that can be applied to data items.

    This code is experimental, meaning that it works with the current version
    but will probably not work "as-is" with future versions of the software.

"""

# standard libraries
import gettext
import math

# third party libraries
import numpy
import scipy.fftpack

# local libraries
# None


_ = gettext.gettext


class DoubleGaussianFilterOperationDelegate(object):

    def __init__(self, api):
        self.__api = api
        self.operation_id = "double-gaussian-filter-operation"
        self.operation_name = _("Double Gaussian Filter")
        self.operation_prefix = _("Double Gaussian Filter of ")
        self.operation_description = [
            {"name": _("Sigma 1"), "property": "sigma1", "type": "scalar", "default": 0.3},
            {"name": _("Sigma 2"), "property": "sigma2", "type": "scalar", "default": 0.3},
            {"name": _("Weight 2"), "property": "weight2", "type": "scalar", "default": 0.3}
        ]

    def can_apply_to_data(self, data_and_metadata):
        return data_and_metadata.is_data_2d and data_and_metadata.is_data_scalar_type

    # process is called to process the data. this version does not change the data shape
    # or data type. if it did, we would need to provide another function to describe the
    # change in shape or data type.
    def get_processed_data_and_metadata(self, data_and_metadata, parameters):
        api = self.__api

        # only works with 2d, scalar data
        assert data_and_metadata.is_data_2d
        assert data_and_metadata.is_data_scalar_type

        # make a copy of the data so that other threads can use data while we're processing
        # otherwise numpy puts a lock on the data.
        data = data_and_metadata.data
        data_copy = data.copy()

        # grab our parameters. ideally this could just access the member variables directly,
        # but it doesn't work that way (yet).
        sigma1 = parameters.get("sigma1")
        sigma2 = parameters.get("sigma2")
        weight2 = parameters.get("weight2")

        # first calculate the FFT
        fft_data = scipy.fftpack.fftshift(scipy.fftpack.fft2(data_copy))

        # next, set up xx, yy arrays to be linear indexes for x and y coordinates ranging
        # from -width/2 to width/2 and -height/2 to height/2.
        yy_min = int(math.floor(-data.shape[0] / 2))
        yy_max = int(math.floor(data.shape[0] / 2))
        xx_min = int(math.floor(-data.shape[1] / 2))
        xx_max = int(math.floor(data.shape[1] / 2))
        xx, yy = numpy.meshgrid(numpy.linspace(yy_min, yy_max, data.shape[0]),
                                numpy.linspace(xx_min, xx_max, data.shape[1]))

        # calculate the pixel distance from the center
        rr = numpy.sqrt(numpy.square(xx) + numpy.square(yy)) / (data.shape[0] * 0.5)

        # finally, apply a filter to the Fourier space data.
        filter = numpy.exp(-0.5 * numpy.square(rr / sigma1)) - (1.0 - weight2) * numpy.exp(
            -0.5 * numpy.square(rr / sigma2))
        filtered_fft_data = fft_data * filter

        # and then do invert FFT and take the real value.
        result = scipy.fftpack.ifft2(scipy.fftpack.ifftshift(filtered_fft_data)).real

        intensity_calibration = data_and_metadata.intensity_calibration
        dimensional_calibrations = data_and_metadata.dimensional_calibrations
        metadata = data_and_metadata.metadata
        return api.create_data_and_metadata_from_data(result, intensity_calibration, dimensional_calibrations, metadata)


class DoubleGaussianExtension(object):

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.swift.extensions.double_gaussian"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        # be sure to keep a reference or it will be closed immediately.
        self.__operation_ref = api.create_unary_operation(DoubleGaussianFilterOperationDelegate(api))

    def close(self):
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__operation_ref.close()
        self.__operation_ref = None
