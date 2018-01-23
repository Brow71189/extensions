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
import uuid

# local libraries
# None


_ = gettext.gettext


class DoubleGaussianFilterAMOperationDelegate(object):

    def __init__(self, api):
        self.__api = api
        self.panel_id = 'DoubleGaussian-Panel'
        self.panel_name = _('Double Gaussian Filter')
        self.panel_positions = ['left', 'right']
        self.panel_position = 'right'
#        self.operation_id = "double-gaussian-filter-operation_am"
#        self.operation_name = _("Double Gaussian Filter AM")
#        self.operation_prefix = _("Double Gaussian Filter of ")
#        self.operation_description = [
#            {"name": _("Sigma 1\n(large)"), "property": "sigma1", "type": "scalar", "default": 0.4},
#            {"name": _("Sigma 2\n(small)"), "property": "sigma2", "type": "scalar", "default": 0.2},
#            {"name": _("Weight 2"), "property": "weight2", "type": "scalar", "default": 0.3},
#            {"name": _("Show Graphene Peak positions"), "property": "show_gpp", "type": "boolean-checkbox",
#             "default": False }
#        ]
        self.fft_data_item = None
        self.small_ellipse_region = None
        self.big_ellipse_region = None
        self.line_profile_data_item = None
        self.interval_region = None
        self.parameters = {'sigma1': 0.4, 'sigma2': 0.2, 'weight': 0.3, 'show_gpp': True}
        self.source_data_item = None

    def create_panel_widget(self, ui, document_controller):

        def sigma_1_field_enter(text):
            if len(text) > 0 and self.source_data_item is not None:
                try:
                    value = float(text)
                except ValueError:
                    pass
                else:
                    if value != self.parameters.get('sigma1'):
                        self.parameters['sigma1'] = value
                        self.update_calculation()

            sigma_1_field.text = '{:g}'.format(self.parameters.get('sigma1'))
            sigma_1_field.select_all()

        def sigma_2_field_enter(text):
            if len(text) > 0 and self.source_data_item is not None:
                try:
                    value = float(text)
                except ValueError:
                    pass
                else:
                    if value != self.parameters.get('sigma2'):
                        self.parameters['sigma2'] = value
                        self.update_calculation()

            sigma_2_field.text = '{:g}'.format(self.parameters.get('sigma2'))
            sigma_2_field.select_all()

        def weight_field_enter(text):
            if len(text) > 0 and self.source_data_item is not None:
                try:
                    value = float(text)
                except ValueError:
                    pass
                else:
                    if value != self.parameters.get('weight'):
                        self.parameters['weight'] = value
                        self.update_calculation()

            weight_field.text = '{:g}'.format(self.parameters.get('weight'))
            weight_field.select_all()

        def show_graphene_positions_changed(check_state):
            if self.source_data_item is not None:
                if show_graphene_positions.checked != self.parameters.get('show_gpp'):
                    self.parameters['show_gpp'] = show_graphene_positions.checked
                    self.update_calculation()
            else:
                show_graphene_positions.checked = self.parameters.get('show_gpp')

        def run_button_clicked():
            self.source_data_item = document_controller.target_data_item
            self.update_calculation()

        column = ui.create_column_widget()
        sigma_1_label = ui.create_label_widget('Sigma 1 (large) ')
        sigma_1_field = ui.create_line_edit_widget()
        sigma_2_label = ui.create_label_widget('Sigma 2 (small) ')
        sigma_2_field = ui.create_line_edit_widget()
        weight_label = ui.create_label_widget('Weight ')
        weight_field = ui.create_line_edit_widget()
        show_graphene_positions = ui.create_check_box_widget('Show graphene peak positions')
        run_button = ui.create_push_button_widget('Connect to data item')
        info_label = ui.create_label_widget('To update values type a new number into the\n' +
                                            'respective field and hit enter.')

        sigma_row = ui.create_row_widget()
        weight_row = ui.create_row_widget()
        run_row = ui.create_row_widget()
        info_row = ui.create_row_widget()

        sigma_row.add_spacing(5)
        sigma_row.add(sigma_1_label)
        sigma_row.add(sigma_1_field)
        sigma_row.add_spacing(10)
        sigma_row.add(sigma_2_label)
        sigma_row.add(sigma_2_field)
        sigma_row.add_spacing(5)
        sigma_row.add_stretch()

        weight_row.add_spacing(5)
        weight_row.add_stretch()
        weight_row.add(weight_label)
        weight_row.add(weight_field)
        weight_row.add_spacing(5)

        run_row.add_spacing(5)
        run_row.add(run_button)
        run_row.add_spacing(10)
        run_row.add_stretch()
        run_row.add(show_graphene_positions)
        run_row.add_spacing(5)

        info_row.add_spacing(5)
        info_row.add(info_label)
        info_row.add_spacing(5)
        info_row.add_stretch()

        column.add_spacing(5)
        column.add(sigma_row)
        column.add(weight_row)
        column.add(run_row)
        column.add_spacing(5)
        column.add(info_row)
        column.add_spacing(5)
        column.add_stretch()

        sigma_1_field.on_editing_finished = sigma_1_field_enter
        sigma_2_field.on_editing_finished = sigma_2_field_enter
        weight_field.on_editing_finished = weight_field_enter
        run_button.on_clicked = run_button_clicked
        show_graphene_positions.on_check_state_changed = show_graphene_positions_changed

        sigma_1_field_enter('')
        sigma_2_field_enter('')
        weight_field_enter('')

        show_graphene_positions.checked = self.parameters.get('show_gpp')

        return column


    def update_calculation(self):
        result_data_item = self.get_result_data_item()
        if result_data_item is not None:
            self.update_metadata(self.source_data_item, 'double_gaussian_filter_am.result_uuid', result_data_item.uuid.hex)
            result_data_item.set_data_and_metadata(self.get_processed_data_and_metadata(self.source_data_item.xdata,
                                                                                        self.parameters))

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
        intensity_calibration = data_and_metadata.intensity_calibration
        dimensional_calibrations = data_and_metadata.dimensional_calibrations
        metadata = data_and_metadata.metadata

        # grab our parameters. ideally this could just access the member variables directly,
        # but it doesn't work that way (yet).
        sigma1 = parameters.get("sigma1")**2
        sigma2 = parameters.get("sigma2")**2
        weight2 = parameters.get("weight")
        show_gpp = parameters.get("show_gpp")

        # first calculate the FFT
        fft_data = scipy.fftpack.fftshift(scipy.fftpack.fft2(data_copy))

        # next, set up xx, yy arrays to be linear indexes for x and y coordinates ranging
        # from -width/2 to width/2 and -height/2 to height/2.
        yy_min = int(math.floor(-data.shape[0] / 2))
        yy_max = int(math.floor(data.shape[0] / 2))
        xx_min = int(math.floor(-data.shape[1] / 2))
        xx_max = int(math.floor(data.shape[1] / 2))
        yy, xx = numpy.meshgrid(numpy.linspace(yy_min, yy_max, data.shape[0]),
                                numpy.linspace(xx_min, xx_max, data.shape[1]),
                                indexing='ij')

        # calculate the pixel distance from the center
        rr = numpy.sqrt(numpy.square(xx/(data.shape[1] * 0.5)) + numpy.square(yy/(data.shape[0] * 0.5)))

        # finally, apply a filter to the Fourier space data.
        filter = numpy.exp(-0.5 * numpy.square(rr / sigma1)) - (1.0 - weight2) * numpy.exp(
            -0.5 * numpy.square(rr / sigma2))

        central_value = fft_data[int(data.shape[0]/2), int(data.shape[1]/2)]
        filtered_fft_data = fft_data * filter
        # Normalize result
        filtered_fft_data[int(data.shape[0]/2), int(data.shape[1]/2)] = central_value

        fft_calibration = self.__api.create_calibration(scale=1/(dimensional_calibrations[0].scale*data.shape[0]),
                                                        units=dimensional_calibrations[1].units + '\u207b\u00b9' if
                                                        dimensional_calibrations[1].units else '')

        # and then do invert FFT and take the real value.
        result = scipy.fftpack.ifft2(scipy.fftpack.ifftshift(filtered_fft_data)).real
        #result = scipy.signal.fftconvolve(data_copy, filter/numpy.sum(filter), mode='same')

        # All following code is just updating the informational data items (line profile of filter and filtered fft)
        if self.fft_data_item is None:
            self.fft_data_item = self.get_fft_data_item()

        if self.line_profile_data_item is None:
            self.line_profile_data_item = self.get_line_data_item()

        if self.fft_data_item is not None:
            for region in self.fft_data_item.regions:
                self.fft_data_item.remove_region(region)
#            if self.small_ellipse_region is not None:
#                try:
#                    self.fft_data_item.remove_region(self.small_ellipse_region)
#                except Exception as detail:
#                    print('Could not remove small ellipse region. Reason: ' + str(detail))
            try:
                self.small_ellipse_region = self.fft_data_item.add_ellipse_region(0.5, 0.5, sigma2, sigma2)
                self.small_ellipse_region.set_property('is_position_locked', True)
                self.small_ellipse_region.set_property('is_shape_locked', True)
            except Exception as detail:
                print('Could not add small ellipse region. Reason: ' + str(detail))

#            if self.big_ellipse_region is not None:
#                try:
#                    self.fft_data_item.remove_region(self.big_ellipse_region)
#                except Exception as detail:
#                    print('Could not remove big ellipse region. Reason: ' + str(detail))
            try:
                self.big_ellipse_region = self.fft_data_item.add_ellipse_region(0.5, 0.5, sigma1, sigma1)
                self.big_ellipse_region.set_property('is_position_locked', True)
                self.big_ellipse_region.set_property('is_shape_locked', True)
            except Exception as detail:
                print('Could not add big ellipse region. Reason: ' + str(detail))

            self.fft_data_item.set_data((numpy.log(numpy.abs(fft_data))*filter).astype(numpy.float32))

        if self.line_profile_data_item is not None:
            try:
                self.line_profile_data_item.set_data((filter[int(filter.shape[0]/2),
                                            int(filter.shape[1]/2):int(filter.shape[1]/2*(1+2*sigma1))]).astype(numpy.float32))
            except Exception as detail:
                print('Could not change line profile data item. Reason: ' + str(detail))
                self.line_profile_data_item = self.__api.library.create_data_item_from_data_and_metadata(
                                                                        self.line_profile_data_item.data_and_metadata,
                                                                        title="Line Profile of Filter")
                self.line_profile_data_item.set_data((filter[int(filter.shape[0]/2),
                                            int(filter.shape[1]/2):int(filter.shape[1]/2*(1+2*sigma1))]).astype(numpy.float32))

        if show_gpp:
            if self.line_profile_data_item is not None:
                for region in self.line_profile_data_item.regions:
                    self.line_profile_data_item.remove_region(region)
                line_length = len(self.line_profile_data_item.data_and_metadata.data)*fft_calibration.scale
#                try:
#                    self.line_profile_data_item.remove_region(self.interval_region)
#                except Exception as detail:
#                    print('Could not remove interval region. Reason: ' + str(detail))
                try:
                    if line_length > 8.13:
                        self.interval_region = self.line_profile_data_item.add_interval_region(4.69/line_length,
                                                                                               8.13/line_length)
                except Exception as detail:
                    print('Could not add interval region. Reason: ' + str(detail))
                    self.line_profile_data_item = self.__api.library.create_data_item_from_data_and_metadata(
                                                                        self.line_profile_data_item.data_and_metadata,
                                                                        title="Line Profile of Filter")
                    if line_length > 8.13:
                        self.interval_region = self.line_profile_data_item.add_interval_region(4.69/line_length,
                                                                                               8.13/line_length)

        if self.interval_region is not None:
            self.interval_region.set_property('label', 'Graphene Peak Positions')

        if not show_gpp and self.interval_region is not None:
            try:
                self.line_profile_data_item.remove_region(self.interval_region)
            except Exception as detail:
                print('Could not remove interval region. Reason: ' + str(detail))
            else:
                self.interval_region = None

        if self.fft_data_item is not None:
            self.fft_data_item.set_dimensional_calibrations([fft_calibration, fft_calibration])
        if self.line_profile_data_item is not None:
            self.line_profile_data_item.set_dimensional_calibrations([fft_calibration])

        return api.create_data_and_metadata_from_data(result.astype(data.dtype), intensity_calibration,
                                                      dimensional_calibrations, metadata)
        #return api.create_data_and_metadata_from_data(filtered_fft_data, intensity_calibration, dimensional_calibrations, metadata)

    def update_metadata(self, data_item, key, value):
        metadata = data_item.metadata
        metadata[key] = value
        data_item.set_metadata(metadata)

    def get_metadata_value(self, data_item, key):
        return data_item.metadata.get(key)

    def update_session_metadata(self, key, value):
        metadata = self.__api.library._document_model.session_metadata.copy()
        metadata[key] = value
        self.__api.library._document_model.session_metadata = metadata

    def get_session_metadata_value(self, key):
        try:
            value = self.__api.library._document_model.session_metadata[key]
        except KeyError:
            return None
        else:
            return value

    def get_fft_data_item(self):
        fft_uuid = self.get_session_metadata_value('double_gaussian_filter_am.fft_uuid')
        fft_data_item = None
        if fft_uuid is not None:
            fft_data_item = self.__api.library.get_data_item_by_uuid(uuid.UUID(fft_uuid))
        if fft_data_item is None:
            fft_data_item = self.__api.library.create_data_item("Filtered FFT")
            self.update_session_metadata('double_gaussian_filter_am.fft_uuid', fft_data_item.uuid.hex)
        return fft_data_item

    def get_line_data_item(self):
        line_uuid = self.get_session_metadata_value('double_gaussian_filter_am.line_uuid')
        line_data_item = None
        if line_uuid is not None:
            line_data_item = self.__api.library.get_data_item_by_uuid(uuid.UUID(line_uuid))
        if line_data_item is None:
            line_data_item = self.__api.library.create_data_item("Line Profile of Filter")
            self.update_session_metadata('double_gaussian_filter_am.line_uuid', line_data_item.uuid.hex)
        return line_data_item

    def get_result_data_item(self):
        if self.source_data_item is None:
            return

        result_uuid = self.get_metadata_value(self.source_data_item, 'double_gaussian_filter_am.result_uuid')

        if result_uuid is not None:
            result_data_item = self.__api.library.get_data_item_by_uuid(uuid.UUID(result_uuid))
        else:
            result_data_item = None

        if result_data_item is None:
            result_data_item = self.__api.library.create_data_item('Double Gaussian Filter of ' + self.source_data_item.title)

        return result_data_item


class DoubleGaussianAMExtension(object):

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.swift.extensions.double_gaussian_am"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        # be sure to keep a reference or it will be closed immediately.
        self.__operation_ref = api.create_panel(DoubleGaussianFilterAMOperationDelegate(api))

    def close(self):
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__operation_ref.close()
        self.__operation_ref = None
