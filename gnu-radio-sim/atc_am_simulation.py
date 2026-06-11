#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: ATC VHF AM Simulation
# Author: ATC Simulation
# Description: ATC VHF AM Transmission Simulation with 16kHz PCM audio
# GNU Radio version: 3.10.4.0

from packaging.version import Version as StrictVersion

if __name__ == "__main__":
    import ctypes
    import sys

    if sys.platform.startswith("linux"):
        try:
            x11 = ctypes.cdll.LoadLibrary("libX11.so")
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import analog
from gnuradio import audio
from gnuradio import blocks
from gnuradio import filter
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from gnuradio.qtgui import Range, GrRangeWidget
from PyQt5 import QtCore


class atc_am_simulation(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "ATC VHF AM Simulation", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("ATC VHF AM Simulation")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme("gnuradio-grc"))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "atc_am_simulation")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.snr_display = snr_display = (
            '"SNR: +inf (no noise active)" if (atm_en + imp_en + imd_en == 0) else "SNR: {:.1f} dB".format(-10.0 * math.log10(max(1e-10, atm_level**2 * atm_en + imp_level**2 * imp_en + imd_level**2 * imd_en)))'
        )
        self.samp_rate = samp_rate = 480000
        self.mod_index = mod_index = 0.85
        self.imp_level = imp_level = 0.05
        self.imp_en = imp_en = 1
        self.imd_level = imd_level = 0.05
        self.imd_en = imd_en = 1
        self.carrier_freq = carrier_freq = 121500
        self.audio_rate = audio_rate = 16000
        self.atm_level = atm_level = 0.1
        self.atm_en = atm_en = 1

        ##################################################
        # Blocks
        ##################################################
        self._imp_level_range = Range(0, 1, 0.005, 0.05, 200)
        self._imp_level_win = GrRangeWidget(
            self._imp_level_range,
            self.set_imp_level,
            "Impulse Noise Level",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
            "value",
        )

        self.top_grid_layout.addWidget(self._imp_level_win, 2, 0, 1, 2)
        for r in range(2, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        _imp_en_check_box = Qt.QCheckBox("Enable Impulse")
        self._imp_en_choices = {True: 1, False: 0}
        self._imp_en_choices_inv = dict((v, k) for k, v in self._imp_en_choices.items())
        self._imp_en_callback = lambda i: Qt.QMetaObject.invokeMethod(
            _imp_en_check_box,
            "setChecked",
            Qt.Q_ARG("bool", self._imp_en_choices_inv[i]),
        )
        self._imp_en_callback(self.imp_en)
        _imp_en_check_box.stateChanged.connect(
            lambda i: self.set_imp_en(self._imp_en_choices[bool(i)])
        )
        self.top_grid_layout.addWidget(_imp_en_check_box, 2, 2, 1, 1)
        for r in range(2, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._imd_level_range = Range(0, 1, 0.005, 0.05, 200)
        self._imd_level_win = GrRangeWidget(
            self._imd_level_range,
            self.set_imd_level,
            "IMD / Channel Noise Level",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
            "value",
        )

        self.top_grid_layout.addWidget(self._imd_level_win, 3, 0, 1, 2)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        _imd_en_check_box = Qt.QCheckBox("Enable IMD/Channel")
        self._imd_en_choices = {True: 1, False: 0}
        self._imd_en_choices_inv = dict((v, k) for k, v in self._imd_en_choices.items())
        self._imd_en_callback = lambda i: Qt.QMetaObject.invokeMethod(
            _imd_en_check_box,
            "setChecked",
            Qt.Q_ARG("bool", self._imd_en_choices_inv[i]),
        )
        self._imd_en_callback(self.imd_en)
        _imd_en_check_box.stateChanged.connect(
            lambda i: self.set_imd_en(self._imd_en_choices[bool(i)])
        )
        self.top_grid_layout.addWidget(_imd_en_check_box, 3, 2, 1, 1)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._atm_level_range = Range(0, 1, 0.005, 0.1, 200)
        self._atm_level_win = GrRangeWidget(
            self._atm_level_range,
            self.set_atm_level,
            "Atmospheric Noise Level",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
            "value",
        )

        self.top_grid_layout.addWidget(self._atm_level_win, 1, 0, 1, 2)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        _atm_en_check_box = Qt.QCheckBox("Enable Atmospheric")
        self._atm_en_choices = {True: 1, False: 0}
        self._atm_en_choices_inv = dict((v, k) for k, v in self._atm_en_choices.items())
        self._atm_en_callback = lambda i: Qt.QMetaObject.invokeMethod(
            _atm_en_check_box,
            "setChecked",
            Qt.Q_ARG("bool", self._atm_en_choices_inv[i]),
        )
        self._atm_en_callback(self.atm_en)
        _atm_en_check_box.stateChanged.connect(
            lambda i: self.set_atm_en(self._atm_en_choices[bool(i)])
        )
        self.top_grid_layout.addWidget(_atm_en_check_box, 1, 2, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._snr_display_tool_bar = Qt.QToolBar(self)

        if None:
            self._snr_display_formatter = None
        else:
            self._snr_display_formatter = lambda x: str(x)

        self._snr_display_tool_bar.addWidget(Qt.QLabel("Signal-to-Noise Ratio"))
        self._snr_display_label = Qt.QLabel(
            str(self._snr_display_formatter(self.snr_display))
        )
        self._snr_display_tool_bar.addWidget(self._snr_display_label)
        self.top_grid_layout.addWidget(self._snr_display_tool_bar, 0, 0, 1, 3)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.sig_imd2 = analog.sig_source_f(
            samp_rate, analog.GR_SIN_WAVE, 123000, 0.5, 0, 0
        )
        self.sig_imd1 = analog.sig_source_f(
            samp_rate, analog.GR_SIN_WAVE, 122000, 0.5, 0, 0
        )
        self.qtgui_time_sink_x_0 = qtgui.time_sink_f(
            1024,  # size
            16000,  # samp_rate
            "Demodulated ATC Audio",  # name
            1,  # number of inputs
            None,  # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0.set_y_label("Amplitude", "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(
            qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, ""
        )
        self.qtgui_time_sink_x_0.enable_autoscale(True)
        self.qtgui_time_sink_x_0.enable_grid(True)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)

        labels = [
            "Received Audio",
            "Signal 2",
            "Signal 3",
            "Signal 4",
            "Signal 5",
            "Signal 6",
            "Signal 7",
            "Signal 8",
            "Signal 9",
            "Signal 10",
        ]
        widths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        colors = [
            "blue",
            "red",
            "green",
            "black",
            "cyan",
            "magenta",
            "yellow",
            "dark red",
            "dark green",
            "dark blue",
        ]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(
            self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget
        )
        self.top_grid_layout.addWidget(self._qtgui_time_sink_x_0_win, 7, 0, 3, 3)
        for r in range(7, 10):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_f(
            1024,  # size
            window.WIN_BLACKMAN_hARRIS,  # wintype
            carrier_freq,  # fc
            samp_rate,  # bw
            "ATC AM Signal Spectrum",  # name
            1,
            None,  # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0.set_y_label("Relative Gain", "dB")
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(True)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)

        self.qtgui_freq_sink_x_0.set_plot_pos_half(not True)

        labels = ["AM Signal + Noise", "", "", "", "", "", "", "", "", ""]
        widths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        colors = [
            "blue",
            "red",
            "green",
            "black",
            "cyan",
            "magenta",
            "yellow",
            "dark red",
            "dark green",
            "dark blue",
        ]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(
            self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget
        )
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_x_0_win, 4, 0, 3, 3)
        for r in range(4, 7):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.noise_imp = analog.noise_source_f(analog.GR_IMPULSE, 1, 7)
        self.noise_imd_bg = analog.noise_source_f(analog.GR_GAUSSIAN, 0.3, 13)
        self.noise_atm = analog.noise_source_f(analog.GR_GAUSSIAN, 1, 42)
        self.mult_imp = blocks.multiply_const_ff((imp_level * imp_en))
        self.mult_imd = blocks.multiply_const_ff((imd_level * imd_en))
        self.mult_atm = blocks.multiply_const_ff((atm_level * atm_en))
        self.mmse_resampler_xx_1 = filter.mmse_resampler_ff(
            0, (float(samp_rate) / float(audio_rate))
        )
        self.mmse_resampler_xx_0 = filter.mmse_resampler_ff(
            0, (float(audio_rate) / float(samp_rate))
        )
        self.low_pass_filter_1 = filter.fir_filter_fff(
            1, firdes.low_pass(1, samp_rate, 3500, 500, window.WIN_HAMMING, 6.76)
        )
        self.low_pass_filter_0 = filter.fir_filter_fff(
            1, firdes.low_pass(1, audio_rate, 3000, 200, window.WIN_HAMMING, 6.76)
        )
        self.high_pass_filter_0 = filter.fir_filter_fff(
            1, firdes.high_pass(1, audio_rate, 300, 100, window.WIN_HAMMING, 6.76)
        )
        self.blocks_wavfile_source_0 = blocks.wavfile_source(
            "/Users/ezka/Documents/LDC2004S13.wav", True
        )
        self.blocks_wavfile_sink_0 = blocks.wavfile_sink(
            "/Users/ezka/Documents/output.wav",
            1,
            16000,
            blocks.FORMAT_WAV,
            blocks.FORMAT_PCM_16,
            False,
        )
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_float * 1, samp_rate, True)
        self.blocks_multiply_xx_0 = blocks.multiply_vff(1)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff(mod_index)
        self.blocks_add_xx_0 = blocks.add_vff(1)
        self.blocks_add_const_vxx_1 = blocks.add_const_ff((-1.0))
        self.blocks_add_const_vxx_0 = blocks.add_const_ff(1.0)
        self.blocks_abs_xx_0 = blocks.abs_ff(1)
        self.audio_sink_0 = audio.sink(16000, "", True)
        self.analog_sig_source_x_0 = analog.sig_source_f(
            samp_rate, analog.GR_SIN_WAVE, carrier_freq, 1, 0, 0
        )
        self.add_imd = blocks.add_vff(1)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.add_imd, 0), (self.mult_imd, 0))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_abs_xx_0, 0), (self.blocks_add_const_vxx_1, 0))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_add_const_vxx_1, 0), (self.low_pass_filter_1, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_abs_xx_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect(
            (self.blocks_multiply_const_vxx_0, 0), (self.blocks_add_const_vxx_0, 0)
        )
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_wavfile_source_0, 0), (self.high_pass_filter_0, 0))
        self.connect((self.high_pass_filter_0, 0), (self.low_pass_filter_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.mmse_resampler_xx_0, 0))
        self.connect((self.low_pass_filter_1, 0), (self.mmse_resampler_xx_1, 0))
        self.connect((self.mmse_resampler_xx_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.mmse_resampler_xx_1, 0), (self.audio_sink_0, 0))
        self.connect((self.mmse_resampler_xx_1, 0), (self.blocks_wavfile_sink_0, 0))
        self.connect((self.mmse_resampler_xx_1, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.mult_atm, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.mult_imd, 0), (self.blocks_add_xx_0, 3))
        self.connect((self.mult_imp, 0), (self.blocks_add_xx_0, 2))
        self.connect((self.noise_atm, 0), (self.mult_atm, 0))
        self.connect((self.noise_imd_bg, 0), (self.add_imd, 2))
        self.connect((self.noise_imp, 0), (self.mult_imp, 0))
        self.connect((self.sig_imd1, 0), (self.add_imd, 0))
        self.connect((self.sig_imd2, 0), (self.add_imd, 1))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "atc_am_simulation")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_snr_display(self):
        return self.snr_display

    def set_snr_display(self, snr_display):
        self.snr_display = snr_display
        Qt.QMetaObject.invokeMethod(
            self._snr_display_label,
            "setText",
            Qt.Q_ARG("QString", str(self._snr_display_formatter(self.snr_display))),
        )

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.mmse_resampler_xx_0.set_resamp_ratio(
            (float(self.audio_rate) / float(self.samp_rate))
        )
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)
        self.sig_imd1.set_sampling_freq(self.samp_rate)
        self.sig_imd2.set_sampling_freq(self.samp_rate)
        self.low_pass_filter_1.set_taps(
            firdes.low_pass(1, self.samp_rate, 3500, 500, window.WIN_HAMMING, 6.76)
        )
        self.mmse_resampler_xx_1.set_resamp_ratio(
            (float(self.samp_rate) / float(self.audio_rate))
        )
        self.qtgui_freq_sink_x_0.set_frequency_range(self.carrier_freq, self.samp_rate)

    def get_mod_index(self):
        return self.mod_index

    def set_mod_index(self, mod_index):
        self.mod_index = mod_index
        self.blocks_multiply_const_vxx_0.set_k(self.mod_index)

    def get_imp_level(self):
        return self.imp_level

    def set_imp_level(self, imp_level):
        self.imp_level = imp_level
        self.mult_imp.set_k((self.imp_level * self.imp_en))

    def get_imp_en(self):
        return self.imp_en

    def set_imp_en(self, imp_en):
        self.imp_en = imp_en
        self._imp_en_callback(self.imp_en)
        self.mult_imp.set_k((self.imp_level * self.imp_en))

    def get_imd_level(self):
        return self.imd_level

    def set_imd_level(self, imd_level):
        self.imd_level = imd_level
        self.mult_imd.set_k((self.imd_level * self.imd_en))

    def get_imd_en(self):
        return self.imd_en

    def set_imd_en(self, imd_en):
        self.imd_en = imd_en
        self._imd_en_callback(self.imd_en)
        self.mult_imd.set_k((self.imd_level * self.imd_en))

    def get_carrier_freq(self):
        return self.carrier_freq

    def set_carrier_freq(self, carrier_freq):
        self.carrier_freq = carrier_freq
        self.analog_sig_source_x_0.set_frequency(self.carrier_freq)
        self.qtgui_freq_sink_x_0.set_frequency_range(self.carrier_freq, self.samp_rate)

    def get_audio_rate(self):
        return self.audio_rate

    def set_audio_rate(self, audio_rate):
        self.audio_rate = audio_rate
        self.high_pass_filter_0.set_taps(
            firdes.high_pass(1, self.audio_rate, 300, 100, window.WIN_HAMMING, 6.76)
        )
        self.low_pass_filter_0.set_taps(
            firdes.low_pass(1, self.audio_rate, 3000, 200, window.WIN_HAMMING, 6.76)
        )
        self.mmse_resampler_xx_0.set_resamp_ratio(
            (float(self.audio_rate) / float(self.samp_rate))
        )
        self.mmse_resampler_xx_1.set_resamp_ratio(
            (float(self.samp_rate) / float(self.audio_rate))
        )

    def get_atm_level(self):
        return self.atm_level

    def set_atm_level(self, atm_level):
        self.atm_level = atm_level
        self.mult_atm.set_k((self.atm_level * self.atm_en))

    def get_atm_en(self):
        return self.atm_en

    def set_atm_en(self, atm_en):
        self.atm_en = atm_en
        self._atm_en_callback(self.atm_en)
        self.mult_atm.set_k((self.atm_level * self.atm_en))


def main(top_block_cls=atc_am_simulation, options=None):
    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string("qtgui", "style", "raster")
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()


if __name__ == "__main__":
    main()
