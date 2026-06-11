#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: ATC VHF AM Transmitter Simulation
# Author: ATC Simulation
# Description: ATC VHF DSB-FC AM Transmitter Simulation with Impulse Noise and Ricean Fading
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
from gnuradio import blocks
from gnuradio import channels
from gnuradio import filter
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from gnuradio import zeromq
from gnuradio.qtgui import Range, GrRangeWidget
from PyQt5 import QtCore


class atc_am_tx(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(
            self, "ATC VHF AM Transmitter Simulation", catch_exceptions=True
        )
        Qt.QWidget.__init__(self)
        self.setWindowTitle("ATC VHF AM Transmitter Simulation")
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

        self.settings = Qt.QSettings("GNU Radio", "atc_am_tx")

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
        self.zmq_address = zmq_address = "tcp://127.0.0.1:55556"
        self.wav_file_path = wav_file_path = "/data/input_audio.wav"
        self.quad_rate = quad_rate = 240000
        self.noise_voltage = noise_voltage = 0.05
        self.mod_index = mod_index = 0.9
        self.impulse_rate = impulse_rate = 50.0
        self.impulse_amplitude = impulse_amplitude = 0.5
        self.if_freq = if_freq = 10000
        self.freq_offset = freq_offset = 0.0
        self.fading_rate = fading_rate = 0.5
        self.fading_offset = fading_offset = 0.7
        self.fading_depth = fading_depth = 0.3
        self.audio_rate = audio_rate = 8000

        ##################################################
        # Blocks
        ##################################################
        self._noise_voltage_range = Range(0.0, 0.5, 0.005, 0.05, 200)
        self._noise_voltage_win = GrRangeWidget(
            self._noise_voltage_range,
            self.set_noise_voltage,
            "AWGN Noise",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
            "value",
        )

        self.top_grid_layout.addWidget(self._noise_voltage_win, 0, 1, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._mod_index_range = Range(0.0, 1.0, 0.01, 0.9, 200)
        self._mod_index_win = GrRangeWidget(
            self._mod_index_range,
            self.set_mod_index,
            "Modulation Index",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
            "value",
        )

        self.top_grid_layout.addWidget(self._mod_index_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._impulse_rate_range = Range(0.0, 500.0, 1.0, 50.0, 200)
        self._impulse_rate_win = GrRangeWidget(
            self._impulse_rate_range,
            self.set_impulse_rate,
            "Impulse Rate (Hz)",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
            "value",
        )

        self.top_grid_layout.addWidget(self._impulse_rate_win, 0, 4, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(4, 5):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._impulse_amplitude_range = Range(0.0, 5.0, 0.05, 0.5, 200)
        self._impulse_amplitude_win = GrRangeWidget(
            self._impulse_amplitude_range,
            self.set_impulse_amplitude,
            "Impulse Amplitude",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
            "value",
        )

        self.top_grid_layout.addWidget(self._impulse_amplitude_win, 0, 3, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(3, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._freq_offset_range = Range(-0.01, 0.01, 0.0001, 0.0, 200)
        self._freq_offset_win = GrRangeWidget(
            self._freq_offset_range,
            self.set_freq_offset,
            "Freq Offset (norm.)",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
            "value",
        )

        self.top_grid_layout.addWidget(self._freq_offset_win, 0, 2, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._fading_rate_range = Range(0.01, 20.0, 0.01, 0.5, 200)
        self._fading_rate_win = GrRangeWidget(
            self._fading_rate_range,
            self.set_fading_rate,
            "Fading Rate (Hz)",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
            "value",
        )

        self.top_grid_layout.addWidget(self._fading_rate_win, 0, 6, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(6, 7):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._fading_offset_range = Range(0.0, 1.0, 0.01, 0.7, 200)
        self._fading_offset_win = GrRangeWidget(
            self._fading_offset_range,
            self.set_fading_offset,
            "Fading Offset",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
            "value",
        )

        self.top_grid_layout.addWidget(self._fading_offset_win, 0, 7, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(7, 8):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._fading_depth_range = Range(0.0, 1.0, 0.01, 0.3, 200)
        self._fading_depth_win = GrRangeWidget(
            self._fading_depth_range,
            self.set_fading_depth,
            "Fading Depth",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
            "value",
        )

        self.top_grid_layout.addWidget(self._fading_depth_win, 0, 5, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(5, 6):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.zeromq_pub_sink_0 = zeromq.pub_sink(
            gr.sizeof_gr_complex, 1, zmq_address, 100, False, (-1), ""
        )
        self.rational_resampler_xxx_0 = filter.rational_resampler_fff(
            interpolation=30, decimation=1, taps=[], fractional_bw=0
        )
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            2048,  # size
            window.WIN_BLACKMAN_hARRIS,  # wintype
            0,  # fc
            quad_rate,  # bw
            "ATC AM TX Spectrum",  # name
            1,
            None,  # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-80), 10)
        self.qtgui_freq_sink_x_0.set_y_label("TX Spectrum", "dB")
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(True)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)

        labels = ["AM + Fading + Impulse", "", "", "", "", "", "", "", "", ""]
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
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_x_0_win, 1, 0, 2, 4)
        for r in range(1, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.channels_channel_model_0 = channels.channel_model(
            noise_voltage=noise_voltage,
            frequency_offset=freq_offset,
            epsilon=1.0,
            taps=[1.0, 0.0, 0.05, 0.0, 0.02],
            noise_seed=0,
            block_tags=False,
        )
        self.blocks_wavfile_source_0 = blocks.wavfile_source(wav_file_path, True)
        self.blocks_throttle_0 = blocks.throttle(
            gr.sizeof_gr_complex * 1, quad_rate, True
        )
        self.blocks_multiply_xx_0 = blocks.multiply_vff(1)
        self.blocks_multiply_fading = blocks.multiply_vcc(1)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff(mod_index)
        self.blocks_keep_one_in_n = blocks.keep_one_in_n(
            gr.sizeof_gr_complex * 1,
            (max(1, int(quad_rate / max(impulse_rate, 0.001)))),
        )
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.blocks_add_impulse = blocks.add_vcc(1)
        self.blocks_add_const_vxx_0 = blocks.add_const_ff(1.0)
        self.band_pass_filter_0 = filter.fir_filter_fff(
            1, firdes.band_pass(1, audio_rate, 300, 3400, 100, window.WIN_KAISER, 6.76)
        )
        self.analog_sig_source_x_0 = analog.sig_source_f(
            quad_rate, analog.GR_COS_WAVE, if_freq, 1, 0, 0
        )
        self.analog_sig_source_fading = analog.sig_source_c(
            quad_rate, analog.GR_SIN_WAVE, fading_rate, fading_depth, fading_offset, 0
        )
        self.analog_noise_source_impulse = analog.noise_source_c(
            analog.GR_UNIFORM, impulse_amplitude, 42
        )

        ##################################################
        # Connections
        ##################################################
        self.connect(
            (self.analog_noise_source_impulse, 0), (self.blocks_keep_one_in_n, 0)
        )
        self.connect(
            (self.analog_sig_source_fading, 0), (self.blocks_multiply_fading, 1)
        )
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect(
            (self.band_pass_filter_0, 0), (self.blocks_multiply_const_vxx_0, 0)
        )
        self.connect(
            (self.blocks_add_const_vxx_0, 0), (self.rational_resampler_xxx_0, 0)
        )
        self.connect((self.blocks_add_impulse, 0), (self.blocks_throttle_0, 0))
        self.connect(
            (self.blocks_float_to_complex_0, 0), (self.blocks_multiply_fading, 0)
        )
        self.connect((self.blocks_keep_one_in_n, 0), (self.blocks_add_impulse, 1))
        self.connect(
            (self.blocks_multiply_const_vxx_0, 0), (self.blocks_add_const_vxx_0, 0)
        )
        self.connect(
            (self.blocks_multiply_fading, 0), (self.channels_channel_model_0, 0)
        )
        self.connect(
            (self.blocks_multiply_xx_0, 0), (self.blocks_float_to_complex_0, 0)
        )
        self.connect((self.blocks_throttle_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.zeromq_pub_sink_0, 0))
        self.connect((self.blocks_wavfile_source_0, 0), (self.band_pass_filter_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_add_impulse, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.blocks_multiply_xx_0, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "atc_am_tx")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_zmq_address(self):
        return self.zmq_address

    def set_zmq_address(self, zmq_address):
        self.zmq_address = zmq_address

    def get_wav_file_path(self):
        return self.wav_file_path

    def set_wav_file_path(self, wav_file_path):
        self.wav_file_path = wav_file_path

    def get_quad_rate(self):
        return self.quad_rate

    def set_quad_rate(self, quad_rate):
        self.quad_rate = quad_rate
        self.analog_sig_source_fading.set_sampling_freq(self.quad_rate)
        self.analog_sig_source_x_0.set_sampling_freq(self.quad_rate)
        self.blocks_keep_one_in_n.set_n(
            (max(1, int(self.quad_rate / max(self.impulse_rate, 0.001))))
        )
        self.blocks_throttle_0.set_sample_rate(self.quad_rate)
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.quad_rate)

    def get_noise_voltage(self):
        return self.noise_voltage

    def set_noise_voltage(self, noise_voltage):
        self.noise_voltage = noise_voltage
        self.channels_channel_model_0.set_noise_voltage(self.noise_voltage)

    def get_mod_index(self):
        return self.mod_index

    def set_mod_index(self, mod_index):
        self.mod_index = mod_index
        self.blocks_multiply_const_vxx_0.set_k(self.mod_index)

    def get_impulse_rate(self):
        return self.impulse_rate

    def set_impulse_rate(self, impulse_rate):
        self.impulse_rate = impulse_rate
        self.blocks_keep_one_in_n.set_n(
            (max(1, int(self.quad_rate / max(self.impulse_rate, 0.001))))
        )

    def get_impulse_amplitude(self):
        return self.impulse_amplitude

    def set_impulse_amplitude(self, impulse_amplitude):
        self.impulse_amplitude = impulse_amplitude
        self.analog_noise_source_impulse.set_amplitude(self.impulse_amplitude)

    def get_if_freq(self):
        return self.if_freq

    def set_if_freq(self, if_freq):
        self.if_freq = if_freq
        self.analog_sig_source_x_0.set_frequency(self.if_freq)

    def get_freq_offset(self):
        return self.freq_offset

    def set_freq_offset(self, freq_offset):
        self.freq_offset = freq_offset
        self.channels_channel_model_0.set_frequency_offset(self.freq_offset)

    def get_fading_rate(self):
        return self.fading_rate

    def set_fading_rate(self, fading_rate):
        self.fading_rate = fading_rate
        self.analog_sig_source_fading.set_frequency(self.fading_rate)

    def get_fading_offset(self):
        return self.fading_offset

    def set_fading_offset(self, fading_offset):
        self.fading_offset = fading_offset
        self.analog_sig_source_fading.set_offset(self.fading_offset)

    def get_fading_depth(self):
        return self.fading_depth

    def set_fading_depth(self, fading_depth):
        self.fading_depth = fading_depth
        self.analog_sig_source_fading.set_amplitude(self.fading_depth)

    def get_audio_rate(self):
        return self.audio_rate

    def set_audio_rate(self, audio_rate):
        self.audio_rate = audio_rate
        self.band_pass_filter_0.set_taps(
            firdes.band_pass(
                1, self.audio_rate, 300, 3400, 100, window.WIN_KAISER, 6.76
            )
        )


def main(top_block_cls=atc_am_tx, options=None):
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
