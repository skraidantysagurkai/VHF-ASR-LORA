#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: ATC VHF AM Receiver Simulation
# Author: ATC Simulation
# Description: ATC VHF DSB-FC AM Receiver Simulation
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
from gnuradio import filter
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from gnuradio import zeromq
from gnuradio.qtgui import Range, GrRangeWidget
from PyQt5 import QtCore


class atc_am_rx(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(
            self, "ATC VHF AM Receiver Simulation", catch_exceptions=True
        )
        Qt.QWidget.__init__(self)
        self.setWindowTitle("ATC VHF AM Receiver Simulation")
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

        self.settings = Qt.QSettings("GNU Radio", "atc_am_rx")

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
        self.volume = volume = 1.0
        self.squelch_threshold = squelch_threshold = -60
        self.quad_rate = quad_rate = 240000
        self.output_wav_path = output_wav_path = "/data/output_audio.wav"
        self.if_freq = if_freq = 10000
        self.channel_bw = channel_bw = 25000
        self.audio_rate = audio_rate = 8000

        ##################################################
        # Blocks
        ##################################################
        self._volume_range = Range(0.0, 3.0, 0.05, 1.0, 200)
        self._volume_win = GrRangeWidget(
            self._volume_range,
            self.set_volume,
            "Volume",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
            "value",
        )

        self.top_grid_layout.addWidget(self._volume_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._squelch_threshold_range = Range(-80, 0, 1, -60, 200)
        self._squelch_threshold_win = GrRangeWidget(
            self._squelch_threshold_range,
            self.set_squelch_threshold,
            "Squelch (dBFS)",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
            "value",
        )

        self.top_grid_layout.addWidget(self._squelch_threshold_win, 0, 1, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.zeromq_sub_source_0 = zeromq.sub_source(
            gr.sizeof_gr_complex, 1, zmq_address, 100, False, (-1), ""
        )
        self.qtgui_time_sink_x_0 = qtgui.time_sink_f(
            1024,  # size
            audio_rate,  # samp_rate
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
            "Demodulated Audio",
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
        self.top_grid_layout.addWidget(self._qtgui_time_sink_x_0_win, 1, 2, 2, 1)
        for r in range(1, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024,  # size
            window.WIN_BLACKMAN_hARRIS,  # wintype
            0,  # fc
            40000,  # bw
            "Received ATC AM Signal",  # name
            1,
            None,  # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-80), 10)
        self.qtgui_freq_sink_x_0.set_y_label("RX Spectrum", "dB")
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(True)
        self.qtgui_freq_sink_x_0.set_fft_average(0.2)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)

        labels = ["Received AM", "", "", "", "", "", "", "", "", ""]
        widths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        colors = [
            "cyan",
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
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_x_0_win, 1, 0, 2, 2)
        for r in range(1, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.freq_xlating_fir_filter_xxx_0 = filter.freq_xlating_fir_filter_ccc(
            6,
            firdes.low_pass(
                1.0, quad_rate, channel_bw / 2, 2500, window.WIN_KAISER, 6.76
            ),
            if_freq,
            quad_rate,
        )
        self.blocks_wavfile_sink_0 = blocks.wavfile_sink(
            output_wav_path,
            1,
            audio_rate,
            blocks.FORMAT_WAV,
            blocks.FORMAT_PCM_16,
            False,
        )
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff(volume)
        self.analog_pwr_squelch_xx_0 = analog.pwr_squelch_cc(
            squelch_threshold, 0.01, 10, True
        )
        self.analog_am_demod_cf_0 = analog.am_demod_cf(
            channel_rate=40000,
            audio_decim=5,
            audio_pass=3400,
            audio_stop=4500,
        )
        self.analog_agc_xx_0 = analog.agc_cc((1e-3), 1.0, 1.0)
        self.analog_agc_xx_0.set_max_gain(65536)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_agc_xx_0, 0), (self.analog_pwr_squelch_xx_0, 0))
        self.connect(
            (self.analog_am_demod_cf_0, 0), (self.blocks_multiply_const_vxx_0, 0)
        )
        self.connect((self.analog_pwr_squelch_xx_0, 0), (self.analog_am_demod_cf_0, 0))
        self.connect(
            (self.blocks_multiply_const_vxx_0, 0), (self.blocks_wavfile_sink_0, 0)
        )
        self.connect(
            (self.blocks_multiply_const_vxx_0, 0), (self.qtgui_time_sink_x_0, 0)
        )
        self.connect((self.freq_xlating_fir_filter_xxx_0, 0), (self.analog_agc_xx_0, 0))
        self.connect(
            (self.freq_xlating_fir_filter_xxx_0, 0), (self.qtgui_freq_sink_x_0, 0)
        )
        self.connect(
            (self.zeromq_sub_source_0, 0), (self.freq_xlating_fir_filter_xxx_0, 0)
        )

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "atc_am_rx")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_zmq_address(self):
        return self.zmq_address

    def set_zmq_address(self, zmq_address):
        self.zmq_address = zmq_address

    def get_volume(self):
        return self.volume

    def set_volume(self, volume):
        self.volume = volume
        self.blocks_multiply_const_vxx_0.set_k(self.volume)

    def get_squelch_threshold(self):
        return self.squelch_threshold

    def set_squelch_threshold(self, squelch_threshold):
        self.squelch_threshold = squelch_threshold
        self.analog_pwr_squelch_xx_0.set_threshold(self.squelch_threshold)

    def get_quad_rate(self):
        return self.quad_rate

    def set_quad_rate(self, quad_rate):
        self.quad_rate = quad_rate
        self.freq_xlating_fir_filter_xxx_0.set_taps(
            firdes.low_pass(
                1.0, self.quad_rate, self.channel_bw / 2, 2500, window.WIN_KAISER, 6.76
            )
        )

    def get_output_wav_path(self):
        return self.output_wav_path

    def set_output_wav_path(self, output_wav_path):
        self.output_wav_path = output_wav_path
        self.blocks_wavfile_sink_0.open(self.output_wav_path)

    def get_if_freq(self):
        return self.if_freq

    def set_if_freq(self, if_freq):
        self.if_freq = if_freq
        self.freq_xlating_fir_filter_xxx_0.set_center_freq(self.if_freq)

    def get_channel_bw(self):
        return self.channel_bw

    def set_channel_bw(self, channel_bw):
        self.channel_bw = channel_bw
        self.freq_xlating_fir_filter_xxx_0.set_taps(
            firdes.low_pass(
                1.0, self.quad_rate, self.channel_bw / 2, 2500, window.WIN_KAISER, 6.76
            )
        )

    def get_audio_rate(self):
        return self.audio_rate

    def set_audio_rate(self, audio_rate):
        self.audio_rate = audio_rate
        self.qtgui_time_sink_x_0.set_samp_rate(self.audio_rate)


def main(top_block_cls=atc_am_rx, options=None):
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
