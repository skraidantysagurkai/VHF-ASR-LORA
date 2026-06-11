#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: VHF Radio Noise Simulation (16 kHz FLAC)
# Description: VHF Radio Communication Simulation with Variable Noise – FLAC I/O at 16 kHz
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
import sip
from gnuradio import analog
from gnuradio import audio
from gnuradio import blocks
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from gnuradio.qtgui import Range, GrRangeWidget
from PyQt5 import QtCore


class vhf_noise_sim(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(
            self, "VHF Radio Noise Simulation (16 kHz FLAC)", catch_exceptions=True
        )
        Qt.QWidget.__init__(self)
        self.setWindowTitle("VHF Radio Noise Simulation (16 kHz FLAC)")
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

        self.settings = Qt.QSettings("GNU Radio", "vhf_noise_sim")

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
        self.snr_db = snr_db = 20
        self.samp_rate = samp_rate = 16000
        self.out_file = out_file = (
            "/Users/ezka/PycharmProjects/VHF-ASR/data/output.flac"
        )
        self.noise_amp = noise_amp = pow(10.0, -snr_db / 20.0)
        self.in_file = in_file = (
            "/Users/ezka/PycharmProjects/VHF-ASR/data/163249/3853-163249-0000.flac"
        )
        self.imp_en = imp_en = 1
        self.imd_en = imd_en = 1
        self.atm_en = atm_en = 1

        ##################################################
        # Blocks
        ##################################################
        _imp_en_check_box = Qt.QCheckBox("Impulse Noise")
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
        self.top_grid_layout.addWidget(_imp_en_check_box, 0, 3, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(3, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        _imd_en_check_box = Qt.QCheckBox("IMD / Channel Noise")
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
        self.top_grid_layout.addWidget(_imd_en_check_box, 0, 4, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(4, 5):
            self.top_grid_layout.setColumnStretch(c, 1)
        _atm_en_check_box = Qt.QCheckBox("Atmospheric Noise")
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
        self.top_grid_layout.addWidget(_atm_en_check_box, 0, 2, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.time_sink = qtgui.time_sink_f(
            1024,  # size
            samp_rate,  # samp_rate
            "VHF Time Domain",  # name
            1,  # number of inputs
            None,  # parent
        )
        self.time_sink.set_update_time(0.10)
        self.time_sink.set_y_axis(-1, 1)

        self.time_sink.set_y_label("Amplitude", "")

        self.time_sink.enable_tags(True)
        self.time_sink.set_trigger_mode(
            qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, ""
        )
        self.time_sink.enable_autoscale(True)
        self.time_sink.enable_grid(True)
        self.time_sink.enable_axis_labels(True)
        self.time_sink.enable_control_panel(False)
        self.time_sink.enable_stem_plot(False)

        labels = ["Signal + Noise", "", "", "", "", "", "", "", "", ""]
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
                self.time_sink.set_line_label(i, "Data {0}".format(i))
            else:
                self.time_sink.set_line_label(i, labels[i])
            self.time_sink.set_line_width(i, widths[i])
            self.time_sink.set_line_color(i, colors[i])
            self.time_sink.set_line_style(i, styles[i])
            self.time_sink.set_line_marker(i, markers[i])
            self.time_sink.set_line_alpha(i, alphas[i])

        self._time_sink_win = sip.wrapinstance(self.time_sink.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._time_sink_win, 4, 0, 3, 5)
        for r in range(4, 7):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 5):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.throttle = blocks.throttle(gr.sizeof_float * 1, samp_rate, True)
        self._snr_db_range = Range(-20, 40, 1, 20, 200)
        self._snr_db_win = GrRangeWidget(
            self._snr_db_range,
            self.set_snr_db,
            "SNR (dB)",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
            "value",
        )

        self.top_grid_layout.addWidget(self._snr_db_win, 0, 0, 1, 2)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.sig_imd2 = analog.sig_source_f(
            samp_rate, analog.GR_SIN_WAVE, 200, 0.5, 0, 0
        )
        self.sig_imd1 = analog.sig_source_f(
            samp_rate, analog.GR_SIN_WAVE, 100, 0.5, 0, 0
        )
        self.noise_imp = analog.noise_source_f(analog.GR_IMPULSE, 1, 7)
        self.noise_imd_bg = analog.noise_source_f(analog.GR_GAUSSIAN, 0.3, 13)
        self.noise_atm = analog.noise_source_f(analog.GR_GAUSSIAN, 1, 42)
        self.mult_imp = blocks.multiply_const_ff((noise_amp * imp_en))
        self.mult_imd = blocks.multiply_const_ff((noise_amp * imd_en))
        self.mult_atm = blocks.multiply_const_ff((noise_amp * atm_en))
        self.freq_sink = qtgui.freq_sink_f(
            1024,  # size
            window.WIN_BLACKMAN_hARRIS,  # wintype
            0,  # fc
            samp_rate,  # bw
            "VHF Spectrum (16 kHz)",  # name
            1,
            None,  # parent
        )
        self.freq_sink.set_update_time(0.10)
        self.freq_sink.set_y_axis((-140), 10)
        self.freq_sink.set_y_label("Relative Gain", "dB")
        self.freq_sink.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.freq_sink.enable_autoscale(False)
        self.freq_sink.enable_grid(True)
        self.freq_sink.set_fft_average(0.2)
        self.freq_sink.enable_axis_labels(True)
        self.freq_sink.enable_control_panel(False)
        self.freq_sink.set_fft_window_normalized(False)

        self.freq_sink.set_plot_pos_half(not True)

        labels = ["Signal + Noise", "", "", "", "", "", "", "", "", ""]
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
                self.freq_sink.set_line_label(i, "Data {0}".format(i))
            else:
                self.freq_sink.set_line_label(i, labels[i])
            self.freq_sink.set_line_width(i, widths[i])
            self.freq_sink.set_line_color(i, colors[i])
            self.freq_sink.set_line_alpha(i, alphas[i])

        self._freq_sink_win = sip.wrapinstance(self.freq_sink.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._freq_sink_win, 1, 0, 3, 5)
        for r in range(1, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 5):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.flac_source = blocks.wavfile_source(in_file, True)
        self.flac_sink = blocks.wavfile_sink(
            out_file, 1, samp_rate, blocks.FORMAT_WAV, blocks.FORMAT_PCM_16, False
        )
        self.audio_out = audio.sink(samp_rate, "", True)
        self.add_main = blocks.add_vff(1)
        self.add_imd = blocks.add_vff(1)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.add_imd, 0), (self.mult_imd, 0))
        self.connect((self.add_main, 0), (self.throttle, 0))
        self.connect((self.flac_source, 0), (self.add_main, 0))
        self.connect((self.mult_atm, 0), (self.add_main, 1))
        self.connect((self.mult_imd, 0), (self.add_main, 3))
        self.connect((self.mult_imp, 0), (self.add_main, 2))
        self.connect((self.noise_atm, 0), (self.mult_atm, 0))
        self.connect((self.noise_imd_bg, 0), (self.add_imd, 2))
        self.connect((self.noise_imp, 0), (self.mult_imp, 0))
        self.connect((self.sig_imd1, 0), (self.add_imd, 0))
        self.connect((self.sig_imd2, 0), (self.add_imd, 1))
        self.connect((self.throttle, 0), (self.audio_out, 0))
        self.connect((self.throttle, 0), (self.flac_sink, 0))
        self.connect((self.throttle, 0), (self.freq_sink, 0))
        self.connect((self.throttle, 0), (self.time_sink, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "vhf_noise_sim")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_snr_db(self):
        return self.snr_db

    def set_snr_db(self, snr_db):
        self.snr_db = snr_db
        self.set_noise_amp(pow(10.0, -self.snr_db / 20.0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.freq_sink.set_frequency_range(0, self.samp_rate)
        self.sig_imd1.set_sampling_freq(self.samp_rate)
        self.sig_imd2.set_sampling_freq(self.samp_rate)
        self.throttle.set_sample_rate(self.samp_rate)
        self.time_sink.set_samp_rate(self.samp_rate)

    def get_out_file(self):
        return self.out_file

    def set_out_file(self, out_file):
        self.out_file = out_file
        self.flac_sink.open(self.out_file)

    def get_noise_amp(self):
        return self.noise_amp

    def set_noise_amp(self, noise_amp):
        self.noise_amp = noise_amp
        self.mult_atm.set_k((self.noise_amp * self.atm_en))
        self.mult_imd.set_k((self.noise_amp * self.imd_en))
        self.mult_imp.set_k((self.noise_amp * self.imp_en))

    def get_in_file(self):
        return self.in_file

    def set_in_file(self, in_file):
        self.in_file = in_file

    def get_imp_en(self):
        return self.imp_en

    def set_imp_en(self, imp_en):
        self.imp_en = imp_en
        self._imp_en_callback(self.imp_en)
        self.mult_imp.set_k((self.noise_amp * self.imp_en))

    def get_imd_en(self):
        return self.imd_en

    def set_imd_en(self, imd_en):
        self.imd_en = imd_en
        self._imd_en_callback(self.imd_en)
        self.mult_imd.set_k((self.noise_amp * self.imd_en))

    def get_atm_en(self):
        return self.atm_en

    def set_atm_en(self, atm_en):
        self.atm_en = atm_en
        self._atm_en_callback(self.atm_en)
        self.mult_atm.set_k((self.noise_amp * self.atm_en))


def main(top_block_cls=vhf_noise_sim, options=None):
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
