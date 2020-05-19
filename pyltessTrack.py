#!/usr/bin/python
#
#   Copyright (C) IMDEA Networks Institute 2019
#   This program is free software: you can redistribute it and/or modify
#
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see http://www.gnu.org/licenses/.
#
#   Authors: Roberto Calvo-Palomino <roberto.calvo [at] imdea [dot] org>
#

import argparse
import json
import datetime
import numpy as np
import subprocess

from foc.pssdrift import get_drift

import warnings
from matplotlib.pyplot import psd, xlabel, ylabel, show

warnings.simplefilter(action = 'ignore', category = FutureWarning)


# Load Zadoff sequencies
def get_zadoff_seqs (filename):
    f = open(filename, 'rb');
    dat = np.fromfile(f, dtype = np.float32)
    dat = dat.astype(np.float32).view(np.complex64)

    return dat


# Constants
VERSION = "1.0-rc1"
RESAMPLE_FACTOR = 20
PSS_STEP = 9600
SEARCH_WINDOW = 150
PREAMBLE = 30

# variables
fs = 1.92e6
fc = 806e6
chan = 0
gain = 30
source = -1

AUX_BUFFER_SIZE = 20 * 1024

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-s", '--source', type = int, dest = 'source', help = "Set default SDR source device to use", default = source)
    parser.add_argument("-f", '--frequency', type = int, dest = 'frequency', help = "Set LTE center frequency of the channel (Hz)", default = fc)
    parser.add_argument("-g", '--gain', type = int, dest = 'gain', help = "Gain", default = gain)
    parser.add_argument("-t", "--time", type = int, dest = 'time', help = "Seconds collecting data on LTE frequency", default = 1)
    parser.add_argument("-j", '--json-file', dest = 'json', type = str, help = "Set the json file where results will be written", default = None)
    parser.add_argument("-i", '--input-file', dest = 'inFile', type = str, help = "Use existing IQ file", default = None)
    parser.add_argument("-d", '--debug', dest = 'debug', help = "enable debug mode with plots", action = 'store_true', default = False)
    args = parser.parse_args()

    print("#########################################")
    print("#      = pyLTESS-Track v" + VERSION + " =       #")
    print("#                                       #")
    print("# A precise and fast frequency offset   #")
    print("# estimation for low-cost SDR platforms #")
    print("#                                       #")
    print("# -- The Electrosense Team              #")
    print("#########################################")

    fc = args.frequency
    gain = args.gain
    sampling_time = args.time

    # Look at for SDR devices
#     sdr_list = SoapySDR.Device.enumerate()
    index = 0

    # Set SDR and read samples

    TOTAL_BUFFER_SIZE = int(fs * args.time)

    print("[LTESSTRACK] Reading for %d seconds at %d MHz with gain=%d ... " % (args.time, args.frequency, args.gain))
    acq_time = datetime.datetime.now()

    data = []

    if args.inFile is None:  # Get fresh data
        # This is unsafe, use rtlsdr wrapper instead
        command = "rtl_sdr -s " + str(fs) + " -f " + str(fc) + " -g 20 -n " + str(TOTAL_BUFFER_SIZE) + " -"
        print("Collecting data with the command: " + command)
        with subprocess.Popen(command, stdout = subprocess.PIPE, shell = True) as proc:
            data = proc.stdout.read()  # Read all at once
    else:  # Read the file given
        with open(args.inFile, "rb") as iqFile:
            data = iqFile.read(TOTAL_BUFFER_SIZE)

    # Convert the sample to float complex values
    samples = np.frombuffer(data, dtype = np.uint8).astype(np.float32).view(np.complex64)

    if (args.debug):
        # use matplotlib to estimate and plot the PSD
        psd(samples, NFFT = 1024, Fs = fs / 1e6, Fc = fc / 1e6)
        xlabel('Frequency (MHz)')
        ylabel('Relative power (dB)')
        show()

    print("[LTESSTRACK] Estimating local oscilator error .... ")

    # load zadoof sequences (in time)
    try:
        Z_sequences = np.array([get_zadoff_seqs("lte/25-Zadoff.bin"), \
        get_zadoff_seqs("lte/29-Zadoff.bin"), \
        get_zadoff_seqs("lte/34-Zadoff.bin")])
    except FileNotFoundError:
        Z_sequences = np.array([get_zadoff_seqs("/usr/share/pyltesstrack/lte/25-Zadoff.bin"), \
        get_zadoff_seqs("/usr/share/pyltesstrack/lte/29-Zadoff.bin"), \
        get_zadoff_seqs("/usr/share/pyltesstrack/lte/34-Zadoff.bin")])

    # Get drift by analyzing the PSS time of arrival
    [PPM, delta_f, confidence] = get_drift(samples, Z_sequences, PREAMBLE, PSS_STEP, SEARCH_WINDOW, RESAMPLE_FACTOR, fs, debug_plot = args.debug)

    print("[LTESSTRACK] Local oscilator error: %.8f PPM - [%.2f Hz], confidence=%.3f" % (PPM, delta_f, confidence))

    if (args.json):
        data = {}
        data['datetime'] = str(acq_time)
        data['fc'] = args.frequency
        data['fs'] = int(fs)
        data['gain'] = args.gain
        data['sampling_time'] = args.time
        data['ppm'] = PPM
        data['confidence'] = confidence

        with open(args.json, 'w', encoding = 'utf-8') as f:
            json.dump(data, f, ensure_ascii = False, indent = 4)

        print("[LTESSTRACK] Results saved in " + args.json)
