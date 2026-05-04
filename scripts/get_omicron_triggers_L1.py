import matplotlib

from gwtrigfind import find_trigger_files
from gwpy.table import (Table, EventTable)
from gwpy.table.filters import in_segmentlist
from gwpy.segments import DataQualityFlag
from gwpy.time import tconvert
from gwpy.timeseries import TimeSeries
from gwpy.segments import Segment, SegmentList

start = 1238166018 #O3a start
end = 1253977218 #O3a end
hoft_channel = 'L1:GDS-CALIB_STRAIN'

omi_files_o3a = find_trigger_files(hoft_channel, 'omicron', start, end)
print(omi_files_o3a[:3])

omi_events_o3a = EventTable.read(omi_files_o3a, tablename='sngl_burst', format='ligolw')
print(omi_events_o3a[:3])

import numpy as np
livingston_o3a_peak_times = np.asarray(omi_events_o3a['peak_time'])
livingston_o3a_durations = np.asarray(omi_events_o3a['duration'])
livingston_o3a_peak_times.shape, livingston_o3a_durations.shape

print(livingston_o3a_peak_times[:10], livingston_o3a_durations[:10])

print(len(livingston_o3a_peak_times), len(livingston_o3a_durations))

np.save('livingston_o3a_peak_times', livingston_o3a_peak_times)
np.save('livingston_o3a_durations', livingston_o3a_durations)

livingston_o3a_peak_times = None
livingston_o3a_durations = None
omi_events_o3a = None

start = 1256655618 #O3b start
end = 1269363618 #O3b end
hoft_channel = 'L1:GDS-CALIB_STRAIN'

omi_files_o3b = find_trigger_files(hoft_channel, 'omicron', start, end)
print(omi_files_o3b[:3])

omi_events_o3b = EventTable.read(omi_files_o3b, tablename='sngl_burst', format='ligolw')
print(omi_events_o3b[:3])

livingston_o3b_peak_times = np.asarray(omi_events_o3b['peak_time'])
livingston_o3b_durations = np.asarray(omi_events_o3b['duration'])
livingston_o3b_peak_times.shape, livingston_o3b_durations.shape

np.save('livingston_o3b_peak_times', livingston_o3b_peak_times)
np.save('livingston_o3b_durations', livingston_o3b_durations)