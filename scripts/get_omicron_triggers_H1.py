import numpy as np

from gwtrigfind import find_trigger_files
from gwpy.table import EventTable

start = 1238166018  # O3a start
end   = 1253977218  # O3a end
hoft_channel = 'H1:GDS-CALIB_STRAIN'

omi_files_o3a = find_trigger_files(hoft_channel, 'omicron', start, end)
print(omi_files_o3a[:3])

omi_events_o3a = EventTable.read(omi_files_o3a, tablename='sngl_burst', format='ligolw')
print(omi_events_o3a[:3])

hanford_o3a_peak_times = np.asarray(omi_events_o3a['peak_time'])
hanford_o3a_durations  = np.asarray(omi_events_o3a['duration'])
print(hanford_o3a_peak_times.shape, hanford_o3a_durations.shape)

np.save('hanford_o3a_peak_times', hanford_o3a_peak_times)
np.save('hanford_o3a_durations',  hanford_o3a_durations)

hanford_o3a_peak_times = None
hanford_o3a_durations  = None
omi_events_o3a         = None

start = 1256655618  # O3b start
end   = 1269363618  # O3b end

omi_files_o3b = find_trigger_files(hoft_channel, 'omicron', start, end)
print(omi_files_o3b[:3])

omi_events_o3b = EventTable.read(omi_files_o3b, tablename='sngl_burst', format='ligolw')
print(omi_events_o3b[:3])

hanford_o3b_peak_times = np.asarray(omi_events_o3b['peak_time'])
hanford_o3b_durations  = np.asarray(omi_events_o3b['duration'])
print(hanford_o3b_peak_times.shape, hanford_o3b_durations.shape)

np.save('hanford_o3b_peak_times', hanford_o3b_peak_times)
np.save('hanford_o3b_durations',  hanford_o3b_durations)
