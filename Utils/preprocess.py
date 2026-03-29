import numpy as np

def prepare_input(study_hours, attendance, mental_health, sleep_hours, part_time_job):
    ptj = 1 if part_time_job == "Yes" else 0
    return np.array([[attendance, mental_health, study_hours, sleep_hours, ptj]])