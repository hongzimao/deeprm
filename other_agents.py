import numpy as np


def get_packer_action(machine, job_slot):
        align_score = 0
        act = len(job_slot.slot)  # if no action available, hold

        for i in range(len(job_slot.slot)):
            new_job = job_slot.slot[i]
            if new_job is not None:  # there is a pending job

                avbl_res = machine.avbl_slot[:new_job.len, :]
                res_left = avbl_res - new_job.res_vec

                if np.all(res_left[:] >= 0):  # enough resource to allocate

                    tmp_align_score = avbl_res[0, :].dot(new_job.res_vec)

                    if tmp_align_score > align_score:
                        align_score = tmp_align_score
                        act = i
        return act


def get_sjf_action(machine, job_slot):
        sjf_score = 0
        act = len(job_slot.slot)  # if no action available, hold

        for i in range(len(job_slot.slot)):
            new_job = job_slot.slot[i]
            if new_job is not None:  # there is a pending job

                avbl_res = machine.avbl_slot[:new_job.len, :]
                res_left = avbl_res - new_job.res_vec

                if np.all(res_left[:] >= 0):  # enough resource to allocate

                    tmp_sjf_score = 1 / float(new_job.len)

                    if tmp_sjf_score > sjf_score:
                        sjf_score = tmp_sjf_score
                        act = i
        return act


def get_packer_sjf_action(machine, job_slot, knob):  # knob controls which to favor, 1 to packer, 0 to sjf

        combined_score = 0
        act = len(job_slot.slot)  # if no action available, hold

        for i in range(len(job_slot.slot)):
            new_job = job_slot.slot[i]
            if new_job is not None:  # there is a pending job

                avbl_res = machine.avbl_slot[:new_job.len, :]
                res_left = avbl_res - new_job.res_vec

                if np.all(res_left[:] >= 0):  # enough resource to allocate

                    tmp_align_score = avbl_res[0, :].dot(new_job.res_vec)
                    tmp_sjf_score = 1 / float(new_job.len)

                    tmp_combined_score = knob * tmp_align_score + (1 - knob) * tmp_sjf_score

                    if tmp_combined_score > combined_score:
                        combined_score = tmp_combined_score
                        act = i
        return act


def get_random_action(job_slot):
    num_act = len(job_slot.slot) + 1  # if no action available,
    act = np.random.randint(num_act)
    return act
