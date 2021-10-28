# Deep learning and array processing libraries
import numpy as np 

def c_persist(sc, is_terminals):
    return np.sum(sc & is_terminals) / np.sum(sc)

def weighted_c_persist(gt, sc, is_terminals):
    classes = np.unique(gt)
    values = np.zeros(len(classes))
    for index, label in enumerate(classes):
        values[index] = np.sum(sc[gt == label] & is_terminals[gt == label]) / np.sum(sc[gt == label])
    values = np.nan_to_num(values)
    return np.mean(values)

def c_withdrawn(sc, is_roots):
    return np.sum(sc & is_roots) / np.sum(sc)

def weighted_c_withdrawn(gt, sc, is_roots):
    classes = np.unique(gt)
    values = np.zeros(len(classes))
    for index, label in enumerate(classes):
        values[index] = np.sum(sc[gt == label] & is_roots[gt == label]) / np.sum(sc[gt == label])
    values = np.nan_to_num(values)
    return np.mean(values)

def c_soften(cp, cw):
    return 1 - (cp + cw)

def weighted_c_soften(wcp, wcw):
    return 1 - (wcp + wcw)
    
def c_corrupt(sc, is_paths):
    return np.sum(sc & np.logical_not(is_paths)) / np.sum(sc)

def weighted_c_corrupt(gt, sc, is_paths):
    classes = np.unique(gt)
    values = np.zeros(len(classes))
    for index, label in enumerate(classes):
        values[index] = np.sum(sc[gt == label] & np.logical_not(is_paths[gt == label])) / np.sum(sc[gt == label])
    values = np.nan_to_num(values)
    return np.mean(values)

def c_softdepth(s_soft, generalized_depths, terminal_depths):
    depth_ratios = generalized_depths / terminal_depths
    return np.sum(depth_ratios[s_soft]) / np.sum(s_soft)

def weighted_c_softdepth():
    pass

def ic_remain(sic, is_subsumers, is_roots):
    return np.sum(np.logical_not(is_subsumers[sic]) & np.logical_not(is_roots[sic])) / np.sum(sic)

def weighted_ic_remain(gt, sic, is_subsumers, is_roots):
    classes = np.unique(gt)
    values = np.zeros(len(classes))
    for index, label in enumerate(classes):
        values[index] = np.sum(np.logical_not(is_subsumers[sic & (gt == label)]) & np.logical_not(is_roots[sic & (gt == label)])) / np.sum(sic & (gt == label))
    values = np.nan_to_num(values)
    return np.mean(values)

def ic_withdrawn(sic, is_roots):
    return np.sum(sic & is_roots) / np.sum(sic)

def weighted_ic_withdrawn(gt, sic, is_roots):
    classes = np.unique(gt)
    values = np.zeros(len(classes))
    for index, label in enumerate(classes):
        values[index] = np.sum(sic[gt == label] & is_roots[gt == label]) / np.sum(sic[gt == label])
    values = np.nan_to_num(values)
    return np.mean(values)

def ic_reform(icr, icw):
    return 1 - (icr + icw)

def weighted_ic_reform(wicr, wicw):
    return 1 - (wicr + wicw)

def ic_refdepth(s_ref, reformed_depths, lcs_depths):
    depth_ratios = reformed_depths / lcs_depths
    return np.sum(depth_ratios[s_ref]) / np.sum(s_ref)

def weighted_ic_refdepth():
    pass

def info_gain(sc, num_classes, num_terminals):
    return np.sum((np.log2(num_classes) - np.log2(num_terminals[sc])) / np.log2(num_classes)) / len(num_terminals)

def weighted_info_gain(gt, sc, num_classes, num_terminals):
    classes = np.unique(gt)
    values = np.zeros(len(classes))
    for index, label in enumerate(classes):
        values[index] = np.sum((np.log2(num_classes) - np.log2(num_terminals[sc & (gt == label)])) / np.log2(num_classes)) / len(sc[gt == label])
    values = np.nan_to_num(values)
    return np.mean(values)

def weighted_accuracy(gt, correct):
    classes = np.unique(gt)
    values = np.zeros(len(classes))
    for index, label in enumerate(classes):
        values[index] = np.sum(correct[gt == label]) / len(correct[gt == label])
    values = np.nan_to_num(values)
    return np.mean(values)

def weighted_valid(gt, is_roots):
    classes = np.unique(gt)
    values = np.zeros(len(classes))
    for index, label in enumerate(classes):
        values[index] = np.sum(np.logical_not(is_roots[gt == label])) / len(is_roots[gt == label]) * 100
    values = np.nan_to_num(values)
    return np.mean(values)

def weighted_specified_stress(gt, stressed, finals):
    classes = np.unique(gt)
    values = np.zeros(len(classes))
    healthy = 0
    for index, label in enumerate(classes):
        if np.sum(stressed[gt == label]) > 0:
            values[index] = np.sum(stressed[gt == label] & (finals[gt == label] != "stressed") & (finals[gt == label] != "unknown")) / np.sum(stressed[gt == label]) * 100
        else:
            healthy = index
    values = np.nan_to_num(values)
    values = np.delete(values, healthy)
    return np.mean(values)

def weighted_posteriors(gt, posteriors):
    classes = np.unique(gt)
    values = np.zeros(len(classes))
    for index, label in enumerate(classes):
        values[index] = np.sum(posteriors[gt == label]) / len(posteriors[gt == label])
    values = np.nan_to_num(values)
    return np.mean(values)
