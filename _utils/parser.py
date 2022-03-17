import numpy as np

def parse_settings(settings):
    cl_model, pr_model, cl_atts = settings.split('._.')
    return parse_cl_atts(cl_atts)
    #return parse_model(cl_model), parse_model(pr_model), parse_cl_atts(cl_atts)

def parse_cl_atts(string):
    if string == "all":
        return lambda x: np.ones(x.shape[1]).astype(bool)
    else:
        # parses <model>._.<model>._.<att_start>-<att_end>
        # SAMPLE: lr._.lr._.0-111
        # Returns class_model, propensity_model, list of attribute indices
        atts = []
        for s in string.split('.'):
            if '-' in s:
                begin, end = s.split('-')
                atts += list(range(int(begin), int(end) + 1))
            else:
                atts += [int(s)]
        return atts

# def parse_model(string):
#     return {
#         "lr": LogisticRegression,
#     }[string]

def read_data(data_files, partitions_file=None, delimiter=None):
    datas = list(map(lambda data_path: np.loadtxt(data_path, delimiter=delimiter), data_files))

    if partitions_file == None:
        return tuple(datas)

    partitions = np.loadtxt(partitions_file, delimiter=delimiter, dtype=int)

    data_per_partition = []
    for partition in sorted(list(set(partitions) - set([0]))):
        data_per_partition.append(tuple(map(lambda data: data[partitions == partition], datas)))

    return data_per_partition