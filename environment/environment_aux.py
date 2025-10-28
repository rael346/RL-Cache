import json
import pickle
import random
import sys
from copy import deepcopy
from multiprocessing import Process, Queue
from os import listdir
from os.path import isfile, join

sys.path.insert(0, "./build")

import numpy as np
from AdaptSizeSim import AdaptSizeSimulator

# from GDSim import GDSimulator
from hurry.filesize import size as hurry_fsize
from LRUSim import LRUSimulator

# from MLSim import MLSimulator
# from S4LRUSim import S4LRUSimulator
from tqdm import tqdm


def dump_cache(cache):
    return {
        "ratings": cache.get_ratings(),
        "sizes": cache.get_sizes(),
        "used space": cache.get_used_space(),
        "cache size": cache.get_cache_size(),
        "L": cache.get_L(),
        "misses": cache.get_misses(),
        "hits": cache.get_hits(),
        "byte misses": cache.get_byte_misses(),
        "byte hits": cache.get_byte_hits(),
        "total rating": cache.get_total_rating(),
        "ml eviction": cache.get_ml_eviction(),
    }


def restore_cache(data, cache):
    cache.set_ratings(data["ratings"])
    cache.set_sizes(data["sizes"])
    cache.set_used_space(data["used space"])
    cache.set_cache_size(data["cache size"])
    cache.set_L(data["L"])
    cache.set_misses(data["misses"])
    cache.set_hits(data["hits"])
    cache.set_byte_misses(data["byte misses"])
    cache.set_byte_hits(data["byte hits"])
    cache.set_total_rating(data["total rating"])
    cache.set_ml_eviction(data["ml eviction"])
    return cache


def write_performance_to_log(log, data, iteration, prefix):
    pstr = ["{:15d}"] + ["{:^15s} {:10.5f}" for _ in data.keys()]
    pstr = " ".join(pstr)
    insertion = [iteration]
    for key in data.keys():
        insertion.append(key)
        insertion.append(data[key])

    pstr = pstr.format(*insertion)
    pstr = " ".join(pstr.split())
    log.write(prefix + " " + pstr + "\n")
    log.flush()


def write_run(log, run_id):
    log.write("RUN" + " " + str(run_id) + "\n")
    log.flush()


def write_accuracy_to_log(log, a, e, step, repetition):
    data = [step, repetition, a, e]
    data = [str(item) for item in data]
    data = " ".join(data)
    log.write("ACCURACY" + " " + data + "\n")
    log.flush()


def to_ts(time_diff):
    time_series = ["{:2f}s", "{:2f}m", "{:2f}h", "{:f}d"]
    data_to_use = []
    seconds = time_diff % 60
    time_diff /= 60

    data_to_use.append(seconds)

    minutes = time_diff % 60
    time_diff /= 60

    if minutes != 0 or time_diff % 60 != 0:
        data_to_use.append(minutes)

    hours = time_diff % 24
    time_diff /= 24

    if hours != 0 or time_diff % 24 != 0:
        data_to_use.append(hours)

    days = time_diff % 24

    if days != 0:
        data_to_use.append(days)

    data_to_print = []
    for i in reversed(range(len(data_to_use))):
        data_to_print.append(time_series[i].format(data_to_use[i]).replace(" ", "0"))
    return ":".join(data_to_print)


def wrapped_f(f, q, *args, **kwargs):
    ret = f(*args, **kwargs)
    q.put(ret)


def threaded(f, daemon=False):
    def wrap(*args, **kwargs):
        q = Queue()

        t = Process(
            target=wrapped_f,
            args=(f, q) + args,
            kwargs=kwargs,
        )
        t.daemon = daemon
        t.start()
        t.result_queue = q
        return t

    return wrap


def copy_object(object_to_copy):
    res = deepcopy(object_to_copy)
    res.set_ratings(object_to_copy.get_ratings())

    return res


def class2name(class_info):
    basic_name = [[class_info["admission"]], [class_info["eviction"]]]
    if class_info["operational mode"] == "" and class_info["random mode"]:
        basic_name.append(["RNG", "DET"])
    else:
        if class_info["operational mode"] != "":
            basic_name.append([class_info["operational mode"]])
    if class_info["size"] != 0:
        basic_name.append([str(class_info["size"])])
    if class_info["UID"] != "":
        basic_name.append([class_info["UID"]])
    possible_names = basic_name[0]
    for lst in basic_name[1:]:
        possible_names = [
            [possible_name + "-" + next_item for possible_name in possible_names]
            for next_item in lst
        ]
        possible_names = sum(possible_names, [])
    return possible_names


def name2class(name):
    name = name.split("-")
    name_admission = name[0]
    name_eviction = name[1]
    operational_mode = ""
    uid = ""
    size = 0
    for item in name[2:]:
        try:
            size = int(item)
            continue
        except ValueError:
            pass
        if item in ["DET", "RNG"]:
            operational_mode = item
            continue
        uid = item

    randomness_type = operational_mode == "RNG"

    admission_random = False
    eviction_random = False

    class_type = None
    admission_index = -1
    eviction_index = -1

    if name_admission == "AS":
        admission_random = False
        admission_index = 5
        class_type = AdaptSizeSimulator
    if name_admission == "AL":
        admission_random = False
        admission_index = 5
    if name_admission == "SH":
        admission_random = False
        admission_index = 2
    if name_admission == "ML":
        admission_random = True
        admission_index = -1

    # if name_eviction == "GDSF":
    #     class_type = GDSimulator
    #     eviction_random = False
    #     eviction_index = 0
    if name_eviction == "LRU":
        if class_type is None:
            class_type = LRUSimulator
        eviction_random = False
        eviction_index = 1
    # if name_eviction == "S4LRU":
    #     class_type = S4LRUSimulator
    #     eviction_random = False
    #     eviction_index = 1
    # if name_eviction == "LFU":
    #     class_type = LRUSimulator
    #     eviction_random = False
    #     eviction_index = 4
    # if name_eviction == "Oracle":
    #     class_type = GDSimulator
    #     eviction_random = False
    #     eviction_index = 3
    # if name_eviction == "ML":
    #     class_type = MLSimulator
    #     eviction_random = True
    #     eviction_index = -1
    # if name_eviction == "MLGD":
    #     class_type = GDSimulator
    #     eviction_random = True
    #     eviction_index = -1
    # if name_eviction == "MLLRU":
    #     class_type = LRUSimulator
    #     eviction_random = True
    #     eviction_index = -1

    assert class_type is not None

    short_name = name_admission + "-" + name_eviction
    short_name_unique = name_admission + "-" + name_eviction
    if uid != "":
        short_name_unique += "-" + uid

    unsized_name_suffix = ""

    if operational_mode != "":
        unsized_name_suffix = "-" + operational_mode
    if uid != "":
        unsized_name_suffix += "-" + uid

    return {
        "class": class_type,
        "admission": name_admission,
        "eviction": name_eviction,
        "size": size,
        "actual size": size * 1024 * 1024,
        "text size": hurry_fsize(size * 1024 * 1024),
        "admission mode": admission_random,
        "admission index": admission_index,
        "eviction mode": eviction_random,
        "eviction index": eviction_index,
        "random mode": eviction_random or admission_random,
        "random": randomness_type,
        "operational mode": operational_mode,
        "short name": short_name,
        "unsized name": short_name + unsized_name_suffix,
        "unique short name": short_name_unique,
        "UID": uid,
    }


def resolve_field(names, field, avoid_value):
    modes_resolved = {}
    for name in names.keys():
        for conflict_name in names[name]:
            info = name2class(conflict_name)
            if info[field] != avoid_value:
                new_name = name + "-" + str(info[field])
            else:
                new_name = name
            if new_name not in modes_resolved.keys():
                modes_resolved[new_name] = []
            modes_resolved[new_name].append(conflict_name)
    return modes_resolved


def clean_names(conflicting_names, name_mapping):
    rest = {}
    for name in conflicting_names.keys():
        if len(conflicting_names[name]) == 1:
            name_mapping[conflicting_names[name][0]] = name
        else:
            rest[name] = conflicting_names[name]
    return rest, name_mapping


def compress_names(names):
    name_mapping = {}
    conflicting_names = {}
    for name in names:
        info = name2class(name)
        if info["short name"] not in conflicting_names.keys():
            conflicting_names[info["short name"]] = []
        conflicting_names[info["short name"]].append(name)
    to_resolve = [
        (True, "operational mode", ""),
        (False, "text size", "0B"),
        (True, "size", 0),
        (True, "UID", ""),
    ]
    conflicting_names, name_mapping = clean_names(conflicting_names, name_mapping)
    for renew, field, value in to_resolve:
        conflicting_names_modified = resolve_field(conflicting_names, field, value)
        conflicting_names_modified, name_mapping_modified = clean_names(
            conflicting_names_modified, name_mapping
        )
        if len(conflicting_names_modified.keys()) == 0:
            return name_mapping_modified
        if renew and len(conflicting_names_modified.keys()) > len(
            conflicting_names.keys()
        ):
            conflicting_names, name_mapping = (
                conflicting_names_modified,
                name_mapping_modified,
            )
    assert False


def extreme_compress(name):
    info = name2class(name)
    replacement_table = {
        "AL": "A",
        "SH": "S",
        "LRU": "R",
        "LFU": "F",
        "GDSF": "G",
        "ML": "M",
        "RNG": "R",
        "DET": "D",
        "AS": "T",
        "Oracle": "O",
        "S4LRU": "S4",
        "SLRU": "S2",
    }
    name_converted = [
        replacement_table[info["admission"]],
        replacement_table[info["eviction"]],
    ]
    if info["operational mode"] != "":
        name_converted.append(replacement_table[info["operational mode"]])
    if info["size"] != 0:
        name_converted.append(info["text size"])
    if info["UID"] != "":
        name_converted.append(info["UID"])
    return "".join(name_converted)


def metric_funct(x, y, alpha):
    return x * alpha + y * (1 - alpha)


def metric(algorithm, alpha):
    return metric_funct(algorithm.byte_hit_rate(), algorithm.hit_rate(), alpha)


def compute_rating(value, total):
    w = (total - 1.0) / 2.0
    return 2.0 ** (float(value - w))


def sampling(prob_matrix):
    maximum_allowed = prob_matrix.shape[1]
    sums = prob_matrix.sum(axis=1)
    prob_matrix = prob_matrix / sums[:, np.newaxis]
    prob_matrix = prob_matrix.T
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return [min(item, maximum_allowed - 1) for item in k]


def get_unique_dict(data, labels=None):
    unique_data, data_counts = np.unique(data, return_counts=True)
    evc_counts = data_counts * 100.0 / len(data)
    data_combined = []
    collected_labels = set()
    for i in range(len(unique_data)):
        data_combined.append((unique_data[i], evc_counts[i]))
        collected_labels.add(unique_data[i])
    for item in labels:
        if item not in collected_labels:
            data_combined.append((item, 0))
    data_combined = sorted(data_combined, key=lambda x: x[0])
    result_data = []
    for item in data_combined:
        result_data.append(item[0])
        result_data.append(item[1])
    return result_data


def generate_predictions(features, index, rng_mode, binarize):
    if index < 0:
        if not rng_mode:
            predictions = np.argmax(features, axis=1)
        else:
            predictions = sampling(features)
    else:
        predictions = features[:, index]
    if binarize:
        predictions = [int(item) for item in predictions]
    return predictions


def generate_data_for_models(
    feature_sets, admission_models, eviction_models, models_mapping, batch_size
):
    decisions_adm = {}
    decisions_evc = {}

    feature_source_mapping = {}

    predictions_adm = []
    predictions_evc = []

    admission_models_predictions = []
    eviction_models_predictions = []

    nml_characteristics_seen = []
    seen = []

    cdk = 0
    operational_keys = []
    classical_keys = []
    for key in models_mapping.keys():
        class_info = name2class(key)
        if class_info["eviction mode"]:
            classical_keys.append(key)
        else:
            operational_keys.append(key)
    for key in classical_keys + operational_keys:
        a, e = models_mapping[key]
        class_info = name2class(key)
        adm_characteristics = str(a) + "|" + str(class_info["admission index"])
        nml_characteristics = "A" + str(a)
        if class_info["admission mode"]:
            adm_characteristics += "|ML"
            nml_characteristics += "|ML"
            if not class_info["random"]:
                adm_characteristics += "|AMAX"
            else:
                adm_characteristics += "|SAMPLE" + str(cdk)
                cdk += 1
        else:
            adm_characteristics += "|CLASSIC"
            nml_characteristics += "|CLASSIC"

        if nml_characteristics not in nml_characteristics_seen:
            if class_info["admission mode"]:
                admission_models_predictions.append(
                    admission_models[a].predict(
                        feature_sets[a], verbose=0, batch_size=batch_size
                    )
                )
            else:
                admission_models_predictions.append(feature_sets[a])
            nml_characteristics_seen.append(nml_characteristics)
        adm_data_index = nml_characteristics_seen.index(nml_characteristics)
        admission_data = admission_models_predictions[adm_data_index]
        if adm_characteristics not in seen:
            seen.append(adm_characteristics)
            predictions = generate_predictions(
                admission_data,
                class_info["admission index"],
                class_info["random"],
                True,
            )
            predictions_adm.append(predictions)

        decisions_adm[key] = seen.index(adm_characteristics)

        feature_source_mapping[key] = [adm_data_index, 0]

    nml_characteristics_seen = []
    seen = []

    cdk = 0

    operational_keys = []
    classical_keys = []
    for key in models_mapping.keys():
        class_info = name2class(key)
        if not class_info["eviction mode"]:
            classical_keys.append(key)
        else:
            operational_keys.append(key)

    for key in classical_keys + operational_keys:
        a, e = models_mapping[key]
        class_info = name2class(key)
        evc_characteristics = str(e) + "|" + str(class_info["eviction index"])
        nml_characteristics = "E" + str(e)
        if class_info["eviction mode"]:
            evc_characteristics += "|ML"
            nml_characteristics += "|ML"
            if not class_info["random"]:
                evc_characteristics += "|AMAX"
            else:
                evc_characteristics += "|SAMPLE" + str(cdk)
                cdk += 1
        else:
            evc_characteristics += "|CLASSIC"
            nml_characteristics += "|CLASSIC"

        if nml_characteristics not in nml_characteristics_seen:
            if class_info["eviction mode"]:
                eviction_models_predictions.append(
                    eviction_models[e].predict(
                        feature_sets[e], verbose=0, batch_size=batch_size
                    )
                )
            else:
                eviction_models_predictions.append(feature_sets[e])
            nml_characteristics_seen.append(nml_characteristics)

        evc_data_index = nml_characteristics_seen.index(nml_characteristics)
        eviction_data = eviction_models_predictions[evc_data_index]
        if evc_characteristics not in seen:
            seen.append(evc_characteristics)
            predictions = generate_predictions(
                eviction_data, class_info["eviction index"], class_info["random"], False
            )
            predictions_evc.append(predictions)
        decisions_evc[key] = seen.index(evc_characteristics)

        feature_source_mapping[key][1] = evc_data_index

    return (
        predictions_adm,
        decisions_adm,
        predictions_evc,
        decisions_evc,
        feature_source_mapping,
    )


def generate_data_for_models_light(
    keys, classical_feature_set, ml_feature_set, adm_model, evc_model, batch_size
):
    decisions_adm = {}
    decisions_evc = {}
    predictions_evc = None
    predictions_adm = None
    amax_adm = None
    amax_evc = None

    for key in keys:
        class_info = name2class(key)
        if class_info["admission mode"]:
            if predictions_adm is None:
                predictions_adm = adm_model.predict(
                    ml_feature_set, verbose=0, batch_size=batch_size
                )
                amax_adm = [int(np.argmax(item)) for item in predictions_adm]
            if class_info["random"]:
                decisions_adm[key] = [int(item) for item in sampling(predictions_adm)]
            else:
                decisions_adm[key] = amax_adm
        else:
            decisions_adm[key] = [
                int(item)
                for item in classical_feature_set[:, class_info["admission index"]]
            ]

        if class_info["eviction mode"]:
            if predictions_evc is None:
                predictions_evc = evc_model.predict(
                    ml_feature_set, verbose=0, batch_size=batch_size
                )
                amax_evc = [
                    compute_rating(np.argmax(item), predictions_evc.shape[1])
                    for item in predictions_evc
                ]
            if class_info["random"]:
                decisions_evc[key] = [
                    compute_rating(item, predictions_evc.shape[1])
                    for item in sampling(predictions_evc)
                ]
            else:
                decisions_evc[key] = amax_evc
        else:
            assert class_info["eviction index"] >= 0
            decisions_evc[key] = classical_feature_set[:, class_info["eviction index"]]
    return decisions_adm, decisions_evc


def select_elites(states, actions, rewards, percentile, max_samples):
    rewards = np.concatenate(rewards, axis=0)
    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)
    if percentile is None:
        percentile_value = np.amax(rewards)
    else:
        percentile_value = np.percentile(rewards, percentile)
    elite_indicies = rewards >= percentile_value
    states = states[elite_indicies]
    actions = actions[elite_indicies]
    samples_used = len(states)
    if samples_used == 0:
        return [], [], max(elite_indicies), 0, 0
    mean_actions = len(elite_indicies) / samples_used
    return states, actions, percentile_value, samples_used, mean_actions


def monte_carlo_sampling(states, actions, rewards, features_embedding, max_action):
    action_mapping = np.diag(np.ones((max_action,)))
    total_actions = np.zeros((features_embedding.shape[0], max_action))
    for i in tqdm(range(len(actions))):
        total_actions_done = np.zeros((features_embedding.shape[0], max_action))
        for j in range(len(actions[i])):
            action = actions[i][j]
            state = states[i][j]
            total_actions_done[state] = action_mapping[action]
        total_actions += total_actions_done * rewards[i]
    total_actions /= len(rewards)
    states_index = []
    for i in range(total_actions.shape[0]):
        pv = np.prod(total_actions[i])
        if pv != 0:
            states_index.append(i)
    X = features_embedding[states_index]
    Y = total_actions[states_index]
    return X, Y


def train_model(
    percentile,
    model,
    rewards,
    states,
    actions,
    predictions,
    features_embedding,
    answers_embedding,
    epochs,
    batch_size,
    max_samples,
    label,
    mc=False,
    verbose=False,
):
    if mc:
        X, Y = monte_carlo_sampling(
            states, actions, rewards, features_embedding, predictions.shape[1]
        )
        model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
        return
    elite_states = []
    elite_actions = []
    percentile_est = 0
    samples = 0
    mean_actions = 0

    while len(elite_states) == 0:
        elite_states, elite_actions, percentile_est, samples, mean_actions = (
            select_elites(states, actions, rewards, percentile, max_samples)
        )
        if len(elite_states) == 0:
            if percentile is None:
                percentile = 100
            if percentile <= 50:
                print("Percentile value dropped below 50, stop training")
                return
            print("Decreasing percentile from", percentile, "to", percentile - 10)
            percentile -= 10

    if verbose:
        data = elite_actions
        unique_sampled = get_unique_dict(data, range(predictions.shape[1]))
        unique_sampled = [
            int(unique_sampled[i]) if i % 2 == 0 else unique_sampled[i]
            for i in range(len(unique_sampled))
        ]
        rstr_random = " ".join(
            ["{:4d} : {:7.3f}%" for _ in range(len(unique_sampled) // 2)]
        ).format(*unique_sampled)

        print(label, rstr_random)

    v = model.fit(
        features_embedding[elite_states],
        answers_embedding[elite_actions],
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
    )

    if verbose:
        rewards_flat = np.concatenate(rewards, axis=0).flatten()
        print(
            (
                "Samples: {:6d}"
                " APS: {:^10f}"
                " \033[94mAccuracy\033[0m: \033[1m{:7.4f}%\033[0m"
                " Loss: {:7.5f}"
                " Mean: {:7.4f}%"
                " Median: {:7.4f}%"
                " Percentile: \033[1m{:7.4f}%\033[0m"
                " Max: {:7.4f}%"
            ).format(
                samples,
                mean_actions,
                100 * np.mean(v.history["acc"][0]),
                np.mean(v.history["loss"][0]),
                100 * np.mean(rewards_flat),
                100 * np.median(rewards_flat),
                100 * percentile_est,
                100 * np.max(rewards_flat),
            )
        )

    return 100 * np.mean(v.history["acc"][0])


def test_algorithms(
    algorithms,
    predictions_adm,
    decisions_adm,
    predictions_evc,
    decisions_evc,
    rows,
    alpha,
    output_file=None,
    base_iteration=0,
    verbose=True,
    keys_to_print=None,
):
    total_size = sum([row["size"] for row in rows])
    total_time = rows[len(rows) - 1]["timestamp"] - rows[0]["timestamp"]
    mean_entropy = np.mean([row["entropy"] for row in rows])

    history = {
        "time": [],
        "flow": 0,
        "alphas": alpha,
        "iterations": len(rows),
        "entropy": mean_entropy,
    }
    keys = sorted(algorithms.keys())

    if keys_to_print is None:
        keys_to_print = keys

    name_mapping = compress_names(keys_to_print)
    for name in name_mapping.keys():
        name_mapping[name] = extreme_compress(name_mapping[name])

    for key in keys:
        history[key] = []

    for i in range(len(rows)):
        for alg in keys:
            algorithms[alg].decide(
                rows[i],
                predictions_evc[decisions_evc[alg]][i],
                predictions_adm[decisions_adm[alg]][i],
            )

        if output_file is not None and i == len(rows) - 1:
            history["flow"] = float(total_size) / (1e-4 + float(total_time))
            history["time"].append(rows[i]["timestamp"])
            for key in keys:
                data_value = [
                    metric(algorithms[key], alpha_value)
                    for alpha_value in sorted(alpha)
                ]
                history[key].append(data_value)

        if verbose and ((i % 1513 == (1513 - 1)) or i == len(rows) - 1):
            print_list_total = []
            values_print_alphas = [base_iteration + i + 1]

            for alpha_value in alpha:
                values = [
                    min(99.99, 100 * metric(algorithms[alg], alpha_value))
                    for alg in keys_to_print
                ]
                best_performance = keys_to_print[values.index(max(values))]
                worst_performance = keys_to_print[values.index(min(values))]
                print_list = ["\033[93m{:2.1f}\033[0m"]
                for name in keys_to_print:
                    if name == best_performance or name == worst_performance:
                        if name == best_performance:
                            print_list.append(
                                "\033[92m{:s}\033[0m : \033[1m{:5.2f}%\033[0m"
                            )
                        else:
                            print_list.append(
                                "\033[91m{:s}\033[0m : \033[1m{:5.2f}%\033[0m"
                            )
                    else:
                        print_list.append(
                            "\033[94m{:s}\033[0m : \033[1m{:5.2f}%\033[0m"
                        )
                subst_vals = [alpha_value]
                for j in range(len(keys_to_print)):
                    subst_vals.append(name_mapping[keys_to_print[j]])
                    subst_vals.append(values[j])
                print_list_total += print_list
                values_print_alphas += subst_vals
            print_string = "   ".join(print_list_total)
            print_string = "I \033[1m{:d}\033[0m " + print_string
            sys.stdout.write("\r" + print_string.format(*values_print_alphas))
            sys.stdout.flush()

    print("")

    if output_file is not None:
        pickle.dump(history, open(output_file, "wb"))


def test_algorithms_light(
    algorithms,
    predictions_adm,
    predictions_evc,
    rows,
    alpha,
    previous_data,
    special_keys,
    base_iteration=0,
    print_at=None,
    verbose=True,
):
    keys = sorted(algorithms.keys())

    name_mapping = compress_names(algorithms.keys())
    # for name in name_mapping.keys():
    #     name_mapping[name] = extreme_compress(name_mapping[name])

    for i in range(len(rows)):
        if verbose and print_at is not None and i == print_at:
            print("")
        for alg in keys:
            algorithms[alg].decide(
                rows[i], predictions_evc[alg][i], predictions_adm[alg][i]
            )

        if verbose and ((i % 1513 == (1513 - 1)) or i == len(rows) - 1):
            values = [min(99.99, 100 * metric(algorithms[alg], alpha)) for alg in keys]
            best_performance = keys[values.index(max(values))]
            if previous_data is not None and i == len(rows) - 1:
                values = [
                    min(
                        99.99, 100 * metric(algorithms[alg], alpha) - previous_data[alg]
                    )
                    for alg in keys
                ]
            print_list = []
            for name in keys:
                if name in special_keys:
                    if name == best_performance:
                        print_list.append("\033[93m{:s}\033[0m \033[1m{:6.2f}%\033[0m")
                    else:
                        print_list.append("\033[92m{:s}\033[0m \033[1m{:6.2f}%\033[0m")
                else:
                    if name == best_performance:
                        print_list.append("\033[93m{:s}\033[0m \033[1m{:6.2f}%\033[0m")
                    else:
                        print_list.append("\033[0m{:s}\033[0m \033[1m{:6.2f}%\033[0m")
            print_string = " | ".join(print_list)
            print_string = "I \033[1m{:10d}\033[0m " + print_string
            subst_vals = [base_iteration + i + 1]
            for i in range(len(keys)):
                subst_vals.append(name_mapping[keys[i]])
                subst_vals.append(values[i])
            sys.stdout.write("\r" + print_string.format(*subst_vals))
            sys.stdout.flush()

    if verbose:
        print("")
    return dict(
        [(alg, min(99.99, 100 * metric(algorithms[alg], alpha))) for alg in keys]
    )


def get_session_features(
    bool_eviction, bool_admission, predictions_evc, predictions_adm
):
    eviction_defined, eviction_deterministic = bool_eviction
    admission_defined, admission_deterministic = bool_admission
    if not eviction_defined:
        if eviction_deterministic:
            eviction_decisions = np.argmax(predictions_evc, axis=1)
        else:
            eviction_decisions = sampling(predictions_evc)
        evc_decision_values = eviction_decisions
        eviction_decisions = [
            compute_rating(eviction_decisions[i], len(predictions_evc[i]))
            for i in range(len(eviction_decisions))
        ]
    else:
        evc_decision_values = None
        eviction_decisions = predictions_evc

    if not admission_defined:
        if admission_deterministic:
            admission_decisions = np.argmax(predictions_adm, axis=1)
        else:
            admission_decisions = sampling(predictions_adm)
        adm_decision_values = admission_decisions
        admission_decisions = [
            int(admission_decisions[i]) for i in range(len(eviction_decisions))
        ]
    else:
        adm_decision_values = None
        admission_decisions = predictions_adm
    return (
        evc_decision_values,
        eviction_decisions,
        adm_decision_values,
        admission_decisions,
    )


def generate_session_continious(
    predictions_evc,
    predictions_adm,
    rows,
    algorithm_template,
    config,
    change_point,
    seed,
    eviction_deterministic=False,
    admission_deterministic=False,
    eviction_defined=False,
    admission_defined=False,
    collect_eviction=True,
    collect_admission=True,
):
    lstates = []
    lactions = []
    lstates_adm = []
    lactions_adm = []

    np.random.seed()
    random.seed()
    # np.random.seed(seed)
    # random.seed(seed)

    decorrelated_metric = config["metric type"] == "decorrelated"
    if decorrelated_metric:
        hit_metric_history_ohr = np.repeat(0.0, change_point)
        hit_metric_history_bhr = np.repeat(0.0, change_point)

        total_metric_history_ohr = np.repeat(0.0, change_point)
        total_metric_history_bhr = np.repeat(0.0, change_point)

    algorithm = copy_object(algorithm_template)

    algorithm.reset()

    alpha = config["alpha"]
    if "use hr" not in config.keys():
        config["use hr"] = True
    count_hr = config["use hr"]

    ratings = algorithm.get_ratings()

    multiplier = 1.0

    reward_hits = 0
    reward_miss = 0

    byte_reward_hits = 0
    byte_reward_miss = 0

    gamma = 1.0

    full_step = len(rows) == change_point

    if config["collect discounted"]:
        multiplier = config["initial gamma"]
        if not full_step:
            gamma = (config["gamma"] / multiplier) ** (1.0 / (len(rows) - change_point))
        else:
            gamma = (config["gamma"] / multiplier) ** (1.0 / (len(rows)))
        if decorrelated_metric:
            multiplier = multiplier / gamma

    (
        evc_decision_values,
        eviction_decisions,
        adm_decision_values,
        admission_decisions,
    ) = get_session_features(
        (eviction_defined, eviction_deterministic),
        (admission_defined, admission_deterministic),
        predictions_evc,
        predictions_adm,
    )

    exchanged = False

    for i in range(len(rows)):
        if decorrelated_metric or (
            config["collect discounted"] and (exchanged or full_step)
        ):
            multiplier *= gamma

        hit = algorithm.decide(rows[i], eviction_decisions[i], admission_decisions[i])
        if decorrelated_metric and not exchanged:
            if hit:
                hit_metric_history_bhr[change_point - i - 1] = (
                    float(rows[i]["size"]) * multiplier
                )
                hit_metric_history_ohr[change_point - i - 1] = 1 * multiplier
            total_metric_history_bhr[change_point - i - 1] = (
                float(rows[i]["size"]) * multiplier
            )
            total_metric_history_ohr[change_point - i - 1] = 1 * multiplier
        if count_hr or (config["collect discounted"] and exchanged) or full_step:
            if hit:
                byte_reward_hits += multiplier * float(rows[i]["size"])
                reward_hits += multiplier
            else:
                byte_reward_miss += multiplier * float(rows[i]["size"])
                reward_miss += multiplier

        if exchanged:
            continue

        if not full_step and (config["change"] and i == change_point - 1):
            if config["change mode"] == "deterministic":
                eviction_deterministic = True
                admission_deterministic = True
            if config["change mode"] == "random":
                np.random.seed(config["seed"])
                random.seed(config["seed"])
            exchanged = True

        if exchanged:
            _, eviction_decisions, _, admission_decisions = get_session_features(
                (eviction_defined, eviction_deterministic),
                (admission_defined, admission_deterministic),
                predictions_evc,
                predictions_adm,
            )
            continue

        if (
            collect_eviction
            and not eviction_deterministic
            and not eviction_defined
            and algorithm.prediction_updated_eviction
        ):
            lstates.append(i)
            lactions.append(evc_decision_values[i])
        if (
            collect_admission
            and not admission_deterministic
            and not admission_defined
            and algorithm.prediction_updated_admission
        ):
            lstates_adm.append(i)
            lactions_adm.append(adm_decision_values[i])

    ohr_reward = reward_hits / (reward_hits + reward_miss)
    bhr_reward = byte_reward_hits / (byte_reward_hits + byte_reward_miss)

    eviction_rating = metric_funct(bhr_reward, ohr_reward, alpha)
    admission_rating = metric_funct(bhr_reward, ohr_reward, alpha)

    if decorrelated_metric:
        hit_metric_history_bhr = byte_reward_hits + np.cumsum(hit_metric_history_bhr)
        hit_metric_history_ohr = reward_hits + np.cumsum(hit_metric_history_ohr)
        total_metric_history_bhr = (
            byte_reward_hits + byte_reward_miss + np.cumsum(total_metric_history_bhr)
        )
        total_metric_history_ohr = (
            reward_hits + reward_miss + np.cumsum(total_metric_history_ohr)
        )
        eviction_rating = np.repeat(0.0, len(lstates))
        admission_rating = np.repeat(0.0, len(lstates_adm))
        for i in range(change_point):
            if i in lstates or i in lstates_adm:
                bhr = (
                    hit_metric_history_bhr[change_point - i - 1]
                    / total_metric_history_bhr[change_point - i - 1]
                )
                ohr = (
                    hit_metric_history_ohr[change_point - i - 1]
                    / total_metric_history_ohr[change_point - i - 1]
                )
                if (i in lstates_adm) and collect_admission:
                    admission_rating[lstates_adm.index(i)] = metric_funct(
                        bhr, ohr, alpha
                    )
                if (i in lstates) and collect_eviction:
                    eviction_rating[lstates.index(i)] = metric_funct(bhr, ohr, alpha)
    else:
        eviction_rating = np.repeat(eviction_rating, len(lstates))
        admission_rating = np.repeat(admission_rating, len(lstates_adm))

    assert algorithm_template.get_ratings() == ratings

    return (
        lstates,
        np.asarray(lactions),
        lstates_adm,
        np.asarray(lactions_adm),
        eviction_rating,
        admission_rating,
    )


def collect_filenames(filepath):
    filenames = [
        (join(filepath, f), int(f.replace(".csv", "")))
        for f in listdir(filepath)
        if isfile(join(filepath, f)) and ".csv" in f and "lock" not in f
    ]
    filenames = sorted(filenames, key=lambda x: x[1])
    return [item[0] for item in filenames]


def load_json(filename):
    return json.load(open(filename, "r"))
