# -*- coding: utf-8 -*-

import pydevd_pycharm
pydevd_pycharm.settrace('207.216.103.218', port=30266, stdoutToServer=True, stderrToServer=True)

import datetime
import json
import time
import numpy as np
from Helpers.data_loader import get_feature_dict, load_gene_expression_data, printProgressBar
import sys

# val = float(0.000000001 / 100000000)
# test = np.asarray([0.000000001 / 100000000, 0.000000001 / 100000000], dtype='float16')
#
# print(test)
# quit(0)

start_time = time.time()
gene_count_data_limit = 978
# target_cell_names = ['VCAP', 'A549', 'A375', 'PC3', 'MCF7', 'HT29']
target_cell_names = ['HT29']
save_xy_path = "TrainData/"
LINCS_data_path = "/home/integra/Data/LINCS/LDS-1191/"  # set this path to your LINCS gctx file
if LINCS_data_path == "":
    print("You need to set the LINCS data path")
    sys.exit()
data_path = "Data/"
dosage = 10


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start


def get_gene_id_dict():
    lm_genes = json.load(open('Data/landmark_genes.json'))
    dict = {}
    for lm_gene in lm_genes:
        # dict[lm_gene['gene_symbol']] = lm_gene['entrez_id']
        dict[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return dict

# get the dictionaries
print(datetime.datetime.now(), "Loading drug and gene features")
drug_features_dict = get_feature_dict('Data/phase1_compounds_morgan_2048.csv')
gene_features_dict = get_feature_dict('Data/go_fingerprints.csv')
cell_name_to_id_dict = get_feature_dict('Data/Phase1_Cell_Line_Metadata.txt', '\t', 2)
experiments_dose_dict = get_feature_dict(LINCS_data_path + 'Metadata/GSE92742_Broad_LINCS_sig_info.txt', '\t', 0)
gene_id_dict = get_gene_id_dict()
covid_genes = get_feature_dict('Data/covid_genes.csv')

lm_gene_entrez_ids = []
for gene in gene_id_dict:
    lm_gene_entrez_ids.append(gene)

print("Loading gene expressions from gctx")
level_5_gctoo = load_gene_expression_data(LINCS_data_path + "Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
                                          lm_gene_entrez_ids)
length = len(level_5_gctoo.col_metadata_df.index)
# length = 10000

for target_cell_name in target_cell_names:
    target_cell_id = cell_name_to_id_dict[target_cell_name][0]

    cell_X = {}  # stores rows used in X
    cell_Y = {}  # stores the highest perturbation for that experiment
    cell_Y_gene_ids = []
    cell_drugs_counts = 0
    repeat_X = {}

    print("Loading experiments")
    # For every experiment
    for i in range(0, length):
        printProgressBar(i, length, prefix='Load experiments progress')
        col_name = level_5_gctoo.col_metadata_df.index[i]
        column = level_5_gctoo.data_df[col_name]

        # parse the time
        start = col_name.rfind("_")
        end = find_nth(col_name, ":", 1)
        exposure_time = col_name[start + 1:end]
        if exposure_time != "24H":  # column counts: 6h 102282, 24h 118597, 48h 1453, 144h 18487, 3h 612
            continue

        # get drug features
        col_name_key = col_name
        if col_name_key not in experiments_dose_dict:
            continue
        experiment_data = experiments_dose_dict[col_name_key]
        drug_id = experiment_data[0]
        if drug_id not in drug_features_dict:
            continue
        drug_features = drug_features_dict[drug_id]

        # parse the dosage unit and value
        dose_unit = experiment_data[5]
        if dose_unit != 'µM':
            # remove any dosages that are not 'µM'. Want to standardize the dosages.
            # column counts: -666 17071, % 2833, uL 238987, uM 205066, ng 1439, ng / uL 2633, ng / mL 5625
            continue
        dose_amt = float(experiment_data[4])
        if dose_amt < dosage - 0.1 or dose_amt > dosage + 0.1:  # 10µM +/- 0.1
            continue

        # parse the cell name
        start = find_nth(col_name, "_", 1)
        end = find_nth(col_name, "_", 2)
        cell_name = col_name[start + 1:end]
        if cell_name != target_cell_name:
            continue

        if cell_name not in cell_name_to_id_dict:
            continue
        cell_id = cell_name_to_id_dict[cell_name][0]

        # go through the covid genes to get the denominator
        pert_sum = 0
        covid_count = 0
        for gene_id in lm_gene_entrez_ids:
            gene_symbol = gene_id_dict[gene_id]
            if gene_symbol not in covid_genes:
                continue
            covid_count += 1
            pert = column[gene_id].astype('float16')
            negative = float(covid_genes[gene_symbol][1]) < 0
            if negative:
                pert_sum -= pert
            else:
                pert_sum += pert
        denominator = (pert_sum/covid_count)/10
        if denominator == 0:
            continue
        for gene_id in lm_gene_entrez_ids:
            gene_symbol = gene_id_dict[gene_id]

            if gene_symbol not in gene_features_dict:
                continue

            pert = column[gene_id].astype('float16')
            # repeat key is used to find the largest perturbation for similar experiments and filter out the rest
            repeat_key = drug_id + "_" + gene_id

            if repeat_key not in cell_X:
                cell_X[repeat_key] = drug_features + gene_features_dict[gene_symbol]
                cell_Y[repeat_key] = []
                cell_Y_gene_ids.append(gene_id)
                cell_drugs_counts += 1
            cell_Y[repeat_key].append((pert/10)/denominator)

    elapsed_time = time.time() - start_time
    print(datetime.datetime.now(), "Time to load data:", elapsed_time)

    print("Sample Size:", len(cell_Y), "Drugs tested:", cell_drugs_counts / gene_count_data_limit)

    # at this point all the perturbation values are stored in cell_Y
    # save the data to be loaded and used multiple times if needed
    model_file_prefix = save_xy_path + target_cell_name
    # save_file = model_file_prefix + "_X"
    # npX = np.asarray(list(cell_X.values()), dtype='float16').astype(dtype='int8')
    # np.savez_compressed(save_file, npX)
    # print("saved", save_file)

    save_file = model_file_prefix + "_Y"
    for key in cell_Y.keys(): # convert from list to average
        vals = cell_Y[key]
        cell_Y[key] = sum(vals) / len(vals)

    keys = list(cell_Y.keys())
    np.savez_compressed('drug_id_gene_id', keys)
    # npY = np.asarray(list(cell_Y.values()), dtype='float16')
    # np.savez_compressed (save_file, npY)
    # print("saved", save_file)


