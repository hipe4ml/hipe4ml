"""
Module used to test the TreeHandler class functionalities
"""
import copy
import pickle
from pathlib import Path

import pandas as pd

import helpers as hp
from hipe4ml.tree_handler import TreeHandler

# globals
SEED = 42


def test_tree_handler():  # pylint: disable=too-many-statements
    """
    Test the TreeHandler class functionalities.
    """
    # define the working directory
    test_dir = Path(__file__).resolve().parent

    # initialize TreeHandler test
    test_data, references = hp.init_handlers_test_workspace(test_dir, 'tree_handler')

    # instantiate tree handler objects
    data_hdlr = TreeHandler(test_data[0], 'treeMLDplus')
    prompt_hdlr = TreeHandler(test_data[1], 'treeMLDplus')
    data_pq_hdlr = TreeHandler(test_data[3])
    prompt_pq_hdlr = TreeHandler(test_data[4])
    mult_hdlr = TreeHandler(test_data[:2], 'treeMLDplus')
    mult_pq_hdlr = TreeHandler(test_data[3:5])
    data_hdlr_large_files = TreeHandler()
    mult_hdlr_large_files = TreeHandler()
    data_hdlr_large_files.get_handler_from_large_file(test_data[0], 'treeMLDplus')
    mult_hdlr_large_files.get_handler_from_large_file(test_data[:2], 'treeMLDplus')

    # open references objects
    reference_data_slice_df = pd.read_pickle(references[0])
    reference_prompt_slice_df = pd.read_pickle(references[1])
    with open(references[2], 'rb') as handle:
        reference_dict = pickle.load(handle)

    # test that data is the same in root and parquet
    assert data_hdlr.get_data_frame().equals(data_pq_hdlr.get_data_frame()), \
        'data Dataframe from parquet file differs from the root file one!'
    assert prompt_hdlr.get_data_frame().equals(prompt_pq_hdlr.get_data_frame()), \
        'prompt Dataframe from parquet file differs from the root file one!'

    # test loading from multiple files
    merged_df = pd.concat([data_hdlr.get_data_frame(), prompt_hdlr.get_data_frame()], ignore_index=True)
    assert mult_hdlr.get_data_frame().equals(merged_df), 'loading of multiple root files not working!'
    merged_pq_df = pd.concat([data_pq_hdlr.get_data_frame(), prompt_pq_hdlr.get_data_frame()], ignore_index=True)
    assert mult_pq_hdlr.get_data_frame().equals(merged_pq_df), 'loading of multiple parquet files not working!'

    # test loading using get_handler_from_large_file
    assert data_hdlr_large_files.get_data_frame().equals(data_hdlr.get_data_frame()), \
        'get_handler_from_large_file() not working for single file'
    assert mult_hdlr_large_files.get_data_frame().equals(mult_hdlr.get_data_frame()), \
        'get_handler_from_large_file() not working for multiple files'

    # define the info dict that will be compared with the reference
    info_dict = {}

    # get the number of candidates in the original data sample
    info_dict['n_data'] = data_hdlr.get_n_cand()
    info_dict['n_prompt'] = prompt_hdlr.get_n_cand()

    # get the original variable list
    info_dict['data_var_list'] = prompt_hdlr.get_var_names()
    info_dict['prompt_var_list'] = prompt_hdlr.get_var_names()

    # shuffle dataframes
    new_hndl = data_hdlr.shuffle_data_frame(size=10, random_state=5, inplace=False)
    copied_hndl = copy.deepcopy(data_hdlr)
    copied_hndl.shuffle_data_frame(size=10, random_state=5, inplace=True)
    assert copied_hndl.get_data_frame().equals(new_hndl.get_data_frame()), \
        'Inplaced dataframe differs from the not inplaced one after shuffling'

    # apply preselections
    preselections_data = '(pt_cand > 1.30 and pt_cand < 42.00) and (inv_mass > 1.6690 and inv_mass < 2.0690)'
    preselections_prompt = '(pt_cand > 1.00 and pt_cand < 25.60) and (inv_mass > 1.8320 and inv_mass < 1.8940)'

    new_hndl = data_hdlr.apply_preselections(preselections_data, inplace=False)
    data_hdlr.apply_preselections(preselections_data)
    assert data_hdlr.get_data_frame().equals(new_hndl.get_data_frame()), \
        'Inplaced dataframe differs from the not inplaced one after the preselections'

    data_hdlr_large_files.get_handler_from_large_file(test_data[0], 'treeMLDplus', preselection=preselections_data)
    assert data_hdlr.get_data_frame().equals(data_hdlr_large_files.get_data_frame()), \
        'get_handler_from_large_file() not working when applying preselections'

    hp.terminate_handlers_test_workspace(test_dir)

    prompt_hdlr.apply_preselections(preselections_prompt)

    # get the number of selected data
    info_dict['n_data_preselected'] = data_hdlr.get_n_cand()
    info_dict['n_prompt_preselected'] = prompt_hdlr.get_n_cand()

    # get the preselections
    info_dict['data_preselections'] = data_hdlr.get_preselections()
    info_dict['prompt_preselections'] = prompt_hdlr.get_preselections()

    # apply dummy eval() on the underlying data frame
    d_len_z_def = 'd_len_z = sqrt(d_len**2 - d_len_xy**2)'
    new_hndl = data_hdlr.eval_data_frame(d_len_z_def, inplace=False)
    data_hdlr.eval_data_frame(d_len_z_def)
    assert data_hdlr.get_data_frame().equals(new_hndl.get_data_frame()), \
        'Inplaced dataframe differs from the not inplaced one after eval'

    prompt_hdlr.eval_data_frame(d_len_z_def)

    # get the new variable list
    info_dict['data_new_var_list'] = prompt_hdlr.get_var_names()
    info_dict['prompt_new_var_list'] = prompt_hdlr.get_var_names()

    # get a random subset of the original data
    data_hdlr = data_hdlr.get_subset(size=3000, rndm_state=SEED)
    prompt_hdlr = prompt_hdlr.get_subset(size=55, rndm_state=SEED)

    # slice both data and prompt data frame respect to the pT
    bins = [[0, 2], [2, 10], [10, 25]]

    data_hdlr.slice_data_frame('pt_cand', bins)
    prompt_hdlr.slice_data_frame('pt_cand', bins)

    # store projection variable and binning
    info_dict['data_proj_variable'] = data_hdlr.get_projection_variable()
    info_dict['prompt_proj_variable'] = prompt_hdlr.get_projection_variable()

    info_dict['data_binning'] = data_hdlr.get_projection_binning()
    info_dict['prompt_binning'] = prompt_hdlr.get_projection_binning()

    # get info from a single data slice
    data_slice_df = data_hdlr.get_slice(2)
    prompt_slice_df = prompt_hdlr.get_slice(2)

    info_dict['n_data_slice'] = len(data_slice_df)
    info_dict['n_prompt_slice'] = len(prompt_slice_df)

    # test info_dict reproduction

    assert info_dict == reference_dict, 'dictionary with the data info differs from the reference!'

    # test sliced data frames reproduction
    assert data_slice_df.equals(reference_data_slice_df), 'data sliced DataFrame differs from the reference!'
    assert prompt_slice_df.equals(reference_prompt_slice_df), 'prompt sliced DataFrame differs from the reference!'
