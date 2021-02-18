"""
Simple module with a class to manage the data used in the analysis
"""
import os.path
import copy
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import uproot


class TreeHandler:
    """
    Class for storing and managing the data of a ROOT tree from a .root file
    or a pandas.DataFrame from a .parquet file
    """

    def __init__(self, file_name=None, tree_name=None, columns_names=None, **kwds):
        """
        Open the file in which the selected tree leaves are converted
        into pandas dataframe columns. If tree_name is not provided file_name is
        assumed to be associated to a .parquet file

        Parameters
        ------------------------------------------------
        file_name: str or list of str
            Name of the input file where the data sit or list of input files

        tree_name: str
            Name of the tree within the input file, must be the same for all files.
            If None the method pandas.read_parquet is called

        columns_name: list
            List of the names of the branches that one wants to analyse. If columns_names is
            not specified all the branches are converted

        **kwds: extra arguments are passed on to the uproot.open or pandas.read_parquet methods:
                https://uproot.readthedocs.io/en/latest/uproot.reading.open.html#uproot.reading.open
                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_parquet.html#pandas.read_parquet
        """
        self._tree = tree_name
        self._full_data_frame = None
        if file_name is not None:
            self._full_data_frame = pd.DataFrame()
            self._files = file_name if isinstance(file_name, list) else [file_name]
            for file in self._files:
                if self._tree is not None:
                    self._full_data_frame = self._full_data_frame.append(
                        uproot.open(f'{file}:{self._tree}', **kwds).arrays(filter_name=columns_names, library='pd'),
                        ignore_index=True)
                else:
                    self._full_data_frame = self._full_data_frame.append(
                        pd.read_parquet(file, columns=columns_names, **kwds), ignore_index=True)
        self._preselections = None
        self._projection_variable = None
        self._projection_binning = None
        self._sliced_df_list = None

    def __getitem__(self, column):
        """
        Access to the elements of the full data frame using
        a dictionary-like syntax. Accessing to the slices
        of the data frame in this way is not supported

        Parameters
        ------------------------------------------------
        column: string or list
            Column name/s of the full data frame

        """
        return self._full_data_frame[column]

    def __len__(self):
        """
        Evaluate the number of entries in the full data frame
        """
        return len(self._full_data_frame)

    def get_handler_from_large_file(self, file_name, tree_name, model_handler=None, preselection='',
                                    output_margin=True, max_workers=None):
        """
        Read a ROOT.TTree in different lazy chuncks. Chuncks are read sequentially or in parallel
        and eventually pre-selections or ML selections are applied. This allows to preserve the
        memory usage and speed-up the reading. Chuncks size is decided automatically

        Parameters
        -----------------------------------------------
        file_name: str or list of str
            Name of the input file where the data sit or list of input files

        tree_name: str
            Name of the tree within the input file, must be the same for all files

        model_handler: hipe4ml ModelHandler
            Model handler to be applied as a preselection on the data contained in the original
            tree. A column named model_output is added to the tree_handler. In case of multi-classification
            a new column is added for each class with name: model_output_{i}

        preselection: str
            String containing the cuts to be applied as preselection on the data contained in the original
            tree. The string syntax is the one required in the pandas.DataFrame.query() method.
            You can refer to variables in the environment by prefixing them with an ‘@’ character like @a + b.
            You can apply ML based preselections like in the example below:
            - "model_output > @score_cut" # binary classification
            - "model_output_0 > @score_cut[0] and model_output_1 <= @score_cut[1]" # multi-classification

        output_margin: bool
            Whether to predict the raw untransformed margin value. If False model
            probabilities are returned

        max_workers: int
            Maximum number of workers employed to read the chuncks. If max_workers is None or not given,
            it will default to the number of processors on the machine, multiplied by 5. If max_workers==-1
            the multi-threading computation is turned off.
            More details in:
            https://docs.python.org/3/library/concurrent.futures.html

        Returns
        -----------------------------------------------
        out: hipe4ml TreeHandler
            TreeHandler from the original files containing informations on the pre-selections applied



        """
        self._files = file_name if isinstance(file_name, list) else [file_name]
        self._tree = tree_name
        self._preselections = preselection
        inputs = [f'{file_name}:{tree_name}' for file_name in self._files]

        executor = ThreadPoolExecutor(max_workers) if max_workers != -1 else None
        iterator = uproot.iterate(inputs, library='pd', decompression_executor=executor,
                                  interpretation_executor=executor)

        result = []
        for data in iterator:
            if model_handler is not None:
                predictions = model_handler.predict(data, output_margin)
                n_classes = model_handler.get_n_classes()
                if n_classes > 2:
                    for i_class in range(n_classes):
                        column_name = f'model_output_{i_class}'
                        data[column_name] = predictions[:, i_class]
                else:
                    column_name = "model_output"
                    data[column_name] = predictions

            if preselection:
                data = data.query(preselection)
            result.append(data)

        result = pd.concat(result)
        self._full_data_frame = result

    def set_data_frame(self, df_orig):
        """
        Set the pandas DataFrame in the TreeHandler

        Parameters
        ------------------------------------------------
        df: pandas.DataFrame
            DataFrame stored in the TreeHandler
        """
        self._full_data_frame = df_orig

    def get_data_frame(self):
        """
        Get the pandas DataFrame stored in the TreeHandler

        Returns
        ------------------------------------------------
        out: pandas.DataFrame
            DataFrame stored in the TreeHandler
        """
        return self._full_data_frame

    def get_preselections(self):
        """
        Get the preselections applied to the stored DataFrame

        Returns
        ------------------------------------------------
        out: str
            String containing the cuts applied to the stored DataFrame
        """
        return self._preselections

    def get_projection_variable(self):
        """
        Get the name of the sliced variable

        Returns
        ------------------------------------------------
        out: str
            Sliced variable
        """
        return self._projection_variable

    def get_projection_binning(self):
        """
        Get the bins used for slicing the DataFrame

        Returns
        ------------------------------------------------
        out: list
            Each element of the list is a list containing the
            bin edges
        """
        return self._projection_binning

    def get_n_cand(self):
        """
        Get the number of candidates stored in the full DataFrame

        Returns
        ------------------------------------------------
        out: int
           Number of candidates
        """
        return len(self._full_data_frame)

    def get_var_names(self):
        """
        Get a list containing the name of the variables

        Returns
        ------------------------------------------------
        out: list
           Names of the variables
        """
        return list(self._full_data_frame.columns)

    def get_slice(self, n_bin):
        """
        Get the n-th slice of the original DataFrame

        Parameters
        ------------------------------------------------
        n_bin: int
            n-th element of _projection_binning list.

        Returns
        ------------------------------------------------
        out: pandas.DataFrame
            N-th Slice of the original DataFrame
        """
        return self._sliced_df_list[n_bin]

    def get_sliced_df_list(self):
        """
        Get the list containing the slices of the orginal
        DataFrame

        Returns
        ------------------------------------------------
        out: list
            List containing the slices of the orginal
            DataFrame
        """
        return self._sliced_df_list

    def apply_preselections(self, preselections, inplace=True, **kwds):
        """
        Apply preselections to data

        Parameters
        ------------------------------------------------
        preselection: str
            String containing the cuts to be applied as preselection on the data contained in the original
            tree. The string syntax is the one required in the pandas.DataFrame.query() method.
            You can refer to variables in the environment by prefixing them with an ‘@’ character like @a + b.

        inplace: bool
            If True, the preselected dataframe replaces the initial dataframe. Otherwise return a copy of the
            preselected df

        **kwds: extra arguments are passed on to the pandas.DataFrame.query method:
                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html#pandas.DataFrame.query

        Returns
        ------------------------------------------------
        out: TreeHandler or None
            If inplace == True return None is returned and the full DataFrame is replaced
        """
        if inplace:
            if self._preselections:
                self._preselections += " and " + preselections
            else:
                self._preselections = preselections
            self._full_data_frame.query(preselections, inplace=True, **kwds)
            return None

        new_hndl = copy.deepcopy(self)
        new_hndl._preselections = preselections  # pylint: disable=W0212
        new_hndl._full_data_frame.query(preselections, inplace=True, **kwds)  # pylint: disable=W0212
        return new_hndl

    def apply_model_handler(self, model_handler, output_margin=True, column_name=None):
        """
        Apply the ML model to data: a new column is added to the DataFrame
        If a list is given the application is performed on the slices.

        Parameters
        ------------------------------------------------
        model_handler: list or hipe4ml model_handler
            If a list of handlers(one for each bin) is provided, the ML
            model is applied to the slices

        output_margin: bool
            Whether to output the raw untransformed margin value.

        column_name: str
            Name of the new column with the model output
        """
        if isinstance(model_handler, list):
            n_class = model_handler[0].get_n_classes()
            sliced = True
        else:
            sliced = False
            n_class = model_handler.get_n_classes()
        if column_name is None:
            if n_class > 2:
                column_name = [f'model_output_{i_class}' for i_class in range(n_class)]
            else:
                column_name = "model_output"

        if sliced:
            for (mod_handl, sliced_df) in zip(model_handler, self._sliced_df_list):
                prediction = mod_handl.predict(sliced_df, output_margin)
                if n_class > 2:
                    for i_class in range(n_class):
                        sliced_df[column_name[i_class]] = prediction[:, i_class]
                else:
                    sliced_df[column_name] = prediction
            return

        prediction = model_handler.predict(self._full_data_frame, output_margin)
        if n_class > 2:
            for i_class in range(n_class):
                self._full_data_frame[column_name[i_class]] = prediction[:, i_class]
            return

        self._full_data_frame[column_name] = prediction

    def get_subset(self, selections=None, frac=None, size=None, rndm_state=None):
        """
        Returns a TreeHandler containing a subset of the data

        Parameters
        ------------------------------------------------
        selection: str
            String containing the cuts to be applied as preselection on the data contained in the original
            tree. The string syntax is the one required in the pandas.DataFrame.query() method.
            You can refer to variables in the environment by prefixing them with an ‘@’ character like @a + b.

        frac: float
            Fraction of candidates to return.

        size: int
            Number of candidates to return. Cannot be used with frac.

        rndm_state: int or numpy.random.RandomState, optional
            Seed for the random number generator (if int), or numpy RandomState object, passed to the
            pandas.DataFrame.sample() method:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html


        Returns
        ------------------------------------------------
        out: TreeHandler
            TreeHandler containing a subset of the current data
        """

        subset = copy.deepcopy(self)

        if selections:
            subset.apply_preselections(selections, inplace=True)
        if frac or size:
            subset.shuffle_data_frame(frac=frac, size=size, inplace=True, random_state=rndm_state)
        return subset

    def slice_data_frame(self, projection_variable, projection_binning, delete_original_df=False):
        """
        Create a list containing slices of the orginal DataFrame.
        The original DataFrame is splitted in N sub-DataFrames following
        the binning(projection_binning) of a given variable(projected_variable)

        Parameters
        ------------------------------------------------
        projection_variable: str
            Name of the variable that will be sliced in the analysis

        projection_binning: list
            Binning of the sliced variable should be given as a list of
            [min, max) values for each bin

        delete_original_df: bool
            If True delete the original DataFrame. Only the
            the slice array will be accessible in this case
        """

        self._projection_variable = projection_variable
        self._projection_binning = projection_binning

        self._sliced_df_list = []
        for ibin in projection_binning:
            bin_mask = np.logical_and(
                self._full_data_frame[projection_variable] >= ibin[0],
                self._full_data_frame[projection_variable] < ibin[1])
            self._sliced_df_list.append(self._full_data_frame[bin_mask].copy())
        if delete_original_df:
            self._full_data_frame = None

    def shuffle_data_frame(self, size=None, frac=None, inplace=True, **kwds):
        """
        Extract a random sample from the DataFrame

        Parameters
        ------------------------------------------------
        size: int
            Number of candidates to return. Cannot be used with frac. Default = 1 if
            frac = None.

        frac: float
            Fraction of candidates to return. Cannot be used with size.

        inplace: bool
            If True the shuffled dataframe replaces the initial dataframe. Otherwise return a copy
            of the shuffled df
        **kwds: extra arguments are passed on to the pandas.DataFrame.sample method:
                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html

        Returns
        ------------------------------------------------
        out: TreeHandler or None
            If inplace == True None is returned and the full DataFrame is replaced
        """

        if inplace:
            self._full_data_frame = self._full_data_frame.sample(size, frac, **kwds)
            return None

        new_hndl = copy.deepcopy(self)
        new_hndl._full_data_frame = self._full_data_frame.sample(size, frac, **kwds)  # pylint: disable=W0212
        return new_hndl

    def eval_data_frame(self, ev_str, inplace=True, **kwds):
        """
        Evaluate a string describing operations on DataFrame columns

        Parameters
        ------------------------------------------------
        ev_str: str
            The expression string to evaluate. The string syntax is the one required in the
            pandas.DataFrame.eval() method.

        inplace: bool
            If the expression contains an assignment, whether to perform the operation inplace and
            mutate the existing DataFrame. Otherwise, a new DataFrame is returned.

        **kwds: extra arguments are passed on to the pandas.DataFrame.eval method:
                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.eval.html

        Returns
        ------------------------------------------------
        out: TreeHandler or None
            if inplace == True None is returned and the full dataframe is evaluated
        """
        if inplace:
            self._full_data_frame.eval(ev_str, inplace=True, **kwds)
            return None

        new_hndl = copy.deepcopy(self)
        new_hndl._full_data_frame.eval(ev_str, inplace=True, **kwds)  # pylint: disable=W0212
        return new_hndl

    def print_summary(self):
        """
        Print information about the TreeHandler object and its
        data members
        """
        print("\nFile name: ", self._files)
        print("Tree name: ", self._tree)
        print("DataFrame head:\n", self._full_data_frame.head(5))
        print("\nPreselections:", self._preselections)
        print("Sliced variable: ", self._projection_variable)
        print("Slices binning: ", self._projection_binning)

    def write_df_to_parquet_files(self, base_file_name="TreeDataFrame", path="./", save_slices=False):
        """
        Write the pandas dataframe to parquet files

        Parameters
        ------------------------------------------------
        base_file_name: str
            Base filename used to save the parquet files

        path: str
            Base path of the output files

        save_slices: bool
            If True and the slices are available, single parquet files for each
            bins are created
        """
        if self._full_data_frame is not None:
            name = os.path.join(path, f"{base_file_name}.parquet.gzip")
            self._full_data_frame.to_parquet(name, compression="gzip")
        else:
            print("\nWarning: original DataFrame not available")
        if save_slices:
            if self._sliced_df_list is not None:
                for ind, i_bin in enumerate(self._projection_binning):
                    name = os.path.join(
                        path, f"{base_file_name}_{self._projection_variable}_{i_bin[0]}_{i_bin[1]}.parquet.gzip")
                    self._sliced_df_list[ind].to_parquet(
                        name, compression="gzip")
            else:
                print("\nWarning: slices not available")
