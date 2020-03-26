"""
Simple module with a class to manage the data used in the analysis
"""
import copy
import numpy as np
import pandas as pd
import uproot


class TreeHandler:
    """
    Class for storing and managing the data of a ROOT tree from a .root file
    or a pandas.DataFrame from a .parquet file
    """

    def __init__(self, file_name, tree_name=None, columns_names=None, **kwds):
        """
        Open the file in which the selected tree leaves are converted
        into pandas dataframe columns. If tree_name is not provided file_name is
        assumed to be associated to a .parquet file

        Parameters
        ------------------------------------------------
        file_name: str
            Name of the input file where the data sit

        tree_name: str
            Name of the tree within the input file. If None the method pandas.read_parquet
            is called

        columns_name: list
            List of the names of the branches that one wants to analyse. If columns_names is
            not specified all the branches are converted

        **kwds: extra arguments are passed on to the uproot.pandas.iterate or the pandas.read_parquet
                methods:
                https://uproot.readthedocs.io/en/latest/opening-files.html#uproot-pandas-iterate
                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_parquet.html#pandas.read_parquet
        """
        if tree_name is not None:
            self._file = uproot.open(file_name)
            self._tree = self._file[tree_name]
            self._full_data_frame = self._tree.pandas.df(
                branches=columns_names, **kwds)
        else:
            self._full_data_frame = pd.read_parquet(
                file_name, columns=columns_names, **kwds)
        self._preselections = None
        self._projection_variable = None
        self._projection_binning = None
        self._sliced_df_list = None

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
        out: pandas.DataFrame or None
            If inplace == True return None is returned and the full DataFrame is replaced
        """
        self._preselections = preselections
        return self._full_data_frame.query(preselections, inplace=inplace, **kwds)

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

    def deepcopy(self, original):
        """
        Performs a deepcopy of another TreeHandler.
        Note: Deepcopy cannot be performed directly on the object itself as we store the
        pointer to the open file.

        Parameters
        ------------------------------------------------
        original: TreeHandler
            TreeHandler to be copied in this object
        """

        self._full_data_frame = copy.deepcopy(original.get_data_frame())
        self._preselections = copy.deepcopy(original.get_preselections())
        self._projection_variable = copy.deepcopy(original.get_projection_variable())
        self._projection_binning = copy.deepcopy(original.get_projection_binning())
        self._sliced_df_list = copy.deepcopy(original.get_sliced_df_list())

    def get_subset(self, selections=None, frac=None, size=None):
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

        Returns
        ------------------------------------------------
        out: TreeHandler
            TreeHandler containing a subset of the current data
        """

        subset = copy.copy(self)
        subset.deepcopy(self)

        if selections:
            subset.apply_preselections(selections, inplace=True)
        if frac or size:
            subset.shuffle_data_frame(frac=frac, size=size, inplace=True)
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
        out: pandas.DataFrame or None
            If inplace == True None is returned and the full DataFrame is replaced
        """
        df_shuf = None
        if inplace:
            self._full_data_frame = self._full_data_frame.sample(size, frac, **kwds)
            return df_shuf

        df_shuf = self._full_data_frame.sample(size, frac, **kwds)
        return df_shuf

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
        out: pandas.DataFrame or None
            if inplace == True None is returned and the full dataframe is evaluated
        """
        return self._full_data_frame.eval(ev_str, inplace=inplace, **kwds)

    def print_summary(self):
        """
        Print information about the TreeHandler object and its
        data members
        """
        print("\nFile name: ", self._file)
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
            self._full_data_frame.to_parquet(
                f"{path}{base_file_name}.parquet.gzip", compression="gzip")
        else:
            print("\nWarning: original DataFrame not available")
        if save_slices:
            if self._sliced_df_list is not None:
                for ind, i_bin in enumerate(self._projection_binning):
                    name = f"{path}{base_file_name}_{self._projection_variable}_{i_bin[0]}_{i_bin[1]}"
                    self._sliced_df_list[ind].to_parquet(f"{name}.parquet.gzip",
                                                         compression="gzip")
            else:
                print("\nWarning: slices not available")
