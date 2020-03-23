"""
Simple module with a class to manage the data used in the analysis
"""

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
            List of the names of the branches that one wants to analyse
        **kwds: extra arguments are passed on to the pandas.read_parquet method

        """
        if tree_name is not None:
            self._file = uproot.open(file_name)
            self._tree = self._file[tree_name]
            self._full_data_frame = self._tree.pandas.df(branches=columns_names)
        else:
            self._full_data_frame = pd.read_parquet(file_name, columns=columns_names, **kwds)
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

    def apply_preselections(self, preselections, inplace=True):
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

        Returns
        ------------------------------------------------
        out: pandas.DataFrame or None
            If inplace == True return None is returned and the full DataFrame is replaced

        """
        df_pres = None
        if inplace:
            self._preselections = preselections
            self._full_data_frame = self._full_data_frame.query(
                self._preselections)
            return df_pres

        df_pres = self._full_data_frame.query(preselections)
        return df_pres

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
            self._sliced_df_list.append(self._full_data_frame[bin_mask])
        if delete_original_df:
            self._full_data_frame = None

    def shuffle_data_frame(self, size=None, frac=None, inplace=True):
        """
        Extract a random sample from the DataFrame

        Parameters
        ------------------------------------------------
        size: int
            Number of candidates to return. Cannot be used with frac. Default = 1 if
            frac = None.

        frac: float
            Fraction of candidates to return. Cannot be used with n.

        inplace: bool
            If True the shuffled dataframe replaces the initial dataframe. Otherwise return a copy
            of the shuffled df

        Returns
        ------------------------------------------------
        out: pandas.DataFrame or None
            If inplace == True None is returned and the full DataFrame is replaced

        """
        df_shuf = None
        if inplace:
            self._full_data_frame = self._full_data_frame.sample(size, frac)
            return df_shuf

        df_shuf = self._full_data_frame.sample(size, frac)
        return df_shuf

    def eval_data_frame(self, ev_str, inplace=True):
        """
        Evaluate a string describing operations on DataFrame columns

        Parameters
        ------------------------------------------------
        preselection: str
            The expression string to evaluate. A combination of a date and a time. Attributes: ()
            The string syntax is the one required in the pandas.DataFrame.eval() method.
        inplace: bool
            If the expression contains an assignment, whether to perform the operation inplace and
            mutate the existing DataFrame. Otherwise, a new DataFrame is returned.

        Returns
        ------------------------------------------------
        out: pandas.DataFrame or None
            if inplace == True None is returned and the full dataframe is evaluated

        """
        df_ev = None
        if inplace:
            self._full_data_frame.eval(ev_str, inplace=True)
            return df_ev

        df_ev = self._full_data_frame.eval(ev_str)
        return df_ev

    def print_summary(self):
        """
        Print information about the DataHandler object and its
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
        self._full_data_frame.to_parquet(
            f"{path}{base_file_name}.parquet.gzip", compression="gzip")
        if save_slices:
            for ind, i_bin in enumerate(self._projection_binning):
                name = f"{path}{base_file_name}_{self._projection_variable}_{i_bin[0]}_{i_bin[1]}"
                self._sliced_df_list[ind].to_parquet(f"{name}.parquet.gzip",
                                                     compression="gzip")
