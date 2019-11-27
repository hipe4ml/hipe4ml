'''Module containing the class to perform the training and the testing of the models '''


import pandas as pd
import uproot
from sklearn.model_selection import train_test_split


class GeneralizedAnalysis:
    ''' Class for training and testing machine learning models starting from ROOT TTrees filled with the candidates
    '''

    def __init__(
            self, mc_file_name, data_file_name, sig_selection=0, bkg_selection=0,
            cent_class=((0, 10),
                        (10, 30),
                        (30, 50),
                        (50, 90))):

        self.cent_class = cent_class.copy()
        self.n_events = [0, 0, 0, 0]

        self.df_signal = uproot.open(mc_file_name)['SignalTable'].pandas.df()
        self.df_generated = uproot.open(mc_file_name)['GenTable'].pandas.df()
        self.df_data = uproot.open(data_file_name)['DataTable'].pandas.df()

        self.df_signal['y'] = 1
        # backup of the data without any selections for the significance
        self.df_data_bkg = self.df_data.copy()
        # dataframe for signal and background with preselection
        if isinstance(sig_selection, str):
            self.df_signal = self.df_signal.query(sig_selection)
        if isinstance(bkg_selection, str):
            self.df_data_bkg = self.df_data_bkg.query(bkg_selection)

        self.hist_centrality = uproot.open(data_file_name)['EventCounter']
        self.n_events = []
        for cent in self.cent_class:
            self.n_events.append(sum(self.hist_centrality[cent[0]+1:cent[1]]))

    def prepare_dataframe(
            self, training_columns, cent_class=(0, 90), pt_range=(0, 10),
            ct_range=(0, 100),
            test=False):
        '''
        Returns the training and the test set with the labels in given pT, ct and
        centrality ranges

        Input
        -----------------------------------------------------------
        training_columns: training columns
        cent_class: centrality range
        pt_range: transverse momentum range
        ct_range: ct range
        test: if True prepare dataframes with only 1000 candidates

        Output
        ------------------------------------------------------------
        train_set: training set dataframe
        y_train: training set label array
        test_set: test set dataframe
        y_test: test set label array


        '''
        data_range_bkg = '{}<ct<{} and {}<HypCandPt<{} and {}<centrality<{}'.format(
            ct_range[0], ct_range[1], pt_range[0], pt_range[1], cent_class[0], cent_class[1])

        columns = training_columns.copy()
        columns.append('InvMass')

        if 'HypCandPt' not in columns:
            columns.append('HypCandPt')
        if 'ct' not in columns:
            columns.append('ct')
        else:
            data_range_sig = data_range_bkg

        bkg = self.df_data_bkg.query(data_range_bkg)
        sig = self.df_signal.query(data_range_sig)

        if test:
            if len(sig) >= 1000:
                sig = sig.sample(n=1000)
            if len(bkg) >= 1000:
                bkg = bkg.sample(n=1000)

            self.df_data = self.df_data.sample(n=1000)

        print('\nnumber of background candidates: {}'.format(len(bkg)))
        print('number of signal candidates: {}\n'.format(len(sig)))

        df_tot = pd.concat([sig, bkg])
        train_set, test_set, y_train, y_test = train_test_split(
            df_tot[columns], df_tot['y'], test_size=0.5, random_state=42)

        return train_set, y_train, test_set, y_test

    def preselection_efficiency(self, cent_class=(0, 90), pt_range=(0, 10), ct_range=(0, 100)):
        '''
        Calculate the pre selection efficiency

        Input
        ---------------------------------------------
        cent_class: centrality range
        pt_range: transverse momentum range
        ct_range: ct range

        Output
        ---------------------------------------------
        eff: pre-selection efficiency

        '''
        ct_min = ct_range[0]
        ct_max = ct_range[1]

        pt_min = pt_range[0]
        pt_max = pt_range[1]

        cent_min = cent_class[0]
        cent_max = cent_class[1]

        total_cut = '{}<ct<{} and {}<HypCandPt<{} and {}<centrality<{}'.format(
            ct_min, ct_max, pt_min, pt_max, cent_min, cent_max)
        total_cut_gen = '{}<ct<{} and {}<pT<{} and {}<centrality<{}'.format(
            ct_min, ct_max, pt_min, pt_max, cent_min, cent_max)

        eff = len(self.df_signal.query(total_cut)) / \
            len(self.df_generated.query(total_cut_gen))

        return eff
