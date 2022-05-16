import sys
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 100000)


class Dataset:
    def __init__(self, location, cols_to_split=None, cols_to_combine=None, **kwargs):
        self.location = location
        self.df = None
        self.developers = []
        self.cols_to_split = cols_to_split
        self.cols_to_combine = cols_to_combine

        # todo put in try/except loop
        self.read()

    def __str__(self):
        return self.df.head(100).to_string()

    def read(self):
        print('reading datset')
        if not type(self.location) == Path:
            self.location = Path(self.location)

        self.df = pd.read_csv(self.location)
        self.df = self.df.loc[self.df['Country'] == 'United States of America']
        print('done reading dataset')

        self.prep()

    def prep(self):
        print('preparing dataset')
        self.split_columns()

        developer_category_shortened = {
            'I am a developer by profession': 'Pro',
            'I am a student who is learning to code': 'Student',
            'I am not primarily a developer, but I write code sometimes as part of my work': 'Part of job',
            'I code primarily as a hobby': 'Hobby',
            'I used to be a developer by profession, but no longer am': 'Former pro',
            'None of these': 'None'
        }

        self.df['MainBranch'] = self.df['MainBranch'].replace(developer_category_shortened)

        self.df['YearsCode'] = self.df['YearsCode'].replace({'Less than 1 year': 1, 'More than 50 years': 50})
        self.df['YearsCode'] = pd.to_numeric(self.df['YearsCode'], errors='raise')
        self.df['YearsCode'] = self.df['YearsCode'].fillna(1)

        self.df.loc[:, 'YearsCodeBin'] = pd.qcut(self.df['YearsCode'], 4,
                                                 labels=["junior", "mid-junior", "mid-senior", "senior"])

        # todo: finish building combine columns functionality
        # self.combine_columns()

    def split_columns(self):
        if self.cols_to_split is not None:
            s = [self.df[col].str.get_dummies(sep=';').add_prefix(f'{col.lower()}_') for col in self.cols_to_split]
            self.df = pd.concat([self.df.drop(self.cols_to_split, axis='columns')] + s, axis=1)

    def combine_columns(self):
        if self.cols_to_combine is not None:
            s = [self.df[col].str.get_dummies(sep=';').add_prefix(f'{col.lower()}_') for col in self.cols_to_split]
            self.df = pd.concat([self.df.drop(self.cols_to_split, axis='columns')] + s, axis=1)

    def summarize(self):
        print(self.df.info())


class Analysis:
    def __init__(self, name=None):
        self.name = name
        self._dataset = None
        self.segmentation_pattern = None
        self.df_sample = None
        self.hc_linkage = None

    @property
    def developers(self):
        return self.dataset.developers

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        if type(value) == Dataset:
            self._dataset = value
        else:
            self._dataset = Dataset(
                value['location'], cols_to_split=value['cols_to_split'], cols_to_combine=value['cols_to_combine']
            )

    def hc_preview(self, sample_size=2000, method='average', optimal_ordering=False, seed=1, columns=None):
        print('preparing preview of hierarchical clustering outputs')
        # sample df
        self.df_sample = self.dataset.df.sample(n=sample_size, random_state=seed)

        # filter to columns of interest
        self.df_sample = self.df_sample[columns]

        # add dummy column for left join
        self.df_sample.loc[:, 'dummy'] = 'dummy'

        # left join
        df = self.df_sample.merge(self.df_sample, on='dummy', how='left')

        # calculate distance
        for column in columns:
            df.loc[df['{}_x'.format(column)] != df['{}_y'.format(column)], "{}_dist".format(column)] = 1
            df["{}_dist".format(column)] = df["{}_dist".format(column)].fillna(0)

        df['Distance'] = df[["{}_dist".format(_) for _ in columns]].sum(axis=1)

        # create initial array
        array = df['Distance'].values.reshape((sample_size, sample_size))

        # cluster array
        d = sch.distance.pdist(array)
        self.hc_linkage = sch.linkage(d, method=method, optimal_ordering=optimal_ordering)
        ind = sch.fcluster(self.hc_linkage, d.max() / 2, 'distance')

        new_order = list(np.argsort(ind))

        new_array = array[new_order]
        new_array = new_array[:, new_order]

        # display initial and clustered array
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(array, ax=axes[0], cmap="Blues_r").set(title='Respondent distances before clustering')
        sns.heatmap(new_array, ax=axes[1], cmap="Blues_r").set(title='Respondent distances after clustering')

        img_path = Path(__file__).parent.parent / 'images'
        fig.savefig(img_path / '{0}_{1}.png'.format(self.name, method))  # save the figure to file
        plt.close(fig)  # close the figure window

    def map_hc_segments(self, num_clusters):
        labels = sch.fcluster(self.hc_linkage, num_clusters, criterion='maxclust')
        self.df_sample.loc[:, 'Segment'] = labels

        unique_labels = np.unique(labels)

        columns_of_interest = [_ for _ in self.df_sample.columns if not _ in ['dummy', 'Segment']]

        for label in unique_labels:
            fig, axes = plt.subplots(1, len(columns_of_interest), figsize=(16, 8))

            for i, col in enumerate(columns_of_interest):
                ax = axes[i]

                if len(self.df_sample[col].unique()) < 10:
                    # todo: clean this up a bit
                    if len(self.df_sample[col].unique()) == 2 and 1 in self.df_sample[col].unique():
                        self.df_sample[col].replace({'0': np.nan, 0: np.nan})

                        toplot = self.df_sample \
                            .loc[self.df_sample['Segment'] == label, col] \
                            .value_counts(sort=False, normalize=True) \
                            .to_frame().T

                        toplot = toplot.reindex(sorted(toplot.columns, reverse=True), axis=1)

                        toplot.plot(kind='bar', stacked=True, ax=ax, legend=False, rot=45)

                    elif 'junior' in self.df_sample[col].unique():

                        toplot = self.df_sample \
                            .loc[self.df_sample['Segment'] == label, col] \
                            .value_counts(sort=False, normalize=True) \
                            .to_frame().T

                        toplot = toplot.reindex(['junior', 'mid-junior', 'mid-senior', 'senior'], axis=1)

                        toplot.plot(kind='bar', stacked=True, ax=ax, legend=False, rot=45)

                        for p, name in zip(ax.patches, toplot.columns):
                            width, height = p.get_width(), p.get_height()
                            x, y = p.get_xy()
                            ax.text(x + width / 2,
                                    y + height / 2,
                                    '{}'.format(name),
                                    horizontalalignment='center',
                                    verticalalignment='center')

                    else:
                        toplot = self.df_sample \
                            .loc[self.df_sample['Segment'] == label, col] \
                            .value_counts(normalize=True) \
                            .to_frame().T

                        toplot = toplot.reindex(['None', 'Hobby', 'Student', 'Part of job', 'Pro', 'Former pro'],
                                                axis=1)

                        toplot.plot(kind='bar', stacked=True, ax=ax, rot=45, legend=False)

                        for p, name in zip(ax.patches, toplot.columns):
                            width, height = p.get_width(), p.get_height()
                            x, y = p.get_xy()
                            ax.text(x + width / 2,
                                    y + height / 2,
                                    '{}'.format(name),
                                    horizontalalignment='center',
                                    verticalalignment='center')

            fig.subplots_adjust(bottom=0.3)
            img_path = Path(__file__).parent.parent / 'images/segment_summaries'
            plt.tight_layout()
            fig.savefig(img_path / 'analysis_{0}_segment_{1}.png'.format(self.name, label))  # save the figure to file
            plt.close(fig)  # close the figure window


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / 'data'
    infolder_2021 = data_path / 'stack-overflow-developer-survey-2021'

    schemafile_2021 = infolder_2021 / r'survey_results_schema.csv'

    dataset_config = {
        'location': infolder_2021 / r'survey_results_public.csv',
        'cols_to_split': [
            'LearnCode', 'DevType', 'LanguageHaveWorkedWith', 'LanguageWantToWorkWith',
            'DatabaseHaveWorkedWith', 'DatabaseWantToWorkWith', 'PlatformHaveWorkedWith', 'PlatformWantToWorkWith',
            'WebframeHaveWorkedWith', 'WebframeWantToWorkWith', 'MiscTechHaveWorkedWith', 'MiscTechWantToWorkWith',
            'ToolsTechHaveWorkedWith', 'ToolsTechWantToWorkWith', 'NEWCollabToolsHaveWorkedWith',
            'NEWCollabToolsWantToWorkWith', 'NEWStuck', 'NEWSOSites', 'Ethnicity', 'MentalHealth'
        ],
        'cols_to_combine': {
            'Combined location': ['Country', 'US_State', 'UK_Country']
        }
    }

    # Question 1: Which developers should we recruit?
    analysis_recruit = Analysis(name='recruit')
    analysis_recruit.dataset = dataset_config

    cluster_cols_recruit = [
        'MainBranch',
        'YearsCodeBin',

        # 'devtype_Academic researcher',
        # 'devtype_Data scientist or machine learning specialist',
        # 'devtype_Database administrator',
        # 'devtype_Designer',
        # 'devtype_DevOps specialist',
        # 'devtype_Developer, QA or test',
        # 'devtype_Developer, back-end',
        # 'devtype_Developer, desktop or enterprise applications',
        # 'devtype_Developer, embedded applications or devices',
        # 'devtype_Developer, front-end',
        # 'devtype_Developer, full-stack',
        # 'devtype_Developer, game or graphics',
        # 'devtype_Developer, mobile',
        # 'devtype_Educator',
        # 'devtype_Engineer, data',
        # 'devtype_Engineer, site reliability',
        # 'devtype_Engineering manager',
        # 'devtype_Marketing or sales professional',
        # 'devtype_Product manager',
        # 'devtype_Scientist',
        # 'devtype_Senior Executive (C-Suite, VP, etc.)',
        # 'devtype_Student',
        # 'devtype_System administrator'

        'languagehaveworkedwith_Python',
        'languagehaveworkedwith_C',
        'languagehaveworkedwith_C++',
        'languagehaveworkedwith_Scala',
        'languagehaveworkedwith_Perl',
        'languagehaveworkedwith_SQL',
    ]

    analysis_recruit.hc_preview(columns=cluster_cols_recruit, method='complete')
    # analysis_recruit.hc_preview(columns=cluster_cols_recruit, method='average')
    # analysis_recruit.hc_preview(columns=cluster_cols_recruit, method='single')
    # analysis_recruit.hc_preview(columns=cluster_cols_recruit, method='median')
    # analysis_recruit.hc_preview(columns=cluster_cols_recruit, method='ward')
    # analysis_recruit.hc_preview(columns=cluster_cols_recruit, method='weighted')

    analysis_recruit.map_hc_segments(5)

    sys.exit()

    # Question 2: Which developers can help us improve our products??
    analysis_improve = Analysis(name='improve')
    analysis_improve.dataset = dataset_config

    cluster_cols_improve = [
        'MainBranch',
        'YearsCodeBin',

        # 'devtype_Academic researcher',
        # 'devtype_Data scientist or machine learning specialist',
        # 'devtype_Database administrator',
        # 'devtype_Designer',
        # 'devtype_DevOps specialist',
        # 'devtype_Developer, QA or test',
        # 'devtype_Developer, back-end',
        # 'devtype_Developer, desktop or enterprise applications',
        'devtype_Developer, embedded applications or devices',
        # 'devtype_Developer, front-end',
        # 'devtype_Developer, full-stack',
        # 'devtype_Developer, game or graphics',
        # 'devtype_Developer, mobile',
        # 'devtype_Educator',
        # 'devtype_Engineer, data',
        'devtype_Engineer, site reliability',
        # 'devtype_Engineering manager',
        # 'devtype_Marketing or sales professional',
        # 'devtype_Product manager',
        # 'devtype_Scientist',
        'devtype_Senior Executive (C-Suite, VP, etc.)',
        # 'devtype_Student',
        # 'devtype_System administrator'

        # 'languagehaveworkedwith_Python',
        # 'languagehaveworkedwith_C',
        # 'languagehaveworkedwith_C++',
        # 'languagehaveworkedwith_Scala',
        # 'languagehaveworkedwith_Perl',
        # 'languagehaveworkedwith_SQL',
    ]

    # analysis_improve.hc_preview(columns=cluster_cols_improve, method='complete')
    # analysis_improve.hc_preview(columns=cluster_cols_improve, method='weighted')
    # analysis_improve.hc_preview(columns=cluster_cols_improve, method='average')
    # analysis_improve.hc_preview(columns=cluster_cols_improve, method='single')
    # analysis_improve.hc_preview(columns=cluster_cols_improve, method='median')
    # analysis_improve.hc_preview(columns=cluster_cols_improve, method='ward')

    # sys.exit()

    # Question 3: Which developers influence purchasing decisions?
    analysis_influence = Analysis(name='influence')
    analysis_influence.dataset = dataset_config
    # todo
    cluster_cols_influence = [
        'MainBranch',
        'YearsCodeBin',

        'devtype_Academic researcher',
        'devtype_Data scientist or machine learning specialist',
        # 'devtype_Database administrator',
        # 'devtype_Designer',
        # 'devtype_DevOps specialist',
        # 'devtype_Developer, QA or test',
        # 'devtype_Developer, back-end',
        # 'devtype_Developer, desktop or enterprise applications',
        'devtype_Developer, embedded applications or devices',
        # 'devtype_Developer, front-end',
        # 'devtype_Developer, full-stack',
        # 'devtype_Developer, game or graphics',
        # 'devtype_Developer, mobile',
        # 'devtype_Educator',
        'devtype_Engineer, data',
        'devtype_Engineer, site reliability',
        # 'devtype_Engineering manager',
        'devtype_Marketing or sales professional',
        'devtype_Product manager',
        # 'devtype_Scientist',
        'devtype_Senior Executive (C-Suite, VP, etc.)',
        # 'devtype_Student',
        # 'devtype_System administrator'

        # 'languagehaveworkedwith_Python',
        # 'languagehaveworkedwith_C',
        # 'languagehaveworkedwith_C++',
        # 'languagehaveworkedwith_Scala',
        # 'languagehaveworkedwith_Perl',
        # 'languagehaveworkedwith_SQL',
    ]

    analysis_influence.hc_preview(columns=cluster_cols_influence, method='complete')
    analysis_influence.hc_preview(columns=cluster_cols_influence, method='weighted')
    analysis_influence.hc_preview(columns=cluster_cols_influence, method='average')
    analysis_influence.hc_preview(columns=cluster_cols_influence, method='single')
    analysis_influence.hc_preview(columns=cluster_cols_influence, method='median')
    analysis_influence.hc_preview(columns=cluster_cols_influence, method='ward')
