"""
module with helper functions used in the tests
"""
import urllib.request
from pathlib import Path


def clean_dir_path(path):
    """
    Remove directory identified by path and all its content recursively.
    Very similar to bash 'rm -r path'
    """
    if path.exists():
        for sub_dir in path.iterdir():
            if sub_dir.is_dir():
                clean_dir_path(sub_dir)
            else:
                sub_dir.unlink()

        path.rmdir()

    else:
        print(f'{path} does not exists, nothing to clear.')


def create_test_dir_tree(base_dir):
    """
    Create the correct directory tree used in the tests.
    """
    tmp_test_dir = base_dir.joinpath('tmp_test')

    data_dir = tmp_test_dir.joinpath('data')
    reference_dir = tmp_test_dir.joinpath('reference')

    tmp_test_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)

    return tmp_test_dir, data_dir, reference_dir


def download_from_url_list(url_list, path):
    """
    Download urls list in a defined destination directory.
    """
    output_list = []

    for url in url_list:
        file_name = url.split('/')[-1]
        destination = f'{path}/{file_name}'

        print(f'Download {url} ...', end='\r')
        urllib.request.urlretrieve(url, destination)
        print(f'Download {url} in {destination}: done!')

        output_list.append(Path(destination))

    return output_list


def download_test_data(path):
    """
    Download .root files used for the tests in a dedicated directory.
    """
    data_url = 'https://raw.github.com/hipe4ml/hipe4ml_tests/master/Dplus7TeV/Bkg_Dpluspp7TeV_pT_1_50.root'
    prompt_url = 'https://raw.github.com/hipe4ml/hipe4ml_tests/master/Dplus7TeV/Prompt_Dpluspp7TeV_pT_1_50.root'
    data_pq_url = ('https://raw.github.com/hipe4ml/hipe4ml_tests/master/Dplus7TeV/'
                   'Bkg_Dpluspp7TeV_pT_1_50.parquet.gzip')
    prompt_pq_url = ('https://raw.github.com/hipe4ml/hipe4ml_tests/master/Dplus7TeV/'
                     'Prompt_Dpluspp7TeV_pT_1_50.parquet.gzip')

    urls = [data_url, prompt_url, data_pq_url, prompt_pq_url]

    return download_from_url_list(urls, path)


def download_test_references(path):
    """
    Download .pickle files used as reference for the tests in dedicated
    directory.
    """

    data_slice_url = 'https://github.com/hipe4ml/hipe4ml_tests/raw/master/references/data_slice.pickle'
    promp_slice_url = 'https://github.com/hipe4ml/hipe4ml_tests/raw/master/references/prompt_slice.pickle'
    reference_dict_url = 'https://github.com/hipe4ml/hipe4ml_tests/raw/master/references/reference_dict.pickle'

    urls = [data_slice_url, promp_slice_url, reference_dict_url]

    return download_from_url_list(urls, path)
