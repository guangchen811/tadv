from cadv_exploration.dataset_download import download_competition


def main():
    # valid_list_sort_by = [
    #     'hotness', 'commentCount', 'dateCreated', 'dateRun', 'relevance',
    #     'scoreAscending', 'scoreDescending', 'viewCount', 'voteCount'
    # ]
    download_competition("playground-series-s4e10", page_size=100, sort_by="scoreAscending")


if __name__ == "__main__":
    main()
