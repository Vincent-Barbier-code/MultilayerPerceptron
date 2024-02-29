"""Main module"""

import data_processing.data_extractor as extractor
from data_visualization import plots


def main():
    """Main function"""
    csv_file = "data/data.csv"
    extracted_data = extractor.extract_data(csv_file)
    if extracted_data is None:
        return
    # print(extracted_data)
    # first_column = extractor.extract_column(extracted_data, 0)
    # print(first_column)

    # for i in range(1, len(extracted_data.columns)):
    #     name = extractor.column_names(i)
    #     print(name)

    # dataframe = extractor.create_dataframe(extracted_data, 1)

    dataframe_normalized = extractor.normalize_dataframe(extracted_data)
    dataframe_normalized[1] = dataframe_normalized[1].map({"M": 1, "B": 0})
    # plots.create_scatter_plots(dataframe_normalized)
    plots.heat_map(dataframe_normalized)


if __name__ == "__main__":
    main()
