"""Main module"""

import data_processing.data_extractor as extractor


def main():
    """Main function"""
    csv_file = "data/data.csv"
    extracted_data = extractor.extract_data(csv_file)

    # print(extracted_data)
    # first_column = extractor.extract_column(extracted_data, 0)
    # print(first_column)

    for i in range(1, len(extracted_data.columns)):
        name = extractor.column_names(i)
        print(name)

    # dataframe = extractor.create_dataframe(extracted_data, 1)
    # print(dataframe)


if __name__ == "__main__":
    main()
