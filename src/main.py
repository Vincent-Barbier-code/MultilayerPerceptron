"""Main module"""

import data_processing.data_extractor as extractor
from data_visualization import plots
from terminal.arg_parsing import arg_parsing


def main():
    """Main function"""
    args = arg_parsing()

    csv_file = args.file if args.file else "data/data.csv"
    if not args.file:
        print("No file path provided, using default file path: data/data.csv")

    extracted_data = extractor.extract_data(csv_file)
    if extracted_data is None:
        return

    dataframe_normalized = extractor.normalize_dataframe(extracted_data)
    dataframe_normalized[1] = dataframe_normalized[1].map({"M": 1, "B": 0})
    # plots.create_scatter_plots(dataframe_normalized)
    # plots.heat_map(dataframe_normalized)

    if args.sc:
        plots.create_scatter_plots(dataframe_normalized)
    if args.hm:
        plots.heat_map(dataframe_normalized)

if __name__ == "__main__":
    main()
