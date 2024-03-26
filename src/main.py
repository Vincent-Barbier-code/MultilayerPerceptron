"""Main module"""

# import data_processing.data_extractor as extractor
# from data_visualization import plots
from terminal.arg_parsing import arg_parsing
from terminal.arg_parsing import execute

# import sys
# sys.path.append('./src')
# sys.path.append('./tests')
# print(sys.path)

def main():
    # print(extractor.extract_data("data/data.csv"))
    """Main function"""
    args = arg_parsing()
    execute(args)
    
    # csv_file = args.file if args.file else "data/data.csv"

    # extracted_data = extractor.extract_data(csv_file)
    # if extracted_data is None:
    #     return

    # dataframe_normalized = extractor.normalize_dataframe(extracted_data)
    # dataframe_normalized[1] = dataframe_normalized[1].map({"M": 1, "B": 0})
    # # plots.create_scatter_plots(dataframe_normalized)
    # # plots.heat_map(dataframe_normalized)

    # if args.sc:
    #     plots.create_scatter_plots(dataframe_normalized)
    # if args.hm:
    #     plots.heat_map(dataframe_normalized)

if __name__ == "__main__":
    main()
